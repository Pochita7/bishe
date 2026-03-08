"""
工具函数 - 为智能体提供搜索、文件处理等能力
融合 MetaGPT 的 Action 概念：每个工具函数对应一个原子 Action
适配 Microsoft Agent Framework (agent-framework)
"""
import json
import os
import base64
import warnings
from typing import Optional

warnings.filterwarnings('ignore', message='.*duckduckgo_search.*')
warnings.filterwarnings('ignore', message='.*GuessedAtParserWarning.*')

from gaia_solver.config import GAIA_ATTACHMENTS_DIR, WORK_DIR, API_KEY, BASE_URL, get_openai_client, VISION_MODEL


# ============================================================
# 1. 网络搜索工具
# ============================================================
def search_web(query: str) -> str:
    """
    搜索互联网信息。DuckDuckGo + Wikipedia。

    Args:
        query: 搜索查询字符串
    Returns:
        搜索结果文本
    """
    results_text = []

    # 1. DuckDuckGo
    try:
        from ddgs import DDGS
        ddg_results = list(DDGS().text(query, max_results=5))
        for r in ddg_results:
            title = r.get('title', '')
            body = r.get('body', '')[:200]  # 截断摘要
            href = r.get('href', '')
            results_text.append(f"[{title}] ({href})\n{body}")
    except Exception as e:
        results_text.append(f"(DuckDuckGo error: {e})")

    # 2. Wikipedia 补充（只取一条精简摘要）
    try:
        import wikipedia
        wikipedia.set_lang('en')
        search_results = wikipedia.search(query, results=2)
        if search_results:
            try:
                page = wikipedia.page(search_results[0], auto_suggest=False)
                results_text.append(f"[Wikipedia: {page.title}]\n{page.summary[:1000]}")
            except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError):
                pass
    except Exception:
        pass

    if not results_text:
        return "No results. Try fetch_webpage with a specific URL."

    return "\n\n".join(results_text[:5])


def _get_wiki_section_from_html(page_title: str, section_name: str) -> str | None:
    """
    回退方案: 通过 Wikipedia HTML 抓取章节内容（含表格数据）。
    当 wikipedia 包的 .content 丢失表格数据时使用。
    """
    import requests
    from bs4 import BeautifulSoup

    url = f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}"
    try:
        resp = requests.get(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }, timeout=20)
        resp.raise_for_status()
    except Exception:
        return None

    soup = BeautifulSoup(resp.text, 'html.parser')

    # 查找目标章节标题 (h2/h3/h4)
    # 现代 Wikipedia 结构: <div class="mw-heading mw-heading2"><h2 id="...">Title</h2>...</div>
    # 旧版结构: <h2><span class="mw-headline">Title</span></h2>
    target_container = None
    heading_level = None
    for heading in soup.find_all(['h2', 'h3', 'h4']):
        headline = heading.get_text(strip=True).replace('[edit]', '').replace('[编辑]', '').strip()
        if section_name.lower() in headline.lower():
            # 检查父元素是否是 mw-heading div (现代结构)
            parent = heading.parent
            if parent and parent.name == 'div' and 'mw-heading' in ' '.join(parent.get('class', [])):
                target_container = parent
            else:
                target_container = heading
            heading_level = heading.name  # 'h2', 'h3', etc.
            break

    if not target_container:
        return None

    # 收集从 container 到下一个同级标题之间的所有内容
    content_parts = []
    for sibling in target_container.find_next_siblings():
        # 检查是否碰到 mw-heading div (现代结构的标题)
        if sibling.name == 'div' and 'mw-heading' in ' '.join(sibling.get('class', [])):
            inner_h = sibling.find(['h2', 'h3', 'h4'])
            if inner_h and inner_h.name <= heading_level:
                break  # 同级或更高级，停止
            # 子标题，记录标题文本并继续
            sub_text = inner_h.get_text(strip=True).replace('[edit]', '').strip() if inner_h else ''
            if sub_text:
                content_parts.append(f"\n### {sub_text}")
            continue
        # 旧版结构: 直接碰到 h-tag
        if sibling.name in ['h2'] or (sibling.name == heading_level and heading_level != 'h2'):
            break
        # 跳过编辑按钮 span
        if sibling.name == 'span' and 'mw-editsection' in ' '.join(sibling.get('class', [])):
            continue
        if sibling.name == 'table':
            # 提取表格 → 文本行
            rows = sibling.find_all('tr')
            for row in rows:
                cells = [cell.get_text(separator=' ', strip=True) for cell in row.find_all(['td', 'th'])]
                if cells and any(c for c in cells):
                    content_parts.append(' | '.join(cells))
        elif sibling.name in ['h3', 'h4']:
            # 子标题
            text = sibling.get_text(strip=True)
            content_parts.append(f"\n### {text}")
        elif sibling.name == 'ul':
            # 列表
            for li in sibling.find_all('li', recursive=False):
                content_parts.append(f"- {li.get_text(separator=' ', strip=True)}")
        else:
            text = sibling.get_text(separator='\n', strip=True)
            if text:
                content_parts.append(text)

    result = '\n'.join(content_parts)
    return result[:8000] if result else None


def search_wikipedia(topic: str, section: str = "") -> str:
    """
    搜索 Wikipedia 特定主题页面，返回详细内容。
    可以指定特定章节名进行精确查找。

    Args:
        topic: Wikipedia 页面名称或主题（英文），如 'Eliud_Kipchoge', 'Moon'
        section: 可选，要查找的特定章节名，如 'Discography', 'Orbit'
    Returns:
        页面摘要和相关章节内容
    """
    # 方案 1: 使用 wikipedia 包（更可靠）
    try:
        import wikipedia
        wikipedia.set_lang('en')
        try:
            page = wikipedia.page(topic, auto_suggest=True)
        except wikipedia.exceptions.DisambiguationError as e:
            # 取消歧义的第一个选项
            page = wikipedia.page(e.options[0], auto_suggest=False)
        except wikipedia.exceptions.PageError:
            # 搜索再试
            results = wikipedia.search(topic, results=3)
            if results:
                page = wikipedia.page(results[0], auto_suggest=False)
            else:
                return f"Wikipedia page '{topic}' not found. Try search_web instead."
        
        content = page.content
        if section:
            # 在全文中查找章节
            import re
            pattern = re.compile(rf'==\s*{re.escape(section)}\s*==', re.IGNORECASE)
            match = pattern.search(content)
            if match:
                start = match.start()
                # 找下一个同级或更高级的标题
                next_heading = re.search(r'\n==\s', content[start+len(match.group()):])
                end = start + len(match.group()) + next_heading.start() if next_heading else len(content)
                section_text = content[start:end][:6000]

                # 检查: 如果提取的章节内容过短（表格数据丢失），回退到 HTML 解析
                # wikipedia 包的 .content 会丢弃 HTML 表格，导致 Discography 等章节为空
                plain_text = re.sub(r'=+\s*.+?\s*=+', '', section_text).strip()
                if len(plain_text) < 200:
                    html_content = _get_wiki_section_from_html(page.title, section)
                    if html_content and len(html_content) > len(plain_text):
                        return f"# {page.title} > {section}\n\n{html_content}"

                return f"# {page.title} > {section}\n\n{section_text}"
            else:
                # 列出可用章节
                headings = re.findall(r'==\s*(.+?)\s*==', content)
                return f"Section '{section}' not found. Available: {', '.join(headings[:20])}"
        
        # 返回全文（限长度）
        return f"# {page.title}\n\n{content[:5000]}"
    except Exception as e:
        pass
    
    # 方案 2: 使用 wikipediaapi 作为后备
    try:
        import wikipediaapi
        wiki = wikipediaapi.Wikipedia('GaiaSolver/1.0 (gaia@example.com)', 'en')

        page = wiki.page(topic.replace(" ", "_"))
        if not page.exists():
            alt_topic = topic.replace("_", " ").title().replace(" ", "_")
            page = wiki.page(alt_topic)

        if not page.exists():
            return f"Wikipedia page '{topic}' not found. Try search_web instead."

        if section:
            found = _find_section_recursive(page.sections, section)
            if found:
                content = f"# {page.title} > {found.title}\n\n{found.text[:5000]}"
                for sub in found.sections:
                    content += f"\n\n### {sub.title}\n{sub.text[:2000]}"
                return content
            else:
                all_sections = _list_sections_recursive(page.sections)
                return f"Section '{section}' not found. Available sections: {', '.join(all_sections)}"

        content_parts = [f"# {page.title}\n\n{page.summary[:3000]}"]
        total_len = len(content_parts[0])
        _collect_sections(page.sections, content_parts, total_len, max_total=10000, depth=0)

        return "\n".join(content_parts)
    except Exception as e:
        return f"Wikipedia error: {str(e)}"


def _find_section_recursive(sections, target: str):
    """递归查找匹配的章节"""
    target_lower = target.lower()
    for sec in sections:
        if target_lower in sec.title.lower():
            return sec
        found = _find_section_recursive(sec.sections, target)
        if found:
            return found
    return None


def _list_sections_recursive(sections, depth=0) -> list:
    """递归列出所有章节标题"""
    result = []
    for sec in sections:
        prefix = "  " * depth
        result.append(f"{prefix}{sec.title}")
        result.extend(_list_sections_recursive(sec.sections, depth + 1))
    return result


def _collect_sections(sections, content_parts, total_len, max_total=8000, depth=0):
    """递归收集章节内容"""
    for sec in sections:
        if total_len > max_total:
            break
        section_text = sec.text[:1000] if sec.text else ""
        if section_text:
            prefix = "#" * (depth + 2)
            entry = f"\n{prefix} {sec.title}\n{section_text}"
            content_parts.append(entry)
            total_len += len(entry)
        # 递归子章节
        _collect_sections(sec.sections, content_parts, total_len, max_total, depth + 1)


def fetch_webpage(url: str) -> str:
    """
    抓取网页内容并提取正文文本。
    适用于需要访问特定 URL 获取信息的场景。

    Args:
        url: 要抓取的网页 URL
    Returns:
        网页的纯文本内容（限制长度）
    """
    try:
        import requests
        from bs4 import BeautifulSoup

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        resp = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')

        # 移除脚本和样式
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()

        # 对 Wikipedia 页面提取含表格的更丰富文本
        if 'wikipedia.org' in url:
            parts = []
            body = soup.find('div', class_='mw-parser-output') or soup.body
            if body:
                for el in body.children:
                    tag = getattr(el, 'name', None)
                    if tag in ['h2', 'h3', 'h4']:
                        headline = el.get_text(strip=True)
                        parts.append(f"\n{'#' * (int(tag[1]))} {headline}")
                    elif tag == 'table':
                        for row in el.find_all('tr'):
                            cells = [c.get_text(separator=' ', strip=True) for c in row.find_all(['td', 'th'])]
                            if cells and any(c for c in cells):
                                parts.append(' | '.join(cells))
                    elif tag == 'ul':
                        for li in el.find_all('li', recursive=False):
                            parts.append(f"- {li.get_text(separator=' ', strip=True)}")
                    elif tag:
                        t = el.get_text(separator='\n', strip=True)
                        if t:
                            parts.append(t)
                text = '\n'.join(parts)
            else:
                text = soup.get_text(separator="\n", strip=True)
            max_len = 12000
        else:
            text = soup.get_text(separator="\n", strip=True)
            max_len = 4000

        # 给学术/参考页面更大的长度限制
        if any(d in url for d in ['arxiv.org', 'github.com', 'cornell.edu', 'merriam-webster.com']):
            max_len = 6000
        if len(text) > max_len:
            text = text[:max_len] + "\n...(truncated)"
        
        # 如果内容太少（可能是 JS 渲染页面），自动降级到浏览器
        if len(text.strip()) < 100:
            return browse_webpage(url)
        return text
    except Exception as e:
        # HTTP 请求失败时自动降级到浏览器
        try:
            return browse_webpage(url)
        except Exception:
            return f"Webpage fetch error: {str(e)}"


def _extract_wiki_topics(query: str) -> list:
    """从查询中提取可能的 Wikipedia 页面名"""
    import re
    topics = []

    # 提取引号中的内容
    quoted = re.findall(r'"([^"]+)"', query)
    topics.extend(quoted)

    # 提取大写开头的连续词组（可能是专有名词）
    proper_nouns = re.findall(r'(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', query)
    topics.extend(proper_nouns)

    # 如果没找到，用前几个有意义的词
    if not topics:
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'of', 'for',
                      'in', 'on', 'at', 'to', 'and', 'or', 'how', 'what', 'who',
                      'when', 'where', 'many', 'much', 'which', 'do', 'does'}
        words = [w for w in query.split() if w.lower() not in stop_words and len(w) > 2]
        if words:
            topics.append(" ".join(words[:3]))

    return topics[:5]


# ============================================================
# 1.5 浏览器交互工具（Playwright）
# ============================================================

# 全局浏览器实例管理（避免重复启动）
_browser_context = {"browser": None, "playwright": None, "page": None}


def _get_browser_page():
    """获取或创建 Playwright 浏览器页面（单例模式）"""
    ctx = _browser_context
    if ctx["page"] is None or ctx["browser"] is None:
        try:
            from playwright.sync_api import sync_playwright
            if ctx["playwright"] is None:
                ctx["playwright"] = sync_playwright().start()
            ctx["browser"] = ctx["playwright"].chromium.launch(
                headless=True,
                args=['--disable-gpu', '--no-sandbox', '--disable-dev-shm-usage']
            )
            ctx["page"] = ctx["browser"].new_page(
                viewport={"width": 1280, "height": 720},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
            )
        except Exception as e:
            return None, f"Browser init error: {e}"
    return ctx["page"], None


def _cleanup_browser():
    """关闭浏览器，释放资源"""
    ctx = _browser_context
    try:
        if ctx["page"]:
            ctx["page"].close()
        if ctx["browser"]:
            ctx["browser"].close()
        if ctx["playwright"]:
            ctx["playwright"].stop()
    except Exception:
        pass
    ctx["page"] = ctx["browser"] = ctx["playwright"] = None


def browse_webpage(url: str, question: str = "") -> str:
    """
    使用真实浏览器打开网页，支持 JavaScript 渲染。
    自动提取文本内容，对于复杂页面还会截图用视觉模型分析。
    比 fetch_webpage 更强大，适合动态页面或需要 JS 渲染的网站。

    Args:
        url: 要访问的网页 URL
        question: 关于网页的具体问题（可选，如果提供则会截图让视觉模型分析）
    Returns:
        网页文本内容，如果提供 question 则附加视觉分析结果
    """
    page, err = _get_browser_page()
    if err:
        return err

    try:
        # 导航到 URL — 先尝试 networkidle，超时则降级为 domcontentloaded
        try:
            page.goto(url, wait_until="networkidle", timeout=20000)
        except Exception:
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=15000)
            except Exception:
                page.goto(url, wait_until="commit", timeout=10000)
        page.wait_for_timeout(1500)  # 额外等待 JS 渲染

        # 提取文本内容
        text_content = page.evaluate("""() => {
            // 移除无用元素
            const removeSelectors = ['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe', '.ad', '.ads', '.advertisement'];
            removeSelectors.forEach(sel => {
                document.querySelectorAll(sel).forEach(el => el.remove());
            });
            return document.body ? document.body.innerText : '';
        }""")

        # 截取合理长度
        max_len = 6000 if any(d in url for d in ['wikipedia.org', 'arxiv.org', 'github.com']) else 4000
        if len(text_content) > max_len:
            text_content = text_content[:max_len] + "\n...(truncated)"

        result = f"[Browser: {page.title()}]\n{text_content}"

        # 如果提供了 question，截图+视觉模型分析
        if question:
            try:
                screenshot_path = os.path.join(WORK_DIR, "browser_screenshot.png")
                page.screenshot(path=screenshot_path, full_page=False)

                import base64 as b64mod
                with open(screenshot_path, 'rb') as f:
                    img_b64 = b64mod.b64encode(f.read()).decode('utf-8')

                from openai import OpenAI
                client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
                resp = client.chat.completions.create(
                    model=VISION_MODEL,
                    messages=[{
                        'role': 'user',
                        'content': [
                            {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{img_b64}'}},
                            {'type': 'text', 'text': f'Look at this webpage screenshot. {question}'}
                        ]
                    }],
                    max_tokens=1000,
                    timeout=60
                )
                vision_answer = resp.choices[0].message.content
                result += f"\n\n[Vision analysis]: {vision_answer}"
            except Exception as e:
                result += f"\n\n[Vision analysis failed: {e}]"

        return result

    except Exception as e:
        return f"Browser error: {str(e)}"


def browser_click(selector: str, wait_after: int = 2000) -> str:
    """
    在当前浏览器页面上点击指定元素，然后返回页面文本内容。
    需要先用 browse_webpage 打开页面。

    Args:
        selector: CSS 选择器（如 'button.submit', '#next', 'a[href*=changelog]'）
        wait_after: 点击后等待毫秒数（默认 2000）
    Returns:
        点击后页面的文本内容
    """
    page, err = _get_browser_page()
    if err:
        return err

    try:
        if not page.url or page.url == "about:blank":
            return "Error: No page loaded. Use browse_webpage first."

        # 等待元素出现
        page.wait_for_selector(selector, timeout=10000)
        page.click(selector)
        page.wait_for_timeout(wait_after)

        # 提取更新后的文本
        text = page.evaluate("""() => {
            ['script','style','nav','header','footer','aside'].forEach(tag => {
                document.querySelectorAll(tag).forEach(el => el.remove());
            });
            return document.body ? document.body.innerText : '';
        }""")

        if len(text) > 4000:
            text = text[:4000] + "\n...(truncated)"

        return f"[After click '{selector}' on {page.url}]\n{text}"
    except Exception as e:
        # 尝试列出可点击元素帮助调试
        try:
            links = page.evaluate("""() => {
                const items = [];
                document.querySelectorAll('a, button, [role=button], [onclick]').forEach(el => {
                    const text = (el.textContent || '').trim().slice(0, 50);
                    const href = el.getAttribute('href') || '';
                    if (text) items.push({text, href, tag: el.tagName});
                });
                return items.slice(0, 20);
            }""")
            hint = "\n".join([f"  {l['tag']}: '{l['text']}' href={l['href']}" for l in links])
            return f"Click failed: {e}\nAvailable clickable elements:\n{hint}"
        except Exception:
            return f"Click error: {str(e)}"


def browser_scroll(direction: str = "down") -> str:
    """
    在当前浏览器页面滚动，然后返回可见区域的文本内容。

    Args:
        direction: 滚动方向，'down'（向下）或 'up'（向上）
    Returns:
        滚动后可见区域的文本内容
    """
    page, err = _get_browser_page()
    if err:
        return err

    try:
        if not page.url or page.url == "about:blank":
            return "Error: No page loaded. Use browse_webpage first."

        # 滚动
        scroll_amount = 600 if direction == "down" else -600
        page.evaluate(f"window.scrollBy(0, {scroll_amount})")
        page.wait_for_timeout(1000)

        # 提取当前可见文本
        text = page.evaluate("""() => {
            ['script','style'].forEach(tag => {
                document.querySelectorAll(tag).forEach(el => el.remove());
            });
            return document.body ? document.body.innerText : '';
        }""")

        if len(text) > 4000:
            text = text[:4000] + "\n...(truncated)"

        scroll_pos = page.evaluate("window.scrollY")
        return f"[Scrolled {direction}, position={scroll_pos}px on {page.url}]\n{text}"
    except Exception as e:
        return f"Scroll error: {str(e)}"


# ============================================================
# 2. 文件读取工具
# ============================================================
def read_text_file(file_name: str) -> str:
    """
    读取附件中的文本文件内容（.txt, .py, .csv 等）。

    Args:
        file_name: GAIA 附件文件名（如 xxx.txt）
    Returns:
        文件的文本内容
    """
    file_path = os.path.join(GAIA_ATTACHMENTS_DIR, file_name)
    if not os.path.exists(file_path):
        return f"Error: File '{file_name}' not found in attachments directory."
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="latin-1") as f:
            content = f.read()
        return content


def read_excel_file(file_name: str, sheet_name: Optional[str] = None) -> str:
    """
    读取附件中的 Excel 文件（.xlsx），返回所有 sheet 的数据摘要。

    Args:
        file_name: GAIA 附件文件名（如 xxx.xlsx）
        sheet_name: 可选，指定要读取的 sheet 名称
    Returns:
        Excel 文件内容的文本表示
    """
    import openpyxl

    file_path = os.path.join(GAIA_ATTACHMENTS_DIR, file_name)
    if not os.path.exists(file_path):
        return f"Error: File '{file_name}' not found."
    try:
        wb = openpyxl.load_workbook(file_path, data_only=True)
        output = []
        sheets_to_read = [sheet_name] if sheet_name else wb.sheetnames
        for sn in sheets_to_read:
            if sn not in wb.sheetnames:
                output.append(f"Sheet '{sn}' not found.")
                continue
            ws = wb[sn]
            output.append(f"=== Sheet: {sn} (rows={ws.max_row}, cols={ws.max_column}) ===")

            # 读取所有数据，包括单元格颜色信息
            for row in ws.iter_rows(min_row=1, max_row=ws.max_row,
                                     max_col=ws.max_column, values_only=False):
                row_data = []
                for cell in row:
                    value = cell.value if cell.value is not None else ""
                    # 提取单元格背景色（某些 GAIA 题目需要颜色信息）
                    fill = cell.fill
                    if fill and fill.fgColor and fill.fgColor.rgb and fill.fgColor.rgb != "00000000":
                        color = fill.fgColor.rgb
                        row_data.append(f"{value}[color:{color}]")
                    else:
                        row_data.append(str(value))
                output.append(" | ".join(row_data))

        wb.close()
        return "\n".join(output)
    except Exception as e:
        return f"Error reading Excel: {str(e)}"


def read_docx_file(file_name: str) -> str:
    """
    读取附件中的 Word 文档（.docx），返回所有段落文本。

    Args:
        file_name: GAIA 附件文件名（如 xxx.docx）
    Returns:
        Word 文档的文本内容
    """
    from docx import Document

    file_path = os.path.join(GAIA_ATTACHMENTS_DIR, file_name)
    if not os.path.exists(file_path):
        return f"Error: File '{file_name}' not found."
    try:
        doc = Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        # 也读取表格内容
        tables_text = []
        for i, table in enumerate(doc.tables):
            tables_text.append(f"\n--- Table {i+1} ---")
            for row in table.rows:
                row_text = [cell.text.strip() for cell in row.cells]
                tables_text.append(" | ".join(row_text))
        return "\n".join(paragraphs) + "\n".join(tables_text)
    except Exception as e:
        return f"Error reading DOCX: {str(e)}"


def read_pptx_file(file_name: str) -> str:
    """
    读取附件中的 PowerPoint 文件（.pptx），返回所有幻灯片的文本。

    Args:
        file_name: GAIA 附件文件名（如 xxx.pptx）
    Returns:
        PPT 文件的文本内容
    """
    try:
        from pptx import Presentation
    except ImportError:
        return "Error: python-pptx is not installed. Run: pip install python-pptx"

    file_path = os.path.join(GAIA_ATTACHMENTS_DIR, file_name)
    if not os.path.exists(file_path):
        return f"Error: File '{file_name}' not found."
    try:
        prs = Presentation(file_path)
        output = []
        for i, slide in enumerate(prs.slides):
            output.append(f"\n=== Slide {i+1} ===")
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for paragraph in shape.text_frame.paragraphs:
                        text = paragraph.text.strip()
                        if text:
                            output.append(text)
                if shape.has_table:
                    table = shape.table
                    for row in table.rows:
                        row_text = [cell.text.strip() for cell in row.cells]
                        output.append(" | ".join(row_text))
        return "\n".join(output)
    except Exception as e:
        return f"Error reading PPTX: {str(e)}"


# ============================================================
# 3. YouTube 视频分析工具（Doubao 视觉模型 + 字幕备选）
# ============================================================
def analyze_youtube_video(url: str, question: str) -> str:
    """
    分析 YouTube 视频内容并回答问题。
    优先使用 Doubao 视觉模型直接分析视频，备选使用字幕提取。

    Args:
        url: YouTube 视频 URL
        question: 关于视频内容的具体问题
    Returns:
        视频分析结果
    """
    import re
    import subprocess
    # 提取 video ID
    vid_match = re.search(r'(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})', url)
    if not vid_match:
        return f"Error: Cannot extract video ID from URL: {url}"
    video_id = vid_match.group(1)

    # === 方法1: Doubao 视觉模型直接分析视频 ===
    video_path = os.path.join(WORK_DIR, f"yt_{video_id}.mp4")
    try:
        if not os.path.exists(video_path):
            yt_dlp_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                        '.venv', 'Scripts', 'yt-dlp.exe')
            if not os.path.exists(yt_dlp_path):
                yt_dlp_path = 'yt-dlp'
            result = subprocess.run([
                yt_dlp_path,
                '-f', 'worst[ext=mp4]',
                '--max-filesize', '20M',
                '-o', video_path,
                '--no-playlist', '--quiet',
                url
            ], capture_output=True, text=True, timeout=120)

        if os.path.exists(video_path) and os.path.getsize(video_path) <= 20 * 1024 * 1024:
            import base64 as b64mod
            with open(video_path, 'rb') as f:
                video_b64 = b64mod.b64encode(f.read()).decode('utf-8')

            from openai import OpenAI
            client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
            resp = client.chat.completions.create(
                model=VISION_MODEL,
                messages=[{
                    'role': 'user',
                    'content': [
                        {'type': 'video_url', 'video_url': {'url': f'data:video/mp4;base64,{video_b64}'}},
                        {'type': 'text', 'text': f'{question}\nBe precise and specific. Give exact details.'}
                    ]
                }],
                max_tokens=1000,
                timeout=120
            )
            answer = resp.choices[0].message.content
            if answer and len(answer.strip()) > 2:
                return f"[Doubao video analysis for {video_id}]\n{answer}"
    except Exception as e:
        pass  # 降级到字幕提取

    # === 方法2: 字幕提取 ===
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.fetch(video_id)
        lines = []
        for snippet in transcript_list:
            text = snippet.text if hasattr(snippet, 'text') else str(snippet.get('text', ''))
            lines.append(text)
        full_text = " ".join(lines)
        if len(full_text) > 4000:
            full_text = full_text[:4000] + "...(truncated)"
        return f"[YouTube transcript for {video_id}]\n{full_text}"
    except Exception:
        pass

    # === 方法3: 搜索备选 ===
    try:
        from ddgs import DDGS
        results = list(DDGS().text(f"youtube {video_id} {question}", max_results=3))
        if results:
            texts = [f"[{r.get('title','')}] {r.get('body','')[:200]}" for r in results]
            return "Video analysis unavailable. Search results:\n" + "\n".join(texts)
    except Exception:
        pass

    return f"Error: Could not analyze video {url}. Try search_web for info about this video."


# ============================================================
# 4. 图片分析工具（使用视觉模型 Doubao-Seed-1.8）
# ============================================================
def analyze_image(file_name: str, question: str) -> str:
    """
    使用视觉模型分析图片，回答关于图片的问题。
    支持棋盘、数学公式、图表、文字识别等。

    Args:
        file_name: GAIA 附件中的图片文件名（如 xxx.png）
        question: 关于图片的具体问题
    Returns:
        视觉模型对图片的分析结果
    """
    from openai import OpenAI

    file_path = os.path.join(GAIA_ATTACHMENTS_DIR, file_name)
    if not os.path.exists(file_path):
        return f"Error: Image '{file_name}' not found."
    try:
        with open(file_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode("utf-8")

        # 根据文件扩展名确定 MIME 类型
        ext = file_name.split(".")[-1].lower()
        mime_map = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "gif": "image/gif", "webp": "image/webp"}
        mime_type = mime_map.get(ext, "image/png")

        # 增强 prompt 以获得更精确的回答
        enhanced_question = (
            f"{question}\n\n"
            "IMPORTANT: Be extremely precise and detailed in your analysis. "
            "If this is a chess position, describe every piece and its exact square. "
            "If this contains text/numbers, transcribe them exactly as shown. "
            "If this is a math problem, read every symbol carefully. "
            "If this is a chart/table, extract all data values precisely."
        )

        client = get_openai_client()
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": enhanced_question},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{img_data}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            max_tokens=4000,
            temperature=0,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error analyzing image: {str(e)}"


# ============================================================
# 4. 音频转文字工具
# ============================================================
def _ensure_ffmpeg():
    """确保能找到 ffmpeg（从 imageio_ffmpeg 获取路径）"""
    import shutil
    if shutil.which("ffmpeg"):
        return shutil.which("ffmpeg")
    try:
        import imageio_ffmpeg
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        if ffmpeg_path and os.path.exists(ffmpeg_path):
            return ffmpeg_path
    except ImportError:
        pass
    return None


def _convert_audio_to_wav(src_path: str) -> str | None:
    """用 ffmpeg 将音频文件转换为 WAV 格式（单声道 16kHz）"""
    ffmpeg = _ensure_ffmpeg()
    if not ffmpeg:
        return None
    import subprocess
    wav_path = src_path.rsplit(".", 1)[0] + "_temp.wav"
    try:
        ret = subprocess.run(
            [ffmpeg, "-i", src_path, "-y", "-ac", "1", "-ar", "16000", wav_path],
            capture_output=True, timeout=30
        )
        if ret.returncode == 0 and os.path.exists(wav_path):
            return wav_path
    except Exception:
        pass
    return None


def transcribe_audio(file_name: str) -> str:
    """
    将音频文件转录为文字。依次尝试多种方案。

    Args:
        file_name: GAIA 附件中的音频文件名（如 xxx.mp3）
    Returns:
        音频的文字转录内容
    """
    file_path = os.path.join(GAIA_ATTACHMENTS_DIR, file_name)
    if not os.path.exists(file_path):
        return f"Error: Audio file '{file_name}' not found."

    # 方案 1：尝试使用 OpenAI 兼容 API
    try:
        client = get_openai_client()
        with open(file_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
            )
        if transcript.text and transcript.text.strip():
            return transcript.text
    except Exception:
        pass

    # 方案 2：使用 ffmpeg 转 WAV + SpeechRecognition + Google Web Speech API
    try:
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        
        # 如果是 mp3/m4a/ogg，用 ffmpeg 转为 wav
        ext = file_name.split(".")[-1].lower()
        wav_path = file_path
        if ext != "wav":
            converted = _convert_audio_to_wav(file_path)
            if converted:
                wav_path = converted
        
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
        
        text = recognizer.recognize_google(audio_data, language="en-US")
        
        # 清理临时文件
        if wav_path != file_path and os.path.exists(wav_path):
            os.unlink(wav_path)
        
        if text and text.strip():
            return text
    except Exception:
        pass

    # 方案 3：使用本地 openai-whisper 模型
    try:
        import whisper
        model = whisper.load_model("base")
        result = model.transcribe(file_path)
        return result["text"]
    except ImportError:
        pass
    except Exception as e:
        return f"Error transcribing audio with local whisper: {str(e)}"

    return (
        "Error: Cannot transcribe audio. "
        "All transcription methods failed. "
        "Try: pip install SpeechRecognition pydub openai-whisper"
    )


# ============================================================
# 5. Python 代码执行工具（用于计算）
# ============================================================
def execute_python(code: str) -> str:
    """
    执行 Python 代码片段并返回输出结果。
    用于数学计算、数据处理等任务。

    Args:
        code: 要执行的 Python 代码
    Returns:
        代码的标准输出或错误信息
    """
    import subprocess
    import sys
    import tempfile

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(code)
            tmp_path = f.name

        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            timeout=30,
            encoding="utf-8",
            errors="replace",
        )
        os.unlink(tmp_path)
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += f"\n[STDERR]: {result.stderr}"
        return output.strip() if output.strip() else "(No output)"
    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out (30s limit)."
    except Exception as e:
        return f"Error executing code: {str(e)}"


# ============================================================
# 6. 通用文件读取入口
# ============================================================
def read_attachment(file_name: str) -> str:
    """
    根据文件扩展名自动选择合适的工具来读取 GAIA 附件。

    Args:
        file_name: GAIA 附件文件名
    Returns:
        文件内容的文本表示
    """
    if not file_name:
        return "No attachment file specified."

    ext = file_name.split(".")[-1].lower()
    if ext in ("txt", "py", "csv", "json", "md"):
        return read_text_file(file_name)
    elif ext in ("xlsx", "xls"):
        return read_excel_file(file_name)
    elif ext == "docx":
        return read_docx_file(file_name)
    elif ext == "pptx":
        return read_pptx_file(file_name)
    elif ext in ("png", "jpg", "jpeg", "gif", "webp"):
        return f"[IMAGE FILE: {file_name}] - Use analyze_image() function to analyze this image."
    elif ext in ("mp3", "wav", "m4a", "ogg"):
        return f"[AUDIO FILE: {file_name}] - Use transcribe_audio() function to get transcription."
    else:
        return f"Unsupported file type: .{ext}"
