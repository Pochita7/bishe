"""诊断: 直接测试 search_wikipedia 和 fetch_webpage 对 Mercedes Sosa 的表现"""
import time
import sys
sys.path.insert(0, ".")

from gaia_solver.tools import search_web, search_wikipedia, fetch_webpage

print("=" * 60)

# Test 1: search_web
print("\n[Test 1] search_web('Mercedes Sosa studio albums 2000 2009')")
t0 = time.time()
try:
    r = search_web("Mercedes Sosa studio albums 2000 2009")
    elapsed = time.time() - t0
    print(f"  耗时: {elapsed:.1f}s, 长度: {len(r)} chars")
    print(f"  前500: {r[:500]}")
except Exception as e:
    print(f"  ERROR: {e} ({time.time()-t0:.1f}s)")

# Test 2: search_wikipedia - full page
print("\n[Test 2] search_wikipedia('Mercedes Sosa')")
t0 = time.time()
try:
    r = search_wikipedia("Mercedes Sosa")
    elapsed = time.time() - t0
    print(f"  耗时: {elapsed:.1f}s, 长度: {len(r)} chars")
    # 看是否包含 discography 信息
    low = r.lower()
    if "discography" in low:
        idx = low.index("discography")
        print(f"  'discography' found at position {idx}")
        print(f"  上下文: ...{r[max(0,idx-100):idx+500]}...")
    else:
        print("  *** 'discography' NOT found in result ***")
    print(f"  前300: {r[:300]}")
except Exception as e:
    print(f"  ERROR: {e} ({time.time()-t0:.1f}s)")

# Test 3: search_wikipedia with section
print("\n[Test 3] search_wikipedia('Mercedes Sosa', section='Discography')")
t0 = time.time()
try:
    r = search_wikipedia("Mercedes Sosa", section="Discography")
    elapsed = time.time() - t0
    print(f"  耗时: {elapsed:.1f}s, 长度: {len(r)} chars")
    print(f"  全文: {r[:2000]}")
except Exception as e:
    print(f"  ERROR: {e} ({time.time()-t0:.1f}s)")

# Test 4: search_wikipedia - discography page
print("\n[Test 4] search_wikipedia('Mercedes Sosa discography')")
t0 = time.time()
try:
    r = search_wikipedia("Mercedes Sosa discography")
    elapsed = time.time() - t0
    print(f"  耗时: {elapsed:.1f}s, 长度: {len(r)} chars")
    low = r.lower()
    if "studio" in low:
        idx = low.index("studio")
        print(f"  'studio' found at position {idx}")
        print(f"  上下文: ...{r[max(0,idx-200):idx+1000]}...")
    print(f"  前500: {r[:500]}")
except Exception as e:
    print(f"  ERROR: {e} ({time.time()-t0:.1f}s)")

# Test 5: fetch_webpage
print("\n[Test 5] fetch_webpage('https://en.wikipedia.org/wiki/Mercedes_Sosa')")
t0 = time.time()
try:
    r = fetch_webpage("https://en.wikipedia.org/wiki/Mercedes_Sosa")
    elapsed = time.time() - t0
    print(f"  耗时: {elapsed:.1f}s, 长度: {len(r)} chars")
    low = r.lower()
    if "discography" in low:
        idx = low.index("discography")
        print(f"  'discography' found at position {idx}")
        print(f"  上下文: ...{r[max(0,idx-100):idx+300]}...")
    else:
        print("  *** 'discography' NOT found in result ***")
except Exception as e:
    print(f"  ERROR: {e} ({time.time()-t0:.1f}s)")

# Test 6: fetch discography page directly
print("\n[Test 6] fetch_webpage('https://en.wikipedia.org/wiki/Mercedes_Sosa_discography')")
t0 = time.time()
try:
    r = fetch_webpage("https://en.wikipedia.org/wiki/Mercedes_Sosa_discography")
    elapsed = time.time() - t0
    print(f"  耗时: {elapsed:.1f}s, 长度: {len(r)} chars")
    low = r.lower()
    if "studio" in low:
        idx = low.index("studio")
        print(f"  上下文: ...{r[max(0,idx-200):idx+1500]}...")
    print(f"  前500: {r[:500]}")
except Exception as e:
    print(f"  ERROR: {e} ({time.time()-t0:.1f}s)")
