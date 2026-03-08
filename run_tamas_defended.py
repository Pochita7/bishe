"""
单独运行 TAMAS Defended 测试（与 Baseline 并行）
"""
import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)

import sys, io, os
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import asyncio
from tamas_adapter.loader import load_all_tamas, extract_task_info
from run_full_benchmark import run_tamas_benchmark, TAMAS_DEFENDED_OUTPUT

async def main():
    all_data = load_all_tamas()
    print(f"\n[TAMAS Defended] 全量 {len(all_data)} 条 × 2 modes = {len(all_data)*2} 次推理")
    await run_tamas_benchmark(all_data, TAMAS_DEFENDED_OUTPUT, use_guardian=True, verbose=True)
    print("\n[TAMAS Defended] 全部完成 ✓")

if __name__ == "__main__":
    asyncio.run(main())
