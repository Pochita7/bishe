"""
下载 GAIA 数据集 (Level 1 validation set)
数据集地址: https://huggingface.co/datasets/gaia-benchmark/GAIA
"""
import os
import json
from datasets import load_dataset

SAVE_DIR = os.path.join(os.path.dirname(__file__), "gaia_data")
os.makedirs(SAVE_DIR, exist_ok=True)

print("正在从 Hugging Face 下载 GAIA 数据集...")
print("数据集: gaia-benchmark/GAIA, 配置: 2023_all")
print("注意: 这是受限数据集，请确保已通过 huggingface-cli login 登录且已申请访问权限")

# 下载 GAIA 数据集 (2023 版本，包含所有 level)
dataset = load_dataset("gaia-benchmark/GAIA", "2023_all")

print(f"\n数据集结构: {dataset}")

# 保存 validation 集
if "validation" in dataset:
    val_data = dataset["validation"]
    print(f"\nValidation 集共 {len(val_data)} 条数据")

    # 筛选 Level 1 (Level 字段可能是字符串或整数)
    level1_data = [row for row in val_data if str(row.get("Level")) == "1"]
    print(f"其中 Level 1 共 {len(level1_data)} 条")

    # 保存为 JSONL 文件
    output_path = os.path.join(SAVE_DIR, "gaia_validation.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for row in val_data:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    print(f"已保存完整 validation 集到: {output_path}")

    # 单独保存 Level 1
    level1_path = os.path.join(SAVE_DIR, "gaia_level1.jsonl")
    with open(level1_path, "w", encoding="utf-8") as f:
        for row in level1_data:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    print(f"已保存 Level 1 数据到: {level1_path}")

    # 保存附件文件（如果有的话）
    attachments_dir = os.path.join(SAVE_DIR, "attachments")
    os.makedirs(attachments_dir, exist_ok=True)

    attachment_count = 0
    for row in val_data:
        file_name = row.get("file_name", "")
        file_bytes = row.get("file_path", None)  # 有些版本字段名不同
        if file_name and file_bytes:
            file_save_path = os.path.join(attachments_dir, file_name)
            try:
                if isinstance(file_bytes, bytes):
                    with open(file_save_path, "wb") as f:
                        f.write(file_bytes)
                    attachment_count += 1
            except Exception as e:
                print(f"  保存附件 {file_name} 失败: {e}")
    
    if attachment_count > 0:
        print(f"已保存 {attachment_count} 个附件到: {attachments_dir}")

    # 打印几条 Level 1 样例
    print("\n===== Level 1 样例 (前3条) =====")
    for i, row in enumerate(level1_data[:3]):
        print(f"\n--- 第 {i+1} 题 ---")
        print(f"  Task ID: {row.get('task_id', 'N/A')}")
        print(f"  Question: {row.get('Question', 'N/A')[:200]}")
        print(f"  Final Answer: {row.get('Final answer', 'N/A')}")
        print(f"  File: {row.get('file_name', '无附件')}")
else:
    print("未找到 validation 集，打印可用的 split:")
    print(dataset.keys())

# 也保存 test 集（如果可用）
if "test" in dataset:
    test_data = dataset["test"]
    test_path = os.path.join(SAVE_DIR, "gaia_test.jsonl")
    with open(test_path, "w", encoding="utf-8") as f:
        for row in test_data:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    print(f"\n已保存 test 集 ({len(test_data)} 条) 到: {test_path}")

print("\n下载完成!")
