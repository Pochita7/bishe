"""验证 Colluding/Contradicting 数据"""
from tamas_adapter.loader import load_all_tamas, extract_task_info, ATTACK_TYPES
print("All attack types:", ATTACK_TYPES)
for atk in ["Colluding", "Contradicting"]:
    data = load_all_tamas(attack_types=[atk], scenarios=["education"])
    if data:
        info = extract_task_info(data[0])
        print(f"\n{atk}: {len(data)} items")
        print(f"  clean_query: {info['clean_query'][:80]}...")
        print(f"  attack_type: {info['attack_type']}")
        print(f"  attack_query len: {len(info['attack_query'])}")
    else:
        print(f"\n{atk}: NO DATA")
