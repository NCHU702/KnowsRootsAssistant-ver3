#!/usr/bin/env python3
"""快速測試查詢類型識別"""

import importlib.util

# 直接導入 layer2_trigger 模組
spec = importlib.util.spec_from_file_location(
    "layer2_trigger",
    "./system_api/layer2_trigger.py"
)
layer2_trigger = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layer2_trigger)
Layer2TriggerDecision = layer2_trigger.Layer2TriggerDecision

# 測試
trigger = Layer2TriggerDecision()

test_queries = [
    ("如何應用深度學習在鼻腔癌", "methodology"),  # 應該是方法論
    ("深度學習在醫療的應用", "application"),       # 應該是應用概覽
    ("深度學習的應用", "application"),            # 應該是應用概覽
    ("有哪些深度學習應用", "overview"),           # 應該是概覽
    ("如何訓練模型", "methodology"),              # 應該是方法論
]

print("=" * 60)
print("查詢類型識別測試")
print("=" * 60)

for query, expected in test_queries:
    result = trigger._analyze_query_intent(query)
    actual = result['type']
    status = "✅" if actual == expected else "❌"
    print(f"{status} '{query}'")
    print(f"   預期: {expected} | 實際: {actual}")
    if actual != expected:
        print(f"   ⚠️  識別錯誤！")
    print()
