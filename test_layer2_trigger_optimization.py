#!/usr/bin/env python3
"""
Layer 2 觸發決策器優化測試

測試優化後的決策邏輯：
1. 綜觀性/大方向問題 → 只需 Layer 1
2. 詳細模型/方法論/特定論文 → 需要 Layer 2
"""

import sys
import importlib.util

# 直接導入 layer2_trigger 模組（避免 __init__.py 的依賴問題）
spec = importlib.util.spec_from_file_location(
    "layer2_trigger",
    "./system_api/layer2_trigger.py"
)
layer2_trigger = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layer2_trigger)
Layer2TriggerDecision = layer2_trigger.Layer2TriggerDecision


def test_layer2_trigger_optimization():
    """測試優化後的 Layer 2 觸發決策"""
    
    print("=" * 70)
    print("🧪 Layer 2 觸發決策優化測試")
    print("=" * 70)
    print()
    
    # 初始化決策器
    trigger = Layer2TriggerDecision()
    
    # 測試案例
    test_cases = [
        # ===== 綜觀性/大方向問題 → 應該只需 Layer 1 =====
        {
            'category': '綜觀性查詢',
            'query': '有哪些深度學習的應用',
            'confidence': 0.50,
            'doc_count': 8,
            'expected': False,  # 不需要 Layer 2
            'reason': '概覽查詢，Layer 1 摘要足夠'
        },
        {
            'category': '綜觀性查詢',
            'query': '列出所有關於人流預測的論文',
            'confidence': 0.48,
            'doc_count': 10,
            'expected': False,
            'reason': '列表查詢，Layer 1 足夠'
        },
        {
            'category': '綜觀性查詢',
            'query': '比較 CNN 和 RNN 的優缺點',
            'confidence': 0.55,
            'doc_count': 6,
            'expected': False,
            'reason': '比較查詢但屬於概覽性質'
        },
        {
            'category': '綜觀性查詢',
            'query': '什麼是遷移學習',
            'confidence': 0.52,
            'doc_count': 5,
            'expected': False,
            'reason': '定義查詢，Layer 1 足夠'
        },
        {
            'category': '綜觀性查詢',
            'query': '深度學習在醫療領域的應用',
            'confidence': 0.50,
            'doc_count': 7,
            'expected': False,
            'reason': '大方向應用查詢，摘要足夠'
        },
        
        # ===== 詳細模型/方法論 → 應該需要 Layer 2 =====
        {
            'category': '模型細節查詢',
            'query': 'ResNet 的殘差連接結構',
            'confidence': 0.65,
            'doc_count': 5,
            'expected': True,  # 需要 Layer 2
            'reason': '查詢模型具體結構'
        },
        {
            'category': '模型細節查詢',
            'query': 'LSTM 模型的參數設置',
            'confidence': 0.60,
            'doc_count': 4,
            'expected': True,
            'reason': '查詢模型參數細節'
        },
        {
            'category': '方法論查詢',
            'query': '如何實現注意力機制',
            'confidence': 0.55,
            'doc_count': 6,
            'expected': True,
            'reason': '查詢具體實現方法'
        },
        {
            'category': '方法論查詢',
            'query': '詳細說明資料增強的步驟',
            'confidence': 0.62,
            'doc_count': 5,
            'expected': True,
            'reason': '需要詳細步驟說明'
        },
        {
            'category': '方法論查詢',
            'query': '深度學習在鼻咽癌辨識的具體方法',
            'confidence': 0.58,
            'doc_count': 3,
            'expected': True,
            'reason': '查詢具體方法和技術細節'
        },
        
        # ===== 特定論文查詢 → 應該需要 Layer 2 =====
        {
            'category': '特定論文查詢',
            'query': '這篇論文用了什麼模型',
            'confidence': 0.60,
            'doc_count': 1,
            'expected': True,
            'reason': '查詢特定論文內容'
        },
        {
            'category': '特定論文查詢',
            'query': '該研究的實驗設計',
            'confidence': 0.55,
            'doc_count': 2,
            'expected': True,
            'reason': '查詢特定研究細節'
        },
        
        # ===== 邊界案例 =====
        {
            'category': '邊界案例',
            'query': '深度學習',
            'confidence': 0.45,
            'doc_count': 15,
            'expected': False,
            'reason': '查詢太寬泛，結果多，Layer 1 足夠'
        },
        {
            'category': '邊界案例',
            'query': '如何訓練神經網路',
            'confidence': 0.50,
            'doc_count': 2,
            'expected': True,
            'reason': '結果少且涉及方法論，需要 Layer 2'
        },
    ]
    
    # 執行測試
    passed = 0
    failed = 0
    
    for i, case in enumerate(test_cases, 1):
        print(f"測試 {i}/{len(test_cases)}: {case['category']}")
        print(f"  查詢: {case['query']}")
        print(f"  Layer 1: {case['doc_count']} 篇論文, 信心度 {case['confidence']:.2f}")
        
        # 模擬 Layer 1 評估結果
        layer1_evaluation = {
            'confidence': case['confidence'],
            'reasoning': ''
        }
        
        # 模擬 Layer 1 文檔
        layer1_docs = [{'dummy': 'doc'}] * case['doc_count']
        
        # 執行決策
        result = trigger.should_trigger_layer2(
            query=case['query'],
            layer1_evaluation=layer1_evaluation,
            layer1_docs=layer1_docs,
            base_threshold=0.6
        )
        
        # 檢查結果
        actual = result['should_trigger']
        expected = case['expected']
        
        if actual == expected:
            print(f"  ✅ 決策正確: {'需要' if actual else '不需要'} Layer 2")
            print(f"  理由: {result['reason']}")
            passed += 1
        else:
            print(f"  ❌ 決策錯誤:")
            print(f"     預期: {'需要' if expected else '不需要'} Layer 2 ({case['reason']})")
            print(f"     實際: {'需要' if actual else '不需要'} Layer 2")
            print(f"     系統理由: {result['reason']}")
            failed += 1
        
        print()
    
    # 總結
    print("=" * 70)
    print("📊 測試總結")
    print("=" * 70)
    print(f"總測試數: {len(test_cases)}")
    print(f"✅ 通過: {passed} ({passed/len(test_cases)*100:.1f}%)")
    print(f"❌ 失敗: {failed} ({failed/len(test_cases)*100:.1f}%)")
    print()
    
    if failed == 0:
        print("🎉 所有測試通過！Layer 2 觸發決策優化成功！")
        return 0
    else:
        print(f"⚠️  有 {failed} 個測試失敗，需要進一步調整")
        return 1


if __name__ == '__main__':
    sys.exit(test_layer2_trigger_optimization())
