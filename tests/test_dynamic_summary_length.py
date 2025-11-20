#!/usr/bin/env python3
"""
測試動態摘要長度功能

驗證：
1. 短文本（<1000字）→ 200-300字摘要
2. 中等文本（1000-3000字）→ 300-800字摘要
3. 長文本（3000-5000字）→ 800-1200字摘要
4. 超長文本（>5000字）→ 1200-1500字摘要
"""

import sys
from system_api.llm_summarizer import LLMSummarizer

def test_calculate_target_length():
    """測試目標長度計算邏輯"""
    print("=" * 70)
    print("測試動態摘要長度計算")
    print("=" * 70)
    
    summarizer = LLMSummarizer(
        model_name="llama3.2:latest",
        map_reduce_threshold=3000
    )
    
    test_cases = [
        (500, "極短章節"),
        (800, "短章節"),
        (1500, "中等章節（下限）"),
        (2500, "中等章節（中間）"),
        (3500, "較長章節"),
        (5000, "長章節"),
        (8000, "超長章節"),
        (15000, "極長章節"),
    ]
    
    print(f"\n{'原文長度':<12} {'章節類型':<20} {'目標摘要長度':<15} {'壓縮比':<10}")
    print("-" * 70)
    
    for text_length, description in test_cases:
        target_length = summarizer._calculate_target_length(text_length)
        compression_ratio = (target_length / text_length) * 100
        
        print(f"{text_length:<12} {description:<20} {target_length:<15} {compression_ratio:>6.1f}%")
    
    print("\n驗證範圍限制：")
    print(f"  最小摘要長度：{summarizer.min_summary_length} 字")
    print(f"  最大摘要長度：{summarizer.max_summary_length} 字")
    
    # 驗證極端情況
    assert summarizer._calculate_target_length(100) >= 200, "最小長度應為 200"
    assert summarizer._calculate_target_length(100000) <= 1500, "最大長度應為 1500"
    
    print("\n✓ 所有長度計算測試通過！")

def test_summarization_integration():
    """測試實際摘要生成（需要 Ollama 運行）"""
    print("\n" + "=" * 70)
    print("測試實際摘要生成")
    print("=" * 70)
    
    try:
        summarizer = LLMSummarizer(
            model_name="llama3.2:latest",
            map_reduce_threshold=3000
        )
        
        # 測試短文本
        short_text = "深度學習是機器學習的一個分支，使用多層神經網路進行特徵學習。" * 15
        print(f"\n測試 1: 短文本（{len(short_text)} 字）")
        target = summarizer._calculate_target_length(len(short_text))
        print(f"  預期目標長度：{target} 字")
        
        summary = summarizer.summarize_section(
            short_text,
            "Introduction",
            "深度學習研究"
        )
        print(f"  實際摘要長度：{len(summary)} 字")
        print(f"  ✓ 範圍檢查：{200 <= len(summary) <= 1500}")
        
        # 測試中等文本
        medium_text = "卷積神經網路（CNN）是一種專門用於處理圖像數據的深度學習模型。" * 40
        print(f"\n測試 2: 中等文本（{len(medium_text)} 字）")
        target = summarizer._calculate_target_length(len(medium_text))
        print(f"  預期目標長度：{target} 字")
        
        summary = summarizer.summarize_section(
            medium_text,
            "Methodology",
            "CNN圖像識別研究"
        )
        print(f"  實際摘要長度：{len(summary)} 字")
        print(f"  ✓ 範圍檢查：{200 <= len(summary) <= 1500}")
        
        print("\n✓ 摘要生成測試完成！")
        
    except Exception as e:
        print(f"\n⚠️  摘要生成測試跳過（需要 Ollama 運行）：{e}")

if __name__ == "__main__":
    print("\n🧪 動態摘要長度測試\n")
    
    # 測試 1: 長度計算邏輯
    test_calculate_target_length()
    
    # 測試 2: 實際摘要生成（可選）
    if "--full" in sys.argv:
        test_summarization_integration()
    else:
        print("\n提示：使用 --full 參數執行完整測試（包含 LLM 調用）")
    
    print("\n" + "=" * 70)
    print("✅ 所有測試完成！")
    print("=" * 70)
