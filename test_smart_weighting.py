#!/usr/bin/env python3
"""
測試智能詞彙權重系統

測試案例：
1. 深度學習在醫療的應用是什麼？ - 期望醫療論文排前面
2. 深度學習在人流的應用 - 期望人流論文排前面
3. 人流預測 - 短查詢但有明確領域詞
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from system_api.hierarchical_rag_system import HierarchicalRAGSystem
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)

print("=" * 80)
print("測試智能詞彙權重系統")
print("=" * 80)

# 測試查詢
test_cases = [
    {
        'query': '深度學習在醫療的應用是什麼？',
        'expected_keywords': ['醫療', '鼻咽癌', '病患'],
        'avoid_keywords': ['PM2.5', '天際線', '路徑']
    },
    {
        'query': '深度學習在人流的應用',
        'expected_keywords': ['人流', '人群'],
        'avoid_keywords': ['PM2.5', '鼻咽癌']
    },
    {
        'query': '人流預測',
        'expected_keywords': ['人流', '人群'],
        'avoid_keywords': ['PM2.5', '路徑']
    },
]

def test_with_smart_weighting(enabled: bool):
    """測試智能權重系統開啟/關閉的差異"""
    
    status = "啟用" if enabled else "停用"
    print(f"\n{'='*80}")
    print(f"智能權重系統: {status}")
    print(f"{'='*80}")
    
    rag = HierarchicalRAGSystem(
        pdf_directory='./data',
        model_name='jcai/llama-3-taiwan-8b-instruct:q4_k_m',
        embedding_model='quentinz/bge-large-zh-v1.5:latest',
        vectorstore_path='./vectorstore',
        config={
            'layer1': {
                'k_documents': 10,
                'similarity_threshold': None
            },
            'hybrid_search': {
                'enabled': True,
                'semantic_weight': 0.5,
                'keyword_weight': 0.5,
                'use_jieba': True,
                'enable_smart_weighting': enabled
            },
            'query_expansion': {'enabled': False},
            'adaptive_weights': {'enabled': False}
        }
    )
    
    for idx, test_case in enumerate(test_cases, 1):
        query = test_case['query']
        expected = test_case['expected_keywords']
        avoid = test_case['avoid_keywords']
        
        print(f"\n{'─'*80}")
        print(f"測試 {idx}: {query}")
        print(f"{'─'*80}")
        print(f"期望關鍵詞: {', '.join(expected)}")
        print(f"避免關鍵詞: {', '.join(avoid)}")
        
        if rag.hybrid_retriever:
            results = rag.hybrid_retriever.hybrid_search(
                query=query,
                k=5,
                semantic_weight=0.5,
                keyword_weight=0.5,
                score_threshold=None,
                return_scores_breakdown=True
            )
            
            print(f"\n檢索結果 (Top 5):")
            
            for i, (doc, scores) in enumerate(results[:5], 1):
                combined, semantic, keyword = scores
                title = doc.metadata.get('title', 'Unknown')
                
                # 檢查是否包含期望/避免關鍵詞
                title_lower = title.lower()
                has_expected = any(kw in title for kw in expected)
                has_avoid = any(kw in title for kw in avoid)
                
                if has_expected:
                    marker = '✓✓✓'
                elif has_avoid:
                    marker = '❌❌'
                else:
                    marker = '   '
                
                print(f"  {i}. {marker} [{combined:.3f}] (語義:{semantic:.3f}, 關鍵詞:{keyword:.3f})")
                print(f"       {title[:70]}")
            
            # 統計
            expected_in_top3 = sum(
                1 for doc, _ in results[:3]
                if any(kw in doc.metadata.get('title', '') for kw in expected)
            )
            avoid_in_top3 = sum(
                1 for doc, _ in results[:3]
                if any(kw in doc.metadata.get('title', '') for kw in avoid)
            )
            
            print(f"\nTop 3 統計:")
            print(f"  ✓ 期望論文: {expected_in_top3}/3")
            print(f"  ❌ 不相關論文: {avoid_in_top3}/3")


def main():
    """主測試流程"""
    
    # 測試 1: 關閉智能權重（基準）
    test_with_smart_weighting(enabled=False)
    
    print("\n\n" + "=" * 80)
    print("等待 3 秒...")
    print("=" * 80)
    import time
    time.sleep(3)
    
    # 測試 2: 開啟智能權重
    test_with_smart_weighting(enabled=True)
    
    print("\n\n" + "=" * 80)
    print("測試完成")
    print("=" * 80)
    print("""
結論:
- 智能權重系統應該能識別「醫療」、「人流」等領域詞
- 提高語義權重，讓系統理解「鼻咽癌」屬於「醫療」
- 降低泛用詞（深度學習、應用）的影響
- 領域詞增強（重複3次）確保關鍵概念被重視
""")


if __name__ == "__main__":
    main()
