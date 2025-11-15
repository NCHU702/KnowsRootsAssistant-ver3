#!/usr/bin/env python3
"""
測試 Layer 搜尋時顯示相似度前 5 高的論文
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from system_api.hierarchical_rag_system import HierarchicalRAGSystem
import logging

# 設定日誌格式
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)

def test_layer_display():
    """測試 Layer 1 和 Layer 2 的相似度顯示"""
    
    print("=" * 80)
    print("測試 Layer 搜尋相似度顯示")
    print("=" * 80)
    
    # 初始化系統
    print("\n初始化系統...")
    rag = HierarchicalRAGSystem(
        pdf_directory='./data',
        model_name='jcai/llama-3-taiwan-8b-instruct:q4_k_m',
        embedding_model='quentinz/bge-large-zh-v1.5:latest',
        vectorstore_path='./vectorstore',
        config={
            'layer1': {
                'k_documents': 10,
                'confidence_threshold': 0.4,  # 降低閾值以觸發 Layer 2
                'similarity_threshold': 0.48
            },
            'layer2': {
                'k_documents': 10,
                'confidence_threshold': 0.8
            },
            'hybrid_search': {
                'enabled': True,
                'semantic_weight': 0.9,
                'keyword_weight': 0.1,
                'use_jieba': True,
                'enable_smart_weighting': True,
                'use_llm_analyzer': True
            },
            'query_expansion': {'enabled': False},
            'adaptive_weights': {'enabled': False}
        }
    )
    
    print("✓ 系統初始化完成\n")
    
    # 測試查詢
    test_queries = [
        "可以用什麼模型處理人流預測",
        "深度學習在醫療的應用",
        "LSTM 模型的優勢"
    ]
    
    for query in test_queries:
        print("\n" + "=" * 80)
        print(f"測試查詢: {query}")
        print("=" * 80)
        
        try:
            # 執行查詢（這會觸發 Layer 1 和可能的 Layer 2）
            result = rag.query(query)
            
            print("\n✓ 查詢完成")
            print(f"使用的層級: {', '.join(result.get('layers_used', []))}")
            print(f"總時間: {result.get('timings', {}).get('total', 0):.2f} 秒")
            
        except Exception as e:
            print(f"❌ 查詢失敗: {e}")
        
        print()

if __name__ == "__main__":
    test_layer_display()
