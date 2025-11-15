#!/usr/bin/env python3
"""
Test script for Combination A solution:
Query Expansion + Adaptive Weight Adjustment

測試短查詢問題的解決方案
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from system_api.hierarchical_rag_system import HierarchicalRAGSystem
from system_api.query_expander import QueryExpander
from system_api.adaptive_weights import AdaptiveWeightAdjuster
from langchain_ollama import OllamaLLM
import logging

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_query_expander():
    """測試查詢擴展器"""
    print("=" * 80)
    print("測試 1: 查詢擴展器 (Query Expander)")
    print("=" * 80)
    
    llm = OllamaLLM(model="jcai/llama-3-taiwan-8b-instruct:q4_k_m")
    expander = QueryExpander(llm=llm, min_word_count=5)
    
    test_queries = [
        "人流預測",           # 2詞 - 應該擴展
        "CNN",               # 1詞 - 應該擴展
        "深度學習",           # 2詞 - 應該擴展
        "如何使用深度學習進行人流預測",  # 7詞 - 不應該擴展
        "YOLO模型在目標檢測中的應用研究",  # 9詞 - 不應該擴展
    ]
    
    for query in test_queries:
        print(f"\n原始查詢: {query}")
        
        # 檢查是否需要擴展
        should_expand = expander.should_expand(query)
        print(f"需要擴展: {should_expand}")
        
        # 嘗試擴展
        if should_expand:
            expanded = expander.expand(query)
            print(f"擴展結果: {expanded}")
            print(f"擴展成功: {expanded != query}")
        else:
            print(f"查詢長度充足，無需擴展")
        
        print("-" * 80)


def test_adaptive_weights():
    """測試自適應權重調整器"""
    print("\n" + "=" * 80)
    print("測試 2: 自適應權重調整器 (Adaptive Weight Adjuster)")
    print("=" * 80)
    
    adjuster = AdaptiveWeightAdjuster()
    
    test_queries = [
        "人流預測",                    # 短查詢
        "CNN",                        # 極短查詢 + 專有名詞
        "深度學習在人流的應用",          # 中等查詢
        "什麼是LSTM模型",               # 問題形式 + 專有名詞
        "如何使用深度學習方法進行城市公共空間的人流量預測與分析研究",  # 長查詢
        "ResNet vs VGG",              # 短查詢 + 多個專有名詞
    ]
    
    for query in test_queries:
        print(f"\n查詢: {query}")
        
        semantic_w, keyword_w, reason = adjuster.adjust_weights(query)
        
        print(f"語義權重: {semantic_w:.2f} ({semantic_w:.1%})")
        print(f"關鍵詞權重: {keyword_w:.2f} ({keyword_w:.1%})")
        print(f"調整原因: {reason}")
        
        # 顯示權重偏好
        if semantic_w > keyword_w:
            print("→ 偏好語義搜尋（理解複雜概念）")
        elif keyword_w > semantic_w:
            print("→ 偏好關鍵詞搜尋（精準匹配）")
        else:
            print("→ 平衡搜尋")
        
        print("-" * 80)


def test_integrated_system():
    """測試完整整合系統"""
    print("\n" + "=" * 80)
    print("測試 3: 完整整合系統 (Combination A)")
    print("=" * 80)
    
    # 初始化系統（啟用所有功能）
    print("\n初始化系統...")
    rag_system = HierarchicalRAGSystem(
        pdf_directory="./data",
        model_name="jcai/llama-3-taiwan-8b-instruct:q4_k_m",
        embedding_model="quentinz/bge-large-zh-v1.5:latest",
        vectorstore_path="./vectorstore",
        config={
            'layer1': {
                'k_documents': 5,
                'similarity_threshold': 0.48,
            },
            'hybrid_search': {
                'enabled': True,
                'semantic_weight': 0.5,
                'keyword_weight': 0.5,
                'use_jieba': True,
            },
            'query_expansion': {
                'enabled': True,
                'min_word_count': 5,
            },
            'adaptive_weights': {
                'enabled': True,
            }
        }
    )
    
    print("✓ 系統初始化完成")
    
    # 測試查詢
    test_queries = [
        "人流預測",      # 短查詢 - 應該擴展 + 提高關鍵詞權重
        "CNN應用",       # 短查詢 + 專有名詞
        "深度學習在人流分析中的應用",  # 中等查詢 - 可能不擴展
    ]
    
    for query in test_queries:
        print("\n" + "=" * 80)
        print(f"測試查詢: {query}")
        print("=" * 80)
        
        try:
            # 執行檢索
            result = rag_system._hierarchical_retrieval(query)
            
            # 顯示增強資訊
            print(f"\n原始查詢: {result.get('original_query', query)}")
            
            if result.get('expanded_query'):
                print(f"擴展查詢: {result['expanded_query']}")
            else:
                print("查詢未擴展")
            
            if result.get('adjusted_weights'):
                weights = result['adjusted_weights']
                print(f"\n權重調整:")
                print(f"  語義: {weights['semantic']:.2f}")
                print(f"  關鍵詞: {weights['keyword']:.2f}")
                print(f"  原因: {weights['reason']}")
            
            # 顯示檢索結果
            if result.get('layer1_docs'):
                print(f"\n檢索到 {len(result['layer1_docs'])} 篇論文:")
                for i, (doc, score) in enumerate(zip(result['layer1_docs'], result.get('layer1_scores', [])), 1):
                    title = doc.metadata.get('title', 'Unknown')
                    print(f"  {i}. {title[:60]}... (score: {score:.3f})")
            
            # 顯示時間統計
            if result.get('timings'):
                print(f"\n執行時間:")
                if 'query_expansion' in result['timings']:
                    print(f"  查詢擴展: {result['timings']['query_expansion']:.3f}s")
                if 'weight_adjustment' in result['timings']:
                    print(f"  權重調整: {result['timings']['weight_adjustment']:.3f}s")
                if 'layer1_retrieval' in result['timings']:
                    print(f"  Layer 1 檢索: {result['timings']['layer1_retrieval']:.3f}s")
            
        except Exception as e:
            logger.error(f"測試失敗: {e}", exc_info=True)
        
        print()


def main():
    """主測試流程"""
    print("開始測試組合A解決方案...")
    print("=" * 80)
    print("功能: 查詢擴展 + 自適應權重調整")
    print("目標: 解決短查詢導致的相似度聚集問題")
    print("=" * 80)
    
    try:
        # 測試 1: 查詢擴展器
        test_query_expander()
        
        # 測試 2: 自適應權重調整器
        test_adaptive_weights()
        
        # 測試 3: 完整整合系統
        test_integrated_system()
        
        print("\n" + "=" * 80)
        print("✓ 所有測試完成")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\n測試已取消")
    except Exception as e:
        logger.error(f"測試過程發生錯誤: {e}", exc_info=True)


if __name__ == "__main__":
    main()
