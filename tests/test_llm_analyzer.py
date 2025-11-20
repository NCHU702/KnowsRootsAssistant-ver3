"""
測試 LLM-based 詞彙分析器

比較三種模式：
1. 無智能權重（基線）
2. 字典式智能權重
3. LLM 自動分析
"""

import os
import sys
import logging
import time
from langchain_ollama import OllamaLLM, OllamaEmbeddings

# 設置路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from system_api.hierarchical_rag_system import HierarchicalRAGSystem

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_query_with_config(query: str, config: dict, mode_name: str):
    """
    測試單一查詢配置
    
    Args:
        query: 查詢字串
        config: 系統配置
        mode_name: 模式名稱
    """
    print(f"\n{'='*80}")
    print(f"模式: {mode_name}")
    print(f"查詢: {query}")
    print(f"{'='*80}")
    
    # 初始化系統
    rag = HierarchicalRAGSystem(config)
    
    # 檢索
    results = rag.hybrid_retriever.hybrid_search(
        query=query,
        k=5,
        semantic_weight=0.5,
        keyword_weight=0.5,
        return_scores_breakdown=True
    )
    
    # 顯示結果
    print(f"\n檢索結果 (Top 5):")
    for i, (doc, (combined, semantic, keyword)) in enumerate(results[:5], 1):
        title = doc.metadata.get('title', 'Unknown')
        
        # 判斷是否相關
        relevant_keywords = ['人流', '人群', '行人', '客流']
        irrelevant_keywords = ['PM2.5', '污染', '天際線', '路徑']
        
        is_relevant = any(kw in title for kw in relevant_keywords)
        is_irrelevant = any(kw in title for kw in irrelevant_keywords)
        
        marker = "✓✓✓" if is_relevant else ("❌❌" if is_irrelevant else "◆◆◆")
        
        print(f"  {i}. {marker} [{combined:.3f}] {title[:50]}...")
        print(f"      (語義={semantic:.3f}, 關鍵詞={keyword:.3f})")
    
    return results


def main():
    """主測試流程"""
    
    print("\n" + "="*80)
    print("LLM vs 字典式 vs 無智能權重 - 詞彙分析對比測試")
    print("="*80)
    
    # 測試查詢
    test_queries = [
        "深度學習在人流的應用",
        "人流預測",
        "深度學習在醫療的應用是什麼？"
    ]
    
    # 基礎配置
    base_config = {
        'pdf_dir': 'data',
        'vectorstore_dir': 'vectorstore',
        'models': {
            'llm': {
                'model': 'jcai/llama-3-taiwan-8b-instruct:q4_k_m',
                'base_url': 'http://localhost:11434',
                'temperature': 0.1
            },
            'embeddings': {
                'model': 'quentinz/bge-large-zh-v1.5:latest',
                'base_url': 'http://localhost:11434'
            }
        },
        'layer1': {
            'k_documents': 10,
            'confidence_threshold': 0.3
        },
        'layer2': {
            'k_documents': 10,
            'confidence_threshold': 0.8
        },
        'query_expansion': {'enabled': False},
        'adaptive_weights': {'enabled': False},
        'performance': {'cache_size': 10}
    }
    
    for query in test_queries:
        print(f"\n\n{'#'*80}")
        print(f"# 測試查詢: {query}")
        print(f"{'#'*80}")
        
        # ========== 模式 1: 無智能權重 ==========
        config1 = base_config.copy()
        config1['hybrid_search'] = {
            'enabled': True,
            'semantic_weight': 0.5,
            'keyword_weight': 0.5,
            'use_jieba': True,
            'enable_smart_weighting': False,  # 停用
            'use_llm_analyzer': False  # 停用
        }
        
        results1 = test_query_with_config(query, config1, "❌ 無智能權重（基線）")
        time.sleep(2)
        
        # ========== 模式 2: 字典式智能權重 ==========
        config2 = base_config.copy()
        config2['hybrid_search'] = {
            'enabled': True,
            'semantic_weight': 0.5,
            'keyword_weight': 0.5,
            'use_jieba': True,
            'enable_smart_weighting': True,   # 啟用字典
            'use_llm_analyzer': False  # 停用 LLM
        }
        
        results2 = test_query_with_config(query, config2, "📚 字典式智能權重")
        time.sleep(2)
        
        # ========== 模式 3: LLM 自動分析 ==========
        config3 = base_config.copy()
        config3['hybrid_search'] = {
            'enabled': True,
            'semantic_weight': 0.5,
            'keyword_weight': 0.5,
            'use_jieba': True,
            'enable_smart_weighting': True,   # Fallback
            'use_llm_analyzer': True  # ✨ 啟用 LLM
        }
        
        results3 = test_query_with_config(query, config3, "✨ LLM 自動分析")
        time.sleep(2)
        
        # ========== 比較結果 ==========
        print(f"\n" + "="*80)
        print(f"結果比較:")
        print(f"="*80)
        
        def count_relevant(results):
            """統計前3名相關論文數"""
            relevant_keywords = ['人流', '人群', '行人', '客流', '醫療', '鼻咽癌', '病患']
            count = 0
            for doc, scores in results[:3]:
                title = doc.metadata.get('title', '')
                if any(kw in title for kw in relevant_keywords):
                    count += 1
            return count
        
        rel1 = count_relevant(results1)
        rel2 = count_relevant(results2)
        rel3 = count_relevant(results3)
        
        print(f"Top 3 相關論文數:")
        print(f"  無智能權重:   {rel1}/3")
        print(f"  字典式權重:   {rel2}/3")
        print(f"  LLM 分析:     {rel3}/3")
        
        if rel3 > rel1:
            improvement = ((rel3 - rel1) / 3) * 100
            print(f"\n✓ LLM 分析改進: +{improvement:.0f}%")
        
        time.sleep(3)
    
    print(f"\n\n{'='*80}")
    print("測試完成！")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
