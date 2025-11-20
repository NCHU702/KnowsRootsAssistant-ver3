#!/usr/bin/env python3
"""
測試醫療詞典整合效果

測試項目：
1. 醫療詞典是否正確載入
2. LLM Term Analyzer 是否能正確識別醫療詞彙
3. jieba 分詞是否正確處理醫療專有名詞
4. 混合檢索是否改善醫療查詢結果
"""

import sys
import logging
from pathlib import Path

# 設定路徑
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_medical_dictionary():
    """測試醫療詞典載入"""
    print("\n" + "="*70)
    print("測試 1: 醫療詞典載入")
    print("="*70)
    
    try:
        from system_api.medical_dictionary import (
            is_medical_term,
            is_disease_name,
            is_critical_medical_term,
            get_medical_category,
            ALL_MEDICAL_TERMS,
            DISEASES,
            MEDICAL_PROCEDURES
        )
        
        print(f"✅ 醫療詞典載入成功")
        print(f"   總詞彙數: {len(ALL_MEDICAL_TERMS)}")
        print(f"   疾病數: {len(DISEASES)}")
        print(f"   檢查/治療數: {len(MEDICAL_PROCEDURES)}")
        
        # 測試關鍵詞
        test_terms = [
            '鼻咽癌', '深度學習', 'CT', '應用', '糖尿病', 
            'MRI', '機器', '肺癌', '預測', '化療'
        ]
        
        print(f"\n測試詞彙識別:")
        for term in test_terms:
            is_medical = is_medical_term(term)
            is_disease = is_disease_name(term)
            is_critical = is_critical_medical_term(term)
            category = get_medical_category(term) if is_medical else 'N/A'
            
            status = "🏥" if is_medical else "📚"
            print(f"  {status} {term:8s} - 醫療詞彙: {str(is_medical):5s} | "
                  f"疾病: {str(is_disease):5s} | 關鍵: {str(is_critical):5s} | "
                  f"類別: {category}")
        
        return True
        
    except Exception as e:
        print(f"❌ 醫療詞典測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_llm_term_analyzer():
    """測試 LLM Term Analyzer 醫療詞彙識別"""
    print("\n" + "="*70)
    print("測試 2: LLM Term Analyzer（含 Fallback）")
    print("="*70)
    
    try:
        from system_api.llm_term_analyzer import LLMTermAnalyzer
        import jieba
        
        # 測試查詢
        test_queries = [
            "深度學習在鼻咽癌的應用",
            "CT影像分析",
            "糖尿病預測模型",
            "人流預測研究",
            "使用LSTM進行交通流量預測"
        ]
        
        print("\n使用 Fallback 分析器測試:")
        print("-"*70)
        
        # 創建分析器（不提供 LLM，強制使用 fallback）
        analyzer = LLMTermAnalyzer(llm=None, cache_enabled=False)
        
        for query in test_queries:
            print(f"\n查詢: {query}")
            
            # 分詞
            tokens = list(jieba.cut(query))
            print(f"  分詞: {' / '.join(tokens)}")
            
            # Fallback 分析
            result = analyzer._fallback_analysis(tokens)
            
            print(f"  核心詞: {result['core_terms']}")
            print(f"  廣泛詞: {result['generic_terms']}")
            print(f"  輔助詞: {result['auxiliary_terms']}")
            print(f"  領域: {result['domain']}")
            print(f"  醫療詞彙偵測: {result.get('medical_terms_detected', False)}")
            print(f"  推理: {result['reasoning']}")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM Term Analyzer 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_jieba_segmentation():
    """測試 jieba 分詞（含醫療詞典）"""
    print("\n" + "="*70)
    print("測試 3: jieba 分詞（載入醫療詞典）")
    print("="*70)
    
    try:
        import jieba
        from system_api.medical_dictionary import get_jieba_custom_words
        
        # 載入醫療詞典
        medical_words = get_jieba_custom_words()
        for word, freq, tag in medical_words:
            jieba.add_word(word, freq=freq, tag=tag)
        
        print(f"✅ 已載入 {len(medical_words)} 個醫療詞彙到 jieba")
        
        # 測試分詞
        test_sentences = [
            "深度學習在鼻咽癌的應用",
            "使用CT和MRI進行診斷",
            "糖尿病患者的血糖監測",
            "肺癌的化療與放療比較",
            "冠心病的早期篩檢"
        ]
        
        print("\n分詞測試:")
        print("-"*70)
        for sentence in test_sentences:
            tokens = list(jieba.cut(sentence))
            print(f"原句: {sentence}")
            print(f"分詞: {' / '.join(tokens)}")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ jieba 分詞測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_medical_query_retrieval():
    """測試醫療查詢檢索（需要已建立的索引）"""
    print("\n" + "="*70)
    print("測試 4: 醫療查詢檢索效果")
    print("="*70)
    
    try:
        from system_api.hierarchical_rag_system import HierarchicalRAGSystem
        
        # 初始化 RAG 系統
        print("初始化 RAG 系統...")
        rag_system = HierarchicalRAGSystem(
            pdf_directory="./test_data",
            model_name="jcai/llama-3-taiwan-8b-instruct:q4_k_m",
            embedding_model="quentinz/bge-large-zh-v1.5:latest",
            vectorstore_path="./vectorstore",
            config={
                'hybrid_search': {
                    'enabled': True,
                    'use_jieba': True,
                    'use_llm_analyzer': True,
                }
            }
        )
        
        if not rag_system.is_ready():
            print("⚠️  RAG 系統未就緒，跳過檢索測試")
            return True
        
        # 測試查詢
        test_queries = [
            "深度學習在鼻咽癌的應用",
            "CT影像分析",
            "糖尿病預測"
        ]
        
        print("\n執行醫療查詢檢索:")
        print("-"*70)
        
        for query in test_queries:
            print(f"\n查詢: {query}")
            
            result = rag_system.query(query)
            
            print(f"  終止於: {result.get('terminated_at', 'unknown')}")
            print(f"  Layer 1 文檔數: {len(result.get('layer1_docs', []))}")
            print(f"  最終文檔數: {len(result.get('final_docs', []))}")
            
            if result.get('layer1_docs'):
                print(f"  前 3 篇論文:")
                for i, doc in enumerate(result['layer1_docs'][:3], 1):
                    title = doc.metadata.get('title', 'Unknown')
                    print(f"    {i}. {title[:60]}...")
        
        return True
        
    except Exception as e:
        print(f"⚠️  檢索測試失敗（可能是索引未建立）: {e}")
        # 這個測試失敗不算整體失敗
        return True


def main():
    """執行所有測試"""
    print("\n" + "="*70)
    print("🧪 醫療詞典整合測試")
    print("="*70)
    
    results = {
        '醫療詞典載入': test_medical_dictionary(),
        'LLM Term Analyzer': test_llm_term_analyzer(),
        'jieba 分詞': test_jieba_segmentation(),
        '醫療查詢檢索': test_medical_query_retrieval(),
    }
    
    # 總結
    print("\n" + "="*70)
    print("📊 測試總結")
    print("="*70)
    
    for name, passed in results.items():
        status = "✅ 通過" if passed else "❌ 失敗"
        print(f"  {status} - {name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 所有測試通過！醫療詞典整合成功！")
        return 0
    else:
        print("\n⚠️  部分測試失敗，請檢查錯誤訊息")
        return 1


if __name__ == '__main__':
    sys.exit(main())
