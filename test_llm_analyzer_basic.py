"""
測試 LLM 詞彙分析器的基本功能

不需要完整的 RAG 系統，只測試 LLM 分析器
"""

import logging
import jieba
from langchain_ollama import OllamaLLM
from system_api.llm_term_analyzer import LLMTermAnalyzer

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def tokenize(text: str):
    """中文分詞"""
    return [t for t in jieba.cut(text) if t.strip()]


def test_llm_analyzer():
    """測試 LLM 詞彙分析器"""
    
    print("\n" + "="*80)
    print("LLM 詞彙分析器功能測試")
    print("="*80)
    
    # 初始化 LLM
    print("\n初始化 LLM...")
    llm = OllamaLLM(
        model='jcai/llama-3-taiwan-8b-instruct:q4_k_m',
        base_url='http://localhost:11434',
        temperature=0.1
    )
    print("✓ LLM 初始化完成")
    
    # 初始化分析器
    print("\n初始化 LLM 詞彙分析器...")
    analyzer = LLMTermAnalyzer(llm=llm, cache_enabled=True)
    print("✓ 分析器初始化完成")
    
    # 測試查詢
    test_queries = [
        "深度學習在醫療的應用是什麼？",
        "深度學習在人流的應用",
        "人流預測",
        "使用CNN進行圖像分類",
        "PM2.5預測模型"
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"查詢: {query}")
        print(f"{'='*80}")
        
        # 分詞
        tokens = tokenize(query)
        print(f"分詞: {', '.join(tokens)}")
        
        # LLM 分析
        print("\n正在分析...")
        analysis = analyzer.analyze_query(query, tokens)
        
        # 顯示分析結果
        print(f"\n核心詞彙 ({len(analysis['core_terms'])}個):")
        if analysis['core_terms']:
            print(f"  → {', '.join(analysis['core_terms'])}")
        else:
            print(f"  → 無")
        
        print(f"\n廣泛詞彙 ({len(analysis['generic_terms'])}個):")
        if analysis['generic_terms']:
            print(f"  → {', '.join(analysis['generic_terms'])}")
        else:
            print(f"  → 無")
        
        print(f"\n輔助詞彙 ({len(analysis['auxiliary_terms'])}個):")
        if analysis['auxiliary_terms']:
            print(f"  → {', '.join(analysis['auxiliary_terms'])}")
        else:
            print(f"  → 無")
        
        print(f"\n領域: {analysis.get('domain', 'Unknown')}")
        print(f"意圖: {analysis.get('intent', 'Unknown')}")
        print(f"\n判斷理由:")
        print(f"  {analysis.get('reasoning', '無')}")
        
        # 計算智能權重
        print(f"\n權重計算:")
        semantic_w, keyword_w, reason = analyzer.compute_smart_weights(
            query=query,
            query_tokens=tokens,
            default_semantic=0.5,
            default_keyword=0.5
        )
        print(f"  語義權重: 0.50 → {semantic_w:.2f}")
        print(f"  關鍵詞權重: 0.50 → {keyword_w:.2f}")
        print(f"  理由: {reason}")
        
        # 詞彙增強
        print(f"\n詞彙增強:")
        boosted = analyzer.boost_terms(
            query_tokens=tokens,
            analysis=analysis,
            core_boost=3,
            generic_reduction=True
        )
        print(f"  原始: {tokens}")
        print(f"  增強: {boosted}")
        
        if analysis.get('is_fallback'):
            print(f"\n⚠️ 注意: 這是 Fallback 分析結果（LLM 回應解析失敗）")
    
    print(f"\n{'='*80}")
    print("測試完成！")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    test_llm_analyzer()
