"""
測試純語義策略（不依賴 BM25）

策略: 廣泛詞權重低 → 語義理解高
      核心詞權重高 → 語義理解高
      
目標: 完全依靠 embedding 的語義理解能力
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


def test_pure_semantic_strategy():
    """測試純語義策略"""
    
    print("\n" + "="*80)
    print("純語義策略測試 - 廣泛詞和核心詞都依靠語義理解")
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
        {
            'query': '深度學習在醫療的應用是什麼？',
            'expected': '醫療相關論文（如鼻咽癌辨識）',
            'avoid': 'PM2.5、天際線等無關論文'
        },
        {
            'query': '深度學習在人流的應用',
            'expected': '人流預測相關論文',
            'avoid': 'PM2.5、醫療等無關論文'
        },
        {
            'query': '人流預測',
            'expected': '人流預測論文',
            'avoid': 'PM2.5、路徑規劃等'
        },
        {
            'query': 'PM2.5預測',
            'expected': 'PM2.5 相關論文',
            'avoid': '醫療、人流等無關論文'
        },
    ]
    
    for test_case in test_queries:
        query = test_case['query']
        
        print(f"\n{'='*80}")
        print(f"查詢: {query}")
        print(f"期望: {test_case['expected']}")
        print(f"避免: {test_case['avoid']}")
        print(f"{'='*80}")
        
        # 分詞
        tokens = tokenize(query)
        print(f"\n分詞: {', '.join(tokens)}")
        
        # LLM 分析
        print("\n正在分析...")
        analysis = analyzer.analyze_query(query, tokens)
        
        # 顯示分析結果
        print(f"\n【LLM 詞彙分析】")
        print(f"核心詞 ({len(analysis['core_terms'])}個): {', '.join(analysis['core_terms']) if analysis['core_terms'] else '無'}")
        print(f"廣泛詞 ({len(analysis['generic_terms'])}個): {', '.join(analysis['generic_terms']) if analysis['generic_terms'] else '無'}")
        print(f"領域: {analysis.get('domain', 'Unknown')}")
        
        core_count = len(analysis['core_terms'])
        generic_count = len(analysis['generic_terms'])
        total = core_count + generic_count
        
        if total > 0:
            core_ratio = core_count / total
            generic_ratio = generic_count / total
            print(f"\n詞彙比例:")
            print(f"  核心詞: {core_ratio:.0%}")
            print(f"  廣泛詞: {generic_ratio:.0%}")
        
        # 計算權重
        print(f"\n【權重計算 - 純語義策略】")
        semantic_w, keyword_w, reason = analyzer.compute_smart_weights(
            query=query,
            query_tokens=tokens,
            default_semantic=0.9,  # 預設就很高
            default_keyword=0.1
        )
        
        print(f"語義權重: 0.90 → {semantic_w:.2f} ({semantic_w*100:.0f}%)")
        print(f"關鍵詞權重: 0.10 → {keyword_w:.2f} ({keyword_w*100:.0f}%)")
        print(f"\n調整理由:")
        for line in reason.split('; '):
            print(f"  • {line}")
        
        # 策略說明
        print(f"\n【策略解釋】")
        if generic_count > 0:
            print(f"✓ 含有廣泛詞 → 不能靠 BM25 精準匹配")
            print(f"  └─ 需要語義理解來區分「深度學習在醫療」vs「深度學習在人流」")
        
        if core_count > 0:
            print(f"✓ 含有核心詞 → 可能以同義詞或相關詞出現")
            print(f"  └─ 如查詢「醫療」，論文標題可能是「鼻咽癌」（同領域）")
        
        if analysis.get('domain') and analysis['domain'] != 'Unknown':
            print(f"✓ 有明確領域 → 需要理解領域內的概念關係")
            print(f"  └─ Embedding 能理解「醫療」和「鼻咽癌」的語義相似性")
        
        print(f"\n{'─'*80}")
        print(f"最終權重: 語義 {semantic_w*100:.0f}% vs 關鍵詞 {keyword_w*100:.0f}%")
        print(f"預期效果: 依靠 embedding 語義理解找到相關論文")
        print(f"{'─'*80}")
    
    print(f"\n{'='*80}")
    print("測試完成！")
    print(f"{'='*80}")
    
    print("\n【純語義策略總結】")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("✓ 廣泛詞（深度學習、應用）→ 語義權重 90%+")
    print("  └─ BM25 會讓所有含「深度學習」的論文分數都高")
    print("  └─ 語義理解能區分「深度學習在醫療」vs「深度學習在人流」")
    print()
    print("✓ 核心詞（醫療、人流）→ 語義權重 90%+")
    print("  └─ 核心詞可能以同義詞出現（醫療→鼻咽癌）")
    print("  └─ 語義理解能找到領域內相關概念")
    print()
    print("✓ 明確領域 → 語義權重 95%+")
    print("  └─ Embedding 能理解領域內的概念層次")
    print()
    print("✓ BM25 僅作輔助參考（5-10%）")
    print("  └─ 避免精準匹配造成的泛用詞干擾")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


if __name__ == "__main__":
    test_pure_semantic_strategy()
