"""
測試擴展後的 ChunkClassifier - 對應 GraphRAG 實體類型
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from system_api.chunk_classifier import ChunkClassifier


def test_expanded_categories():
    """測試新增的分類類別"""
    print("\n" + "="*80)
    print("🧪 測試擴展後的 ChunkClassifier (對應 GraphRAG 實體)")
    print("="*80)
    
    classifier = ChunkClassifier()
    
    # 測試案例：每個類別都有中英文範例
    test_cases = [
        {
            'category': 'background',
            'text': 'This paper introduces a novel approach to neural machine translation. '
                   'In recent years, transformer models have become dominant in NLP tasks. '
                   '本文介紹了一種新的神經機器翻譯方法。',
            'expected': 'background'
        },
        {
            'category': 'method',
            'text': 'We propose a new attention mechanism based on multi-head self-attention. '
                   'The architecture consists of 12 transformer layers with 768 hidden dimensions. '
                   '我們提出了基於多頭自注意力的新架構，模型包含12層變換器。',
            'expected': 'method'
        },
        {
            'category': 'dataset',
            'text': 'We evaluated our model on the WMT14 English-German dataset containing 4.5M sentence pairs. '
                   'The training set includes examples from news articles and parliamentary proceedings. '
                   '我們在 WMT14 英德數據集上評估模型，該數據集包含450萬句對。',
            'expected': 'dataset'
        },
        {
            'category': 'metric',
            'text': 'We report BLEU scores, ROUGE-L, and perplexity for all experiments. '
                   'The model achieves 28.4 BLEU score with 95% confidence interval. '
                   '我們報告了 BLEU 分數、ROUGE-L 和困惑度等評估指標。',
            'expected': 'metric'
        },
        {
            'category': 'domain',
            'text': 'Our approach is particularly effective in medical domain and legal text translation. '
                   'The method can be applied to various NLP tasks including question answering. '
                   '我們的方法在醫療領域和法律文本翻譯中特別有效。',
            'expected': 'domain'
        },
        {
            'category': 'results',
            'text': 'The experimental results show that our model outperforms the baseline by 3.2 BLEU points. '
                   'We observe significant improvements across all test sets with p<0.01. '
                   '實驗結果顯示我們的模型優於基準3.2個BLEU點，所有測試集上都有顯著改善。',
            'expected': 'results'
        }
    ]
    
    print("\n📊 分類測試結果：")
    print("-" * 80)
    
    correct = 0
    total = len(test_cases)
    
    for i, case in enumerate(test_cases, 1):
        result = classifier.classify_chunk(case['text'], chunk_id=f"test_{i}")
        predicted = result['chunk_type']
        expected = case['expected']
        confidence = result['confidence']
        
        is_correct = predicted == expected
        if is_correct:
            correct += 1
        
        status = "✅" if is_correct else "❌"
        
        print(f"\n{status} Test {i}: {case['category'].upper()}")
        print(f"   Expected: {expected}")
        print(f"   Predicted: {predicted} (confidence: {confidence:.2f})")
        print(f"   Text preview: {case['text'][:80]}...")
        
        if not is_correct:
            print(f"   ⚠️  Scores: {result.get('scores', {})}")
    
    print("\n" + "="*80)
    print(f"✅ Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
    print("="*80)
    
    return correct == total


def test_chunk_type_determination():
    """測試 GraphIntegrator 的 chunk type 映射"""
    print("\n" + "="*80)
    print("🧪 測試 GraphIntegrator chunk type 映射")
    print("="*80)
    
    from system_api.graph_integrator import GraphIntegrator
    
    # Mock LLM
    class MockLLM:
        def invoke(self, prompt):
            class Response:
                content = '{"answer": "test", "confidence": 0.8, "needs_details": false, "missing": []}'
            return Response()
    
    integrator = GraphIntegrator(MockLLM())
    
    test_queries = [
        {
            'query': 'What dataset did they use?',
            'missing': [],
            'expected': ['dataset']
        },
        {
            'query': '這篇論文用了什麼方法？',
            'missing': [],
            'expected': ['method']
        },
        {
            'query': 'What metrics were reported?',
            'missing': ['指標'],
            'expected': ['metric']
        },
        {
            'query': '性能表現如何？',
            'missing': [],
            'expected': ['results', 'metric']
        },
        {
            'query': 'Which domain is this applied in?',
            'missing': [],
            'expected': ['domain']
        },
        {
            'query': 'Tell me about this paper',
            'missing': ['詳細資訊'],
            'expected': ['method', 'results']
        }
    ]
    
    print("\n📋 Chunk Type 映射測試：")
    print("-" * 80)
    
    for i, test in enumerate(test_queries, 1):
        result = integrator.determine_chunk_types_for_query(
            test['query'], 
            test['missing']
        )
        
        # Check if all expected types are present
        expected_set = set(test['expected'])
        result_set = set(result)
        
        is_correct = expected_set.issubset(result_set)
        status = "✅" if is_correct else "⚠️"
        
        print(f"\n{status} Query {i}: {test['query']}")
        print(f"   Missing info: {test['missing']}")
        print(f"   Expected (min): {test['expected']}")
        print(f"   Returned: {result}")
        
        if not is_correct:
            missing_types = expected_set - result_set
            print(f"   ❌ Missing types: {missing_types}")
    
    print("\n" + "="*80)


def test_category_coverage():
    """測試所有新類別都有關鍵詞"""
    print("\n" + "="*80)
    print("🧪 測試分類類別覆蓋度")
    print("="*80)
    
    from system_api.chunk_classifier import ChunkClassifier
    
    expected_categories = ['background', 'method', 'dataset', 'metric', 'domain', 'results']
    
    print("\n📚 已定義的分類類別：")
    for category in ChunkClassifier.PATTERNS.keys():
        keywords = ChunkClassifier.PATTERNS[category]['keywords']
        print(f"   ✓ {category}: {len(keywords)} keywords")
    
    missing = set(expected_categories) - set(ChunkClassifier.PATTERNS.keys())
    if missing:
        print(f"\n   ❌ Missing categories: {missing}")
        return False
    else:
        print(f"\n   ✅ All expected categories present!")
        return True


if __name__ == '__main__':
    print("\n" + "🔬 ChunkClassifier 擴展功能測試 - GraphRAG 實體對齊")
    print("="*80)
    
    # Run all tests
    test1_pass = test_expanded_categories()
    test2_pass = test_category_coverage()
    test_chunk_type_determination()
    
    print("\n" + "="*80)
    if test1_pass and test2_pass:
        print("🎉 所有測試通過！ChunkClassifier 已擴展為 6 個類別，對應 GraphRAG 實體。")
    else:
        print("⚠️  部分測試失敗，需要調整關鍵詞權重或模式。")
    print("="*80 + "\n")
