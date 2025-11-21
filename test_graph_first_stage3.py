#!/usr/bin/env python3
"""
測試 GraphIntegrator 功能（使用 mock LLM）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockLLM:
    """模擬 LLM 用於測試"""
    
    def invoke(self, prompt):
        """模擬 LLM 回應"""
        # 根據 prompt 中的內容返回不同的模擬回應
        if 'MNIST' in prompt:
            return json.dumps({
                "answer": "根據知識圖譜，有 3 篇論文使用了 MNIST 資料集進行圖像分類實驗。這些論文主要使用 CNN 和 LSTM 等深度學習方法。",
                "confidence": 0.75,
                "needs_details": True,
                "missing": ["具體模型參數", "訓練配置", "詳細實驗結果"]
            }, ensure_ascii=False)
        elif 'LSTM' in prompt:
            return json.dumps({
                "answer": "知識圖譜顯示有多篇論文採用 LSTM 模型進行時間序列預測，包括 GRU 等變體。",
                "confidence": 0.65,
                "needs_details": True,
                "missing": ["LSTM 層數", "隱藏單元數量", "具體架構細節"]
            }, ensure_ascii=False)
        else:
            return json.dumps({
                "answer": "根據圖譜資訊，找到相關論文但缺少詳細內容。",
                "confidence": 0.5,
                "needs_details": True,
                "missing": ["詳細方法", "實驗設定"]
            }, ensure_ascii=False)


def test_integration_with_results():
    """測試有 Graph 結果的整合"""
    logger.info("=" * 70)
    logger.info("🧪 測試 1: 整合 Graph 查詢結果")
    logger.info("=" * 70)
    
    from system_api.graph_integrator import GraphIntegrator
    
    # 建立 integrator
    mock_llm = MockLLM()
    integrator = GraphIntegrator(mock_llm)
    
    # 模擬 Graph 查詢結果
    graph_results = [
        {
            'paper_id': 'paper_mnist_1',
            'title': 'Image Classification with CNN on MNIST',
            'score': 0.85,
            'matched_entities': ['MNIST', 'CNN', 'ImageNet'],
            'entity_types': ['Dataset', 'Method'],
            'evidence': 'Matched dataset: MNIST',
            'relationship_count': 3
        },
        {
            'paper_id': 'paper_mnist_2',
            'title': 'Deep Learning for Handwritten Digit Recognition',
            'score': 0.78,
            'matched_entities': ['MNIST', 'LSTM'],
            'entity_types': ['Dataset', 'Method'],
            'evidence': 'Matched dataset: MNIST',
            'relationship_count': 2
        }
    ]
    
    # 執行整合
    query = "哪些論文使用了 MNIST 資料集？"
    result = integrator.integrate_graph_results(query, graph_results)
    
    logger.info(f"\n📋 查詢: {query}")
    logger.info(f"\n整合結果:")
    logger.info(f"  Text: {result['text'][:150]}...")
    logger.info(f"  Confidence: {result['confidence']:.2f}")
    logger.info(f"  Should descend: {result['should_descend']}")
    logger.info(f"  Referenced papers: {result['referenced_papers']}")
    logger.info(f"  Missing info: {result['missing_info']}")
    logger.info(f"  Notes: {result['notes']}")
    
    # 驗證結果結構
    assert 'text' in result
    assert 'confidence' in result
    assert 'should_descend' in result
    assert 'referenced_papers' in result
    assert 'missing_info' in result
    assert isinstance(result['confidence'], float)
    assert 0 <= result['confidence'] <= 1
    
    logger.info("\n  ✓ 結果結構正確")
    logger.info("✅ 有結果的整合測試通過")
    
    return True


def test_integration_without_results():
    """測試無 Graph 結果的整合"""
    logger.info("\n" + "=" * 70)
    logger.info("🧪 測試 2: 無 Graph 結果的處理")
    logger.info("=" * 70)
    
    from system_api.graph_integrator import GraphIntegrator
    
    mock_llm = MockLLM()
    integrator = GraphIntegrator(mock_llm)
    
    # 空結果
    query = "一個找不到結果的查詢"
    result = integrator.integrate_graph_results(query, [])
    
    logger.info(f"\n📋 查詢: {query}")
    logger.info(f"\n整合結果:")
    logger.info(f"  Text: {result['text']}")
    logger.info(f"  Confidence: {result['confidence']:.2f}")
    logger.info(f"  Should descend: {result['should_descend']}")
    
    # 驗證無結果時的行為
    assert result['confidence'] == 0.0
    assert result['should_descend'] == True  # 應該嘗試 Layer2
    assert '沒有找到' in result['text'] or 'not found' in result['text'].lower()
    
    logger.info("\n  ✓ 正確處理無結果情況")
    logger.info("✅ 無結果處理測試通過")
    
    return True


def test_determine_chunk_types():
    """測試決定 chunk types"""
    logger.info("\n" + "=" * 70)
    logger.info("🧪 測試 3: 決定檢索的 Chunk Types")
    logger.info("=" * 70)
    
    from system_api.graph_integrator import GraphIntegrator
    
    mock_llm = MockLLM()
    integrator = GraphIntegrator(mock_llm)
    
    # Test cases
    test_cases = [
        {
            'query': '使用了哪些資料集？',
            'missing': ['dataset details'],
            'expected': ['experiment']
        },
        {
            'query': 'LSTM 模型的架構是什麼？',
            'missing': ['architecture details'],
            'expected': ['method']
        },
        {
            'query': '實驗結果如何？',
            'missing': ['performance metrics'],
            'expected': ['results']
        },
        {
            'query': '這篇論文的概述',
            'missing': [],
            'expected': ['summary']
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        logger.info(f"\n📋 Case {i}: {case['query']}")
        chunk_types = integrator.determine_chunk_types_for_query(
            case['query'],
            case['missing']
        )
        
        logger.info(f"  Missing: {case['missing']}")
        logger.info(f"  Determined: {chunk_types}")
        logger.info(f"  Expected: {case['expected']}")
        
        # 驗證至少包含預期的類型之一
        assert any(t in chunk_types for t in case['expected']), \
            f"Expected chunk types not found for case {i}"
        
        logger.info(f"  ✓ Case {i} 通過")
    
    logger.info("\n✅ Chunk Types 決策測試通過")
    return True


def test_confidence_and_descend_logic():
    """測試信心分數和下降邏輯"""
    logger.info("\n" + "=" * 70)
    logger.info("🧪 測試 4: 信心分數和下降決策")
    logger.info("=" * 70)
    
    from system_api.graph_integrator import GraphIntegrator
    
    # 建立不同信心水平的 mock LLM
    class HighConfidenceLLM:
        def invoke(self, prompt):
            return json.dumps({
                "answer": "完整的答案，包含所有必要資訊。",
                "confidence": 0.95,
                "needs_details": False,
                "missing": []
            }, ensure_ascii=False)
    
    class LowConfidenceLLM:
        def invoke(self, prompt):
            return json.dumps({
                "answer": "部分答案，缺少詳細資訊。",
                "confidence": 0.4,
                "needs_details": True,
                "missing": ["detailed information"]
            }, ensure_ascii=False)
    
    # 測試高信心情況
    logger.info("\n📋 Test 4.1: 高信心（不需下降）")
    integrator_high = GraphIntegrator(HighConfidenceLLM())
    result_high = integrator_high.integrate_graph_results(
        "test query",
        [{'paper_id': 'p1', 'title': 'Test', 'score': 0.9, 
          'matched_entities': ['E1'], 'entity_types': ['T1'], 
          'evidence': 'test', 'relationship_count': 1}]
    )
    
    logger.info(f"  Confidence: {result_high['confidence']:.2f}")
    logger.info(f"  Should descend: {result_high['should_descend']}")
    
    # 高信心且不需要詳情時，不應下降
    # 但我們的邏輯是 confidence < 0.7 才下降，0.95 > 0.7 且 needs_details=False
    # 所以 should_descend 應為 False
    assert result_high['confidence'] >= 0.7
    assert result_high['should_descend'] == False  # 不需要下降
    logger.info("  ✓ 高信心情況正確")
    
    # 測試低信心情況
    logger.info("\n📋 Test 4.2: 低信心（需要下降）")
    integrator_low = GraphIntegrator(LowConfidenceLLM())
    result_low = integrator_low.integrate_graph_results(
        "test query",
        [{'paper_id': 'p1', 'title': 'Test', 'score': 0.9,
          'matched_entities': ['E1'], 'entity_types': ['T1'],
          'evidence': 'test', 'relationship_count': 1}]
    )
    
    logger.info(f"  Confidence: {result_low['confidence']:.2f}")
    logger.info(f"  Should descend: {result_low['should_descend']}")
    
    # 低信心或需要詳情時，應下降
    assert result_low['should_descend'] == True
    logger.info("  ✓ 低信心情況正確")
    
    logger.info("\n✅ 信心分數和下降決策測試通過")
    return True


def main():
    """執行所有測試"""
    logger.info("\n🚀 開始測試 GraphIntegrator")
    logger.info("=" * 70)
    
    try:
        # 測試 1: 有結果的整合
        if not test_integration_with_results():
            logger.error("❌ 有結果的整合測試失敗")
            return False
        
        # 測試 2: 無結果的處理
        if not test_integration_without_results():
            logger.error("❌ 無結果處理測試失敗")
            return False
        
        # 測試 3: Chunk types 決策
        if not test_determine_chunk_types():
            logger.error("❌ Chunk types 決策測試失敗")
            return False
        
        # 測試 4: 信心分數和下降邏輯
        if not test_confidence_and_descend_logic():
            logger.error("❌ 信心分數測試失敗")
            return False
        
        logger.info("\n" + "=" * 70)
        logger.info("🎉 所有 GraphIntegrator 測試通過！")
        logger.info("=" * 70)
        logger.info("\n✅ GraphIntegrator 功能驗證完成:")
        logger.info("   1. ✓ 整合 Graph 結果生成答案")
        logger.info("   2. ✓ 處理無結果情況")
        logger.info("   3. ✓ 根據查詢決定 Chunk Types")
        logger.info("   4. ✓ 信心分數評估和下降決策")
        logger.info("\n📝 下一步:")
        logger.info("   - 整合到 HierarchicalRAGSystem")
        logger.info("   - 建立完整的 Graph-first 流程")
        logger.info("=" * 70)
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ 測試失敗: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
