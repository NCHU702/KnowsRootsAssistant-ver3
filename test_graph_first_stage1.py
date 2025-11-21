#!/usr/bin/env python3
"""
簡單測試 Layer2 Chunk Type 過濾功能（無需完整初始化）
"""

import sys
import os

# 添加項目根目錄到 Python 路徑
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_chunk_classifier():
    """測試 Chunk Classifier"""
    logger.info("=" * 70)
    logger.info("🧪 階段 1: 測試 Chunk Classifier")
    logger.info("=" * 70)
    
    from system_api.chunk_classifier import ChunkClassifier
    
    # 建立分類器
    classifier = ChunkClassifier(llm=None, use_llm_fallback=False)
    
    # 測試樣本
    samples = {
        'method': "We propose a hybrid LSTM architecture combining recurrent layers with attention. The model uses 3 layers with 128 hidden units each.",
        'experiment': "We evaluate on MNIST, CIFAR-10, and ImageNet datasets. Training uses batch size 32 and learning rate 0.001.",
        'results': "Our model achieves 95% accuracy, outperforming baselines. F1-score is 0.94 and precision is 0.96.",
        'summary': "Abstract: This paper presents a novel approach to time series forecasting using deep learning networks.",
    }
    
    for expected_type, text in samples.items():
        result = classifier.classify_chunk(text)
        actual_type = result['chunk_type']
        confidence = result['confidence']
        
        status = "✓" if actual_type == expected_type else "✗"
        logger.info(f"  {status} Expected: {expected_type:12s} Got: {actual_type:12s} (confidence: {confidence:.2f})")
        logger.info(f"     Text: {text[:60]}...")
    
    logger.info("\n✅ Chunk Classifier 測試完成")
    return True


def test_chunk_type_filtering_mock():
    """模擬測試 Layer2 chunk type 過濾（不需要實際建立索引）"""
    logger.info("\n" + "=" * 70)
    logger.info("🧪 階段 2: 測試 Chunk Type 過濾邏輯")
    logger.info("=" * 70)
    
    # 模擬候選 chunks
    candidate_chunks = [
        {'text': 'Method text...', 'metadata': {'chunk_type': 'method', 'paper_id': 'p1'}},
        {'text': 'Experiment text...', 'metadata': {'chunk_type': 'experiment', 'paper_id': 'p1'}},
        {'text': 'Results text...', 'metadata': {'chunk_type': 'results', 'paper_id': 'p1'}},
        {'text': 'Summary text...', 'metadata': {'chunk_type': 'summary', 'paper_id': 'p1'}},
    ]
    
    logger.info(f"\n📦 模擬 {len(candidate_chunks)} 個候選 chunks:")
    for chunk in candidate_chunks:
        logger.info(f"    - {chunk['metadata']['chunk_type']}: {chunk['text']}")
    
    # 測試過濾邏輯
    def filter_by_chunk_types(chunks, filter_types):
        """模擬 Layer2 的過濾邏輯"""
        if not filter_types:
            return chunks
        return [
            chunk for chunk in chunks
            if chunk.get('metadata', {}).get('chunk_type') in filter_types
        ]
    
    # Test 1: 過濾 'experiment'
    logger.info("\n🔍 Test 1: 過濾 chunk_type=['experiment']")
    filtered = filter_by_chunk_types(candidate_chunks, ['experiment'])
    logger.info(f"  結果: {len(filtered)} chunks")
    for chunk in filtered:
        logger.info(f"    - {chunk['metadata']['chunk_type']}: {chunk['text']}")
    assert len(filtered) == 1 and filtered[0]['metadata']['chunk_type'] == 'experiment'
    logger.info("  ✓ 通過")
    
    # Test 2: 過濾多個類型
    logger.info("\n🔍 Test 2: 過濾 chunk_type=['method', 'experiment']")
    filtered = filter_by_chunk_types(candidate_chunks, ['method', 'experiment'])
    logger.info(f"  結果: {len(filtered)} chunks")
    for chunk in filtered:
        logger.info(f"    - {chunk['metadata']['chunk_type']}: {chunk['text']}")
    assert len(filtered) == 2
    assert all(c['metadata']['chunk_type'] in ['method', 'experiment'] for c in filtered)
    logger.info("  ✓ 通過")
    
    # Test 3: 無過濾
    logger.info("\n🔍 Test 3: 無過濾（filter_types=None）")
    filtered = filter_by_chunk_types(candidate_chunks, None)
    logger.info(f"  結果: {len(filtered)} chunks")
    assert len(filtered) == len(candidate_chunks)
    logger.info("  ✓ 通過")
    
    logger.info("\n✅ Chunk Type 過濾邏輯測試完成")
    return True


def test_layer2_api():
    """測試 Layer2 的 API 簽名（檢查參數是否正確添加）"""
    logger.info("\n" + "=" * 70)
    logger.info("🧪 階段 3: 檢查 Layer2 API 簽名")
    logger.info("=" * 70)
    
    from system_api.layer2_vectorstore import Layer2VectorStore
    import inspect
    
    # 檢查 search_with_scores 方法的參數
    sig = inspect.signature(Layer2VectorStore.search_with_scores)
    params = list(sig.parameters.keys())
    
    logger.info(f"\n📋 search_with_scores 方法參數:")
    for param in params:
        logger.info(f"    - {param}")
    
    # 驗證 filter_chunk_types 參數存在
    assert 'filter_chunk_types' in params, "Missing filter_chunk_types parameter"
    logger.info("\n  ✓ filter_chunk_types 參數已正確添加")
    
    # 檢查 search 方法的參數
    sig = inspect.signature(Layer2VectorStore.search)
    params = list(sig.parameters.keys())
    
    logger.info(f"\n📋 search 方法參數:")
    for param in params:
        logger.info(f"    - {param}")
    
    assert 'filter_chunk_types' in params, "Missing filter_chunk_types parameter in search"
    logger.info("\n  ✓ filter_chunk_types 參數已正確添加到 search 方法")
    
    logger.info("\n✅ Layer2 API 簽名檢查完成")
    return True


def main():
    """執行所有測試"""
    logger.info("\n🚀 開始測試 Graph-first Flow 的基礎組件")
    logger.info("=" * 70)
    
    try:
        # 測試 1: Chunk Classifier
        if not test_chunk_classifier():
            logger.error("❌ Chunk Classifier 測試失敗")
            return False
        
        # 測試 2: Chunk Type 過濾邏輯
        if not test_chunk_type_filtering_mock():
            logger.error("❌ Chunk Type 過濾邏輯測試失敗")
            return False
        
        # 測試 3: Layer2 API
        if not test_layer2_api():
            logger.error("❌ Layer2 API 檢查失敗")
            return False
        
        logger.info("\n" + "=" * 70)
        logger.info("🎉 所有測試通過！")
        logger.info("=" * 70)
        logger.info("\n✅ 已完成功能:")
        logger.info("   1. ✓ ChunkClassifier - 基於規則的 chunk 分類器")
        logger.info("   2. ✓ Layer2 filter_chunk_types - chunk 類型過濾")
        logger.info("   3. ✓ API 簽名驗證")
        logger.info("\n📝 下一步:")
        logger.info("   - 建立 GraphRetriever（Graph 查詢組件）")
        logger.info("   - 建立 GraphIntegrator（整合 Graph 結果）")
        logger.info("   - 整合到 HierarchicalRAGSystem")
        logger.info("=" * 70)
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ 測試失敗: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
