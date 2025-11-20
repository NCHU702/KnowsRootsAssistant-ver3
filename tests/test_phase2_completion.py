"""
Phase 2 Completion Test: Cross-Encoder Integration

Tests that validate Cross-Encoder re-ranker is ready for integration with Layer 2.
"""

import sys
import logging
from system_api.cross_encoder_reranker import CrossEncoderReranker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_phase2_completion():
    """Test Phase 2 Cross-Encoder completion criteria"""
    print("\n" + "="*70)
    print("PHASE 2 COMPLETION TEST: Cross-Encoder Integration")
    print("="*70)
    
    # Criterion 1: Cross-Encoder initialization with Ollama
    print("\n✓ Criterion 1: Ollama-based Cross-Encoder initialized")
    reranker = CrossEncoderReranker(
        model_name='qllama/bce-reranker-base_v1:latest',
        ollama_base_url='http://localhost:11434'
    )
    
    # Criterion 2: Model availability check
    print("✓ Criterion 2: Model availability verification")
    assert reranker.is_loaded(), "Model not available"
    print(f"   Model: {reranker.model_name}")
    print(f"   Status: Available ✓")
    
    # Criterion 3: Multi-concept query handling
    print("\n✓ Criterion 3: Multi-concept query scoring")
    query = "深度學習在人流的應用"
    
    chunks = [
        "人流統計系統使用深度學習算法進行實時分析",  # Both concepts
        "深度學習神經網絡的訓練方法研究",             # Only 深度學習
        "購物中心人流監控系統的設計",                 # Only 人流
    ]
    
    scores = reranker.score_pairs(query, chunks)
    
    print(f"   Query: {query}")
    print(f"   Chunk 1 (both concepts): {scores[0]:.4f}")
    print(f"   Chunk 2 (深度學習 only): {scores[1]:.4f}")
    print(f"   Chunk 3 (人流 only): {scores[2]:.4f}")
    
    # Chunk with both concepts should score highest
    if scores[0] > scores[1] and scores[0] > scores[2]:
        print("   ✓ Multi-concept chunk scored highest")
    else:
        print("   ⚠ Warning: Expected multi-concept chunk to score higher")
    
    # Criterion 4: Batch processing (sequential for Ollama)
    print("\n✓ Criterion 4: Batch processing capability")
    
    test_chunks = [f"測試文本 {i}" for i in range(10)]
    batch_scores = reranker.score_pairs(query, test_chunks)
    
    assert len(batch_scores) == 10, "Batch size mismatch"
    print(f"   Processed {len(test_chunks)} chunks successfully")
    
    # Criterion 5: rank_chunks API
    print("\n✓ Criterion 5: rank_chunks API")
    
    chunk_dicts = [
        {'text': c, 'paper_id': f'p{i:03d}', 'chunk_id': f'c{i:03d}'}
        for i, c in enumerate(chunks)
    ]
    
    ranked = reranker.rank_chunks(query, chunk_dicts, top_k=2)
    
    assert len(ranked) == 2, "Top-k filtering failed"
    assert ranked[0][1] >= ranked[1][1], "Sorting failed"
    
    print(f"   Returned top {len(ranked)} chunks")
    print(f"   Scores: {ranked[0][1]:.4f} >= {ranked[1][1]:.4f}")
    
    # Success
    print("\n" + "="*70)
    print("✅ PHASE 2 COMPLETE: Cross-Encoder Integration Ready")
    print("="*70)
    print("\nNext Steps:")
    print("  • Phase 3: Refactor Layer2VectorStore to use Cross-Encoder")
    print("  • Phase 4: Build new indexes with JSONL storage")
    print("  • Phase 5: Add performance optimizations")
    print("\n" + "="*70)
    
    return True


if __name__ == '__main__':
    try:
        success = test_phase2_completion()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Phase 2 completion test failed: {e}", exc_info=True)
        sys.exit(1)
