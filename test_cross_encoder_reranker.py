"""
Test Cross-Encoder Reranker

Basic tests for Cross-Encoder re-ranking functionality.
"""

import logging
from system_api.cross_encoder_reranker import CrossEncoderReranker, TFIDFFallbackReranker

# Setup logging
logging.basicConfig(level=logging.INFO)


def test_reranker_initialization():
    """Test reranker initialization"""
    print("\n" + "="*60)
    print("Test 1: Reranker Initialization")
    print("="*60)
    
    reranker = CrossEncoderReranker(
        model_name='qllama/bce-reranker-base_v1:latest',
        ollama_base_url='http://localhost:11434',
        max_length=512,
        timeout=30
    )
    
    info = reranker.get_model_info()
    print(f"Model info: {info}")
    
    assert 'model_name' in info
    assert 'ollama_base_url' in info
    
    print("✓ Initialization test passed")


def test_score_pairs():
    """Test scoring query-chunk pairs"""
    print("\n" + "="*60)
    print("Test 2: Score Query-Chunk Pairs")
    print("="*60)
    
    reranker = CrossEncoderReranker(
        model_name='qllama/bce-reranker-base_v1:latest'
    )
    
    query = "深度學習在人流的應用"
    chunks = [
        "深度學習模型在人流預測中的應用研究",  # Highly relevant
        "深度學習的數學基礎知識",               # Partially relevant
        "人流分析的統計方法",                   # Partially relevant
        "天氣預報系統設計"                      # Not relevant
    ]
    
    print(f"\nQuery: {query}")
    print(f"Chunks to score: {len(chunks)}")
    
    scores = reranker.score_pairs(query, chunks)
    
    print(f"\nScores:")
    for i, (chunk, score) in enumerate(zip(chunks, scores)):
        print(f"  {i+1}. Score: {score:.4f} - {chunk[:40]}...")
    
    # Most relevant should have highest score
    assert len(scores) == 4
    assert scores[0] > scores[1]  # Highly relevant > partially relevant
    assert scores[0] > scores[3]  # Highly relevant > not relevant
    
    print("\n✓ Scoring test passed")


def test_rank_chunks():
    """Test ranking chunks"""
    print("\n" + "="*60)
    print("Test 3: Rank Chunks")
    print("="*60)
    
    reranker = CrossEncoderReranker(
        model_name='qllama/bce-reranker-base_v1:latest'
    )
    
    query = "機器學習在醫療診斷的應用"
    
    chunks = [
        {
            'paper_id': 'p001',
            'chunk_id': 'p001_c000',
            'text': '機器學習算法在醫療影像診斷中的應用與評估'
        },
        {
            'paper_id': 'p002',
            'chunk_id': 'p002_c000',
            'text': '深度學習的理論基礎'
        },
        {
            'paper_id': 'p003',
            'chunk_id': 'p003_c000',
            'text': '醫療診斷系統的設計原則'
        }
    ]
    
    print(f"\nQuery: {query}")
    print(f"Chunks to rank: {len(chunks)}")
    
    ranked = reranker.rank_chunks(query, chunks, top_k=3)
    
    print(f"\nRanked results:")
    for i, (chunk, score) in enumerate(ranked):
        print(f"  {i+1}. Score: {score:.4f}")
        print(f"     Paper: {chunk['paper_id']}")
        print(f"     Text: {chunk['text'][:50]}...")
    
    # First result should be most relevant (contains both "機器學習" and "醫療診斷")
    assert len(ranked) == 3
    assert ranked[0][1] > ranked[1][1]  # Top score > second score
    
    print("\n✓ Ranking test passed")


def test_tfidf_fallback():
    """Test TF-IDF fallback reranker"""
    print("\n" + "="*60)
    print("Test 4: TF-IDF Fallback Reranker")
    print("="*60)
    
    fallback = TFIDFFallbackReranker()
    
    query = "深度學習"
    chunks = [
        "深度學習神經網絡",
        "機器學習算法",
        "天氣預報"
    ]
    
    scores = fallback.score_pairs(query, chunks)
    
    print(f"\nFallback scores:")
    for i, (chunk, score) in enumerate(zip(chunks, scores)):
        print(f"  {i+1}. Score: {score:.4f} - {chunk}")
    
    assert len(scores) == 3
    assert scores[0] > scores[2]  # "深度學習" more relevant than "天氣預報"
    
    print("\n✓ Fallback test passed")


def test_empty_chunks():
    """Test handling of empty chunks"""
    print("\n" + "="*60)
    print("Test 5: Empty Chunks Handling")
    print("="*60)
    
    reranker = CrossEncoderReranker(
        model_name='qllama/bce-reranker-base_v1:latest'
    )
    
    query = "test query"
    
    # Test empty list
    scores = reranker.score_pairs(query, [])
    assert scores == []
    
    ranked = reranker.rank_chunks(query, [], top_k=5)
    assert ranked == []
    
    print("✓ Empty chunks test passed")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Running Cross-Encoder Reranker Tests")
    print("="*60)
    
    try:
        test_reranker_initialization()
        test_score_pairs()
        test_rank_chunks()
        test_tfidf_fallback()
        test_empty_chunks()
        
        print("\n" + "="*60)
        print("✅ All Cross-Encoder tests passed!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
