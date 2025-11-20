"""
Integration Test: Layer 2 Re-ranking vs FAISS

Tests the difference between FAISS mode and Cross-Encoder re-ranking mode.
"""

import sys
import os
import logging
import time
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from system_api.layer2_vectorstore import Layer2VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_documents():
    """Create test documents with various relevance patterns"""
    docs = [
        # High relevance: Contains both "深度學習" and "人流"
        Document(
            page_content="深度學習模型在人流預測系統中的應用研究。本文探討如何使用CNN和LSTM網絡分析商場人流數據，實現精準的客流量預測。",
            metadata={'paper_id': 'p001', 'chunk_id': 'p001_c000', 'chunk_index': 0}
        ),
        Document(
            page_content="基於深度學習的智能人流監控系統設計。系統利用計算機視覺技術實時追蹤人群移動，並通過深度神經網絡進行分析。",
            metadata={'paper_id': 'p002', 'chunk_id': 'p002_c000', 'chunk_index': 0}
        ),
        
        # Medium relevance: Only "深度學習"
        Document(
            page_content="深度學習在圖像識別領域的最新進展。卷積神經網絡（CNN）已成為計算機視覺任務的標準工具。",
            metadata={'paper_id': 'p003', 'chunk_id': 'p003_c000', 'chunk_index': 0}
        ),
        Document(
            page_content="深度學習模型訓練技巧與優化方法。包括學習率調整、正則化技術、資料增強等策略。",
            metadata={'paper_id': 'p004', 'chunk_id': 'p004_c000', 'chunk_index': 0}
        ),
        
        # Medium relevance: Only "人流"
        Document(
            page_content="傳統人流統計方法的局限性分析。紅外線感應器和壓力感測器在複雜場景下準確率不足。",
            metadata={'paper_id': 'p005', 'chunk_id': 'p005_c000', 'chunk_index': 0}
        ),
        Document(
            page_content="購物中心人流管理系統設計。通過入口計數器和監控攝像頭統計客流量，優化商場營運。",
            metadata={'paper_id': 'p006', 'chunk_id': 'p006_c000', 'chunk_index': 0}
        ),
        
        # Low relevance: Unrelated
        Document(
            page_content="自然語言處理技術在聊天機器人中的應用。Transformer模型已成為NLP領域的主流架構。",
            metadata={'paper_id': 'p007', 'chunk_id': 'p007_c000', 'chunk_index': 0}
        ),
        Document(
            page_content="推薦系統的協同過濾算法研究。基於用戶行為數據進行個性化推薦，提升用戶體驗。",
            metadata={'paper_id': 'p008', 'chunk_id': 'p008_c000', 'chunk_index': 0}
        ),
    ]
    
    return docs


def test_faiss_mode():
    """Test Layer 2 with FAISS mode (original behavior)"""
    print("\n" + "="*70)
    print("TEST 1: FAISS Mode (Bi-Encoder)")
    print("="*70)
    
    try:
        # Initialize embeddings
        embeddings = OllamaEmbeddings(model="embeddinggemma:latest")
        
        # Create Layer 2 with FAISS mode
        layer2 = Layer2VectorStore(
            embeddings=embeddings,
            vectorstore_path="./test_vectorstore/layer2_faiss",
            use_reranker=False
        )
        
        # Build index
        docs = create_test_documents()
        print(f"\n1. Building FAISS index with {len(docs)} documents...")
        success = layer2.build_index(docs)
        
        if not success:
            print("   ✗ Failed to build FAISS index")
            return False
        
        print("   ✓ FAISS index built successfully")
        
        # Test query
        query = "深度學習在人流的應用"
        print(f"\n2. Query: '{query}'")
        print("   Expected: Documents with BOTH concepts should rank higher\n")
        
        # Search
        start_time = time.time()
        results = layer2.search(query, k=5)
        duration = time.time() - start_time
        
        print(f"3. Results (FAISS mode) - {duration:.3f}s:")
        print("-" * 70)
        
        for i, doc in enumerate(results):
            paper_id = doc.metadata.get('paper_id', 'unknown')
            text_preview = doc.page_content[:80] + "..." if len(doc.page_content) > 80 else doc.page_content
            
            # Check relevance
            has_dl = "深度學習" in doc.page_content
            has_pf = "人流" in doc.page_content
            relevance = "🟢 Both" if (has_dl and has_pf) else ("🟡 One" if (has_dl or has_pf) else "🔴 None")
            
            print(f"   [{i+1}] {relevance} | {paper_id}")
            print(f"       {text_preview}")
        
        # Analyze results
        top3_has_both = sum(1 for doc in results[:3] if "深度學習" in doc.page_content and "人流" in doc.page_content)
        
        print(f"\n4. Analysis:")
        print(f"   Top-3 with BOTH concepts: {top3_has_both}/3")
        
        if top3_has_both >= 2:
            print("   ✓ FAISS mode: Reasonably good for multi-concept queries")
        else:
            print("   ⚠ FAISS mode: May miss multi-concept relationships")
        
        return True
        
    except Exception as e:
        logger.error(f"FAISS mode test failed: {e}", exc_info=True)
        return False


def test_reranking_mode():
    """Test Layer 2 with Cross-Encoder re-ranking"""
    print("\n" + "="*70)
    print("TEST 2: Re-ranking Mode (Cross-Encoder)")
    print("="*70)
    
    try:
        # Initialize embeddings (not used in re-ranking mode, but required for API)
        embeddings = OllamaEmbeddings(model="embeddinggemma:latest")
        
        # Create Layer 2 with re-ranking mode
        layer2 = Layer2VectorStore(
            embeddings=embeddings,
            vectorstore_path="./test_vectorstore/layer2_reranking",
            use_reranker=True,
            reranker_config={
                'model': 'qllama/bce-reranker-base_v1:latest',
                'ollama_base_url': 'http://localhost:11434',
                'max_candidates': 100
            }
        )
        
        # Build index (stores in JSONL)
        docs = create_test_documents()
        print(f"\n1. Building document store with {len(docs)} documents...")
        success = layer2.build_index(docs)
        
        if not success:
            print("   ✗ Failed to build document store")
            return False
        
        print("   ✓ Document store built successfully")
        
        # Test query
        query = "深度學習在人流的應用"
        print(f"\n2. Query: '{query}'")
        print("   Expected: Documents with BOTH concepts should rank HIGHEST\n")
        
        # Search with re-ranking
        start_time = time.time()
        results = layer2.search(query, k=5)
        duration = time.time() - start_time
        
        print(f"3. Results (Re-ranking mode) - {duration:.3f}s:")
        print("-" * 70)
        
        for i, doc in enumerate(results):
            paper_id = doc.metadata.get('paper_id', 'unknown')
            score = doc.metadata.get('rerank_score', 0.0)
            text_preview = doc.page_content[:80] + "..." if len(doc.page_content) > 80 else doc.page_content
            
            # Check relevance
            has_dl = "深度學習" in doc.page_content
            has_pf = "人流" in doc.page_content
            relevance = "🟢 Both" if (has_dl and has_pf) else ("🟡 One" if (has_dl or has_pf) else "🔴 None")
            
            print(f"   [{i+1}] {relevance} | Score: {score:.4f} | {paper_id}")
            print(f"       {text_preview}")
        
        # Analyze results
        top3_has_both = sum(1 for doc in results[:3] if "深度學習" in doc.page_content and "人流" in doc.page_content)
        
        print(f"\n4. Analysis:")
        print(f"   Top-3 with BOTH concepts: {top3_has_both}/3")
        
        if top3_has_both >= 2:
            print("   ✓ Re-ranking mode: Excellent multi-concept understanding")
        else:
            print("   ⚠ Re-ranking mode: Unexpected results")
        
        return True
        
    except Exception as e:
        logger.error(f"Re-ranking mode test failed: {e}", exc_info=True)
        return False


def main():
    """Run integration tests"""
    print("\n" + "="*70)
    print("Layer 2 Re-ranking Integration Test")
    print("="*70)
    print("\nThis test compares FAISS (Bi-Encoder) vs Cross-Encoder Re-ranking")
    print("for multi-concept queries like '深度學習在人流的應用'")
    
    results = {}
    
    # Test FAISS mode
    try:
        results['FAISS'] = test_faiss_mode()
    except Exception as e:
        logger.error(f"FAISS test error: {e}")
        results['FAISS'] = False
    
    # Test Re-ranking mode
    try:
        results['Reranking'] = test_reranking_mode()
    except Exception as e:
        logger.error(f"Re-ranking test error: {e}")
        results['Reranking'] = False
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for mode, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status} | {mode} Mode")
    
    print("\n" + "="*70)
    print("\nConclusion:")
    print("  - FAISS: Fast, good for general queries")
    print("  - Re-ranking: Slower, better for multi-concept queries")
    print("  - Trade-off: Latency vs Accuracy")
    print("="*70)
    
    # Cleanup
    print("\n💡 TIP: Clean up test files with:")
    print("   rm -rf ./test_vectorstore/")
    
    return all(results.values())


if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)
