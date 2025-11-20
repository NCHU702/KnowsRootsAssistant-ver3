"""
Test Migration Tool

Creates a small FAISS index and tests migration to JSONL.
"""

import sys
import os
import shutil
import logging
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from system_api.layer2_vectorstore import Layer2VectorStore
from system_api.layer2_document_store import JSONLDocumentStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_migration():
    """Test the migration process"""
    print("\n" + "="*70)
    print("Migration Tool Test")
    print("="*70)
    
    test_dir = "./test_migration"
    
    try:
        # Clean up any existing test data
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        
        # Step 1: Create test FAISS index
        print("\n1. Creating test FAISS index...")
        
        embeddings = OllamaEmbeddings(model="embeddinggemma:latest")
        
        layer2_faiss = Layer2VectorStore(
            embeddings=embeddings,
            vectorstore_path=f"{test_dir}/faiss",
            use_reranker=False
        )
        
        # Create test documents (need to be > 50 chars)
        test_docs = [
            Document(
                page_content="深度學習在圖像識別領域的應用非常廣泛，包括人臉識別、物體檢測、圖像分割等多種任務。CNN架構在這些領域取得了顯著的成果，並持續推動技術進步。",
                metadata={'paper_id': 'p001', 'chunk_id': 'p001_c000', 'chunk_index': 0}
            ),
            Document(
                page_content="卷積神經網絡（CNN）是深度學習中最重要的架構之一，特別適合處理圖像數據。它通過卷積層、池化層和全連接層的組合來提取特徵並進行分類。",
                metadata={'paper_id': 'p001', 'chunk_id': 'p001_c001', 'chunk_index': 1}
            ),
            Document(
                page_content="循環神經網絡（RNN）適合處理序列數據，如文本和時間序列。LSTM和GRU是RNN的改進版本，能夠更好地處理長期依賴問題。",
                metadata={'paper_id': 'p002', 'chunk_id': 'p002_c000', 'chunk_index': 0}
            ),
        ]
        
        success = layer2_faiss.build_index(test_docs)
        
        if not success:
            print("   ✗ Failed to build FAISS index")
            return False
        
        print(f"   ✓ Created FAISS index with {len(test_docs)} documents")
        
        # Step 2: Test extraction
        print("\n2. Testing document extraction...")
        
        layer2_loaded = Layer2VectorStore(
            embeddings=embeddings,
            vectorstore_path=f"{test_dir}/faiss",
            use_reranker=False
        )
        
        layer2_loaded.load()
        
        # Extract documents
        if not layer2_loaded.vectorstore:
            print("   ✗ FAISS vectorstore not loaded")
            return False
        
        docstore = layer2_loaded.vectorstore.docstore
        extracted_docs = [docstore._dict[doc_id] for doc_id in docstore._dict.keys()]
        
        print(f"   ✓ Extracted {len(extracted_docs)} documents")
        
        # Step 3: Test migration to JSONL
        print("\n3. Testing migration to JSONL...")
        
        jsonl_path = f"{test_dir}/chunks.jsonl"
        doc_store = JSONLDocumentStore(file_path=jsonl_path)
        
        success = doc_store.store_chunks(extracted_docs)
        
        if not success:
            print("   ✗ Failed to store in JSONL")
            return False
        
        stats = doc_store.get_stats()
        print(f"   ✓ Migrated to JSONL")
        print(f"     Chunks: {stats['chunk_count']}")
        print(f"     Papers: {stats['paper_count']}")
        
        # Step 4: Verify migration
        print("\n4. Verifying migration...")
        
        loaded_chunks = doc_store.get_chunks_by_paper_ids(None)
        
        if len(loaded_chunks) != len(test_docs):
            print(f"   ✗ Count mismatch: {len(loaded_chunks)} != {len(test_docs)}")
            return False
        
        # Check content
        for i, (orig, loaded) in enumerate(zip(test_docs, loaded_chunks)):
            if orig.page_content != loaded.get('text', ''):
                print(f"   ✗ Content mismatch at index {i}")
                return False
            
            if orig.metadata.get('paper_id') != loaded.get('metadata', {}).get('paper_id'):
                print(f"   ✗ Metadata mismatch at index {i}")
                return False
        
        print("   ✓ Verification passed")
        
        # Step 5: Test loading with Layer2VectorStore in reranking mode
        print("\n5. Testing Layer2VectorStore with reranking mode...")
        
        layer2_reranking = Layer2VectorStore(
            embeddings=embeddings,
            vectorstore_path=test_dir,
            use_reranker=True,
            reranker_config={
                'model': 'qllama/bce-reranker-base_v1:latest',
                'ollama_base_url': 'http://localhost:11434'
            }
        )
        
        # Build index in reranking mode
        success = layer2_reranking.build_index(test_docs)
        
        if not success:
            print("   ✗ Failed to build in reranking mode")
            return False
        
        # Search
        results = layer2_reranking.search("深度學習", k=2)
        
        if not results:
            print("   ✗ No search results")
            return False
        
        print(f"   ✓ Search returned {len(results)} results")
        for i, doc in enumerate(results):
            score = doc.metadata.get('rerank_score', 0.0)
            print(f"     [{i+1}] Score: {score:.4f} - {doc.page_content[:50]}...")
        
        # Success
        print("\n" + "="*70)
        print("✅ MIGRATION TOOL TEST PASSED")
        print("="*70)
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return False
    
    finally:
        # Cleanup
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            print(f"\n🧹 Cleaned up test directory: {test_dir}")


if __name__ == '__main__':
    success = test_migration()
    sys.exit(0 if success else 1)
