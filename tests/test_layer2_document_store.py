"""
Test Layer 2 Document Store

Tests for JSONL-based document store implementation.
"""

import os
import tempfile
from langchain_core.documents import Document

from system_api.layer2_document_store import Layer2DocumentStore, JSONLDocumentStore


def test_jsonl_store_basic():
    """Test basic JSONL document store operations"""
    
    # Create temp file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
        temp_path = f.name
    
    try:
        # Initialize store
        store = JSONLDocumentStore(file_path=temp_path, cache_in_memory=False)
        
        # Create test documents
        docs = [
            Document(
                page_content="深度學習的基礎知識",
                metadata={
                    'paper_id': 'p001',
                    'chunk_id': 'p001_c000',
                    'chunk_index': 0
                }
            ),
            Document(
                page_content="卷積神經網絡的應用",
                metadata={
                    'paper_id': 'p001',
                    'chunk_id': 'p001_c001',
                    'chunk_index': 1
                }
            ),
            Document(
                page_content="人流分析方法",
                metadata={
                    'paper_id': 'p002',
                    'chunk_id': 'p002_c000',
                    'chunk_index': 0
                }
            )
        ]
        
        # Test store_chunks
        assert store.store_chunks(docs) == True
        
        # Test get_chunk_count
        assert store.get_chunk_count() == 3
        
        # Test get_paper_count
        assert store.get_paper_count() == 2
        
        # Test get_chunks_by_paper_ids
        chunks_p001 = store.get_chunks_by_paper_ids(['p001'])
        assert len(chunks_p001) == 2
        assert chunks_p001[0]['paper_id'] == 'p001'
        assert chunks_p001[1]['paper_id'] == 'p001'
        
        chunks_p002 = store.get_chunks_by_paper_ids(['p002'])
        assert len(chunks_p002) == 1
        assert chunks_p002[0]['paper_id'] == 'p002'
        
        # Test multiple paper IDs
        chunks_both = store.get_chunks_by_paper_ids(['p001', 'p002'])
        assert len(chunks_both) == 3
        
        # Test clear
        assert store.clear() == True
        assert store.get_chunk_count() == 0
        
        print("✓ All basic tests passed")
        
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_jsonl_store_chinese_text():
    """Test JSONL store with Chinese text"""
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
        temp_path = f.name
    
    try:
        store = JSONLDocumentStore(file_path=temp_path, cache_in_memory=False)
        
        # Chinese text with metadata
        docs = [
            Document(
                page_content="深度學習在人流預測中的應用研究",
                metadata={
                    'paper_id': 'p003',
                    'chunk_id': 'p003_c000',
                    'chunk_index': 0,
                    'title': '深度學習與人流分析',
                    'page': 1
                }
            )
        ]
        
        assert store.store_chunks(docs) == True
        
        chunks = store.get_chunks_by_paper_ids(['p003'])
        assert len(chunks) == 1
        assert "深度學習" in chunks[0]['text']
        assert "人流預測" in chunks[0]['text']
        assert chunks[0]['metadata']['title'] == '深度學習與人流分析'
        
        print("✓ Chinese text test passed")
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_jsonl_store_with_cache():
    """Test JSONL store with memory caching"""
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
        temp_path = f.name
    
    try:
        # Enable caching with low limit to test loading
        store = JSONLDocumentStore(
            file_path=temp_path,
            cache_in_memory=True,
            cache_size_limit_mb=10
        )
        
        # Create test documents
        docs = [
            Document(
                page_content=f"Chunk {i}",
                metadata={
                    'paper_id': f'p{i//10:03d}',
                    'chunk_id': f'p{i//10:03d}_c{i%10:03d}',
                    'chunk_index': i%10
                }
            )
            for i in range(100)
        ]
        
        assert store.store_chunks(docs) == True
        
        # Should load into cache
        stats = store.get_stats()
        print(f"Cache stats: {stats}")
        
        # Test retrieval from cache
        chunks = store.get_chunks_by_paper_ids(['p000', 'p001'])
        assert len(chunks) == 20  # 10 chunks per paper
        
        print("✓ Cache test passed")
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_jsonl_store_empty_paper_ids():
    """Test handling of empty or missing paper IDs"""
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
        temp_path = f.name
    
    try:
        store = JSONLDocumentStore(file_path=temp_path, cache_in_memory=False)
        
        docs = [
            Document(
                page_content="Test",
                metadata={'paper_id': 'p001', 'chunk_id': 'c001', 'chunk_index': 0}
            )
        ]
        
        assert store.store_chunks(docs) == True
        
        # Test with non-existent paper IDs
        chunks = store.get_chunks_by_paper_ids(['p999'])
        assert len(chunks) == 0
        
        # Test with empty list
        chunks = store.get_chunks_by_paper_ids([])
        assert len(chunks) == 0
        
        print("✓ Empty paper IDs test passed")
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == '__main__':
    print("Running Layer 2 Document Store Tests\n")
    print("=" * 60)
    
    test_jsonl_store_basic()
    test_jsonl_store_chinese_text()
    test_jsonl_store_with_cache()
    test_jsonl_store_empty_paper_ids()
    
    print("=" * 60)
    print("\n✅ All document store tests passed!")
