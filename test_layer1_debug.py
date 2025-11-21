#!/usr/bin/env python3
"""
Debug Layer 1 VectorStore Search Issue
診斷 Layer 1 搜尋問題
"""

import logging
from system_api.layer1_vectorstore import Layer1VectorStore
from langchain_ollama import OllamaEmbeddings

# 設置詳細日誌
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_layer1_search():
    """測試 Layer 1 搜尋功能"""
    
    logger.info("="*80)
    logger.info("🔍 Testing Layer 1 VectorStore Search")
    logger.info("="*80)
    
    # 初始化 embeddings
    logger.info("Initializing embeddings...")
    embeddings = OllamaEmbeddings(
        model="quentinz/bge-large-zh-v1.5:latest",
        base_url="http://localhost:11434"
    )
    logger.info("✓ Embeddings initialized")
    
    # 初始化 Layer 1
    logger.info("\nInitializing Layer 1 VectorStore...")
    layer1 = Layer1VectorStore(
        embeddings=embeddings,
        vectorstore_path="./vectorstore/layer1"
    )
    
    # 載入 index
    logger.info("\nLoading index...")
    success = layer1.load()
    logger.info(f"Load result: {success}")
    
    if not success:
        logger.error("❌ Failed to load index")
        return
    
    logger.info(f"✓ Index loaded: {layer1._paper_count} papers")
    
    # 檢查 vectorstore
    logger.info(f"\nVectorStore object: {layer1.vectorstore}")
    logger.info(f"VectorStore type: {type(layer1.vectorstore)}")
    
    # 測試簡單搜尋
    logger.info("\n" + "="*80)
    logger.info("Test 1: Simple search without threshold")
    logger.info("="*80)
    
    test_query = "澳門公車"
    logger.info(f"Query: {test_query}")
    
    try:
        results = layer1.search_with_scores(
            query=test_query,
            k=3
        )
        
        logger.info(f"\n✓ Search successful! Found {len(results)} results")
        for i, (doc, score) in enumerate(results, 1):
            logger.info(f"\n{i}. Score: {score:.4f}")
            logger.info(f"   Title: {doc.metadata.get('title', 'N/A')}")
            logger.info(f"   Content preview: {doc.page_content[:100]}...")
            
    except Exception as e:
        logger.error(f"❌ Search failed: {e}")
        logger.exception("Full traceback:")
    
    # 測試帶閾值的搜尋
    logger.info("\n" + "="*80)
    logger.info("Test 2: Search with threshold")
    logger.info("="*80)
    
    try:
        results = layer1.search_with_scores(
            query=test_query,
            k=10,
            score_threshold=0.7
        )
        
        logger.info(f"\n✓ Search successful! Found {len(results)} results above threshold 0.7")
        for i, (doc, score) in enumerate(results, 1):
            logger.info(f"\n{i}. Score: {score:.4f}")
            logger.info(f"   Title: {doc.metadata.get('title', 'N/A')}")
            
    except Exception as e:
        logger.error(f"❌ Search with threshold failed: {e}")
        logger.exception("Full traceback:")
    
    # 測試 HybridRetriever 使用的方式
    logger.info("\n" + "="*80)
    logger.info("Test 3: Test embedding generation")
    logger.info("="*80)
    
    try:
        logger.info(f"Generating embedding for query: {test_query}")
        # Test if embeddings work
        test_embed = embeddings.embed_query(test_query)
        logger.info(f"✓ Embedding generated: dimension={len(test_embed)}")
        
    except Exception as e:
        logger.error(f"❌ Embedding generation failed: {e}")
        logger.exception("Full traceback:")


if __name__ == "__main__":
    test_layer1_search()
