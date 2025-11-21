#!/usr/bin/env python3
"""
Debug HybridRetriever with actual query
診斷 HybridRetriever 實際查詢問題
"""

import logging
from system_api.hierarchical_rag_system import HierarchicalRAGSystem

# 設置詳細日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_actual_query():
    """測試實際查詢"""
    
    logger.info("="*80)
    logger.info("🔍 Testing Actual Query with Routing")
    logger.info("="*80)
    
    # 使用與 test_routing_e2e 相同的配置，但使用正確的模型
    config = {
        'model_name': "jcai/llama-3-taiwan-8b-instruct:q4_k_m",
        'embeddings_model': "quentinz/bge-large-zh-v1.5:latest",  # 改回原本的模型
        'pdf_directory': "./data",
        'vectorstore_path': "./vectorstore",
        'query_routing': {
            'enabled': True,
            'default_route': 'rag',
            'confidence_threshold': 0.7,
        },
        'graph_first': {
            'enabled': False,
        },
        'layer1': {
            'k_documents': 15,
            'confidence_threshold': 0.7,
            'similarity_threshold': None,  # 不使用閾值過濾
        },
    }
    
    logger.info("Initializing RAG System...")
    rag_system = HierarchicalRAGSystem(config=config)
    
    logger.info("✓ RAG System initialized")
    logger.info("")
    
    # 測試查詢
    test_query = "總結澳門公車軌跡辨識這篇論文"
    
    logger.info("="*80)
    logger.info(f"Query: {test_query}")
    logger.info("="*80)
    
    try:
        # 使用 return_metadata=True 獲取完整資訊
        response = rag_system.query(test_query, return_metadata=True)
        
        # 檢查路由決策
        routing_info = response.get('routing', {})
        logger.info("")
        logger.info("📍 Routing Decision:")
        logger.info(f"   Route: {routing_info.get('route', 'N/A').upper()}")
        logger.info(f"   Confidence: {routing_info.get('confidence', 0):.2f}")
        logger.info(f"   Reasoning: {routing_info.get('reasoning', 'N/A')}")
        logger.info(f"   Skip Graph: {response.get('skip_graph_for_this_query', False)}")
        
        # 檢查 Layer 1 結果
        layer1_results = response.get('layer1_results', [])
        logger.info("")
        logger.info(f"📊 Layer 1 Results: {len(layer1_results)} papers")
        for i, result in enumerate(layer1_results[:3], 1):
            logger.info(f"   {i}. {result.get('title', 'N/A')[:60]}")
            logger.info(f"      Score: {result.get('score', 0):.4f}")
        
        # 檢查 Layer 2 結果
        layer2_chunks = response.get('layer2_chunks', [])
        logger.info("")
        logger.info(f"📄 Layer 2 Chunks: {len(layer2_chunks)} chunks")
        
        # 顯示回答
        answer = response.get('answer', 'N/A')
        logger.info("")
        logger.info("💬 Answer:")
        logger.info(f"   {answer[:300]}...")
        
        logger.info("")
        logger.info("="*80)
        if len(layer1_results) > 0:
            logger.info("✅ Query executed successfully with results!")
        else:
            logger.info("⚠️ Query executed but no results found")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"❌ Query failed: {e}")
        logger.exception("Full traceback:")


if __name__ == "__main__":
    test_actual_query()
