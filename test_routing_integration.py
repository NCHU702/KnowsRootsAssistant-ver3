"""
測試智能查詢路由器整合

驗證路由器是否正確整合到 HierarchicalRAGSystem 中
"""

import logging
import sys
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from system_api.hierarchical_rag_system import HierarchicalRAGSystem

def test_routing_integration():
    """測試路由器整合"""
    
    logger.info("="*80)
    logger.info("🧪 Testing Query Router Integration")
    logger.info("="*80)
    
    # Initialize RAG system with routing enabled
    try:
        rag_system = HierarchicalRAGSystem(
            pdf_directory="./data",
            model_name="jcai/llama-3-taiwan-8b-instruct:q4_k_m",
            embedding_model="quentinz/bge-large-zh-v1.5:latest",
            vectorstore_path="./vectorstore",
            config={
                'query_routing': {
                    'enabled': True,
                    'default_route': 'rag',
                    'confidence_threshold': 0.7,
                },
                'graph_first': {
                    'enabled': True,
                },
                'layer1': {
                    'k_documents': 10,
                    'confidence_threshold': 0.7,
                },
                'layer2': {
                    'k_documents': 3,
                    'confidence_threshold': 0.8,
                },
            }
        )
        logger.info("✓ RAG System initialized successfully")
        logger.info(f"  Query Router: {'Enabled' if rag_system.query_router else 'Disabled'}")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG system: {e}", exc_info=True)
        return False
    
    # Test queries
    test_queries = [
        ("哪些論文使用 CNN?", "graph"),  # Expected: graph
        ("總結澳門公車軌跡辨識這篇論文", "rag"),  # Expected: rag
        ("比較不同論文的研究方法", "graph"),  # Expected: graph
        ("詳細說明論文的實驗設計", "rag"),  # Expected: rag
    ]
    
    logger.info("\n" + "="*80)
    logger.info("Testing Routing Decisions")
    logger.info("="*80)
    
    for query, expected_route in test_queries:
        logger.info(f"\n📝 Query: {query}")
        logger.info(f"   Expected Route: {expected_route.upper()}")
        logger.info("-" * 80)
        
        try:
            # Test routing (without full retrieval)
            if rag_system.query_router:
                routing_decision = rag_system.query_router.route(query)
                actual_route = routing_decision['route']
                confidence = routing_decision['confidence']
                reasoning = routing_decision['reasoning']
                
                match = "✅" if actual_route == expected_route else "❌"
                logger.info(f"{match} Actual Route: {actual_route.upper()}")
                logger.info(f"   Confidence: {confidence:.2f}")
                logger.info(f"   Reasoning: {reasoning}")
                
                if actual_route != expected_route:
                    logger.warning(f"   ⚠️  Mismatch! Expected {expected_route.upper()}, got {actual_route.upper()}")
            else:
                logger.error("   ❌ Query Router not initialized!")
                
        except Exception as e:
            logger.error(f"   ❌ Error during routing: {e}", exc_info=True)
    
    logger.info("\n" + "="*80)
    logger.info("✅ Test Complete")
    logger.info("="*80)
    
    return True


if __name__ == "__main__":
    test_routing_integration()
