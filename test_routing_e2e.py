#!/usr/bin/env python3
"""
End-to-End Test for Query Routing Integration
測試完整的路由系統：從查詢輸入到最終回答

Tests:
1. Cross-paper query → Should use Graph routing → Graph stage executes
2. Single-paper query → Should use RAG routing → Graph stage skipped
"""

import logging
from system_api.hierarchical_rag_system import HierarchicalRAGSystem

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_routing_e2e():
    """測試完整的路由系統"""
    
    logger.info("="*80)
    logger.info("🧪 End-to-End Query Routing Test")
    logger.info("="*80)
    
    # 初始化系統
    config = {
        'model_name': "jcai/llama-3-taiwan-8b-instruct:q4_k_m",
        'embeddings_model': "quentinz/bge-large-zh-v1.5:latest",
        'pdf_directory': "./data",
        'vectorstore_path': "./vectorstore",
        'query_routing': {
            'enabled': True,  # ✨ 啟用智能查詢路由
            'default_route': 'rag',
            'confidence_threshold': 0.7,
        },
        'graph_first': {
            'enabled': False,  # Graph-first 關閉，由 router 決定
        }
    }
    
    rag_system = HierarchicalRAGSystem(config=config)
    
    logger.info("✓ RAG System initialized")
    logger.info(f"  Query Router: {'Enabled' if rag_system.query_router else 'Disabled'}")
    logger.info(f"  Graph Retriever: {'Available' if rag_system.graph_retriever else 'Not Available'}")
    logger.info("")
    
    # 測試案例
    test_cases = [
        {
            'query': '哪些論文使用 CNN?',
            'expected_route': 'GRAPH',
            'description': '跨論文結構化查詢（比較多篇論文）',
            'expected_behavior': 'Should attempt Graph retrieval (if available)'
        },
        {
            'query': '總結澳門公車軌跡辨識這篇論文',
            'expected_route': 'RAG',
            'description': '單一論文詳細內容檢索',
            'expected_behavior': 'Should skip Graph stage, use Layer 1+2 only'
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        logger.info("="*80)
        logger.info(f"Test Case {i}/{len(test_cases)}")
        logger.info("="*80)
        logger.info(f"📝 Query: {test_case['query']}")
        logger.info(f"   Description: {test_case['description']}")
        logger.info(f"   Expected Route: {test_case['expected_route']}")
        logger.info(f"   Expected Behavior: {test_case['expected_behavior']}")
        logger.info("-"*80)
        
        try:
            # 執行查詢 (return_metadata=True 以獲取完整結果)
            response = rag_system.query(test_case['query'], return_metadata=True)
            
            # 檢查路由決策
            routing_info = response.get('routing', {})
            actual_route = routing_info.get('route', 'UNKNOWN').upper()
            confidence = routing_info.get('confidence', 0.0)
            reasoning = routing_info.get('reasoning', 'N/A')
            
            # 檢查是否跳過 Graph
            skip_graph = response.get('skip_graph_for_this_query', False)
            
            # 檢查檢索階段
            retrieval_stages = []
            if response.get('graph_results'):
                retrieval_stages.append('Graph')
            if response.get('layer1_results'):
                retrieval_stages.append('Layer1')
            if response.get('layer2_chunks'):
                retrieval_stages.append('Layer2')
            
            logger.info("")
            logger.info("📊 Routing Decision:")
            logger.info(f"   Route: {actual_route}")
            logger.info(f"   Confidence: {confidence:.2f}")
            logger.info(f"   Reasoning: {reasoning}")
            logger.info(f"   Skip Graph: {skip_graph}")
            logger.info("")
            logger.info("🔍 Retrieval Stages Executed:")
            for stage in retrieval_stages:
                logger.info(f"   ✓ {stage}")
            if not retrieval_stages:
                logger.info("   (None)")
            logger.info("")
            
            # 驗證路由
            route_correct = actual_route == test_case['expected_route']
            
            # 驗證行為
            if test_case['expected_route'] == 'GRAPH':
                # Graph 查詢應該嘗試使用 Graph（如果可用）
                behavior_correct = not skip_graph
                behavior_msg = "Graph stage attempted" if not skip_graph else "⚠️ Graph stage skipped (unexpected)"
            else:  # RAG
                # RAG 查詢應該跳過 Graph
                behavior_correct = skip_graph
                behavior_msg = "Graph stage skipped correctly" if skip_graph else "⚠️ Graph stage executed (unexpected)"
            
            logger.info(f"✅ Route Match: {route_correct} ({actual_route} vs {test_case['expected_route']})")
            logger.info(f"{'✅' if behavior_correct else '⚠️'} Behavior: {behavior_msg}")
            
            # 顯示回答摘要
            answer = response.get('answer', 'N/A')
            logger.info("")
            logger.info("💬 Answer Preview:")
            logger.info(f"   {answer[:200]}..." if len(answer) > 200 else f"   {answer}")
            
            results.append({
                'query': test_case['query'],
                'expected_route': test_case['expected_route'],
                'actual_route': actual_route,
                'route_correct': route_correct,
                'behavior_correct': behavior_correct,
                'confidence': confidence,
                'stages': retrieval_stages,
                'skip_graph': skip_graph
            })
            
        except Exception as e:
            logger.error(f"❌ Test failed with error: {str(e)}")
            logger.exception(e)
            results.append({
                'query': test_case['query'],
                'expected_route': test_case['expected_route'],
                'error': str(e)
            })
        
        logger.info("")
    
    # 總結
    logger.info("="*80)
    logger.info("📊 Test Summary")
    logger.info("="*80)
    
    total_tests = len(results)
    successful_routes = sum(1 for r in results if r.get('route_correct', False))
    successful_behaviors = sum(1 for r in results if r.get('behavior_correct', False))
    
    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"Correct Routes: {successful_routes}/{total_tests}")
    logger.info(f"Correct Behaviors: {successful_behaviors}/{total_tests}")
    logger.info("")
    
    for i, result in enumerate(results, 1):
        if 'error' in result:
            logger.info(f"{i}. ❌ {result['query'][:50]}... - ERROR")
        else:
            route_icon = "✅" if result['route_correct'] else "❌"
            behavior_icon = "✅" if result['behavior_correct'] else "⚠️"
            logger.info(f"{i}. {route_icon} {behavior_icon} {result['query'][:50]}...")
            logger.info(f"   Route: {result['actual_route']} (expected: {result['expected_route']})")
            logger.info(f"   Stages: {', '.join(result['stages']) if result['stages'] else 'None'}")
            logger.info(f"   Skip Graph: {result['skip_graph']}")
    
    logger.info("")
    logger.info("="*80)
    if successful_routes == total_tests and successful_behaviors == total_tests:
        logger.info("✅ All Tests Passed!")
    else:
        logger.info("⚠️ Some Tests Failed - Review logs above")
    logger.info("="*80)
    
    return results


if __name__ == "__main__":
    test_routing_e2e()
