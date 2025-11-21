#!/usr/bin/env python3
"""
Graph-Enabled Query Routing Test
測試啟用 Graph 檢索的完整路由系統

Prerequisites:
- Neo4j running on bolt://localhost:7687
- Graph database populated with paper metadata

Tests:
1. Cross-paper query → Router classifies as GRAPH → Graph retrieval executes
2. Single-paper query → Router classifies as RAG → Traditional RAG flow
"""

import logging
from system_api.hierarchical_rag_system import HierarchicalRAGSystem
from system_api.graph_manager import GraphManager
from system_api.graph_extractor import GraphDataExtractor
from langchain_ollama import OllamaLLM

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_graph_enabled_routing():
    """測試啟用 Graph 的路由系統"""
    
    logger.info("="*80)
    logger.info("🧪 Graph-Enabled Query Routing Test")
    logger.info("="*80)
    
    # 配置 Neo4j 連接
    neo4j_config = {
        'url': "bolt://localhost:7687",
        'username': "neo4j",
        'password': "password"  # 使用與 test_neo4j_connection.py 相同的密碼
    }
    
    # 初始化 LLM（用於 Graph queries）
    llm = OllamaLLM(model="jcai/llama-3-taiwan-8b-instruct:q4_k_m")
    
    logger.info("🔗 Initializing Graph components...")
    
    # 初始化 GraphManager
    try:
        graph_manager = GraphManager(
            llm=llm,
            url=neo4j_config['url'],
            username=neo4j_config['username'],
            password=neo4j_config['password']
        )
        logger.info("✓ GraphManager connected to Neo4j")
    except Exception as e:
        logger.error(f"❌ Failed to connect to Neo4j: {e}")
        logger.error("Please ensure Neo4j is running and credentials are correct")
        return
    
    # 初始化 GraphDataExtractor
    graph_extractor = GraphDataExtractor(llm=llm)
    logger.info("✓ GraphDataExtractor initialized")
    
    # 初始化 RAG 系統（帶 Graph 支持）
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
            'enabled': False,  # 由 router 決定何時使用 Graph
        }
    }
    
    rag_system = HierarchicalRAGSystem(config=config)
    
    # 附加 Graph 組件到 index_manager
    rag_system.index_manager.set_graph_components(graph_manager, graph_extractor)
    logger.info("✓ Graph components attached to index_manager")
    
    # 初始化 GraphRetriever（現在 graph_manager 可用）
    if rag_system.index_manager.graph_manager:
        from system_api.graph_retriever import GraphRetriever
        from system_api.graph_integrator import GraphIntegrator
        
        rag_system.graph_retriever = GraphRetriever(rag_system.index_manager.graph_manager)
        rag_system.graph_integrator = GraphIntegrator(llm=llm)  # GraphIntegrator 需要 llm
        logger.info("✓ GraphRetriever and GraphIntegrator initialized")
    else:
        logger.error("❌ graph_manager not available, Graph retrieval disabled")
        return
    
    logger.info("")
    logger.info("✓ System fully initialized with Graph support")
    logger.info(f"  Query Router: {'Enabled' if rag_system.query_router else 'Disabled'}")
    logger.info(f"  Graph Retriever: {'Available' if rag_system.graph_retriever else 'Not Available'}")
    logger.info(f"  Graph Manager: {'Connected' if rag_system.index_manager.graph_manager else 'Disconnected'}")
    logger.info("")
    
    # 測試案例
    test_cases = [
        {
            'query': "哪些論文使用 CNN?",
            'description': "跨論文結構化查詢（比較多篇論文）",
            'expected_route': 'GRAPH',
            'expected_behavior': "Should execute Graph retrieval"
        },
        {
            'query': "總結澳門公車軌跡辨識這篇論文",
            'description': "單一論文詳細內容檢索",
            'expected_route': 'RAG',
            'expected_behavior': "Should skip Graph, use Layer 1+2 only"
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
            # 執行查詢（返回完整 metadata）
            response = rag_system.query(test_case['query'], return_metadata=True)
            
            # 提取路由資訊
            # The hierarchical system stores routing under 'routing' (older tests used 'route_decision')
            route_decision = response.get('routing', response.get('route_decision', {}))
            # preferred_route may be set as well; fallback to UNKNOWN when missing
            route = (route_decision.get('route') or response.get('preferred_route') or 'UNKNOWN').upper()
            confidence = route_decision.get('confidence', 0.0)
            reasoning = route_decision.get('reasoning', response.get('routing_reasoning', 'N/A'))
            # skip flag may be stored as 'skip_graph_for_this_query' or 'skip_graph'
            skip_graph = response.get('skip_graph_for_this_query', response.get('skip_graph', False))

            # 提取執行的檢索階段 (layers_used is the authoritative list)
            retrieval_stages = response.get('layers_used', response.get('retrieval_stages_executed', []))
            
            logger.info("")
            logger.info("📊 Routing Decision:")
            logger.info(f"   Route: {route}")
            logger.info(f"   Confidence: {confidence:.2f}")
            logger.info(f"   Reasoning: {reasoning}")
            logger.info(f"   Skip Graph: {skip_graph}")
            logger.info("")
            logger.info("🔍 Retrieval Stages Executed:")
            if retrieval_stages:
                for stage in retrieval_stages:
                    logger.info(f"   - {stage}")
            else:
                logger.info("   (None)")
            logger.info("")
            
            # 驗證結果
            route_match = (route == test_case['expected_route'])
            
            # 檢查行為是否符合預期
            if test_case['expected_route'] == 'GRAPH':
                # GRAPH query 應該嘗試執行 Graph retrieval
                behavior_correct = ('graph' in retrieval_stages or 
                                   any('graph' in stage.lower() for stage in retrieval_stages))
            else:  # RAG
                # RAG query 應該跳過 Graph stage
                behavior_correct = skip_graph and 'graph' not in [s.lower() for s in retrieval_stages]
            
            logger.info(f"{'✅' if route_match else '❌'} Route Match: {route_match} ({route} vs {test_case['expected_route']})")
            logger.info(f"{'✅' if behavior_correct else '❌'} Behavior: {'Graph stage executed' if not skip_graph and 'graph' in str(retrieval_stages).lower() else 'Graph stage skipped' if skip_graph else 'Graph stage attempted'}")
            logger.info("")
            
            # 顯示回答預覽
            answer = response.get('answer', 'No answer generated')
            logger.info("💬 Answer Preview:")
            logger.info(f"   {answer[:200]}{'...' if len(answer) > 200 else ''}")
            logger.info("")
            
            results.append({
                'query': test_case['query'],
                'route': route,
                'expected_route': test_case['expected_route'],
                'route_match': route_match,
                'behavior_correct': behavior_correct,
                'stages': retrieval_stages,
                'skip_graph': skip_graph
            })
            
        except Exception as e:
            logger.error(f"❌ Test failed with error: {e}")
            logger.exception("Full traceback:")
            results.append({
                'query': test_case['query'],
                'route': 'ERROR',
                'expected_route': test_case['expected_route'],
                'route_match': False,
                'behavior_correct': False,
                'error': str(e)
            })
    
    # 總結
    logger.info("="*80)
    logger.info("📊 Test Summary")
    logger.info("="*80)
    logger.info(f"Total Tests: {len(results)}")
    correct_routes = sum(1 for r in results if r['route_match'])
    correct_behaviors = sum(1 for r in results if r.get('behavior_correct', False))
    logger.info(f"Correct Routes: {correct_routes}/{len(results)}")
    logger.info(f"Correct Behaviors: {correct_behaviors}/{len(results)}")
    logger.info("")
    
    for i, result in enumerate(results, 1):
        route_icon = '✅' if result['route_match'] else '❌'
        behavior_icon = '✅' if result.get('behavior_correct', False) else '❌'
        logger.info(f"{i}. {route_icon} {behavior_icon} {result['query'][:40]}...")
        logger.info(f"   Route: {result['route']} (expected: {result['expected_route']})")
        if 'stages' in result:
            logger.info(f"   Stages: {result.get('stages', [])}")
        if 'skip_graph' in result:
            logger.info(f"   Skip Graph: {result['skip_graph']}")
        if 'error' in result:
            logger.info(f"   Error: {result['error']}")
    
    logger.info("")
    if correct_routes == len(results) and correct_behaviors == len(results):
        logger.info("="*80)
        logger.info("✅ All Tests Passed!")
        logger.info("="*80)
    else:
        logger.info("="*80)
        logger.info(f"⚠️  Some tests failed: {len(results) - correct_routes} route errors, {len(results) - correct_behaviors} behavior errors")
        logger.info("="*80)


if __name__ == "__main__":
    test_graph_enabled_routing()
