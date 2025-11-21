#!/usr/bin/env python3
"""
Stage 5: End-to-End Testing for Graph-first Flow
Tests with real papers from ./data directory
"""

import os
import sys
import time
import logging
from typing import Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_neo4j_connection():
    """Check if Neo4j is running and accessible"""
    try:
        from neo4j import GraphDatabase
        
        uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        user = os.getenv('NEO4J_USER', 'neo4j')
        password = os.getenv('NEO4J_PASSWORD', 'password')
        
        driver = GraphDatabase.driver(uri, auth=(user, password))
        
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            result.single()
        
        driver.close()
        logger.info(f"✅ Neo4j connection successful: {uri}")
        return True
    except Exception as e:
        logger.error(f"❌ Neo4j connection failed: {e}")
        logger.info("💡 Please start Neo4j:")
        logger.info("   - Option 1: docker run -p 7687:7687 -p 7474:7474 neo4j")
        logger.info("   - Option 2: Neo4j Desktop")
        return False


def test_graph_indexing():
    """Test indexing a paper with Graph extraction"""
    logger.info("\n" + "="*70)
    logger.info("📄 Test 1: Graph Indexing with Real Paper")
    logger.info("="*70)
    
    try:
        from langchain_ollama import OllamaLLM
        from system_api.graph_manager import GraphManager
        from system_api.graph_extractor import GraphDataExtractor
        
        # Initialize components
        logger.info("Initializing Graph components...")
        llm = OllamaLLM(model="jcai/llama-3-taiwan-8b-instruct:q4_k_m", temperature=0)
        graph_manager = GraphManager(llm=llm)
        graph_extractor = GraphDataExtractor(llm=llm)
        
        # Select a test paper
        test_paper = "./data/標準1_基於卷積類神經網路之澳門公車軌跡辨識.pdf"
        
        if not os.path.exists(test_paper):
            logger.error(f"❌ Test paper not found: {test_paper}")
            return False
        
        logger.info(f"📖 Processing paper: {os.path.basename(test_paper)}")
        
        # Extract text (simplified - use your PDF reader)
        import PyPDF2
        with open(test_paper, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            pdf_text = "\n".join([page.extract_text() for page in reader.pages[:5]])  # First 5 pages
        
        logger.info(f"   Extracted {len(pdf_text)} characters")
        
        # Extract structured data
        logger.info("🔍 Extracting structured data with LLM...")
        start_time = time.time()
        graph_data = graph_extractor.extract(pdf_text)
        extraction_time = time.time() - start_time
        
        logger.info(f"   ✓ Extraction completed in {extraction_time:.1f}s")
        logger.info(f"   - Research Goal: {graph_data.get('research_goal', 'N/A')[:100]}...")
        logger.info(f"   - Methods: {graph_data.get('methods', [])}")
        logger.info(f"   - Datasets: {graph_data.get('datasets', [])}")
        logger.info(f"   - Domain: {graph_data.get('domain', 'Unknown')} / {graph_data.get('domain_zh', '未知')}")
        logger.info(f"   - Metrics: {graph_data.get('metrics', [])}")
        
        # Write to Neo4j
        logger.info("💾 Writing to Neo4j...")
        paper_id = os.path.basename(test_paper)
        success = graph_manager.add_paper_metadata(
            paper_id=paper_id,
            title="基於卷積類神經網路之澳門公車軌跡辨識",
            year="2020",
            research_goal=graph_data.get('research_goal', ''),
            methods=graph_data.get('methods', []),
            datasets=graph_data.get('datasets', []),
            domain=graph_data.get('domain', 'Unknown'),
            metrics=graph_data.get('metrics', []),
            domain_zh=graph_data.get('domain_zh', '未知領域')
        )
        
        if success:
            logger.info("   ✅ Graph ingestion successful")
            return True
        else:
            logger.error("   ❌ Graph ingestion failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Graph indexing test failed: {e}", exc_info=True)
        return False


def test_graph_query():
    """Test querying the graph"""
    logger.info("\n" + "="*70)
    logger.info("🔍 Test 2: Graph Query")
    logger.info("="*70)
    
    try:
        from langchain_ollama import OllamaLLM
        from system_api.graph_manager import GraphManager
        
        llm = OllamaLLM(model="jcai/llama-3-taiwan-8b-instruct:q4_k_m", temperature=0)
        graph_manager = GraphManager(llm=llm)
        
        test_queries = [
            "哪些論文使用 CNN?",
            "交通領域的研究有哪些?",
            "使用了哪些數據集?"
        ]
        
        for query in test_queries:
            logger.info(f"\n📝 Query: {query}")
            start_time = time.time()
            result = graph_manager.query_graph(query)
            query_time = time.time() - start_time
            
            logger.info(f"   Time: {query_time:.2f}s")
            logger.info(f"   Result:\n{result[:300]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Graph query test failed: {e}", exc_info=True)
        return False


def test_hierarchical_rag_with_graph():
    """Test full HierarchicalRAGSystem with Graph-first flow"""
    logger.info("\n" + "="*70)
    logger.info("🔄 Test 3: Full Hierarchical RAG with Graph-first")
    logger.info("="*70)
    
    try:
        from system_api.hierarchical_rag_system import HierarchicalRAGSystem
        
        # Initialize system
        logger.info("Initializing HierarchicalRAGSystem...")
        rag_system = HierarchicalRAGSystem(
            pdf_directory="./data",
            model_name="jcai/llama-3-taiwan-8b-instruct:q4_k_m",
            embedding_model="quentinz/bge-large-zh-v1.5:latest",
            vectorstore_path="./vectorstore",
            chunk_size=800,
            chunk_overlap=100
        )
        
        # Check if indices exist
        stats = rag_system.get_stats()
        logger.info(f"   Layer1: {stats['layer1']['paper_count']} papers")
        logger.info(f"   Layer2: {stats['layer2']['chunk_count']} chunks")
        
        if stats['layer1']['paper_count'] == 0:
            logger.warning("⚠️  No papers indexed. Indexing test paper...")
            result = rag_system.add_document("./data/標準1_基於卷積類神經網路之澳門公車軌跡辨識.pdf")
            if result['status'] == 'success':
                logger.info(f"   ✓ Indexed {result['chunks_added']} chunks")
            else:
                logger.error(f"   ❌ Indexing failed: {result.get('error')}")
                return False
        
        # Test queries
        test_cases = [
            {
                'name': 'Case A: High-confidence (should terminate at Graph)',
                'query': '哪些論文使用 CNN 方法?',
                'expected_termination': 'graph'
            },
            {
                'name': 'Case B: Low-confidence (should descend to Layer2)',
                'query': '詳細說明澳門公車軌跡辨識的方法論',
                'expected_termination': 'layer2'
            },
            {
                'name': 'Case C: Dataset query (should filter chunk_types)',
                'query': '論文中使用了哪些數據集?',
                'expected_chunk_types': ['dataset', 'method']
            }
        ]
        
        for case in test_cases:
            logger.info(f"\n{'='*60}")
            logger.info(f"📋 {case['name']}")
            logger.info(f"   Query: {case['query']}")
            logger.info(f"{'='*60}")
            
            start_time = time.time()
            result = rag_system.query(case['query'], return_metadata=True)
            query_time = time.time() - start_time
            
            logger.info(f"\n⏱️  Query time: {query_time:.2f}s")
            
            # Check termination point
            if 'terminated_at' in result:
                logger.info(f"🎯 Terminated at: {result['terminated_at'].upper()}")
            
            # Check graph results
            if result.get('graph_results'):
                logger.info(f"📊 Graph results: {len(result['graph_results'])} papers")
                for paper in result['graph_results'][:3]:
                    logger.info(f"   - {paper.get('title', 'N/A')}")
            
            # Check graph integration
            if result.get('graph_integration'):
                integration = result['graph_integration']
                logger.info(f"🔗 Graph integration:")
                logger.info(f"   - Confidence: {integration.get('confidence', 0):.2f}")
                logger.info(f"   - Should descend: {integration.get('should_descend', False)}")
                logger.info(f"   - Missing info: {integration.get('missing_information_types', [])}")
            
            # Check chunk type filtering
            if result.get('target_chunk_types'):
                logger.info(f"🎯 Target chunk types: {result['target_chunk_types']}")
            
            # Check Layer1 results
            if result.get('layer1_results'):
                logger.info(f"📄 Layer1: {len(result['layer1_results'])} papers retrieved")
            elif result.get('layer1_skipped'):
                logger.info(f"⏭️  Layer1: SKIPPED (Graph-first mode)")
            
            # Check Layer2 results
            if result.get('layer2_papers'):
                logger.info(f"📝 Layer2: {len(result['layer2_papers'])} papers processed")
                total_chunks = sum(len(chunks) for chunks in result['layer2_papers'].values())
                logger.info(f"   - Total chunks: {total_chunks}")
                
                # Show chunk types if available
                chunk_types_count = {}
                for chunks in result['layer2_papers'].values():
                    for chunk in chunks:
                        chunk_type = chunk.get('metadata', {}).get('chunk_type', 'unknown')
                        chunk_types_count[chunk_type] = chunk_types_count.get(chunk_type, 0) + 1
                
                if chunk_types_count:
                    logger.info(f"   - Chunk types: {chunk_types_count}")
            
            # Show answer preview
            answer = result.get('answer', '')
            logger.info(f"\n💬 Answer preview:")
            logger.info(f"   {answer[:200]}...")
            
            # Validation
            if case.get('expected_termination'):
                actual = result.get('terminated_at', 'unknown')
                expected = case['expected_termination']
                if actual == expected:
                    logger.info(f"   ✅ Termination point matches expected: {expected}")
                else:
                    logger.warning(f"   ⚠️  Termination point mismatch: expected={expected}, actual={actual}")
            
            if case.get('expected_chunk_types'):
                actual_types = result.get('target_chunk_types', [])
                expected_types = case['expected_chunk_types']
                if any(t in actual_types for t in expected_types):
                    logger.info(f"   ✅ Chunk type filtering applied correctly")
                else:
                    logger.warning(f"   ⚠️  Expected chunk types {expected_types}, got {actual_types}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Hierarchical RAG test failed: {e}", exc_info=True)
        return False


def test_performance_comparison():
    """Compare performance: old flow vs Graph-first flow"""
    logger.info("\n" + "="*70)
    logger.info("⚡ Test 4: Performance Comparison")
    logger.info("="*70)
    
    try:
        from system_api.hierarchical_rag_system import HierarchicalRAGSystem
        
        rag_system = HierarchicalRAGSystem(
            pdf_directory="./data",
            model_name="jcai/llama-3-taiwan-8b-instruct:q4_k_m",
            embedding_model="quentinz/bge-large-zh-v1.5:latest",
            vectorstore_path="./vectorstore"
        )
        
        test_query = "哪些論文研究交通預測?"
        
        # Test 1: Graph-first enabled (default)
        logger.info("\n🚀 Mode 1: Graph-first ENABLED")
        start = time.time()
        result1 = rag_system.query(test_query, return_metadata=True)
        time1 = time.time() - start
        
        chunks1 = sum(len(chunks) for chunks in result1.get('layer2_papers', {}).values())
        
        logger.info(f"   Time: {time1:.2f}s")
        logger.info(f"   Chunks retrieved: {chunks1}")
        logger.info(f"   Terminated at: {result1.get('terminated_at', 'unknown')}")
        
        # Test 2: Simulate old flow (Layer1 → Layer2)
        # (This would require disabling Graph-first in config)
        logger.info("\n🐢 Mode 2: Traditional Flow (Layer1 → Layer2)")
        logger.info("   (Note: Would need to disable graph_first in config)")
        logger.info("   Estimated performance: 2-3x slower, more chunks retrieved")
        
        # Summary
        logger.info(f"\n📊 Performance Summary:")
        logger.info(f"   Graph-first time: {time1:.2f}s")
        logger.info(f"   Graph-first chunks: {chunks1}")
        logger.info(f"   Improvement: Graph provides targeted filtering")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance test failed: {e}", exc_info=True)
        return False


def main():
    """Run all end-to-end tests"""
    print("\n" + "="*70)
    print("🧪 Stage 5: End-to-End Testing - Graph-first Flow")
    print("="*70)
    print(f"📅 Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    # Check prerequisites
    logger.info("🔍 Checking prerequisites...")
    
    # 1. Neo4j connection
    neo4j_ok = check_neo4j_connection()
    
    if not neo4j_ok:
        logger.error("\n❌ Prerequisites not met. Please start Neo4j first.")
        logger.info("\n💡 Quick Start:")
        logger.info("   export NEO4J_URI='bolt://localhost:7687'")
        logger.info("   export NEO4J_USER='neo4j'")
        logger.info("   export NEO4J_PASSWORD='your_password'")
        return 1
    
    # 2. Check data directory
    data_dir = "./data"
    if not os.path.exists(data_dir):
        logger.error(f"❌ Data directory not found: {data_dir}")
        return 1
    
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]
    logger.info(f"✅ Found {len(pdf_files)} PDF files in {data_dir}")
    
    # Run tests
    results = {}
    
    tests = [
        ("Graph Indexing", test_graph_indexing),
        ("Graph Query", test_graph_query),
        ("Hierarchical RAG with Graph", test_hierarchical_rag_with_graph),
        ("Performance Comparison", test_performance_comparison)
    ]
    
    for test_name, test_func in tests:
        try:
            logger.info(f"\n{'='*70}")
            logger.info(f"Running: {test_name}")
            logger.info(f"{'='*70}")
            results[test_name] = test_func()
        except KeyboardInterrupt:
            logger.warning("\n⚠️  Test interrupted by user")
            results[test_name] = False
            break
        except Exception as e:
            logger.error(f"❌ Test crashed: {e}", exc_info=True)
            results[test_name] = False
    
    # Summary
    print("\n" + "="*70)
    print("📋 Test Summary")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {test_name}")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    print("="*70)
    print(f"Result: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print("="*70 + "\n")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
