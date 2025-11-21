#!/usr/bin/env python3
"""
Quick verification script to test:
1. Cypher fallback strategy
2. trigger_decision bug fix
"""

import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_cypher_fallback():
    """Test the Cypher fallback strategy with a query that should trigger fallback"""
    print("\n" + "="*70)
    print("🧪 Test 1: Cypher Fallback Strategy")
    print("="*70)
    
    try:
        from langchain_ollama import OllamaLLM
        from system_api.graph_manager import GraphManager
        
        llm = OllamaLLM(model="jcai/llama-3-taiwan-8b-instruct:q4_k_m", temperature=0)
        graph_manager = GraphManager(llm=llm)
        
        # This query should return empty on first try (strict Cypher) and succeed on fallback
        test_query = "哪些論文使用 CNN?"
        
        logger.info(f"\n📝 Testing query: {test_query}")
        logger.info("   Expected: Should trigger fallback and return paper title")
        
        result = graph_manager.query_graph(test_query)
        
        logger.info(f"\n✅ Result: {result[:200]}...")
        
        if "基於卷積類神經網路" in result or "CNN" in result:
            print("✅ PASS - Cypher fallback strategy working")
            return True
        else:
            print("⚠️  UNCERTAIN - Result doesn't contain expected content")
            print(f"   Got: {result[:100]}")
            return False
            
    except Exception as e:
        print(f"❌ FAIL - Cypher fallback test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_trigger_decision_fix():
    """Test that trigger_decision bug is fixed in hierarchical retrieval"""
    print("\n" + "="*70)
    print("🧪 Test 2: trigger_decision Bug Fix")
    print("="*70)
    
    try:
        from system_api.hierarchical_rag_system import HierarchicalRAGSystem
        
        logger.info("Initializing HierarchicalRAGSystem...")
        rag = HierarchicalRAGSystem(
            model_name="jcai/llama-3-taiwan-8b-instruct:q4_k_m",
            embedding_model="quentinz/bge-large-zh-v1.5:latest",
            pdf_directory="./data",
            vectorstore_dir="./vectorstore",
            enable_graph=False  # Disable graph to test traditional Layer1→Layer2 flow
        )
        
        test_query = "CNN 方法的應用"
        
        logger.info(f"\n📝 Testing query: {test_query}")
        logger.info("   Expected: Should complete without UnboundLocalError")
        
        result = rag.query(test_query, return_metadata=True)
        
        if 'error' in result:
            print(f"❌ FAIL - Query returned error: {result['error']}")
            return False
        
        if result.get('layers_used'):
            print(f"✅ PASS - Query completed successfully")
            print(f"   Layers used: {result['layers_used']}")
            print(f"   Retrieved docs: {len(result.get('final_docs', []))}")
            return True
        else:
            print("⚠️  UNCERTAIN - Query completed but no layers recorded")
            return False
            
    except UnboundLocalError as e:
        print(f"❌ FAIL - UnboundLocalError still occurs: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"❌ FAIL - Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*70)
    print("🚀 Quick Verification - Stage 5 Fixes")
    print("="*70)
    
    results = []
    
    # Test 1: Cypher fallback
    try:
        result1 = test_cypher_fallback()
        results.append(("Cypher Fallback", result1))
    except KeyboardInterrupt:
        print("\n⚠️  Test 1 cancelled by user")
        results.append(("Cypher Fallback", False))
    
    # Test 2: trigger_decision fix
    try:
        result2 = test_trigger_decision_fix()
        results.append(("trigger_decision Fix", result2))
    except KeyboardInterrupt:
        print("\n⚠️  Test 2 cancelled by user")
        results.append(("trigger_decision Fix", False))
    
    # Summary
    print("\n" + "="*70)
    print("📊 Verification Summary")
    print("="*70)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {test_name}")
    
    total_passed = sum(1 for _, passed in results if passed)
    print("="*70)
    print(f"Result: {total_passed}/{len(results)} tests passed")
    print("="*70)
    
    return 0 if all(passed for _, passed in results) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Testing interrupted by user")
        sys.exit(1)
