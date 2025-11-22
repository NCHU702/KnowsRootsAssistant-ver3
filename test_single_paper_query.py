"""
Test Single Paper Query Mode

Tests the new single paper query functionality:
1. Router correctly identifies single paper queries
2. Layer1 correctly identifies the target paper
3. Abstract sufficiency check works
4. Layer2 retrieves chunks with semantic filtering
"""

import logging
import sys
from system_api.hierarchical_rag_system import HierarchicalRAGSystem

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_single_paper_query():
    """Test single paper query mode"""
    
    print("=" * 80)
    print("Testing Single Paper Query Mode")
    print("=" * 80)
    
    # Initialize RAG system
    logger.info("Initializing Hierarchical RAG System...")
    rag_system = HierarchicalRAGSystem(
        pdf_directory="./test_data",
        model_name="jcai/llama-3-taiwan-8b-instruct:q4_k_m",
        embedding_model="quentinz/bge-large-zh-v1.5:latest",
        vectorstore_path="./vectorstore",
        chunk_size=800,
        chunk_overlap=100
    )
    
    # Load existing indices
    if not rag_system.layer1.is_initialized:
        logger.error("Layer 1 not initialized! Please build index first.")
        return
    
    if not rag_system.layer2.is_initialized:
        logger.error("Layer 2 not initialized! Please build index first.")
        return
    
    logger.info(f"Layer 1: {rag_system.layer1.get_stats()}")
    logger.info(f"Layer 2: {rag_system.layer2.get_stats()}")
    
    # Test queries
    test_queries = [
        # Single paper queries (should use new mode)
        "芒果分類那篇論文用什麼方法？",
        "交通流量預測的論文使用什麼數據集？",
        "Summarize the paper about traffic flow prediction",
        "What methods does the LSTM paper use?",
        
        # More specific single paper queries
        "那篇用深度學習做芒果分類的論文訓練參數是什麼？",
        "Tell me about the results in the mango classification paper",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print("\n" + "=" * 80)
        print(f"Test Query {i}: {query}")
        print("=" * 80)
        
        try:
            # Test with metadata
            result = rag_system.query_single_paper(query, return_metadata=True)
            
            print(f"\n✓ Query completed")
            print(f"  Mode: {result.get('mode')}")
            print(f"  Status: {result.get('status')}")
            print(f"  Layers used: {result.get('layers_used')}")
            print(f"  Terminated at: {result.get('terminated_at')}")
            
            if 'target_paper' in result:
                paper = result['target_paper']
                print(f"\n📄 Target Paper:")
                print(f"  Title: {paper['title']}")
                print(f"  Score: {paper['score']:.4f}")
            
            if 'abstract_evaluation' in result:
                eval_result = result['abstract_evaluation']
                print(f"\n📊 Abstract Evaluation:")
                print(f"  Confidence: {eval_result['confidence']:.2f}")
                print(f"  Should continue: {eval_result['should_continue']}")
            
            if 'layer2_chunk_types' in result and result['layer2_chunk_types']:
                print(f"\n🏷️  Chunk types retrieved: {result['layer2_chunk_types']}")
            
            print(f"\n💬 Answer:")
            print(f"  {result.get('answer', 'No answer')[:200]}...")
            
            print(f"\n⏱️  Timings:")
            for key, value in result.get('timings', {}).items():
                print(f"  {key}: {value:.3f}s")
            
        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 80)
    print("Testing completed")
    print("=" * 80)


def test_streaming():
    """Test streaming mode"""
    
    print("\n" + "=" * 80)
    print("Testing Single Paper Query Streaming Mode")
    print("=" * 80)
    
    # Initialize RAG system
    rag_system = HierarchicalRAGSystem(
        pdf_directory="./test_data",
        model_name="jcai/llama-3-taiwan-8b-instruct:q4_k_m",
        embedding_model="quentinz/bge-large-zh-v1.5:latest",
        vectorstore_path="./vectorstore"
    )
    
    query = "芒果分類那篇論文用什麼方法？"
    print(f"\nQuery: {query}")
    print("\nStreaming answer:")
    print("-" * 80)
    
    try:
        chunk_count = 0
        for chunk in rag_system.query_single_paper_stream(query):
            print(chunk, end='', flush=True)
            chunk_count += 1
        
        print("\n" + "-" * 80)
        print(f"✓ Streaming completed ({chunk_count} chunks)")
        
    except Exception as e:
        logger.error(f"Streaming failed: {e}", exc_info=True)
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    try:
        # Test blocking mode
        test_single_paper_query()
        
        # Test streaming mode
        test_streaming()
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)
