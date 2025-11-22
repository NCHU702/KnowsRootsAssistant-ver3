"""
Quick test for single paper query with chunk type filtering
"""

import logging
from system_api.hierarchical_rag_system import HierarchicalRAGSystem

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_quick():
    """Quick test"""
    
    print("=" * 80)
    print("Quick Single Paper Query Test")
    print("=" * 80)
    
    # Initialize RAG system
    rag_system = HierarchicalRAGSystem(
        pdf_directory="./test_data",
        model_name="jcai/llama-3-taiwan-8b-instruct:q4_k_m",
        embedding_model="quentinz/bge-large-zh-v1.5:latest",
        vectorstore_path="./vectorstore"
    )
    
    # Test query
    query = "芒果分類那篇論文用什麼方法？"
    print(f"\nQuery: {query}")
    print("-" * 80)
    
    # Check if graph_integrator is initialized
    print(f"\nGraph Integrator initialized: {rag_system.graph_integrator is not None}")
    print(f"Chunk Classifier initialized: {rag_system.chunk_classifier is not None}")
    
    # Run query
    result = rag_system.query_single_paper(query, return_metadata=True)
    
    print(f"\n✓ Query completed")
    print(f"  Target: {result.get('target_paper', {}).get('title', 'N/A')}")
    print(f"  Terminated at: {result.get('terminated_at')}")
    print(f"  Chunk types used: {result.get('layer2_chunk_types', 'None')}")
    
    print(f"\n💬 Answer (first 300 chars):")
    print(f"  {result.get('answer', 'No answer')[:300]}...")
    
    print(f"\n⏱️  Timings:")
    for key, value in result.get('timings', {}).items():
        print(f"  {key}: {value:.3f}s")


if __name__ == "__main__":
    test_quick()
