"""
Test Layer2 section filtering specifically
"""

import logging
from system_api.hierarchical_rag_system import HierarchicalRAGSystem

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_layer2_filtering():
    """Test that Layer2 section filtering works correctly"""
    
    print("=" * 80)
    print("Test Layer2 Section Filtering")
    print("=" * 80)
    
    # Initialize RAG system
    rag_system = HierarchicalRAGSystem(
        pdf_directory="./test_data",
        model_name="jcai/llama-3-taiwan-8b-instruct:q4_k_m",
        embedding_model="quentinz/bge-large-zh-v1.5:latest",
        vectorstore_path="./vectorstore",
        config={
            'layer1': {
                'confidence_threshold': 0.3,  # 降低阈值强制进入 Layer2
            }
        }
    )
    
    # Test queries that should trigger Layer2
    queries = [
        "芒果分類論文用什麼訓練參數？",  # 训练参数（需要详细 Methodology）
        "芒果分類論文的實驗結果準確率是多少？",  # 实验结果（需要 Results）
        "交通流量預測論文使用什麼數據集？",  # 数据集（需要 Dataset）
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}: {query}")
        print(f"{'='*80}\n")
        
        result = rag_system.query_single_paper(query, return_metadata=True)
        
        print(f"✓ Query completed")
        print(f"  Target: {result.get('target_paper', {}).get('title', 'N/A')[:60]}...")
        print(f"  Terminated at: {result.get('terminated_at')}")
        print(f"  Section names used: {result.get('layer2_chunk_types', 'None')}")
        
        if result.get('terminated_at') == 'layer2':
            layer2_docs = result.get('layer2_docs', [])
            print(f"  Retrieved chunks: {len(layer2_docs)}")
            
            if layer2_docs:
                print(f"\n  Chunk sections retrieved:")
                for j, doc in enumerate(layer2_docs[:3], 1):
                    section = doc.metadata.get('section_name', 'Unknown')
                    preview = doc.page_content[:60].replace('\n', ' ')
                    print(f"    {j}. [{section}] {preview}...")
        
        print(f"\n  Answer preview:")
        print(f"    {result.get('answer', 'No answer')[:200]}...")


if __name__ == "__main__":
    test_layer2_filtering()
