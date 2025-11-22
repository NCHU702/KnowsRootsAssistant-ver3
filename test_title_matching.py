"""
Test Layer1 title matching for exact paper ID queries
"""

from system_api.hierarchical_rag_system import HierarchicalRAGSystem

# Initialize system
print("Initializing RAG system...")
rag_system = HierarchicalRAGSystem(
    pdf_directory='./test_data',
    model_name='jcai/llama-3-taiwan-8b-instruct:q4_k_m',
    embedding_model='quentinz/bge-large-zh-v1.5:latest',
    vectorstore_path='./vectorstore'
)

test_queries = [
    "請仔細介紹鼻癌腫塊辨識_20251106_145324這篇論文在做什麼",
    "芒果分類那篇論文用什麼方法？",
    "基礎5_應用集成式深度學習模型進行芒果分類辨識這篇論文的方法是什麼？"
]

print("\n" + "="*80)
print("Testing Layer1 Title Matching")
print("="*80)

for i, query in enumerate(test_queries, 1):
    print(f"\n{'-'*80}")
    print(f"Test {i}: {query}")
    print('-'*80)
    
    # Call query_single_paper to see which paper is identified
    result = rag_system.query_single_paper(query, return_metadata=True)
    
    if result['status'] == 'success':
        target = result.get('target_paper', {})
        print(f"✓ Identified Paper:")
        print(f"  Title: {target.get('paper_id', 'Unknown')}")
        print(f"  Score: {target.get('score', 0):.4f}")
        print(f"  Method: {target.get('method', 'Unknown')}")
    else:
        print(f"❌ Failed: {result.get('message', 'Unknown error')}")

print("\n" + "="*80)
print("✓ Tests completed")
print("="*80)
