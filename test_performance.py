"""
Quick test to measure where the time is spent
"""

import time
from system_api.hierarchical_rag_system import HierarchicalRAGSystem

print("Initializing RAG system...")
init_start = time.time()
rag_system = HierarchicalRAGSystem(
    pdf_directory='./test_data',
    model_name='jcai/llama-3-taiwan-8b-instruct:q4_k_m',
    embedding_model='quentinz/bge-large-zh-v1.5:latest',
    vectorstore_path='./vectorstore'
)
print(f"✓ Initialization took {time.time() - init_start:.2f}s\n")

query = "請仔細介紹鼻癌腫塊辨識_20251106_145324這篇論文在做什麼"

print("="*80)
print(f"Query: {query}")
print("="*80)

# Test just Layer1 search
print("\n1. Testing Layer1 search (k=5)...")
t1 = time.time()
layer1_results = rag_system.layer1.search_with_scores(query=query, k=5)
t1_elapsed = time.time() - t1
print(f"   ✓ Found {len(layer1_results)} results in {t1_elapsed:.3f}s")
for i, (doc, score) in enumerate(layer1_results, 1):
    paper_id = doc.metadata['paper_id']
    print(f"   {i}. [{score:.4f}] {paper_id[:50]}...")

# Test title matching
print("\n2. Testing title matching...")
t2 = time.time()
found = False
for idx, (doc, score) in enumerate(layer1_results):
    paper_id = doc.metadata['paper_id']
    if paper_id in query or query in paper_id:
        print(f"   ✓ Match found at position {idx+1}: {paper_id}")
        found = True
        break
t2_elapsed = time.time() - t2
print(f"   Time: {t2_elapsed:.6f}s")
if not found:
    print("   ✗ No title match in top-5")

print("\n3. Full query_single_paper call...")
t3 = time.time()
result = rag_system.query_single_paper(query, return_metadata=True)
t3_elapsed = time.time() - t3
print(f"   ✓ Total query time: {t3_elapsed:.2f}s")
if result['status'] == 'success':
    print(f"   Paper: {result['target_paper']['paper_id']}")
    print(f"   Terminated at: {result.get('terminated_at', 'unknown')}")

print("\n" + "="*80)
print("Time breakdown:")
print(f"  Layer1 search: {t1_elapsed:.3f}s")
print(f"  Title matching: {t2_elapsed:.6f}s")
print(f"  Full query: {t3_elapsed:.2f}s")
print("="*80)
