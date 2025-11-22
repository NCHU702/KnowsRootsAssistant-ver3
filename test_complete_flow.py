"""
Complete End-to-End Test for Single Paper Query Flow
Tests: Router → AssistantCall → Layer1 → Layer2
"""

import time
from system_api.hierarchical_rag_system import HierarchicalRAGSystem

print("="*80)
print("Complete Single Paper Query Flow Test")
print("="*80)

# Initialize system
print("\n[1] Initializing RAG system...")
init_start = time.time()
rag_system = HierarchicalRAGSystem(
    pdf_directory='./test_data',
    model_name='jcai/llama-3-taiwan-8b-instruct:q4_k_m',
    embedding_model='quentinz/bge-large-zh-v1.5:latest',
    vectorstore_path='./vectorstore'
)
print(f"✓ Initialization: {time.time() - init_start:.2f}s\n")

# Test queries
test_cases = [
    {
        'query': '請仔細介紹鼻癌腫塊辨識_20251106_145324這篇論文在做什麼',
        'expected_paper': '鼻癌腫塊辨識_20251106_145324',
        'description': 'Exact paper name in query (should skip abstract check)'
    },
    {
        'query': '芒果分類那篇論文用什麼方法？',
        'expected_paper': '基礎5_應用集成式深度學習模型進行芒果分類辨識',
        'description': 'Partial description (should use abstract check)'
    }
]

for i, test in enumerate(test_cases, 1):
    print("="*80)
    print(f"[Test {i}] {test['description']}")
    print("="*80)
    print(f"Query: {test['query']}")
    print(f"Expected Paper: {test['expected_paper']}")
    print("-"*80)
    
    # Call query_single_paper with timing
    start_time = time.time()
    result = rag_system.query_single_paper(test['query'], return_metadata=True)
    total_time = time.time() - start_time
    
    # Check results
    if result['status'] == 'success':
        identified_paper = result['target_paper']['paper_id']
        terminated_at = result.get('terminated_at', 'unknown')
        
        print(f"\n✓ Query completed in {total_time:.2f}s")
        print(f"\nResults:")
        print(f"  Identified Paper: {identified_paper}")
        print(f"  Match: {'✓ CORRECT' if test['expected_paper'] in identified_paper else '✗ WRONG'}")
        print(f"  Terminated at: {terminated_at}")
        print(f"  Layers used: {result['layers_used']}")
        
        # Timing breakdown
        print(f"\nTiming Breakdown:")
        for phase, duration in result['timings'].items():
            print(f"  {phase}: {duration:.3f}s")
        
        # Abstract evaluation
        if 'abstract_evaluation' in result:
            eval = result['abstract_evaluation']
            print(f"\nAbstract Evaluation:")
            print(f"  Confidence: {eval.get('confidence', 0):.2f}")
            print(f"  Should continue to Layer2: {eval.get('should_continue', False)}")
            print(f"  Reasoning: {eval.get('reasoning', 'N/A')[:80]}...")
        
        # Layer2 details
        if 'layer2_docs' in result:
            print(f"\nLayer2 Details:")
            print(f"  Chunks retrieved: {len(result['layer2_docs'])}")
            print(f"  Chunk types filter: {result.get('layer2_chunk_types', 'None')}")
        
        # Answer preview
        answer = result.get('answer', '')
        print(f"\n💬 Answer (first 200 chars):")
        print(f"  {answer[:200]}...")
        
    else:
        print(f"✗ Query failed: {result.get('message', 'Unknown error')}")
    
    print()

print("\n" + "="*80)
print("✓ All tests completed")
print("="*80)
