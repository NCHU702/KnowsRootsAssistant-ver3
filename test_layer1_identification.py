"""
Test Layer1 paper identification for the nasopharyngeal cancer query
"""

from system_api.hierarchical_rag_system import HierarchicalRAGSystem

# Initialize system
rag_system = HierarchicalRAGSystem(
    pdf_directory='./test_data',
    model_name='jcai/llama-3-taiwan-8b-instruct:q4_k_m',
    embedding_model='quentinz/bge-large-zh-v1.5:latest',
    vectorstore_path='./vectorstore'
)

query = "請仔細介紹鼻癌腫塊辨識_20251106_145324這篇論文在做什麼"

print("=" * 80)
print(f"Query: {query}")
print("=" * 80)

# Test Layer1 search
from langchain_core.documents import Document
results = rag_system.layer1.vectorstore.similarity_search_with_relevance_scores(query, k=5)

print(f"\nLayer1 Top-5 Results:")
for i, (doc, score) in enumerate(results, 1):
    title = doc.metadata.get('title', doc.metadata.get('paper_id', 'Unknown'))
    print(f"{i}. [{score:.4f}] {title}")
    if i <= 2:
        abstract_preview = doc.page_content[:100].replace('\n', ' ')
        print(f"   Abstract: {abstract_preview}...")
print()

# Check if exact title match is in results
target_title = "鼻癌腫塊辨識_20251106_145324"
for i, (doc, score) in enumerate(results, 1):
    title = doc.metadata.get('title', doc.metadata.get('paper_id', ''))
    if target_title in title or title in target_title:
        print(f"✓ Found exact match at position {i} with score {score:.4f}")
        break
else:
    print(f"❌ Exact match '{target_title}' not in top-5!")
    print(f"\n Possible reasons:")
    print(f"  1. Title in query doesn't match paper_id in vectorstore")
    print(f"  2. Embedding similarity too low")
    print(f"  3. Abstract doesn't contain relevant keywords")
