import os
from system_api.hierarchical_rag_system import HierarchicalRAGSystem

# 初始化系統
rag_system = HierarchicalRAGSystem(
    pdf_directory="./data",
    model_name="llama3.2:latest",
    embedding_model="embeddinggemma:latest",
    vectorstore_path="./vectorstore"
)

# 測試查詢
test_queries = [
    "深度學習在醫療的應用",
    "鼻咽癌腫塊辨識",
    "神經網路預測"
]

print("=" * 80)
print("Embedding Quality Diagnosis")
print("=" * 80)

for query in test_queries:
    print(f"\n查詢: {query}")
    print("-" * 80)
    
    # Layer 1 檢索
    results = rag_system.layer1.search_with_scores(query, k=5, similarity_threshold=None)
    
    for i, (doc, score) in enumerate(results, 1):
        print(f"{i}. {doc.metadata.get('title', 'Unknown')[:50]}")
        print(f"   相似度: {score:.4f}")
        print(f"   摘要前100字: {doc.page_content[:100]}...")
        print()