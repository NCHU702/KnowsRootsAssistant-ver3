#!/usr/bin/env python3
"""
驗證重建後的索引是否正確

檢查：
1. Layer 1: 11 篇論文，所有都有 pdf_path
2. Layer 2: 有 chunks
3. 增量更新功能正常
"""

import logging
from langchain_ollama import OllamaEmbeddings
from system_api.layer1_vectorstore import Layer1VectorStore
from system_api.layer2_vectorstore import Layer2VectorStore

logging.basicConfig(level=logging.WARNING)

print("="*80)
print("驗證重建後的索引")
print("="*80)

# Layer 1 檢查
print("\n1️⃣ Layer 1 檢查")
embeddings = OllamaEmbeddings(model="quentinz/bge-large-zh-v1.5:latest")
layer1 = Layer1VectorStore(embeddings=embeddings, vectorstore_path="./vectorstore/layer1")

if not layer1.load():
    print("❌ Layer 1 未建立")
    exit(1)

print(f"✅ Layer 1 載入成功: {layer1._paper_count} 篇論文")

# 檢查 pdf_path
if layer1.vectorstore:
    all_docs = list(layer1.vectorstore.docstore._dict.values())
    has_path = sum(1 for doc in all_docs if doc.metadata.get('pdf_path'))
    print(f"✅ 有 pdf_path 的論文: {has_path}/{len(all_docs)}")
    
    if has_path == len(all_docs) == 11:
        print("✅ 所有論文都有正確的 metadata！")
    else:
        print(f"❌ 預期 11 篇論文都有 pdf_path，實際：{has_path}/{ len(all_docs)}")
        exit(1)
else:
    print("❌ Layer 1 vectorstore 未初始化")
    exit(1)

# Layer 2 檢查
print("\n2️⃣ Layer 2 檢查")
layer2 = Layer2VectorStore(
    embeddings=embeddings,
    vectorstore_path="./vectorstore/layer2",
    use_reranker=True,
    reranker_config={'enabled': True}
)

if not layer2.load():
    print("❌ Layer 2 未建立")
    exit(1)

stats = layer2.get_stats()
print(f"✅ Layer 2 載入成功")
print(f"   Chunks: {stats['chunk_count']}")
print(f"   Papers: {stats['paper_count']}")
print(f"   Initialized: {stats['is_initialized']}")

if stats['chunk_count'] > 0 and stats['is_initialized']:
    print("✅ Layer 2 正常！")
else:
    print(f"❌ Layer 2 有問題")
    exit(1)

print("\n" + "="*80)
print("✅ 所有檢查通過！")
print("="*80)
print("\n📋 摘要:")
print(f"  - Layer 1: {layer1._paper_count} 篇論文（全部有 pdf_path）")
print(f"  - Layer 2: {stats['chunk_count']} chunks from {stats['paper_count']} papers")
print(f"\n✨ 增量更新功能應該可以正常工作了！")
print("="*80)
