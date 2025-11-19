#!/usr/bin/env python3
"""
檢查 Layer 1 索引中的所有文檔 metadata
"""

import logging
from langchain_ollama import OllamaEmbeddings
from system_api.layer1_vectorstore import Layer1VectorStore

logging.basicConfig(level=logging.WARNING)

embeddings = OllamaEmbeddings(model="quentinz/bge-large-zh-v1.5:latest")
layer1 = Layer1VectorStore(embeddings=embeddings, vectorstore_path="./vectorstore/layer1")
layer1.load()

print("="*80)
print(f"Layer 1 索引分析: 共 {layer1._paper_count} 篇論文")
print("="*80)

if layer1.vectorstore:
    docstore = layer1.vectorstore.docstore
    all_docs = list(docstore._dict.values())
    
    has_pdf_path = 0
    no_pdf_path = 0
    
    print("\n📄 所有文檔:")
    for i, doc in enumerate(all_docs, 1):
        paper_id = doc.metadata.get('paper_id', 'Unknown')
        title = doc.metadata.get('title', 'Unknown')
        pdf_path = doc.metadata.get('pdf_path', '')
        
        if pdf_path:
            has_pdf_path += 1
            status = "✅"
        else:
            no_pdf_path += 1
            status = "❌"
        
        print(f"{i:2}. {status} [{paper_id[:16]}] {title[:60]}")
        if not pdf_path:
            print(f"     ⚠️  缺少 pdf_path")
    
    print("\n" + "="*80)
    print(f"統計:")
    print(f"  ✅ 有 pdf_path: {has_pdf_path} 個")
    print(f"  ❌ 無 pdf_path: {no_pdf_path} 個")
    print(f"  📊 總計: {len(all_docs)} 個")
    print("="*80)
