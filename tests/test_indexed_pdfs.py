#!/usr/bin/env python3
"""
Test Indexed PDFs Detection

驗證系統能否正確檢測已索引的 PDF
"""

import logging
from langchain_ollama import OllamaEmbeddings
from system_api.layer1_vectorstore import Layer1VectorStore

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def test_indexed_pdfs():
    """測試已索引 PDF 檢測"""
    
    print("="*80)
    print("Testing Indexed PDFs Detection")
    print("="*80)
    
    # 初始化 embeddings
    embeddings = OllamaEmbeddings(model="quentinz/bge-large-zh-v1.5:latest")
    
    # 初始化 Layer 1
    print("\n1️⃣ Loading Layer 1 index...")
    layer1 = Layer1VectorStore(
        embeddings=embeddings,
        vectorstore_path="./vectorstore/layer1"
    )
    
    load_success = layer1.load()
    if not load_success:
        print("❌ Failed to load Layer 1 index")
        return False
    
    print(f"✅ Layer 1 loaded: {layer1._paper_count} papers")
    
    # 嘗試獲取所有已索引的 PDF 路徑
    print("\n2️⃣ Extracting indexed PDF paths...")
    indexed_pdfs = set()
    
    if layer1.vectorstore:
        try:
            # 方法 1: 嘗試 _dict
            docstore = layer1.vectorstore.docstore
            print(f"   Docstore type: {type(docstore)}")
            print(f"   Docstore attributes: {dir(docstore)}")
            
            if hasattr(docstore, '_dict'):
                print("   ✓ Using _dict method")
                all_docs = list(docstore._dict.values())
                for doc in all_docs:
                    pdf_path = doc.metadata.get('pdf_path', '')
                    if pdf_path:
                        indexed_pdfs.add(pdf_path)
            else:
                print("   ⚠️  No _dict attribute found")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n3️⃣ Results:")
    print(f"   Indexed PDFs found: {len(indexed_pdfs)}")
    
    if indexed_pdfs:
        print(f"\n   All Indexed PDFs:")
        for i, pdf in enumerate(sorted(indexed_pdfs), 1):
            print(f"   {i}. {pdf}")
    else:
        print("   ❌ No PDFs found - this is the problem!")
    
    print("\n" + "="*80)
    return len(indexed_pdfs) > 0


if __name__ == "__main__":
    success = test_indexed_pdfs()
    exit(0 if success else 1)
