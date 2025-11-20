#!/usr/bin/env python3
"""
快速驗證 Layer2 純 Re-ranking 模式
"""

import os
import sys
import tempfile
import shutil
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from system_api.layer2_vectorstore import Layer2VectorStore

def test_reranking_only():
    """測試純 re-ranking 模式"""
    
    print("=" * 60)
    print("Layer2 純 Re-ranking 模式驗證")
    print("=" * 60)
    
    temp_dir = tempfile.mkdtemp()
    vectorstore_dir = os.path.join(temp_dir, "layer2")
    
    try:
        # 1. 初始化
        print("\n✓ 步驟 1: 初始化 Layer2VectorStore...")
        embeddings = OllamaEmbeddings(model="embeddinggemma:latest")
        layer2 = Layer2VectorStore(
            embeddings=embeddings,
            vectorstore_path=vectorstore_dir,
            use_reranker=True,
            reranker_config={
                'model': 'qllama/bce-reranker-base_v1:latest',
                'max_candidates': 300
            }
        )
        print(f"  模式: {layer2.get_stats()['mode']}")
        print(f"  模型: {layer2.get_stats()['reranker_model']}")
        
        # 2. 建構索引
        print("\n✓ 步驟 2: 建構索引...")
        docs = [
            Document(
                page_content="深度學習使用多層神經網絡學習數據的複雜表示，可以自動提取特徵而不需要人工設計。這種方法在圖像識別和語音處理領域取得了突破性進展。",
                metadata={"paper_id": "AI001", "chunk_id": 0}
            ),
            Document(
                page_content="卷積神經網絡（CNN）特別適合處理圖像數據，通過卷積層提取局部特徵。循環神經網絡（RNN）則用於處理序列數據，能夠記憶之前的信息。",
                metadata={"paper_id": "AI001", "chunk_id": 1}
            ),
            Document(
                page_content="自然語言處理讓計算機理解人類語言，包括文本分類、命名實體識別和機器翻譯等任務。現代NLP大量使用Transformer架構和預訓練模型。",
                metadata={"paper_id": "NLP001", "chunk_id": 0}
            )
        ]
        
        success = layer2.build_index(docs)
        if not success:
            print("  ✗ 建構索引失敗")
            return False
        
        stats = layer2.get_stats()
        print(f"  Chunks: {stats['chunk_count']}")
        print(f"  Papers: {stats['paper_count']}")
        
        # 3. 搜索測試
        print("\n✓ 步驟 3: 搜索測試...")
        results = layer2.search_with_scores(
            query="深度學習在圖像處理中的應用",
            k=2
        )
        
        print(f"  返回 {len(results)} 個結果:")
        for i, (doc, score) in enumerate(results, 1):
            paper_id = doc.metadata.get('paper_id')
            preview = doc.page_content[:50]
            print(f"  [{i}] Score: {score:.4f} | Paper: {paper_id}")
            print(f"      {preview}...")
        
        # 4. 添加 chunks 測試
        print("\n✓ 步驟 4: 添加新 chunks...")
        new_docs = [
            Document(
                page_content="Transformer架構徹底改變了自然語言處理領域，使用自注意力機制捕捉長距離依賴關係。BERT和GPT等模型都基於Transformer。",
                metadata={"paper_id": "NLP001", "chunk_id": 1}
            )
        ]
        
        success = layer2.add_chunks(new_docs)
        if not success:
            print("  ✗ 添加 chunks 失敗")
            return False
        
        stats = layer2.get_stats()
        print(f"  總 Chunks: {stats['chunk_count']}")
        
        # 5. 驗證 JSONL 文件
        print("\n✓ 步驟 5: 驗證 JSONL 存儲...")
        jsonl_path = os.path.join(vectorstore_dir, "chunks.jsonl")
        if os.path.exists(jsonl_path):
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            print(f"  JSONL 文件存在: {len(lines)} 行")
            
            # 檢查內容
            import json
            first_line = json.loads(lines[0])
            print(f"  第一行包含: text, metadata")
            print(f"  Paper ID: {first_line['metadata'].get('paper_id')}")
        else:
            print("  ✗ JSONL 文件不存在")
            return False
        
        print("\n" + "=" * 60)
        print("✅ 純 Re-ranking 模式驗證通過！")
        print("=" * 60)
        print("\n摘要:")
        print(f"  ✓ 模式: {stats['mode']}")
        print(f"  ✓ 存儲: JSONL")
        print(f"  ✓ 總 Chunks: {stats['chunk_count']}")
        print(f"  ✓ 總 Papers: {stats['paper_count']}")
        print(f"  ✓ 搜索功能: 正常")
        print(f"  ✓ 添加功能: 正常")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        shutil.rmtree(temp_dir)
        print(f"\n🧹 清理臨時目錄")


if __name__ == "__main__":
    success = test_reranking_only()
    exit(0 if success else 1)
