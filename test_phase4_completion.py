#!/usr/bin/env python3
"""
Phase 4 Completion Test
Tests Layer2VectorStore add_chunks() in re-ranking mode

This test validates:
1. add_chunks() can append documents in re-ranking mode  
2. Document count tracking is accurate
3. Search works after adding chunks
4. Both papers can be found in index
"""

import os
import shutil
import tempfile
import json
from pathlib import Path
from langchain_core.documents import Document

# Import components
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from system_api.layer2_vectorstore import Layer2VectorStore
from langchain_ollama import OllamaEmbeddings


def test_phase4_completion():
    """Test Layer2VectorStore add_chunks() with re-ranking mode"""
    
    print("=" * 60)
    print("Phase 4 Completion Test")
    print("Testing add_chunks() in re-ranking mode")
    print("=" * 60)
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    l2_vectorstore_dir = os.path.join(temp_dir, "l2_vectorstore")
    os.makedirs(l2_vectorstore_dir, exist_ok=True)
    
    try:
        # Step 1: Create Layer2 with re-ranking mode
        print("\n1️⃣  Creating Layer2VectorStore with re-ranking mode...")
        embeddings = OllamaEmbeddings(model="embeddinggemma:latest")
        layer2 = Layer2VectorStore(
            embeddings=embeddings,
            vectorstore_path=l2_vectorstore_dir,
            use_reranker=True  # Enable re-ranking mode
        )
        print("   ✓ Layer 2 created (re-ranking mode)")
        jsonl_path = os.path.join(l2_vectorstore_dir, "chunks.jsonl")
        print(f"   ✓ Document store path: {jsonl_path}")
        
        # Step 2: Add first batch of chunks
        print("\n2️⃣  Adding first batch of chunks...")
        batch1_chunks = [
            Document(
                page_content="深度學習是機器學習的一個分支，它使用多層神經網絡來學習數據的複雜表示。深度學習模型可以自動從原始數據中提取特徵，不需要人工設計特徵。",
                metadata={"paper_id": "DL2024001", "chunk_id": 0}
            ),
            Document(
                page_content="卷積神經網絡（CNN）用於圖像處理和計算機視覺任務。循環神經網絡（RNN）用於序列數據和自然語言處理。變換器（Transformer）用於大型語言模型和機器翻譯。",
                metadata={"paper_id": "DL2024001", "chunk_id": 1}
            ),
            Document(
                page_content="深度學習在語音識別、圖像分類、自然語言理解等領域取得了突破性進展。現代深度學習系統能夠處理複雜的現實世界問題並達到人類水平的性能。",
                metadata={"paper_id": "DL2024001", "chunk_id": 2}
            )
        ]
        
        success = layer2.add_chunks(batch1_chunks)
        if success:
            print(f"   ✓ Added {len(batch1_chunks)} chunks for paper DL2024001")
        else:
            print("   ✗ Failed to add first batch")
            return False
        
        # Step 3: Verify JSONL file was created
        print("\n3️⃣  Verifying JSONL storage...")
        if os.path.exists(jsonl_path):
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            chunk_count = len(lines)
            print(f"   ✓ JSONL file exists with {chunk_count} chunks")
            
            # Check first chunk
            first_chunk = json.loads(lines[0])
            print(f"   ✓ First chunk: {first_chunk['text'][:50]}...")
            print(f"   ✓ Metadata: paper_id={first_chunk['metadata'].get('paper_id')}")
        else:
            print("   ✗ JSONL file not found")
            return False
        
        # Step 4: Add second batch of chunks (different paper)
        print("\n4️⃣  Adding second batch of chunks...")
        batch2_chunks = [
            Document(
                page_content="自然語言處理（NLP）是人工智能的一個重要分支，致力於讓計算機理解、解釋和生成人類語言。NLP技術使機器能夠處理和分析大量的自然語言數據。",
                metadata={"paper_id": "NLP2024001", "chunk_id": 0}
            ),
            Document(
                page_content="文本分類判斷文本的類別或情感。命名實體識別識別文本中的人名、地名、組織名等。機器翻譯將一種語言翻譯成另一種語言。問答系統根據問題從文本中找到答案。",
                metadata={"paper_id": "NLP2024001", "chunk_id": 1}
            ),
            Document(
                page_content="現代NLP技術大量使用預訓練語言模型，如BERT、GPT等。這些模型在大規模文本語料上進行預訓練，然後在特定任務上進行微調，顯著提升了性能。",
                metadata={"paper_id": "NLP2024001", "chunk_id": 2}
            )
        ]
        
        success = layer2.add_chunks(batch2_chunks)
        if success:
            print(f"   ✓ Added {len(batch2_chunks)} chunks for paper NLP2024001")
        else:
            print("   ✗ Failed to add second batch")
            return False
        
        # Step 5: Verify chunk count increased
        print("\n5️⃣  Verifying document count...")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            new_lines = f.readlines()
        new_chunk_count = len(new_lines)
        
        expected_count = len(batch1_chunks) + len(batch2_chunks)
        if new_chunk_count == expected_count:
            print(f"   ✓ Chunk count correct: {new_chunk_count} (expected {expected_count})")
        else:
            print(f"   ✗ Chunk count mismatch: {new_chunk_count} (expected {expected_count})")
            return False
        
        # Step 6: Test re-ranking search
        print("\n6️⃣  Testing re-ranking search...")
        results = layer2.search_with_scores(
            query="深度學習在圖像處理中的應用",
            k=3,
            paper_ids=None
        )
        
        if results:
            print(f"   ✓ Search returned {len(results)} results")
            for i, (doc, score) in enumerate(results, 1):
                paper_id = doc.metadata.get('paper_id', 'Unknown')
                preview = doc.page_content[:60].replace('\n', ' ')
                print(f"   [{i}] Score: {score:.4f} | Paper: {paper_id}")
                print(f"       {preview}...")
        else:
            print("   ✗ No search results returned")
            return False
        
        # Step 7: Verify both paper IDs are in index
        print("\n7️⃣  Verifying both papers are in index...")
        paper_ids_found = set()
        for line in new_lines:
            chunk = json.loads(line)
            paper_id = chunk['metadata'].get('paper_id')
            if paper_id:
                paper_ids_found.add(paper_id)
        
        expected_papers = {"DL2024001", "NLP2024001"}
        if expected_papers == paper_ids_found:
            print(f"   ✓ Both papers found: {', '.join(sorted(paper_ids_found))}")
        else:
            print(f"   ✗ Paper IDs mismatch")
            print(f"     Expected: {expected_papers}")
            print(f"     Found: {paper_ids_found}")
            return False
        
        # All tests passed
        print("\n" + "=" * 60)
        print("✅ PHASE 4 COMPLETION TEST PASSED")
        print("=" * 60)
        print("\nValidated:")
        print("  ✓ add_chunks() can append documents in re-ranking mode")
        print("  ✓ JSONL storage works correctly")
        print("  ✓ Document count tracking is accurate")
        print("  ✓ Re-ranking search works after adding chunks")
        print("  ✓ Multiple papers can be indexed")
        print(f"  ✓ Total chunks indexed: {new_chunk_count}")
        print(f"  ✓ Total papers indexed: {len(paper_ids_found)}")
        print("\n📦 Phase 4 (Index Building Pipeline) is COMPLETE!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        try:
            shutil.rmtree(temp_dir)
            print(f"\n🧹 Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            print(f"⚠️  Failed to clean up: {e}")


if __name__ == "__main__":
    success = test_phase4_completion()
    exit(0 if success else 1)
