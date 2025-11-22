"""
Debug: Test the actual document store get_chunks_by_paper_ids method
"""

import json
from system_api.layer2_document_store import JSONLDocumentStore

# Initialize document store
doc_store = JSONLDocumentStore(file_path='vectorstore/layer2/chunks.jsonl')

# Test paper ID
paper_id = "基礎5_應用集成式深度學習模型進行芒果分類辨識"

print(f"Testing paper_id: {paper_id}")
print(f"\n1. Getting chunks by paper_ids=['{paper_id}']:")
chunks = doc_store.get_chunks_by_paper_ids([paper_id])
print(f"   Got {len(chunks)} chunks")

print(f"\n2. Checking first chunk structure:")
if chunks:
    first_chunk = chunks[0]
    print(f"   Keys: {first_chunk.keys()}")
    print(f"   metadata keys: {first_chunk.get('metadata', {}).keys()}")
    print(f"   section_name: {first_chunk.get('metadata', {}).get('section_name')}")
    print(f"   chunk_type: {first_chunk.get('metadata', {}).get('chunk_type')}")

print(f"\n3. Testing filter logic:")
filter_types = ['Methodology']
filtered = [
    chunk for chunk in chunks
    if (chunk.get('metadata', {}).get('chunk_type') in filter_types or
        chunk.get('metadata', {}).get('section_name') in filter_types)
]
print(f"   Filter types: {filter_types}")
print(f"   Filtered: {len(filtered)} chunks")

if filtered:
    print(f"\n4. Filtered chunk sections:")
    for i, chunk in enumerate(filtered[:5], 1):
        section = chunk.get('metadata', {}).get('section_name')
        chunk_type = chunk.get('metadata', {}).get('chunk_type')
        text_preview = chunk.get('text', '')[:60]
        print(f"   {i}. section={section}, type={chunk_type}")
        print(f"      text: {text_preview}...")
