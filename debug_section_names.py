"""
Debug: Check the actual section names in the identified paper's chunks
"""

import json
from system_api.layer2_document_store import JSONLDocumentStore

# Initialize document store
doc_store = JSONLDocumentStore(file_path='vectorstore/layer2/chunks.jsonl')

# Test paper ID
paper_id = "基礎4_應用混合式深度學習開發適應於小資料集之鼻咽癌分類模型"

print(f"Checking paper: {paper_id}")
print(f"\n1. Getting all chunks for this paper:")
chunks = doc_store.get_chunks_by_paper_ids([paper_id])
print(f"   Total chunks: {len(chunks)}")

if chunks:
    print(f"\n2. Section names in these chunks:")
    sections = set()
    for chunk in chunks:
        section = chunk.get('metadata', {}).get('section_name')
        sections.add(section)
        print(f"   - {section}")
    
    print(f"\n3. Testing filter: ['Introduction', 'Results', 'Methodology', 'Dataset']")
    filter_types = ['Introduction', 'Results', 'Methodology', 'Dataset']
    
    filtered = [
        chunk for chunk in chunks
        if (chunk.get('metadata', {}).get('chunk_type') in filter_types or
            chunk.get('metadata', {}).get('section_name') in filter_types)
    ]
    
    print(f"   Filtered count: {len(filtered)}")
    
    if not filtered:
        print(f"\n4. Why filtering failed:")
        print(f"   Filter expects: {filter_types}")
        print(f"   Actual sections: {sorted(sections)}")
        print(f"   → No match!")
