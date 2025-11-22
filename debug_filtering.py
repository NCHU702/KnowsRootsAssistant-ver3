"""
Debug script to test chunk filtering logic
"""

import json

# Load chunks for the mango paper
paper_id = "基礎5_應用集成式深度學習模型進行芒果分類辨識"
chunks = []

with open('vectorstore/layer2/chunks.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        if data['paper_id'] == paper_id:
            chunks.append(data)

print(f"Total chunks for paper: {len(chunks)}")
print(f"\nSection names:")
for chunk in chunks:
    section = chunk['metadata'].get('section_name', 'N/A')
    chunk_type = chunk['metadata'].get('chunk_type', 'N/A')
    print(f"  - section_name: {section}, chunk_type: {chunk_type}")

# Test filtering
filter_types = ['Dataset', 'Introduction', 'Methodology', 'Results']
print(f"\nFilter types: {filter_types}")

filtered = [
    chunk for chunk in chunks
    if (chunk.get('metadata', {}).get('chunk_type') in filter_types or
        chunk.get('metadata', {}).get('section_name') in filter_types)
]

print(f"Filtered chunks: {len(filtered)}")
print(f"\nFiltered sections:")
for chunk in filtered:
    section = chunk['metadata'].get('section_name', 'N/A')
    print(f"  - {section}")
