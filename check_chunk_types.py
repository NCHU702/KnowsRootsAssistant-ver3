"""
檢查芒果論文的 chunk 類型
"""
import json

layer2_file = "./vectorstore/layer2/chunks.jsonl"

try:
    # Read JSONL file
    chunks = []
    with open(layer2_file, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
    
    print(f"Total chunks in Layer2: {len(chunks)}")
    print("\n" + "="*80)
    
    # Find 芒果論文 chunks
    mango_chunks = []
    for chunk in chunks:
        content = chunk.get('content', '')
        metadata = chunk.get('metadata', {})
        title = metadata.get('title', '')
        
        if '芒果' in content or '芒果' in title:
            mango_chunks.append(chunk)
    
    print(f"\n找到 {len(mango_chunks)} 個芒果論文的 chunks")
    print("="*80)
    
    if mango_chunks:
        # Display chunk types
        chunk_type_counts = {}
        for chunk in mango_chunks:
            metadata = chunk.get('metadata', {})
            chunk_type = metadata.get('chunk_type', 'unknown')
            chunk_type_counts[chunk_type] = chunk_type_counts.get(chunk_type, 0) + 1
        
        print("\nChunk Type 分布:")
        for chunk_type, count in sorted(chunk_type_counts.items()):
            print(f"  {chunk_type}: {count} chunks")
        
        print("\n" + "="*80)
        print("前 5 個 chunks 的詳細資訊:")
        print("="*80)
        
        for idx, chunk in enumerate(mango_chunks[:5], 1):
            chunk_id = chunk.get('chunk_id', 'unknown')
            metadata = chunk.get('metadata', {})
            content = chunk.get('content', '')
            
            print(f"\n{idx}. Chunk ID: {chunk_id}")
            print(f"   Metadata: {metadata}")
            print(f"   Content preview: {content[:200]}...")
            print("-"*80)
    else:
        print("\n⚠️  芒果論文沒有在 Layer2 中！")
        print("\n讓我們看看 Layer2 中有什麼論文：")
        print("="*80)
        
        # Group by paper_id
        papers = {}
        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            paper_id = metadata.get('paper_id', 'unknown')
            title = metadata.get('title', 'unknown')
            
            if paper_id not in papers:
                papers[paper_id] = {
                    'title': title,
                    'chunk_count': 0,
                    'chunk_types': []
                }
            
            papers[paper_id]['chunk_count'] += 1
            chunk_type = metadata.get('chunk_type', 'unknown')
            papers[paper_id]['chunk_types'].append(chunk_type)
        
        print(f"\nLayer2 中的論文 ({len(papers)} 篇):")
        for paper_id, info in papers.items():
            chunk_types_str = ', '.join(set(info['chunk_types']))
            print(f"\n  Paper ID: {paper_id[:60]}...")
            print(f"  Title: {info['title'][:80]}...")
            print(f"  Chunks: {info['chunk_count']}")
            print(f"  Chunk types: {chunk_types_str}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
