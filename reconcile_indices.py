"""
Index Reconciliation Script

檢查文件系統、Layer1 索引、Layer2 索引和 Neo4j 之間的一致性。
找出重複、缺失或不一致的記錄。

Usage:
    python reconcile_indices.py [--fix]
"""

import sys
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Any
import hashlib

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_filesystem_pdfs(data_dir: str = "./test_data") -> Dict[str, Dict[str, Any]]:
    """獲取文件系統中的所有 PDF 檔案"""
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.warning(f"Directory not found: {data_dir}")
        return {}
    
    pdfs = {}
    for pdf_file in data_path.glob("*.pdf"):
        filename = pdf_file.name
        file_path = str(pdf_file.absolute())
        
        # Generate paper_id using the same method as in the system
        paper_id = hashlib.md5(file_path.encode()).hexdigest()[:16]
        
        pdfs[filename] = {
            'path': file_path,
            'paper_id': paper_id,
            'size': pdf_file.stat().st_size,
            'modified': pdf_file.stat().st_mtime
        }
    
    return pdfs

def get_layer1_papers(vectorstore_path: str = "./vectorstore/layer1") -> Dict[str, Dict[str, Any]]:
    """從 Layer1 索引中讀取已索引的論文"""
    layer1_path = Path(vectorstore_path)
    papers = {}
    
    # Try to read from abstract.faiss and abstract.pkl (Layer1 uses these names)
    index_file = layer1_path / "abstract.faiss"
    pkl_file = layer1_path / "abstract.pkl"
    
    if not index_file.exists() or not pkl_file.exists():
        logger.warning(f"Layer1 index files not found at {vectorstore_path}")
        return papers
    
    try:
        # Load using FAISS
        from langchain_community.vectorstores import FAISS
        from langchain_ollama import OllamaEmbeddings
        
        embeddings = OllamaEmbeddings(
            model="quentinz/bge-large-zh-v1.5:latest",
            base_url="http://localhost:11434"
        )
        
        vectorstore = FAISS.load_local(
            str(layer1_path),
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Get all documents
        docs = vectorstore.docstore._dict
        logger.info(f"Found {len(docs)} documents in Layer1 vectorstore")
        
        for doc_id, doc in docs.items():
            metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            paper_id = metadata.get('paper_id', 'unknown')
            title = metadata.get('title', 'Unknown')
            source = metadata.get('source_file', metadata.get('pdf_path', 'Unknown'))
            
            papers[paper_id] = {
                'title': title,
                'source': source,
                'doc_id': doc_id,
                'metadata': metadata
            }
        
    except Exception as e:
        logger.error(f"Failed to read Layer1 index: {e}")
    
    return papers

def get_layer2_chunks(vectorstore_path: str = "./vectorstore/layer2") -> Dict[str, List[str]]:
    """從 Layer2 索引中讀取已索引的 chunks (按 paper_id 分組)"""
    chunks_by_paper = {}
    
    # Try to read chunks.jsonl
    chunks_file = Path(vectorstore_path) / "chunks.jsonl"
    
    if not chunks_file.exists():
        logger.warning(f"Layer2 chunks file not found: {chunks_file}")
        return chunks_by_paper
    
    try:
        with open(chunks_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    chunk_data = json.loads(line)
                    paper_id = chunk_data.get('metadata', {}).get('paper_id', 'unknown')
                    chunk_id = chunk_data.get('metadata', {}).get('chunk_id', f'chunk_{line_num}')
                    
                    if paper_id not in chunks_by_paper:
                        chunks_by_paper[paper_id] = []
                    chunks_by_paper[paper_id].append(chunk_id)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON at line {line_num}: {e}")
        
        logger.info(f"Found chunks for {len(chunks_by_paper)} papers in Layer2")
        
    except Exception as e:
        logger.error(f"Failed to read Layer2 chunks: {e}")
    
    return chunks_by_paper

def get_neo4j_papers() -> Dict[str, Dict[str, Any]]:
    """從 Neo4j 獲取已索引的論文"""
    papers = {}
    
    try:
        from langchain_community.graphs import Neo4jGraph
        
        graph = Neo4jGraph(
            url="bolt://localhost:7687",
            username="neo4j",
            password="password"
        )
        
        query = """
        MATCH (p:Paper)
        RETURN p.paper_id as paper_id, p.title as title, p.year as year
        ORDER BY p.title
        """
        
        results = graph.query(query)
        
        for row in results:
            paper_id = row['paper_id']
            papers[paper_id] = {
                'title': row['title'],
                'year': row['year']
            }
        
        logger.info(f"Found {len(papers)} papers in Neo4j")
        
    except Exception as e:
        logger.error(f"Failed to query Neo4j: {e}")
    
    return papers

def analyze_consistency(
    filesystem_pdfs: Dict,
    layer1_papers: Dict,
    layer2_chunks: Dict,
    neo4j_papers: Dict
) -> Dict[str, Any]:
    """分析各數據源之間的一致性"""
    
    # Extract keys
    fs_files = set(filesystem_pdfs.keys())
    l1_papers = set(layer1_papers.keys())
    l2_papers = set(layer2_chunks.keys())
    neo4j_ids = set(neo4j_papers.keys())
    
    # Find discrepancies
    analysis = {
        'counts': {
            'filesystem': len(fs_files),
            'layer1': len(l1_papers),
            'layer2': len(l2_papers),
            'neo4j': len(neo4j_ids)
        },
        'missing_in_layer1': [],
        'missing_in_layer2': [],
        'missing_in_neo4j': [],
        'orphaned_in_layer1': [],
        'orphaned_in_layer2': [],
        'orphaned_in_neo4j': [],
        'paper_details': {}
    }
    
    # Check for missing papers (in filesystem but not in indices)
    for filename in fs_files:
        pdf_info = filesystem_pdfs[filename]
        paper_id = pdf_info['paper_id']
        
        analysis['paper_details'][filename] = {
            'paper_id': paper_id,
            'path': pdf_info['path'],
            'in_layer1': paper_id in l1_papers,
            'in_layer2': paper_id in l2_papers,
            'in_neo4j': paper_id in neo4j_papers or filename in neo4j_ids,
            'layer2_chunks': len(layer2_chunks.get(paper_id, []))
        }
        
        if paper_id not in l1_papers:
            analysis['missing_in_layer1'].append(filename)
        
        if paper_id not in l2_papers:
            analysis['missing_in_layer2'].append(filename)
        
        if paper_id not in neo4j_ids and filename not in neo4j_ids:
            analysis['missing_in_neo4j'].append(filename)
    
    # Check for orphaned entries (in indices but not in filesystem)
    for paper_id in l1_papers:
        source = layer1_papers[paper_id].get('source', '')
        if not any(source.endswith(f) for f in fs_files):
            analysis['orphaned_in_layer1'].append({
                'paper_id': paper_id,
                'title': layer1_papers[paper_id].get('title'),
                'source': source
            })
    
    for paper_id in l2_papers:
        if not any(pdf['paper_id'] == paper_id for pdf in filesystem_pdfs.values()):
            analysis['orphaned_in_layer2'].append({
                'paper_id': paper_id,
                'chunks': len(layer2_chunks[paper_id])
            })
    
    for paper_id in neo4j_ids:
        if not any(pdf['paper_id'] == paper_id or paper_id == filename for filename, pdf in filesystem_pdfs.items()):
            analysis['orphaned_in_neo4j'].append({
                'paper_id': paper_id,
                'title': neo4j_papers[paper_id].get('title')
            })
    
    return analysis

def print_report(analysis: Dict[str, Any]):
    """輸出分析報告"""
    print("\n" + "="*80)
    print("INDEX RECONCILIATION REPORT")
    print("="*80)
    
    counts = analysis['counts']
    print(f"\n📊 COUNTS:")
    print(f"  Filesystem PDFs:     {counts['filesystem']}")
    print(f"  Layer1 Papers:       {counts['layer1']}")
    print(f"  Layer2 Papers:       {counts['layer2']}")
    print(f"  Neo4j Papers:        {counts['neo4j']}")
    
    # Check consistency
    all_equal = (counts['filesystem'] == counts['layer1'] == counts['layer2'] == counts['neo4j'])
    if all_equal:
        print("\n✅ All counts match! Indices appear consistent.")
    else:
        print("\n⚠️  COUNTS MISMATCH DETECTED!")
    
    # Missing papers
    if analysis['missing_in_layer1']:
        print(f"\n❌ MISSING IN LAYER1 ({len(analysis['missing_in_layer1'])} files):")
        for filename in analysis['missing_in_layer1']:
            print(f"  - {filename}")
    
    if analysis['missing_in_layer2']:
        print(f"\n❌ MISSING IN LAYER2 ({len(analysis['missing_in_layer2'])} files):")
        for filename in analysis['missing_in_layer2']:
            print(f"  - {filename}")
    
    if analysis['missing_in_neo4j']:
        print(f"\n❌ MISSING IN NEO4J ({len(analysis['missing_in_neo4j'])} files):")
        for filename in analysis['missing_in_neo4j']:
            print(f"  - {filename}")
    
    # Orphaned entries
    if analysis['orphaned_in_layer1']:
        print(f"\n🗑️  ORPHANED IN LAYER1 ({len(analysis['orphaned_in_layer1'])} entries):")
        for entry in analysis['orphaned_in_layer1']:
            print(f"  - paper_id={entry['paper_id']}, title={entry['title']}, source={entry['source']}")
    
    if analysis['orphaned_in_layer2']:
        print(f"\n🗑️  ORPHANED IN LAYER2 ({len(analysis['orphaned_in_layer2'])} entries):")
        for entry in analysis['orphaned_in_layer2']:
            print(f"  - paper_id={entry['paper_id']}, chunks={entry['chunks']}")
    
    if analysis['orphaned_in_neo4j']:
        print(f"\n🗑️  ORPHANED IN NEO4J ({len(analysis['orphaned_in_neo4j'])} entries):")
        for entry in analysis['orphaned_in_neo4j']:
            print(f"  - paper_id={entry['paper_id']}, title={entry['title']}")
    
    # Detailed paper status
    print(f"\n📄 PAPER-BY-PAPER STATUS:")
    for filename, details in analysis['paper_details'].items():
        status_icons = []
        if details['in_layer1']:
            status_icons.append("L1✓")
        else:
            status_icons.append("L1✗")
        
        if details['in_layer2']:
            status_icons.append(f"L2✓({details['layer2_chunks']})")
        else:
            status_icons.append("L2✗")
        
        if details['in_neo4j']:
            status_icons.append("Neo4j✓")
        else:
            status_icons.append("Neo4j✗")
        
        status = " | ".join(status_icons)
        print(f"  {filename:40s} [{status}]")
    
    print("\n" + "="*80)

def main():
    """主程式"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Reconcile indices consistency')
    parser.add_argument('--data-dir', default='./test_data', help='PDF directory to check')
    parser.add_argument('--fix', action='store_true', help='Attempt to fix inconsistencies (NOT IMPLEMENTED YET)')
    args = parser.parse_args()
    
    logger.info("Starting index reconciliation...")
    
    # Gather data from all sources
    logger.info(f"Scanning filesystem: {args.data_dir}")
    filesystem_pdfs = get_filesystem_pdfs(args.data_dir)
    
    logger.info("Reading Layer1 index...")
    layer1_papers = get_layer1_papers()
    
    logger.info("Reading Layer2 chunks...")
    layer2_chunks = get_layer2_chunks()
    
    logger.info("Querying Neo4j...")
    neo4j_papers = get_neo4j_papers()
    
    # Analyze
    analysis = analyze_consistency(filesystem_pdfs, layer1_papers, layer2_chunks, neo4j_papers)
    
    # Print report
    print_report(analysis)
    
    # Save to JSON
    output_file = "reconciliation_report.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    logger.info(f"\n💾 Full report saved to: {output_file}")
    
    if args.fix:
        logger.warning("--fix option not implemented yet. Please manually review the report.")

if __name__ == '__main__':
    main()
