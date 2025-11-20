"""
Batch Graph Indexing Script

Indexes all existing papers in ./data or ./test_data to Neo4j Graph database.
Run this after enabling Graph RAG to index historical papers.

Usage:
    python batch_index_to_graph.py [--data-dir ./data] [--test-only]
"""

import sys
import os
import logging
from pathlib import Path
from typing import List, Dict, Any
import PyPDF2

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_ollama import OllamaLLM
from system_api.graph_manager import GraphManager
from system_api.graph_extractor import GraphDataExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path: str, max_pages: int = 10) -> str:
    """Extract text from PDF (first N pages)"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            pages_to_read = min(max_pages, total_pages)
            
            text = ""
            for page_num in range(pages_to_read):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
            
            return text
    except Exception as e:
        logger.error(f"Failed to extract text from {pdf_path}: {e}")
        return ""

def batch_index_papers(
    data_dir: str,
    model_name: str = "jcai/llama-3-taiwan-8b-instruct:q4_k_m",
    test_only: bool = False
) -> Dict[str, Any]:
    """
    Batch index all PDFs in data_dir to Neo4j Graph
    
    Args:
        data_dir: Directory containing PDF files
        model_name: LLM model name for extraction
        test_only: If True, only show what would be indexed without writing
    
    Returns:
        Dict with statistics
    """
    # Initialize components
    logger.info("Initializing LLM and Graph components...")
    try:
        llm = OllamaLLM(model=model_name, temperature=0, base_url="http://localhost:11434")
        graph_manager = GraphManager(llm=llm)
        graph_extractor = GraphDataExtractor(llm=llm)
        
        if not graph_manager.graph:
            logger.error("❌ Failed to connect to Neo4j. Please start Neo4j first.")
            return {'status': 'error', 'message': 'Neo4j connection failed'}
        
        logger.info("✓ Components initialized successfully")
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        return {'status': 'error', 'message': str(e)}
    
    # Find all PDFs
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.error(f"❌ Directory not found: {data_dir}")
        return {'status': 'error', 'message': f'Directory not found: {data_dir}'}
    
    pdf_files = list(data_path.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {data_dir}")
    
    if len(pdf_files) == 0:
        logger.warning("No PDF files found")
        return {'status': 'success', 'papers_indexed': 0}
    
    if test_only:
        logger.info("🔍 TEST MODE - Will show what would be indexed:")
        for pdf_file in pdf_files:
            logger.info(f"  - {pdf_file.name}")
        return {'status': 'test', 'papers_found': len(pdf_files)}
    
    # Process each PDF
    success_count = 0
    error_count = 0
    results = []
    
    for idx, pdf_file in enumerate(pdf_files, 1):
        filename = pdf_file.name
        logger.info(f"\n[{idx}/{len(pdf_files)}] Processing: {filename}")
        
        try:
            # Extract text
            logger.info("  Extracting PDF text...")
            pdf_text = extract_text_from_pdf(str(pdf_file), max_pages=10)
            
            if not pdf_text or len(pdf_text.strip()) < 100:
                logger.warning(f"  ⚠️  Insufficient text extracted, skipping")
                error_count += 1
                results.append({'filename': filename, 'status': 'skipped', 'reason': 'insufficient_text'})
                continue
            
            # Extract metadata
            logger.info("  Extracting metadata with LLM...")
            graph_data = graph_extractor.extract(pdf_text)
            
            logger.info(f"  Extracted: domain={graph_data.get('domain')}, "
                       f"methods={len(graph_data.get('methods', []))}, "
                       f"datasets={len(graph_data.get('datasets', []))}")
            
            # Write to Neo4j (with bilingual domain support)
            logger.info("  Writing to Neo4j...")
            success = graph_manager.add_paper_metadata(
                paper_id=filename,
                title=filename.replace('.pdf', ''),
                year="Unknown",  # Could extract from filename or PDF if needed
                research_goal=graph_data.get('research_goal', ''),
                methods=graph_data.get('methods', []),
                datasets=graph_data.get('datasets', []),
                domain=graph_data.get('domain', 'Unknown'),
                metrics=graph_data.get('metrics', []),
                domain_zh=graph_data.get('domain_zh', '未知領域')
            )
            
            if success:
                logger.info(f"  ✅ Successfully indexed to Graph")
                success_count += 1
                results.append({'filename': filename, 'status': 'success', 'data': graph_data})
            else:
                logger.error(f"  ❌ Failed to write to Neo4j")
                error_count += 1
                results.append({'filename': filename, 'status': 'failed', 'reason': 'neo4j_write_failed'})
                
        except Exception as e:
            logger.error(f"  ❌ Error processing {filename}: {e}")
            error_count += 1
            results.append({'filename': filename, 'status': 'error', 'reason': str(e)})
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("📊 Batch Indexing Complete")
    logger.info("="*60)
    logger.info(f"  Total files: {len(pdf_files)}")
    logger.info(f"  ✅ Success: {success_count}")
    logger.info(f"  ❌ Errors: {error_count}")
    logger.info("="*60)
    
    return {
        'status': 'success',
        'total': len(pdf_files),
        'success': success_count,
        'errors': error_count,
        'results': results
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch index PDFs to Neo4j Graph')
    parser.add_argument('--data-dir', default='./test_data', help='Directory containing PDFs')
    parser.add_argument('--test-only', action='store_true', help='Test mode - show files without indexing')
    parser.add_argument('--model', default='jcai/llama-3-taiwan-8b-instruct:q4_k_m', help='LLM model name')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🚀 Batch Graph Indexing Tool")
    print("="*60)
    print(f"  Data Directory: {args.data_dir}")
    print(f"  Model: {args.model}")
    print(f"  Mode: {'TEST ONLY' if args.test_only else 'LIVE INDEXING'}")
    print("="*60 + "\n")
    
    result = batch_index_papers(
        data_dir=args.data_dir,
        model_name=args.model,
        test_only=args.test_only
    )
    
    if result['status'] == 'success':
        print("\n✅ Batch indexing completed successfully!")
        print(f"\n💡 Verify in Neo4j Browser:")
        print("   MATCH (n)-->(m) RETURN n, m LIMIT 50")
    elif result['status'] == 'test':
        print("\n🔍 Test completed. Run without --test-only to index.")
    else:
        print(f"\n❌ Batch indexing failed: {result.get('message')}")
        sys.exit(1)
