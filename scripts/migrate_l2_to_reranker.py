#!/usr/bin/env python3
"""
Migration Tool: FAISS Index → JSONL Document Store

Converts existing Layer 2 FAISS index to JSONL format for Cross-Encoder re-ranking.

Usage:
    python scripts/migrate_l2_to_reranker.py --vectorstore ./vectorstore/layer2
    
    # Or with custom output path:
    python scripts/migrate_l2_to_reranker.py --vectorstore ./vectorstore/layer2 --output ./vectorstore/layer2_reranking
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from langchain_ollama import OllamaEmbeddings
from system_api.layer2_vectorstore import Layer2VectorStore
from system_api.layer2_document_store import JSONLDocumentStore
from langchain_core.documents import Document

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_faiss_index(vectorstore_path: str) -> Layer2VectorStore:
    """
    Load existing FAISS-based Layer 2 index
    
    Args:
        vectorstore_path: Path to FAISS vectorstore
        
    Returns:
        Layer2VectorStore instance with loaded FAISS index
    """
    logger.info(f"Loading FAISS index from: {vectorstore_path}")
    
    # Initialize embeddings (needed for API but not used)
    embeddings = OllamaEmbeddings(model="embeddinggemma:latest")
    
    # Create Layer2VectorStore in FAISS mode
    layer2 = Layer2VectorStore(
        embeddings=embeddings,
        vectorstore_path=vectorstore_path,
        use_reranker=False
    )
    
    # Load existing index
    success = layer2.load()
    
    if not success:
        raise RuntimeError(f"Failed to load FAISS index from {vectorstore_path}")
    
    logger.info(f"✓ Loaded FAISS index")
    logger.info(f"  Chunks: {layer2._chunk_count}")
    logger.info(f"  Papers: {layer2._paper_count}")
    
    return layer2


def extract_all_documents(layer2: Layer2VectorStore) -> list[Document]:
    """
    Extract all documents from FAISS vectorstore
    
    Args:
        layer2: Layer2VectorStore instance with loaded FAISS
        
    Returns:
        List of Document objects
    """
    logger.info("Extracting documents from FAISS index...")
    
    if not layer2.vectorstore:
        raise RuntimeError("FAISS vectorstore not initialized")
    
    try:
        # Access the internal document store from FAISS
        # FAISS stores documents in docstore
        docstore = layer2.vectorstore.docstore
        
        documents = []
        for doc_id in docstore._dict.keys():
            doc = docstore._dict[doc_id]
            documents.append(doc)
        
        logger.info(f"✓ Extracted {len(documents)} documents")
        
        return documents
        
    except Exception as e:
        logger.error(f"Failed to extract documents: {e}", exc_info=True)
        raise


def migrate_to_jsonl(
    documents: list[Document],
    output_path: str
) -> bool:
    """
    Migrate documents to JSONL document store
    
    Args:
        documents: List of Document objects to migrate
        output_path: Path to output JSONL file
        
    Returns:
        True if successful
    """
    logger.info(f"Migrating {len(documents)} documents to JSONL...")
    logger.info(f"  Output: {output_path}")
    
    try:
        # Create document store
        doc_store = JSONLDocumentStore(
            file_path=output_path,
            cache_in_memory=True
        )
        
        # Store all documents
        success = doc_store.store_chunks(documents)
        
        if not success:
            logger.error("Failed to store documents")
            return False
        
        # Verify
        stats = doc_store.get_stats()
        logger.info(f"✓ Migration complete")
        logger.info(f"  Chunks stored: {stats['chunk_count']}")
        logger.info(f"  Papers: {stats['paper_count']}")
        logger.info(f"  Cache status: {stats['cache_status']}")
        
        # Verify count matches
        if stats['chunk_count'] != len(documents):
            logger.warning(f"Chunk count mismatch: {stats['chunk_count']} != {len(documents)}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        return False


def verify_migration(
    original_docs: list[Document],
    jsonl_path: str
) -> bool:
    """
    Verify migration by comparing original and migrated documents
    
    Args:
        original_docs: Original documents from FAISS
        jsonl_path: Path to JSONL file
        
    Returns:
        True if verification passed
    """
    logger.info("Verifying migration...")
    
    try:
        # Load from JSONL
        doc_store = JSONLDocumentStore(file_path=jsonl_path)
        migrated_chunks = doc_store.get_chunks_by_paper_ids(None)
        
        # Check count
        if len(migrated_chunks) != len(original_docs):
            logger.error(f"Count mismatch: {len(migrated_chunks)} != {len(original_docs)}")
            return False
        
        # Sample check: verify first 10 documents
        sample_size = min(10, len(original_docs))
        logger.info(f"  Sampling {sample_size} documents for verification...")
        
        for i in range(sample_size):
            orig = original_docs[i]
            migrated = migrated_chunks[i]
            
            # Check text
            if orig.page_content != migrated.get('text', ''):
                logger.error(f"Text mismatch at index {i}")
                return False
            
            # Check metadata keys
            orig_keys = set(orig.metadata.keys())
            migrated_keys = set(migrated.get('metadata', {}).keys())
            
            if orig_keys != migrated_keys:
                logger.warning(f"Metadata keys differ at index {i}: {orig_keys} vs {migrated_keys}")
        
        logger.info("✓ Verification passed")
        return True
        
    except Exception as e:
        logger.error(f"Verification failed: {e}", exc_info=True)
        return False


def main():
    """Main migration function"""
    parser = argparse.ArgumentParser(
        description='Migrate Layer 2 FAISS index to JSONL format for re-ranking'
    )
    parser.add_argument(
        '--vectorstore',
        type=str,
        required=True,
        help='Path to existing FAISS vectorstore directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for JSONL file (default: <vectorstore>/chunks.jsonl)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify migration after completion'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be migrated without actually migrating'
    )
    
    args = parser.parse_args()
    
    # Validate input path
    if not os.path.exists(args.vectorstore):
        logger.error(f"Vectorstore path does not exist: {args.vectorstore}")
        return 1
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(args.vectorstore, "chunks.jsonl")
    
    # Check if output already exists
    if os.path.exists(output_path) and not args.dry_run:
        response = input(f"Output file already exists: {output_path}\nOverwrite? (y/n): ")
        if response.lower() != 'y':
            logger.info("Migration cancelled")
            return 0
    
    logger.info("="*70)
    logger.info("FAISS → JSONL Migration Tool")
    logger.info("="*70)
    logger.info(f"Source: {args.vectorstore}")
    logger.info(f"Target: {output_path}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info("="*70)
    
    try:
        # Step 1: Load FAISS index
        layer2 = load_faiss_index(args.vectorstore)
        
        # Step 2: Extract documents
        documents = extract_all_documents(layer2)
        
        if args.dry_run:
            logger.info("\n" + "="*70)
            logger.info("DRY RUN: Would migrate the following:")
            logger.info(f"  Total documents: {len(documents)}")
            
            # Show sample
            if documents:
                logger.info(f"\n  Sample document:")
                sample = documents[0]
                logger.info(f"    Text: {sample.page_content[:100]}...")
                logger.info(f"    Metadata: {sample.metadata}")
            
            logger.info("="*70)
            logger.info("✓ Dry run complete (no files modified)")
            return 0
        
        # Step 3: Migrate to JSONL
        success = migrate_to_jsonl(documents, output_path)
        
        if not success:
            logger.error("Migration failed")
            return 1
        
        # Step 4: Verify (optional)
        if args.verify:
            logger.info("")
            verify_success = verify_migration(documents, output_path)
            if not verify_success:
                logger.error("Verification failed")
                return 1
        
        # Success summary
        logger.info("\n" + "="*70)
        logger.info("✅ MIGRATION COMPLETE")
        logger.info("="*70)
        logger.info(f"Migrated: {len(documents)} chunks")
        logger.info(f"Output: {output_path}")
        logger.info(f"Size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
        logger.info("\nNext steps:")
        logger.info("  1. Update config: HIERARCHICAL_RAG_CONFIG['layer2_reranking']['enabled'] = True")
        logger.info("  2. Restart system to use re-ranking mode")
        logger.info("="*70)
        
        return 0
        
    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
