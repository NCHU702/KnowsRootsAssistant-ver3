"""
Index Manager Module

Manages transactional updates to hierarchical indices with backup and rollback capability.
Ensures data consistency when adding or removing documents.
"""

import os
import shutil
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

from system_api.layer1_vectorstore import Layer1VectorStore
from system_api.layer2_vectorstore import Layer2VectorStore
from system_api.abstract_extractor import AbstractExtractor

logger = logging.getLogger(__name__)


class IndexManager:
    """
    Manage hierarchical index updates with transactional safety
    
    Features:
    - Automatic backup before updates
    - Atomic add/remove operations
    - Rollback on failure
    - Backup rotation (keep last N backups)
    """
    
    def __init__(
        self,
        layer1: Layer1VectorStore,
        layer2: Layer2VectorStore,
        abstract_extractor: AbstractExtractor,
        backup_dir: str = "./vectorstore/backups",
        max_backups: int = 5
    ):
        """
        Initialize Index Manager
        
        Args:
            layer1: Layer 1 vectorstore instance
            layer2: Layer 2 vectorstore instance
            abstract_extractor: Abstract extractor instance
            backup_dir: Directory for backups
            max_backups: Maximum number of backups to keep
        """
        self.layer1 = layer1
        self.layer2 = layer2
        self.abstract_extractor = abstract_extractor
        self.backup_dir = backup_dir
        self.max_backups = max_backups
        
        # Ensure backup directory exists
        os.makedirs(backup_dir, exist_ok=True)
        
        logger.info(f"Index Manager initialized")
        logger.info(f"  Backup directory: {backup_dir}")
        logger.info(f"  Max backups: {max_backups}")
    
    def add_document(
        self,
        pdf_path: str,
        pdf_text: str,
        paper_id: str,
        paper_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add a document to both layers with transactional safety
        
        Args:
            pdf_path: Path to PDF file
            pdf_text: Extracted PDF text
            paper_id: Unique paper identifier
            paper_metadata: Optional metadata (title, authors, year, etc.)
            
        Returns:
            Dictionary with operation result
        """
        start_time = time.time()
        logger.info("="*60)
        logger.info(f"Adding document: {paper_id}")
        logger.info("="*60)
        
        backup_id = None
        
        try:
            # Step 1: Create backup
            logger.info("Step 1: Creating backup...")
            backup_id = self._create_backup()
            logger.info(f"  ✓ Backup created: {backup_id}")
            
            # Step 2: Extract abstract
            logger.info("Step 2: Extracting abstract...")
            abstract_result = self.abstract_extractor.extract(
                pdf_path=pdf_path,
                pdf_text=pdf_text,
                paper_id=paper_id
            )
            
            if not abstract_result:
                raise ValueError("Failed to extract abstract")
            
            # Merge metadata
            if paper_metadata:
                abstract_result.update(paper_metadata)
            
            logger.info(f"  ✓ Abstract extracted: {abstract_result['source']} (confidence: {abstract_result['confidence']:.2f})")
            
            # Step 3: Create chunks
            logger.info("Step 3: Creating chunks...")
            from langchain_core.documents import Document
            
            chunks = self.layer2.text_splitter.create_documents(
                texts=[pdf_text],
                metadatas=[{
                    'paper_id': paper_id,
                    'pdf_path': pdf_path,
                    'source': 'index_manager_add'
                }]
            )
            
            # Add chunk metadata
            for idx, chunk in enumerate(chunks):
                chunk.metadata['chunk_id'] = f"{paper_id}_chunk_{idx}"
                chunk.metadata['chunk_index'] = idx
            
            logger.info(f"  ✓ Created {len(chunks)} chunks")
            
            # Step 4: Add to Layer 1
            logger.info("Step 4: Adding to Layer 1...")
            success = self.layer1.add_abstract(abstract_result)
            
            if not success:
                raise ValueError("Failed to add abstract to Layer 1")
            
            logger.info("  ✓ Added to Layer 1")
            
            # Step 5: Add to Layer 2
            logger.info("Step 5: Adding to Layer 2...")
            success = self.layer2.add_chunks(chunks)
            
            if not success:
                raise ValueError("Failed to add chunks to Layer 2")
            
            logger.info(f"  ✓ Added {len(chunks)} chunks to Layer 2")
            
            # Step 6: Save indices
            logger.info("Step 6: Saving indices...")
            self.layer1.save()
            self.layer2.save()
            logger.info("  ✓ Indices saved")
            
            # Clean up old backups
            self._rotate_backups()
            
            duration = time.time() - start_time
            
            logger.info("="*60)
            logger.info(f"Document added successfully in {duration:.2f}s")
            logger.info("="*60)
            
            return {
                'status': 'success',
                'paper_id': paper_id,
                'chunks_added': len(chunks),
                'abstract_source': abstract_result['source'],
                'abstract_confidence': abstract_result['confidence'],
                'duration': duration,
                'backup_id': backup_id
            }
            
        except Exception as e:
            logger.error(f"Failed to add document: {e}", exc_info=True)
            
            # Rollback
            if backup_id:
                logger.warning("Attempting rollback...")
                rollback_success = self._rollback(backup_id)
                if rollback_success:
                    logger.info("✓ Rollback successful")
                else:
                    logger.error("✗ Rollback failed - indices may be corrupted!")
            
            return {
                'status': 'error',
                'paper_id': paper_id,
                'error': str(e),
                'backup_id': backup_id
            }
    
    def remove_document(self, paper_id: str) -> Dict[str, Any]:
        """
        Remove a document from both layers with transactional safety
        
        Args:
            paper_id: Paper identifier to remove
            
        Returns:
            Dictionary with operation result
        """
        start_time = time.time()
        logger.info("="*60)
        logger.info(f"Removing document: {paper_id}")
        logger.info("="*60)
        
        backup_id = None
        
        try:
            # Step 1: Create backup
            logger.info("Step 1: Creating backup...")
            backup_id = self._create_backup()
            logger.info(f"  ✓ Backup created: {backup_id}")
            
            # Step 2: Remove from Layer 1
            logger.info("Step 2: Removing from Layer 1...")
            success = self.layer1.remove_abstract(paper_id)
            
            if not success:
                logger.warning("Paper not found in Layer 1 or removal failed")
            else:
                logger.info("  ✓ Removed from Layer 1")
            
            # Step 3: Remove from Layer 2
            logger.info("Step 3: Removing from Layer 2...")
            success = self.layer2.remove_chunks(paper_id)
            
            if not success:
                logger.warning("Paper not found in Layer 2 or removal failed")
            else:
                logger.info("  ✓ Removed from Layer 2")
            
            # Step 4: Save indices
            logger.info("Step 4: Saving indices...")
            self.layer1.save()
            self.layer2.save()
            logger.info("  ✓ Indices saved")
            
            # Clean up old backups
            self._rotate_backups()
            
            duration = time.time() - start_time
            
            logger.info("="*60)
            logger.info(f"Document removed successfully in {duration:.2f}s")
            logger.info("="*60)
            
            return {
                'status': 'success',
                'paper_id': paper_id,
                'duration': duration,
                'backup_id': backup_id
            }
            
        except Exception as e:
            logger.error(f"Failed to remove document: {e}", exc_info=True)
            
            # Rollback
            if backup_id:
                logger.warning("Attempting rollback...")
                rollback_success = self._rollback(backup_id)
                if rollback_success:
                    logger.info("✓ Rollback successful")
                else:
                    logger.error("✗ Rollback failed - indices may be corrupted!")
            
            return {
                'status': 'error',
                'paper_id': paper_id,
                'error': str(e),
                'backup_id': backup_id
            }
    
    def update_document(
        self,
        paper_id: str,
        pdf_path: str,
        pdf_text: str,
        paper_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Update a document (remove old + add new) with transactional safety
        
        Args:
            paper_id: Paper identifier
            pdf_path: Path to PDF file
            pdf_text: Extracted PDF text
            paper_metadata: Optional metadata
            
        Returns:
            Dictionary with operation result
        """
        logger.info(f"Updating document: {paper_id}")
        
        backup_id = None
        
        try:
            # Create backup
            backup_id = self._create_backup()
            
            # Remove old version
            remove_result = self.remove_document(paper_id)
            if remove_result['status'] == 'error':
                raise ValueError(f"Failed to remove old document: {remove_result['error']}")
            
            # Add new version
            add_result = self.add_document(pdf_path, pdf_text, paper_id, paper_metadata)
            if add_result['status'] == 'error':
                raise ValueError(f"Failed to add new document: {add_result['error']}")
            
            return {
                'status': 'success',
                'paper_id': paper_id,
                'chunks_added': add_result['chunks_added'],
                'backup_id': backup_id
            }
            
        except Exception as e:
            logger.error(f"Failed to update document: {e}")
            
            if backup_id:
                self._rollback(backup_id)
            
            return {
                'status': 'error',
                'paper_id': paper_id,
                'error': str(e),
                'backup_id': backup_id
            }
    
    def _create_backup(self) -> str:
        """
        Create backup of current indices
        
        Returns:
            Backup ID (timestamp)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_id = f"backup_{timestamp}"
        backup_path = os.path.join(self.backup_dir, backup_id)
        
        os.makedirs(backup_path, exist_ok=True)
        
        try:
            # Backup Layer 1
            layer1_src = self.layer1.vectorstore_path
            layer1_dst = os.path.join(backup_path, "layer1")
            
            if os.path.exists(layer1_src):
                shutil.copytree(layer1_src, layer1_dst)
                logger.debug(f"  Backed up Layer 1: {layer1_dst}")
            
            # Backup Layer 2
            layer2_src = self.layer2.vectorstore_path
            layer2_dst = os.path.join(backup_path, "layer2")
            
            if os.path.exists(layer2_src):
                shutil.copytree(layer2_src, layer2_dst)
                logger.debug(f"  Backed up Layer 2: {layer2_dst}")
            
            return backup_id
            
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            # Clean up partial backup
            if os.path.exists(backup_path):
                shutil.rmtree(backup_path)
            raise
    
    def _rollback(self, backup_id: str) -> bool:
        """
        Rollback to a specific backup
        
        Args:
            backup_id: Backup identifier
            
        Returns:
            True if rollback successful, False otherwise
        """
        backup_path = os.path.join(self.backup_dir, backup_id)
        
        if not os.path.exists(backup_path):
            logger.error(f"Backup not found: {backup_id}")
            return False
        
        try:
            # Rollback Layer 1
            layer1_backup = os.path.join(backup_path, "layer1")
            if os.path.exists(layer1_backup):
                # Remove current
                if os.path.exists(self.layer1.vectorstore_path):
                    shutil.rmtree(self.layer1.vectorstore_path)
                
                # Restore backup
                shutil.copytree(layer1_backup, self.layer1.vectorstore_path)
                
                # Reload
                self.layer1.load()
                logger.info("  ✓ Layer 1 rolled back")
            
            # Rollback Layer 2
            layer2_backup = os.path.join(backup_path, "layer2")
            if os.path.exists(layer2_backup):
                # Remove current
                if os.path.exists(self.layer2.vectorstore_path):
                    shutil.rmtree(self.layer2.vectorstore_path)
                
                # Restore backup
                shutil.copytree(layer2_backup, self.layer2.vectorstore_path)
                
                # Reload
                self.layer2.load()
                logger.info("  ✓ Layer 2 rolled back")
            
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def _rotate_backups(self):
        """Remove old backups, keeping only max_backups most recent"""
        try:
            backups = []
            for item in os.listdir(self.backup_dir):
                if item.startswith("backup_"):
                    backup_path = os.path.join(self.backup_dir, item)
                    if os.path.isdir(backup_path):
                        backups.append((item, os.path.getctime(backup_path)))
            
            # Sort by creation time (newest first)
            backups.sort(key=lambda x: x[1], reverse=True)
            
            # Remove old backups
            if len(backups) > self.max_backups:
                for backup_id, _ in backups[self.max_backups:]:
                    backup_path = os.path.join(self.backup_dir, backup_id)
                    shutil.rmtree(backup_path)
                    logger.debug(f"  Removed old backup: {backup_id}")
                
                logger.info(f"  Backup rotation: kept {self.max_backups}, removed {len(backups) - self.max_backups}")
        
        except Exception as e:
            logger.warning(f"Backup rotation failed: {e}")
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """
        List all available backups
        
        Returns:
            List of backup information dictionaries
        """
        backups = []
        
        try:
            for item in os.listdir(self.backup_dir):
                if item.startswith("backup_"):
                    backup_path = os.path.join(self.backup_dir, item)
                    if os.path.isdir(backup_path):
                        stat = os.stat(backup_path)
                        backups.append({
                            'backup_id': item,
                            'created_at': datetime.fromtimestamp(stat.st_ctime),
                            'size_bytes': self._get_dir_size(backup_path),
                            'path': backup_path
                        })
            
            # Sort by creation time (newest first)
            backups.sort(key=lambda x: x['created_at'], reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to list backups: {e}")
        
        return backups
    
    def _get_dir_size(self, path: str) -> int:
        """Get total size of directory in bytes"""
        total = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total += os.path.getsize(filepath)
        except Exception as e:
            logger.warning(f"Failed to calculate directory size: {e}")
        
        return total
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index manager statistics"""
        backups = self.list_backups()
        
        return {
            'backup_dir': self.backup_dir,
            'max_backups': self.max_backups,
            'current_backups': len(backups),
            'total_backup_size_mb': sum(b['size_bytes'] for b in backups) / (1024 * 1024),
            'latest_backup': backups[0] if backups else None
        }
