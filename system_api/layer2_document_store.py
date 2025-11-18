"""
Layer 2 Document Store Module

Provides abstract interface and implementations for storing/retrieving document chunks
without vector embeddings. This replaces FAISS for Layer 2 storage in the re-ranking architecture.
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class Layer2DocumentStore(ABC):
    """
    Abstract interface for Layer 2 document storage
    
    Implementations must support:
    - Storing chunks with metadata
    - Fast retrieval by paper IDs
    - Counting operations
    """
    
    @abstractmethod
    def store_chunks(self, documents: List[Document]) -> bool:
        """
        Store document chunks in the store
        
        Args:
            documents: List of Document objects with metadata
                Required metadata fields:
                - paper_id: str
                - chunk_id: str
                - chunk_index: int
                
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_chunks_by_paper_ids(self, paper_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve all chunks belonging to specified papers
        
        Args:
            paper_ids: List of paper IDs to filter by
            
        Returns:
            List of dictionaries with keys:
            - paper_id: str
            - chunk_id: str
            - chunk_index: int
            - text: str (chunk content)
            - metadata: dict (all metadata)
        """
        pass
    
    @abstractmethod
    def get_chunk_count(self) -> int:
        """
        Get total number of chunks in store
        
        Returns:
            Number of chunks
        """
        pass
    
    @abstractmethod
    def get_paper_count(self) -> int:
        """
        Get total number of unique papers in store
        
        Returns:
            Number of unique papers
        """
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """
        Clear all data from the store
        
        Returns:
            True if successful, False otherwise
        """
        pass


class JSONLDocumentStore(Layer2DocumentStore):
    """
    JSONL-based document store implementation
    
    Stores chunks as JSON Lines format (one JSON object per line).
    Simple, human-readable, and requires no external dependencies.
    
    Suitable for corpus sizes up to ~50k chunks (~1GB file).
    """
    
    def __init__(
        self,
        file_path: str = "./vectorstore/layer2/chunks.jsonl",
        cache_in_memory: bool = True,
        cache_size_limit_mb: int = 100
    ):
        """
        Initialize JSONL document store
        
        Args:
            file_path: Path to JSONL file
            cache_in_memory: If True, load entire file into memory when size < limit
            cache_size_limit_mb: Maximum file size (MB) to cache in memory
        """
        self.file_path = file_path
        self.cache_in_memory = cache_in_memory
        self.cache_size_limit_mb = cache_size_limit_mb
        
        # In-memory cache
        self._cache: Optional[List[Dict[str, Any]]] = None
        self._cache_loaded = False
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        logger.info(f"Initialized JSONL Document Store: {file_path}")
        if cache_in_memory:
            logger.info(f"  Memory caching enabled (limit: {cache_size_limit_mb}MB)")
    
    def store_chunks(self, documents: List[Document]) -> bool:
        """
        Store documents as JSONL (one JSON object per line)
        
        Args:
            documents: List of Document objects
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not documents:
                logger.warning("No documents provided to store")
                return False
            
            logger.info(f"Storing {len(documents)} chunks to JSONL: {self.file_path}")
            
            # Write JSONL file
            with open(self.file_path, 'w', encoding='utf-8') as f:
                for doc in documents:
                    # Extract required fields
                    chunk_data = {
                        'paper_id': doc.metadata.get('paper_id', ''),
                        'chunk_id': doc.metadata.get('chunk_id', ''),
                        'chunk_index': doc.metadata.get('chunk_index', 0),
                        'text': doc.page_content,
                        'metadata': doc.metadata
                    }
                    
                    # Validate required fields
                    if not chunk_data['paper_id']:
                        logger.warning(f"Skipping chunk without paper_id")
                        continue
                    
                    # Write JSON line
                    f.write(json.dumps(chunk_data, ensure_ascii=False) + '\n')
            
            # Clear cache to force reload
            self._cache = None
            self._cache_loaded = False
            
            logger.info(f"✓ Successfully stored {len(documents)} chunks")
            
            # Load cache if enabled
            if self.cache_in_memory:
                self._load_cache()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store chunks: {e}", exc_info=True)
            return False
    
    def get_chunks_by_paper_ids(self, paper_ids: Optional[List[str]]) -> List[Dict[str, Any]]:
        """
        Retrieve chunks filtered by paper IDs
        
        Args:
            paper_ids: List of paper IDs, or None to get all chunks
            
        Returns:
            List of chunk dictionaries
        """
        try:
            # If None, return all chunks
            if paper_ids is None:
                if self._use_cache():
                    logger.debug(f"Retrieved all {len(self._cache)} chunks from cache")
                    return self._cache.copy()
                else:
                    # Read all from file
                    return self._read_all_chunks()
            
            if not paper_ids:
                logger.warning("Empty paper IDs list")
                return []
            
            paper_id_set = set(paper_ids)
            
            # Use cache if available
            if self._use_cache():
                chunks = [
                    chunk for chunk in self._cache
                    if chunk['paper_id'] in paper_id_set
                ]
                logger.debug(f"Retrieved {len(chunks)} chunks from cache for {len(paper_ids)} papers")
                return chunks
            
            # Read from file
            chunks = []
            if not os.path.exists(self.file_path):
                logger.warning(f"JSONL file not found: {self.file_path}")
                return []
            
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        chunk_data = json.loads(line)
                        if chunk_data.get('paper_id') in paper_id_set:
                            chunks.append(chunk_data)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON line: {e}")
                        continue
            
            logger.debug(f"Retrieved {len(chunks)} chunks from file for {len(paper_ids)} papers")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to retrieve chunks: {e}", exc_info=True)
            return []
    
    def _read_all_chunks(self) -> List[Dict[str, Any]]:
        """
        Read all chunks from file
        
        Returns:
            List of all chunk dictionaries
        """
        try:
            if not os.path.exists(self.file_path):
                return []
            
            chunks = []
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        chunk_data = json.loads(line)
                        chunks.append(chunk_data)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON line: {e}")
                        continue
            
            logger.debug(f"Retrieved all {len(chunks)} chunks from file")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to read all chunks: {e}", exc_info=True)
            return []
    
    def get_chunk_count(self) -> int:
        """
        Count total chunks in store
        
        Returns:
            Number of chunks
        """
        try:
            # Use cache if available
            if self._use_cache():
                return len(self._cache)
            
            # Count from file
            if not os.path.exists(self.file_path):
                return 0
            
            count = 0
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        count += 1
            
            return count
            
        except Exception as e:
            logger.error(f"Failed to count chunks: {e}", exc_info=True)
            return 0
    
    def get_paper_count(self) -> int:
        """
        Count unique papers in store
        
        Returns:
            Number of unique papers
        """
        try:
            # Use cache if available
            if self._use_cache():
                paper_ids = set(chunk['paper_id'] for chunk in self._cache)
                return len(paper_ids)
            
            # Count from file
            if not os.path.exists(self.file_path):
                return 0
            
            paper_ids = set()
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        chunk_data = json.loads(line)
                        paper_ids.add(chunk_data.get('paper_id', ''))
                    except json.JSONDecodeError:
                        continue
            
            return len(paper_ids)
            
        except Exception as e:
            logger.error(f"Failed to count papers: {e}", exc_info=True)
            return 0
    
    def clear(self) -> bool:
        """
        Clear JSONL file and cache
        
        Returns:
            True if successful
        """
        try:
            if os.path.exists(self.file_path):
                os.remove(self.file_path)
                logger.info(f"Cleared JSONL file: {self.file_path}")
            
            self._cache = None
            self._cache_loaded = False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear store: {e}", exc_info=True)
            return False
    
    def _use_cache(self) -> bool:
        """
        Check if cache is available for use
        
        Returns:
            True if cache is loaded and usable
        """
        if not self.cache_in_memory:
            return False
        
        if self._cache_loaded and self._cache is not None:
            return True
        
        # Try to load cache
        if not self._cache_loaded:
            self._load_cache()
        
        return self._cache is not None
    
    def _load_cache(self):
        """
        Load entire JSONL file into memory if size permits
        """
        try:
            if not os.path.exists(self.file_path):
                self._cache_loaded = True
                return
            
            # Check file size
            file_size_mb = os.path.getsize(self.file_path) / (1024 * 1024)
            
            if file_size_mb > self.cache_size_limit_mb:
                logger.info(f"JSONL file too large to cache ({file_size_mb:.1f}MB > {self.cache_size_limit_mb}MB)")
                self._cache_loaded = True
                return
            
            # Load into memory
            logger.info(f"Loading JSONL into memory ({file_size_mb:.1f}MB)...")
            chunks = []
            
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        chunk_data = json.loads(line)
                        chunks.append(chunk_data)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON line: {e}")
                        continue
            
            self._cache = chunks
            self._cache_loaded = True
            
            logger.info(f"✓ Loaded {len(chunks)} chunks into memory")
            
        except Exception as e:
            logger.error(f"Failed to load cache: {e}", exc_info=True)
            self._cache_loaded = True  # Mark as attempted
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get store statistics
        
        Returns:
            Dictionary with statistics
        """
        return {
            'file_path': self.file_path,
            'chunk_count': self.get_chunk_count(),
            'paper_count': self.get_paper_count(),
            'cache_enabled': self.cache_in_memory,
            'cache_loaded': self._cache_loaded,
            'cache_size': len(self._cache) if self._cache else 0
        }
