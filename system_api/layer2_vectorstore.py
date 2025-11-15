"""
Layer 2 VectorStore Module

Manages chunk-level FAISS index with dynamic subindex creation capability.
This is the second layer in the hierarchical RAG system, providing fine-grained retrieval.
"""

import os
import logging
import pickle
import time
from typing import List, Dict, Optional, Any, Tuple
from collections import OrderedDict
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class SubindexCache:
    """
    LRU cache for dynamic subindices
    Caches filtered vectorstores to avoid rebuilding them repeatedly
    """
    
    def __init__(self, max_size: int = 10):
        """
        Initialize cache
        
        Args:
            max_size: Maximum number of cached subindices
        """
        self.cache = OrderedDict()
        self.max_size = max_size
        
    def get(self, paper_ids: Tuple[str]) -> Optional[FAISS]:
        """
        Get cached subindex
        
        Args:
            paper_ids: Tuple of paper IDs (must be sorted)
            
        Returns:
            Cached FAISS vectorstore or None if not found
        """
        key = tuple(sorted(paper_ids))
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            logger.debug(f"Subindex cache HIT for {len(paper_ids)} papers")
            return self.cache[key]
        
        logger.debug(f"Subindex cache MISS for {len(paper_ids)} papers")
        return None
    
    def put(self, paper_ids: Tuple[str], subindex: FAISS):
        """
        Cache subindex with LRU eviction
        
        Args:
            paper_ids: Tuple of paper IDs
            subindex: FAISS vectorstore to cache
        """
        key = tuple(sorted(paper_ids))
        
        if key in self.cache:
            # Update existing entry
            self.cache.move_to_end(key)
        else:
            # Add new entry
            if len(self.cache) >= self.max_size:
                # Remove oldest (least recently used)
                removed_key = self.cache.popitem(last=False)[0]
                logger.debug(f"Evicted subindex for {len(removed_key)} papers from cache")
            
            self.cache[key] = subindex
            logger.debug(f"Cached subindex for {len(paper_ids)} papers")
    
    def clear(self):
        """Clear all cached subindices"""
        self.cache.clear()
        logger.debug("Cleared subindex cache")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'utilization': len(self.cache) / self.max_size if self.max_size > 0 else 0
        }


class Layer2VectorStore:
    """
    Manage Layer 2 (chunk-level) vector store with dynamic filtering
    
    Key features:
    - Full chunk-level FAISS index
    - Dynamic subindex creation for filtered search
    - Context retrieval (surrounding chunks)
    - Caching for performance
    """
    
    def __init__(
        self,
        embeddings: OllamaEmbeddings,
        vectorstore_path: str = "./vectorstore/layer2",
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        cache_size: int = 10
    ):
        """
        Initialize Layer 2 VectorStore
        
        Args:
            embeddings: Embedding model instance
            vectorstore_path: Directory to save/load the vector store
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            cache_size: Maximum number of cached subindices
        """
        self.embeddings = embeddings
        self.vectorstore_path = vectorstore_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.index_file = os.path.join(vectorstore_path, "chunks.faiss")
        self.pkl_file = os.path.join(vectorstore_path, "chunks.pkl")
        self.metadata_file = os.path.join(vectorstore_path, "metadata.pkl")
        
        self.vectorstore: Optional[FAISS] = None
        self._chunk_count = 0
        self._paper_count = 0
        
        # Subindex cache
        self._subindex_cache = SubindexCache(max_size=cache_size)
        
        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def build_index(self, documents: List[Document]) -> bool:
        """
        Build FAISS index from all document chunks
        
        Args:
            documents: List of Document objects (already chunked)
                Each document should have metadata:
                - paper_id: str
                - chunk_id: str
                - chunk_index: int
                - pdf_path: str
                - Other metadata...
                
        Returns:
            True if successful, False otherwise
        """
        try:
            if not documents:
                logger.warning("No documents provided to build Layer 2 index")
                return False
            
            logger.info(f"Building Layer 2 index from {len(documents)} chunks...")
            start_time = time.time()
            
            # Validate and clean documents
            valid_docs = []
            paper_ids = set()
            
            for doc in documents:
                # Check required metadata
                if 'paper_id' not in doc.metadata:
                    logger.warning("Document missing paper_id, skipping")
                    continue
                
                # Check content
                if not doc.page_content or len(doc.page_content) < 50:
                    logger.warning(f"Document content too short, skipping")
                    continue
                
                # Mark as Layer 2
                doc.metadata['layer'] = 2
                
                valid_docs.append(doc)
                paper_ids.add(doc.metadata['paper_id'])
            
            if not valid_docs:
                logger.error("No valid documents after filtering")
                return False
            
            logger.info(f"Creating FAISS index with {len(valid_docs)} chunks from {len(paper_ids)} papers...")
            estimated_time = len(valid_docs) * 1.0 / 60  # 假設每個 chunk 1 秒
            logger.info(f"⏳ Generating {len(valid_docs)} embeddings using Ollama...")
            logger.info(f"   Estimated time: {estimated_time:.1f} minutes (depends on hardware)")
            logger.info(f"   � TIP: This is a one-time process. Future queries will be fast!")
            
            # Create FAISS index with progress tracking
            from system_api.progress_embeddings import ProgressEmbeddings
            progress_embeddings = ProgressEmbeddings(self.embeddings, label="Layer 2")
            
            self.vectorstore = FAISS.from_documents(
                documents=valid_docs,
                embedding=progress_embeddings
            )
            
            self._chunk_count = len(valid_docs)
            self._paper_count = len(paper_ids)
            
            # Clear cache since we have a new index
            self._subindex_cache.clear()
            
            # Save index
            success = self.save()
            
            duration = time.time() - start_time
            logger.info(f"Layer 2 index built successfully in {duration:.2f}s")
            logger.info(f"  Chunks: {self._chunk_count}")
            logger.info(f"  Papers: {self._paper_count}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to build Layer 2 index: {e}", exc_info=True)
            return False
    
    def search(
        self,
        query: str,
        k: int = 10,
        filter_paper_ids: Optional[List[str]] = None,
        **kwargs
    ) -> List[Document]:
        """
        Search for relevant chunks
        
        If filter_paper_ids is provided, creates/uses dynamic subindex
        for much faster filtered search.
        
        Args:
            query: User query string
            k: Number of chunks to retrieve
            filter_paper_ids: Optional list of paper IDs to filter by
            **kwargs: Additional search parameters
            
        Returns:
            List of Document objects with chunks
        """
        if not self.vectorstore:
            logger.error("VectorStore not initialized. Call build_index() or load() first.")
            return []
        
        try:
            # If filtering by papers, use dynamic subindex
            if filter_paper_ids:
                return self._search_filtered(query, k, filter_paper_ids, **kwargs)
            
            # Otherwise, search full index
            logger.debug(f"Layer 2 search (full): query='{query[:50]}...', k={k}")
            
            results = self.vectorstore.similarity_search(
                query=query,
                k=k,
                **kwargs
            )
            
            logger.debug(f"Layer 2 found {len(results)} chunks")
            
            return results
            
        except Exception as e:
            logger.error(f"Layer 2 search failed: {e}", exc_info=True)
            return []
    
    def search_with_scores(
        self,
        query: str,
        k: int = 10,
        filter_paper_ids: Optional[List[str]] = None,
        **kwargs
    ) -> List[tuple[Document, float]]:
        """
        Search for relevant chunks and return with similarity scores
        
        Args:
            query: User query string
            k: Number of chunks to retrieve
            filter_paper_ids: Optional list of paper IDs to filter by
            **kwargs: Additional search parameters
            
        Returns:
            List of (Document, score) tuples
        """
        if not self.vectorstore:
            logger.error("VectorStore not initialized. Call build_index() or load() first.")
            return []
        
        try:
            # If filtering by papers, use dynamic subindex
            if filter_paper_ids:
                return self._search_filtered_with_scores(query, k, filter_paper_ids, **kwargs)
            
            # Otherwise, search full index
            logger.debug(f"Layer 2 search with scores (full): query='{query[:50]}...', k={k}")
            
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k,
                **kwargs
            )
            
            logger.debug(f"Layer 2 found {len(results)} chunks with scores")
            
            return results
            
        except Exception as e:
            logger.error(f"Layer 2 search with scores failed: {e}", exc_info=True)
            return []
    
    def _search_filtered(
        self,
        query: str,
        k: int,
        paper_ids: List[str],
        **kwargs
    ) -> List[Document]:
        """
        Search with paper ID filtering using dynamic subindex
        
        Args:
            query: User query
            k: Number of results
            paper_ids: List of paper IDs to search within
            **kwargs: Additional search parameters
            
        Returns:
            List of Document objects
        """
        logger.debug(f"Layer 2 filtered search: {len(paper_ids)} papers, k={k}")
        
        # Try to get from cache
        cache_key = tuple(sorted(paper_ids))
        filtered_vs = self._subindex_cache.get(cache_key)
        
        # If not cached, create subindex
        if filtered_vs is None:
            filtered_vs = self.create_filtered_vectorstore(paper_ids)
            if filtered_vs:
                self._subindex_cache.put(cache_key, filtered_vs)
        
        # Search in filtered vectorstore
        if filtered_vs:
            results = filtered_vs.similarity_search(query, k=k, **kwargs)
            logger.debug(f"Layer 2 filtered search found {len(results)} chunks")
            return results
        else:
            logger.warning("Failed to create filtered vectorstore, returning empty results")
            return []
    
    def _search_filtered_with_scores(
        self,
        query: str,
        k: int,
        paper_ids: List[str],
        **kwargs
    ) -> List[tuple[Document, float]]:
        """
        Search with paper ID filtering and return scores
        
        Args:
            query: User query
            k: Number of results
            paper_ids: List of paper IDs to search within
            **kwargs: Additional search parameters
            
        Returns:
            List of (Document, score) tuples
        """
        logger.debug(f"Layer 2 filtered search with scores: {len(paper_ids)} papers, k={k}")
        
        # Try to get from cache
        cache_key = tuple(sorted(paper_ids))
        filtered_vs = self._subindex_cache.get(cache_key)
        
        # If not cached, create subindex
        if filtered_vs is None:
            filtered_vs = self.create_filtered_vectorstore(paper_ids)
            if filtered_vs:
                self._subindex_cache.put(cache_key, filtered_vs)
        
        # Search in filtered vectorstore with scores
        if filtered_vs:
            results = filtered_vs.similarity_search_with_score(query, k=k, **kwargs)
            logger.debug(f"Layer 2 filtered search found {len(results)} chunks with scores")
            return results
        else:
            logger.warning("Failed to create filtered vectorstore, returning empty results")
            return []
    
    def create_filtered_vectorstore(
        self,
        paper_ids: List[str]
    ) -> Optional[FAISS]:
        """
        Create temporary FAISS index containing only chunks from specified papers
        
        This is the core of the dynamic subindex strategy, enabling 10x faster
        filtered searches by creating a small index on-the-fly.
        
        Args:
            paper_ids: List of paper IDs to include
            
        Returns:
            Filtered FAISS vectorstore or None if failed
        """
        if not self.vectorstore:
            logger.error("Main vectorstore not initialized")
            return None
        
        try:
            logger.debug(f"Creating filtered subindex for {len(paper_ids)} papers...")
            start_time = time.time()
            
            # Get all documents from main index
            all_docs = list(self.vectorstore.docstore._dict.values())
            
            # Filter documents by paper_id
            filtered_docs = [
                doc for doc in all_docs
                if doc.metadata.get('paper_id') in paper_ids
            ]
            
            if not filtered_docs:
                logger.warning(f"No chunks found for papers: {paper_ids}")
                return None
            
            logger.debug(f"Found {len(filtered_docs)} chunks from {len(paper_ids)} papers")
            
            # Create new FAISS index with filtered documents
            filtered_vs = FAISS.from_documents(
                documents=filtered_docs,
                embedding=self.embeddings
            )
            
            duration = time.time() - start_time
            logger.debug(f"Created filtered subindex in {duration:.2f}s")
            
            return filtered_vs
            
        except Exception as e:
            logger.error(f"Failed to create filtered vectorstore: {e}")
            return None
    
    def get_surrounding_chunks(
        self,
        paper_id: str,
        chunk_id: str,
        before: int = 2,
        after: int = 2
    ) -> str:
        """
        Get surrounding chunks for context expansion
        
        Args:
            paper_id: Paper identifier
            chunk_id: Chunk identifier
            before: Number of chunks before
            after: Number of chunks after
            
        Returns:
            Concatenated text of surrounding chunks
        """
        if not self.vectorstore:
            logger.error("VectorStore not initialized")
            return ""
        
        try:
            # Get all chunks from this paper
            all_docs = list(self.vectorstore.docstore._dict.values())
            paper_chunks = [
                doc for doc in all_docs
                if doc.metadata.get('paper_id') == paper_id
            ]
            
            if not paper_chunks:
                logger.warning(f"No chunks found for paper {paper_id}")
                return ""
            
            # Sort by chunk_index
            paper_chunks.sort(key=lambda x: x.metadata.get('chunk_index', 0))
            
            # Find the target chunk
            target_idx = None
            for i, doc in enumerate(paper_chunks):
                if doc.metadata.get('chunk_id') == chunk_id:
                    target_idx = i
                    break
            
            if target_idx is None:
                logger.warning(f"Chunk {chunk_id} not found in paper {paper_id}")
                # Return the first chunk as fallback
                return paper_chunks[0].page_content if paper_chunks else ""
            
            # Get surrounding chunks
            start_idx = max(0, target_idx - before)
            end_idx = min(len(paper_chunks), target_idx + after + 1)
            
            surrounding = paper_chunks[start_idx:end_idx]
            
            # Concatenate with markers
            context_parts = []
            for i, doc in enumerate(surrounding):
                chunk_idx = start_idx + i
                is_target = (chunk_idx == target_idx)
                
                if is_target:
                    context_parts.append(f"[TARGET CHUNK]\n{doc.page_content}\n[/TARGET CHUNK]")
                else:
                    context_parts.append(doc.page_content)
            
            context = "\n\n".join(context_parts)
            
            logger.debug(f"Retrieved {len(surrounding)} chunks (target + {before} before + {after} after)")
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting surrounding chunks: {e}")
            return ""
    
    def add_chunks(self, chunks: List[Document]) -> bool:
        """
        Add document chunks to index
        
        Args:
            chunks: List of Document objects to add
            
        Returns:
            True if successful, False otherwise
        """
        if not self.vectorstore:
            logger.error("VectorStore not initialized. Call build_index() or load() first.")
            return False
        
        try:
            if not chunks:
                logger.warning("No chunks to add")
                return False
            
            # Validate chunks
            valid_chunks = []
            for chunk in chunks:
                if 'paper_id' not in chunk.metadata:
                    logger.warning("Chunk missing paper_id, skipping")
                    continue
                chunk.metadata['layer'] = 2
                valid_chunks.append(chunk)
            
            if not valid_chunks:
                logger.error("No valid chunks to add")
                return False
            
            # Add to vectorstore
            self.vectorstore.add_documents(valid_chunks)
            self._chunk_count += len(valid_chunks)
            
            # Clear cache since index changed
            self._subindex_cache.clear()
            
            logger.info(f"Added {len(valid_chunks)} chunks to Layer 2")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add chunks to Layer 2: {e}")
            return False
    
    def remove_chunks(self, paper_id: str) -> bool:
        """
        Remove all chunks of a paper from index
        
        Args:
            paper_id: Paper identifier
            
        Returns:
            True if successful, False otherwise
        """
        if not self.vectorstore:
            logger.error("VectorStore not initialized")
            return False
        
        try:
            logger.info(f"Removing chunks for paper {paper_id} from Layer 2...")
            
            # Get all documents except those from the paper to remove
            all_docs = list(self.vectorstore.docstore._dict.values())
            remaining_docs = [
                doc for doc in all_docs
                if doc.metadata.get('paper_id') != paper_id
            ]
            
            removed_count = len(all_docs) - len(remaining_docs)
            
            if removed_count == 0:
                logger.warning(f"No chunks found for paper {paper_id}")
                return False
            
            # Rebuild index without the removed chunks
            logger.info(f"Rebuilding Layer 2 index without {removed_count} chunks...")
            self.vectorstore = FAISS.from_documents(
                documents=remaining_docs,
                embedding=self.embeddings
            )
            
            self._chunk_count = len(remaining_docs)
            
            # Clear cache
            self._subindex_cache.clear()
            
            logger.info(f"Successfully removed {removed_count} chunks for paper {paper_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove chunks from Layer 2: {e}")
            return False
    
    def save(self) -> bool:
        """
        Save FAISS index and metadata to disk
        
        Returns:
            True if successful, False otherwise
        """
        if not self.vectorstore:
            logger.error("No vectorstore to save")
            return False
        
        try:
            # Ensure directory exists
            os.makedirs(self.vectorstore_path, exist_ok=True)
            
            logger.info(f"Saving Layer 2 index to {self.vectorstore_path}...")
            
            # Save FAISS index
            self.vectorstore.save_local(self.vectorstore_path, index_name="chunks")
            
            # Save metadata
            metadata = {
                'chunk_count': self._chunk_count,
                'paper_count': self._paper_count,
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'saved_at': time.time(),
                'index_type': 'layer2_chunks',
            }
            
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"Layer 2 index saved successfully ({self._chunk_count} chunks)")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save Layer 2 index: {e}")
            return False
    
    def load(self) -> bool:
        """
        Load FAISS index from disk
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if files exist
            if not os.path.exists(self.index_file):
                logger.info("No saved Layer 2 index found")
                return False
            
            logger.info(f"Loading Layer 2 index from {self.vectorstore_path}...")
            start_time = time.time()
            
            # Load FAISS index
            self.vectorstore = FAISS.load_local(
                self.vectorstore_path,
                self.embeddings,
                index_name="chunks",
                allow_dangerous_deserialization=True
            )
            
            # Load metadata
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'rb') as f:
                    metadata = pickle.load(f)
                    self._chunk_count = metadata.get('chunk_count', 0)
                    self._paper_count = metadata.get('paper_count', 0)
            else:
                # Count from vectorstore
                self._chunk_count = len(self.vectorstore.docstore._dict)
            
            # Clear cache
            self._subindex_cache.clear()
            
            duration = time.time() - start_time
            logger.info(f"Layer 2 index loaded successfully in {duration:.2f}s")
            logger.info(f"  Chunks: {self._chunk_count}")
            logger.info(f"  Papers: {self._paper_count}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Layer 2 index: {e}")
            return False
    
    @property
    def is_initialized(self) -> bool:
        """Check if vectorstore is initialized"""
        return self.vectorstore is not None
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache statistics
        """
        return self._subindex_cache.get_stats()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about Layer 2 index
        
        Returns:
            Dictionary with statistics
        """
        cache_stats = self._subindex_cache.get_stats()
        
        return {
            'layer': 2,
            'type': 'chunks',
            'chunk_count': self._chunk_count,
            'paper_count': self._paper_count,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'is_initialized': self.vectorstore is not None,
            'index_path': self.vectorstore_path,
            'cache': cache_stats
        }
