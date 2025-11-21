"""
Layer 1 VectorStore Module

Manages abstract-level FAISS index for fast paper-level retrieval.
This is the first layer in the hierarchical RAG system.
"""

import os
import logging
import pickle
import time
from typing import List, Dict, Optional, Any
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class Layer1VectorStore:
    """
    Manage Layer 1 (abstract-level) vector store
    
    This index contains one vector per paper (the abstract),
    enabling fast coarse filtering of relevant papers.
    """
    
    def __init__(
        self,
        embeddings: OllamaEmbeddings,
        vectorstore_path: str = "./vectorstore/layer1"
    ):
        """
        Initialize Layer 1 VectorStore
        
        Args:
            embeddings: Embedding model instance
            vectorstore_path: Directory to save/load the vector store
        """
        self.embeddings = embeddings
        self.vectorstore_path = vectorstore_path
        self.index_file = os.path.join(vectorstore_path, "abstract.faiss")
        self.pkl_file = os.path.join(vectorstore_path, "abstract.pkl")
        self.metadata_file = os.path.join(vectorstore_path, "metadata.pkl")
        
        self.vectorstore: Optional[FAISS] = None
        self._paper_count = 0
        
    def build_index(self, abstracts: List[Dict[str, Any]]) -> bool:
        """
        Build FAISS index from paper abstracts
        
        Args:
            abstracts: List of abstract dicts with keys:
                - abstract: str (abstract text)
                - paper_id: str (unique identifier)
                - title: str (optional, paper title)
                - authors: List[str] (optional)
                - year: int (optional)
                - pdf_path: str (path to PDF)
                - Other metadata...
                
        Returns:
            True if successful, False otherwise
        """
        try:
            if not abstracts:
                logger.warning("No abstracts provided to build index")
                return False
            
            logger.info(f"Building Layer 1 index from {len(abstracts)} abstracts...")
            start_time = time.time()
            
            # Create Document objects
            documents = []
            for abstract_dict in abstracts:
                # Extract abstract text
                abstract_text = abstract_dict.get('abstract', '')
                if not abstract_text or len(abstract_text) < 50:
                    logger.warning(f"Skipping paper {abstract_dict.get('paper_id')} - abstract too short")
                    continue
                
                # Prepare metadata
                metadata = {
                    'paper_id': abstract_dict.get('paper_id'),
                    'title': abstract_dict.get('title', ''),
                    'authors': abstract_dict.get('authors', []),
                    'year': abstract_dict.get('year', None),
                    'pdf_path': abstract_dict.get('pdf_path', ''),
                    'abstract_source': abstract_dict.get('source', 'unknown'),
                    'abstract_confidence': abstract_dict.get('confidence', 0.0),
                    'layer': 1,  # Mark as Layer 1 document
                }
                
                # Create document
                doc = Document(
                    page_content=abstract_text,
                    metadata=metadata
                )
                documents.append(doc)
            
            if not documents:
                logger.error("No valid documents after filtering")
                return False
            
            logger.info(f"Creating FAISS index with {len(documents)} abstracts...")
            
            # Use progress wrapper for real-time updates
            from system_api.progress_embeddings import ProgressEmbeddings
            progress_embeddings = ProgressEmbeddings(self.embeddings, label="Layer 1")
            
            # Create FAISS index with progress tracking
            self.vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=progress_embeddings
            )
            
            self._paper_count = len(documents)
            
            # Save index
            success = self.save()
            
            duration = time.time() - start_time
            logger.info(f"Layer 1 index built successfully in {duration:.2f}s")
            logger.info(f"  Papers indexed: {self._paper_count}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to build Layer 1 index: {e}", exc_info=True)
            return False
    
    def search(
        self,
        query: str,
        k: int = 15,
        score_threshold: Optional[float] = None,
        **kwargs
    ) -> List[Document]:
        """
        Search for relevant papers by abstract similarity
        
        Args:
            query: User query string
            k: Maximum number of papers to retrieve (default: 15)
            score_threshold: Optional similarity score threshold (0-1).
                           Only return papers with similarity >= threshold.
                           If provided, returns papers passing threshold (up to k).
            **kwargs: Additional search parameters
            
        Returns:
            List of Document objects with paper abstracts and metadata
        """
        if not self.vectorstore:
            logger.error("VectorStore not initialized. Call build_index() or load() first.")
            return []
        
        try:
            logger.debug(f"Layer 1 search: query='{query[:50]}...', k={k}, threshold={score_threshold}")
            
            # 使用 search_with_scores 來支援閾值過濾
            if score_threshold is not None:
                results_with_scores = self.search_with_scores(
                    query=query,
                    k=k,
                    score_threshold=score_threshold,
                    **kwargs
                )
                results = [doc for doc, score in results_with_scores]
            else:
                results = self.vectorstore.similarity_search(
                    query=query,
                    k=k,
                    **kwargs
                )
            
            logger.debug(f"Layer 1 found {len(results)} papers")
            
            return results
            
        except Exception as e:
            logger.error(f"Layer 1 search failed: {e}", exc_info=True)
            return []
    
    def search_with_scores(
        self,
        query: str,
        k: int = 15,
        score_threshold: Optional[float] = None,
        **kwargs
    ) -> List[tuple]:
        """
        Search with similarity scores and optional threshold filtering
        
        Args:
            query: User query string
            k: Maximum number of papers to retrieve (used if no threshold)
            score_threshold: Optional similarity score threshold (0-1). 
                           Only return papers with score >= threshold.
                           If provided, k is used as max limit.
            **kwargs: Additional search parameters
            
        Returns:
            List of (Document, score) tuples, filtered by threshold if provided
        """
        if not self.vectorstore:
            logger.error("VectorStore not initialized")
            return []
        
        try:
            # 如果有閾值，需要獲取所有文檔來進行過濾
            if score_threshold is not None:
                # 獲取所有文檔（使用總數）
                fetch_k = self._paper_count if self._paper_count else 100
                logger.info(f"Layer 1: Using threshold {score_threshold:.2f}, fetching {fetch_k} papers for filtering")
            else:
                # 沒有閾值，只取 top-k
                fetch_k = k
            
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=fetch_k,
                **kwargs
            )
            
            # 如果有閾值，進行過濾（不限制數量）
            if score_threshold is not None:
                # FAISS 返回的是 L2 距離的平方
                # 對於正規化的向量: cosine_similarity ≈ 1 - (L2_distance² / 2)
                # 但更簡單的方式：較小的距離 = 較高的相似度
                # 我們使用: similarity = 1 / (1 + sqrt(distance))
                filtered_results = []
                for idx, (doc, distance) in enumerate(results):
                    # 轉換為相似度分數 (0-1)
                    # 使用平方根來減緩距離的增長
                    import math
                    similarity = 1 / (1 + math.sqrt(distance))
                    
                    # 調試：顯示前幾個論文的分數
                    if idx < 5:  # 顯示前 5 個
                        logger.info(f"  Paper {idx+1}: {doc.metadata.get('title', 'Unknown')[:50]}")
                        logger.info(f"    Distance={distance:.4f}, Similarity={similarity:.4f}, Threshold={score_threshold:.2f}, Pass={similarity >= score_threshold}")
                    
                    # 只保留超過閾值的結果
                    if similarity >= score_threshold:
                        filtered_results.append((doc, similarity))
                
                # 按相似度排序（從高到低）
                filtered_results.sort(key=lambda x: x[1], reverse=True)
                
                logger.info(f"Layer 1: {len(results)} total papers → {len(filtered_results)} papers after threshold {score_threshold:.2f}")
                
                return filtered_results
            else:
                # 沒有閾值，返回 top-k（轉換分數）
                import math
                converted_results = []
                for doc, distance in results[:k]:
                    similarity = 1 / (1 + math.sqrt(distance))
                    converted_results.append((doc, similarity))
                
                logger.debug(f"Layer 1 found {len(converted_results)} papers with scores")
                
                return converted_results
            
        except Exception as e:
            logger.error(f"Layer 1 search with scores failed: {e}")
            logger.exception("Full traceback:")
            return []
    
    def add_abstract(self, abstract_dict: Dict[str, Any]) -> bool:
        """
        Add single abstract to index
        
        Args:
            abstract_dict: Abstract dictionary (same format as build_index)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            abstract_text = abstract_dict.get('abstract', '')
            if not abstract_text or len(abstract_text) < 50:
                logger.warning("Abstract too short, skipping")
                return False
            
            # Prepare metadata
            metadata = {
                'paper_id': abstract_dict.get('paper_id'),
                'title': abstract_dict.get('title', ''),
                'authors': abstract_dict.get('authors', []),
                'year': abstract_dict.get('year', None),
                'pdf_path': abstract_dict.get('pdf_path', ''),
                'abstract_source': abstract_dict.get('source', 'unknown'),
                'abstract_confidence': abstract_dict.get('confidence', 0.0),
                'layer': 1,
            }
            
            # Create document
            doc = Document(page_content=abstract_text, metadata=metadata)
            
            # If vectorstore not initialized, create it with first document
            if not self.vectorstore:
                logger.info("Initializing Layer 1 vectorstore with first document...")
                from system_api.progress_embeddings import ProgressEmbeddings
                progress_embeddings = ProgressEmbeddings(self.embeddings, label="Layer 1")
                self.vectorstore = FAISS.from_documents([doc], embedding=progress_embeddings)
                self._paper_count = 1
                logger.info("  ✓ Layer 1 vectorstore initialized")
            else:
                # Add to existing index
                self.vectorstore.add_documents([doc])
                self._paper_count += 1
            
            logger.info(f"Added abstract for paper {abstract_dict.get('paper_id')} to Layer 1")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add abstract to Layer 1: {e}")
            return False
    
    def remove_abstract(self, paper_id: str) -> bool:
        """
        Remove abstract from index by paper_id
        
        Note: FAISS doesn't support efficient deletion.
        This requires rebuilding the index without the specified paper.
        
        Args:
            paper_id: Paper identifier to remove
            
        Returns:
            True if successful, False otherwise
        """
        if not self.vectorstore:
            logger.error("VectorStore not initialized")
            return False
        
        try:
            logger.info(f"Removing paper {paper_id} from Layer 1...")
            
            # Get all documents except the one to remove
            all_docs = list(self.vectorstore.docstore._dict.values())
            remaining_docs = [
                doc for doc in all_docs
                if doc.metadata.get('paper_id') != paper_id
            ]
            
            if len(remaining_docs) == len(all_docs):
                logger.warning(f"Paper {paper_id} not found in Layer 1")
                return False
            
            # Rebuild index without the removed paper
            logger.info(f"Rebuilding Layer 1 index with {len(remaining_docs)} papers...")
            self.vectorstore = FAISS.from_documents(
                documents=remaining_docs,
                embedding=self.embeddings
            )
            
            self._paper_count = len(remaining_docs)
            
            logger.info(f"Successfully removed paper {paper_id} from Layer 1")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove abstract from Layer 1: {e}")
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
            
            logger.info(f"Saving Layer 1 index to {self.vectorstore_path}...")
            
            # Save FAISS index
            self.vectorstore.save_local(self.vectorstore_path, index_name="abstract")
            
            # Save metadata
            metadata = {
                'paper_count': self._paper_count,
                'saved_at': time.time(),
                'index_type': 'layer1_abstract',
            }
            
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"Layer 1 index saved successfully ({self._paper_count} papers)")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save Layer 1 index: {e}")
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
                logger.info("No saved Layer 1 index found")
                return False
            
            logger.info(f"Loading Layer 1 index from {self.vectorstore_path}...")
            start_time = time.time()
            
            # Load FAISS index
            self.vectorstore = FAISS.load_local(
                self.vectorstore_path,
                self.embeddings,
                index_name="abstract",
                allow_dangerous_deserialization=True
            )
            
            # Load metadata
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'rb') as f:
                    metadata = pickle.load(f)
                    self._paper_count = metadata.get('paper_count', 0)
            else:
                # Count from vectorstore
                self._paper_count = len(self.vectorstore.docstore._dict)
            
            duration = time.time() - start_time
            logger.info(f"Layer 1 index loaded successfully in {duration:.2f}s")
            logger.info(f"  Papers: {self._paper_count}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Layer 1 index: {e}")
            return False
    
    @property
    def is_initialized(self) -> bool:
        """Check if vectorstore is initialized"""
        return self.vectorstore is not None
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about Layer 1 index
        
        Returns:
            Dictionary with statistics
        """
        return {
            'layer': 1,
            'type': 'abstract',
            'paper_count': self._paper_count,
            'is_initialized': self.vectorstore is not None,
            'index_path': self.vectorstore_path,
        }
    
    def get_paper_by_id(self, paper_id: str) -> Optional[Document]:
        """
        Get paper abstract by paper_id
        
        Args:
            paper_id: Paper identifier
            
        Returns:
            Document object or None if not found
        """
        if not self.vectorstore:
            return None
        
        try:
            all_docs = list(self.vectorstore.docstore._dict.values())
            for doc in all_docs:
                if doc.metadata.get('paper_id') == paper_id:
                    return doc
            return None
        except Exception as e:
            logger.error(f"Error getting paper by ID: {e}")
            return None
