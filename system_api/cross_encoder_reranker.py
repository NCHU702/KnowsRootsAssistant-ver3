"""
Cross-Encoder Re-ranker Module

Provides Cross-Encoder model for re-ranking (query, document) pairs.
Uses Ollama's /api/generate endpoint with BCE-reranker model.
Returns relevance scores in range [0, 1] where higher values indicate stronger relevance.
Based on official dengcao/bce-reranker-base_v1 usage pattern.
"""

import logging
import time
import requests
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Cross-Encoder re-ranker using Ollama's BCE-reranker model
    
    Scores (query, document) pairs to capture semantic relationships
    that Bi-Encoders cannot understand.
    
    Uses Ollama /api/generate endpoint with BCE Reranker prompt format:
      "query: {query}\ndocument: {document}\nscore:"
    
    Returns relevance scores in range [0, 1]:
      - 1.0: Perfectly relevant
      - 0.5: Moderately relevant
      - 0.0: Completely irrelevant
    """
    
    def __init__(
        self,
        model_name: str = 'qllama/bce-reranker-base_v1:latest',
        ollama_base_url: str = 'http://localhost:11434',
        max_length: int = 512,
        timeout: int = 30
    ):
        """
        Initialize Cross-Encoder re-ranker with Ollama
        
        Args:
            model_name: Ollama model name
            ollama_base_url: Ollama API base URL
            max_length: Maximum sequence length
            timeout: Request timeout in seconds
        """
        self.model_name = model_name
        self.ollama_base_url = ollama_base_url
        self.max_length = max_length
        self.timeout = timeout
        
        self._model_checked = False
        
        logger.info(f"Initialized Ollama Cross-Encoder Re-ranker")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Ollama URL: {ollama_base_url}")
    
    def _check_model_available(self) -> bool:
        """
        Check if Ollama model is available
        
        Returns:
            True if model exists
        """
        if self._model_checked:
            return True
        
        try:
            response = requests.get(
                f"{self.ollama_base_url}/api/tags",
                timeout=5
            )
            
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m.get('name', '') for m in models]
                
                # Check if reranker model exists
                base_name = self.model_name.split(':')[0]
                if any(base_name in name for name in model_names):
                    logger.info(f"✓ Ollama model available: {self.model_name}")
                    self._model_checked = True
                    return True
                else:
                    logger.warning(f"Model not found: {self.model_name}")
                    logger.info(f"Available models: {model_names}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error checking Ollama: {e}")
            return False
    
    def _compute_similarity(self, query: str, chunk_text: str) -> float:
        """
        Compute relevance score for (query, chunk) pair using BCE Reranker
        
        Uses Ollama's /api/generate endpoint with BCE Reranker prompt format.
        Based on official dengcao/bce-reranker-base_v1 usage pattern.
        
        Args:
            query: Query string
            chunk_text: Chunk text
            
        Returns:
            Relevance score (typically 0-1 range, higher = more relevant)
        """
        try:
            # Truncate if needed
            if len(query) > self.max_length:
                query = query[:self.max_length]
            if len(chunk_text) > self.max_length:
                chunk_text = chunk_text[:self.max_length]
            
            # 官方 BCE Reranker 格式
            prompt = f"query: {query}\ndocument: {chunk_text}\nscore:"
            
            # 調用 Ollama generate API
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.0,  # 確保輸出穩定
                        "num_predict": 10    # 只需要數字分數
                    }
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '').strip()
                
                # 嘗試解析分數
                # BCE Reranker 可能返回 "score: 0.95" 或 "0.95"
                try:
                    # 提取 ":" 後面的數字
                    if ':' in response_text:
                        score_str = response_text.split(':')[-1].strip()
                    else:
                        score_str = response_text
                    
                    # 提取第一個數字
                    import re
                    numbers = re.findall(r'\d+\.?\d*', score_str)
                    if numbers:
                        score = float(numbers[0])
                        # BCE Reranker 通常輸出 0-1 範圍，但確保在有效範圍內
                        score = max(0.0, min(1.0, score))
                        return score
                except (ValueError, IndexError):
                    pass
            
            return 0.0  # 失敗時返回最低分數
            
        except Exception as e:
            logger.debug(f"Error computing similarity: {e}")
            return 0.0
    
    def score_pairs(
        self,
        query: str,
        chunks: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> List[float]:
        """
        Score (query, chunk) pairs using BCE Reranker
        
        Args:
            query: Query string
            chunks: List of chunk text strings
            batch_size: Not used (kept for API compatibility)
            show_progress: Not used (kept for API compatibility)
            
        Returns:
            List of relevance scores in range [0, 1]
            Higher scores indicate stronger relevance
        """
        if not chunks:
            return []
        
        # Check model availability
        if not self._check_model_available():
            logger.warning("Model unavailable, returning neutral scores")
            return [0.0] * len(chunks)  # 中性分數（-1 到 1 範圍）
        
        try:
            logger.debug(f"Scoring {len(chunks)} chunks with Ollama re-ranker")
            start_time = time.time()
            
            scores = []
            for i, chunk_text in enumerate(chunks):
                if not chunk_text:
                    scores.append(0.0)
                    continue
                
                score = self._compute_similarity(query, chunk_text)
                scores.append(score)
                
                if (i + 1) % 10 == 0:
                    logger.debug(f"  Scored {i+1}/{len(chunks)} chunks...")
            
            duration = time.time() - start_time
            logger.debug(f"✓ Scored {len(chunks)} chunks in {duration:.2f}s")
            
            return scores
            
        except Exception as e:
            logger.error(f"Failed to score pairs: {e}", exc_info=True)
            return [0.0] * len(chunks)  # 中性分數（-1 到 1 範圍）
    
    def rank_chunks(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Rank chunks by relevance and return top K
        
        Args:
            query: Query string
            chunks: List of chunk dicts with 'text' field
            top_k: Number of top chunks to return
            
        Returns:
            List of (chunk, score) tuples, sorted by score descending
        """
        if not chunks:
            return []
        
        try:
            # Extract text from chunks
            chunk_texts = [chunk.get('text', '') for chunk in chunks]
            
            # Score all chunks
            scores = self.score_pairs(query, chunk_texts)
            
            # Zip chunks with scores
            chunk_scores = list(zip(chunks, scores))
            
            # Sort by score descending
            chunk_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Return top K
            top_chunks = chunk_scores[:top_k]
            
            logger.debug(f"Ranked {len(chunks)} chunks, returning top {len(top_chunks)}")
            if top_chunks:
                logger.debug(f"  Top score: {top_chunks[0][1]:.4f}")
                logger.debug(f"  Bottom score: {top_chunks[-1][1]:.4f}")
            
            return top_chunks
            
        except Exception as e:
            logger.error(f"Failed to rank chunks: {e}", exc_info=True)
            return []
    
    def is_loaded(self) -> bool:
        """
        Check if model is available
        
        Returns:
            True if model is available
        """
        return self._check_model_available()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information
        
        Returns:
            Dictionary with model details
        """
        return {
            'model_name': self.model_name,
            'ollama_base_url': self.ollama_base_url,
            'max_length': self.max_length,
            'is_loaded': self.is_loaded()
        }


class TFIDFFallbackReranker:
    """
    Simple TF-IDF based fallback re-ranker
    
    Used when Cross-Encoder fails to load or encounters errors.
    Provides basic relevance scoring as emergency fallback.
    """
    
    def __init__(self):
        """Initialize TF-IDF fallback"""
        logger.warning("Using TF-IDF fallback reranker (reduced accuracy)")
        self._model = None
    
    def score_pairs(self, query: str, chunks: List[str], **kwargs) -> List[float]:
        """
        Score using simple keyword matching
        
        Args:
            query: Query string
            chunks: List of chunk texts
            
        Returns:
            List of relevance scores
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            if self._model is None:
                # Use character-level for Chinese text
                self._model = TfidfVectorizer(analyzer='char', ngram_range=(1, 2))
            
            # Fit on query + chunks
            all_texts = [query] + chunks
            tfidf_matrix = self._model.fit_transform(all_texts)
            
            # Compute similarity between query (first row) and chunks
            query_vec = tfidf_matrix[0:1]
            chunk_vecs = tfidf_matrix[1:]
            
            scores = cosine_similarity(query_vec, chunk_vecs)[0]
            
            return scores.tolist()
            
        except Exception as e:
            logger.error(f"TF-IDF fallback failed: {e}")
            # Last resort: return equal scores
            return [0.5] * len(chunks)
    
    def rank_chunks(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Rank using TF-IDF"""
        chunk_texts = [chunk.get('text', '') for chunk in chunks]
        scores = self.score_pairs(query, chunk_texts)
        
        # Combine and sort
        chunk_scores = list(zip(chunks, scores))
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        
        return chunk_scores[:top_k]
    
    def is_loaded(self) -> bool:
        """Always ready"""
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get info"""
        return {
            'model_name': 'TF-IDF Fallback',
            'device': 'cpu',
            'is_loaded': True
        }
