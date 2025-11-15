"""
Progress-tracking wrapper for embeddings with real-time updates
"""

import logging
import time
from typing import List
from langchain_ollama import OllamaEmbeddings

logger = logging.getLogger(__name__)


class ProgressEmbeddings:
    """
    Wrapper around OllamaEmbeddings that shows progress during batch embedding
    """
    
    def __init__(
        self, 
        base_embeddings: OllamaEmbeddings, 
        label: str = "Embeddings",
        progress_interval: int = 10
    ):
        """
        Initialize progress wrapper
        
        Args:
            base_embeddings: Original OllamaEmbeddings instance
            label: Label for progress messages (e.g., "Layer 1", "Layer 2")
            progress_interval: Show progress every N documents (default: 10)
        """
        self.base_embeddings = base_embeddings
        self.label = label
        self.progress_interval = progress_interval
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed documents with progress tracking and error handling
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        total = len(texts)
        logger.info(f"[{self.label}] Starting embedding for {total} documents...")
        
        start_time = time.time()
        embeddings = []
        
        # Use smaller batch size to avoid Ollama crashes
        # Progress interval is for display, actual batch size should be smaller
        batch_size = min(5, self.progress_interval)  # Max 5 at a time to avoid EOF errors
        display_interval = self.progress_interval
        
        for i in range(0, total, batch_size):
            batch_end = min(i + batch_size, total)
            batch_texts = texts[i:batch_end]
            
            # Embed this batch with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    batch_embeddings = self.base_embeddings.embed_documents(batch_texts)
                    embeddings.extend(batch_embeddings)
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"[{self.label}] Batch {i}-{batch_end} failed (attempt {attempt+1}/{max_retries}): {str(e)[:100]}"
                        )
                        logger.warning(f"[{self.label}] Retrying in 2 seconds...")
                        time.sleep(2)
                    else:
                        logger.error(
                            f"[{self.label}] Batch {i}-{batch_end} failed after {max_retries} attempts: {e}"
                        )
                        raise  # Re-raise after all retries exhausted
            
            # Display progress at intervals
            processed = len(embeddings)
            if processed % display_interval == 0 or processed == total:
                elapsed = time.time() - start_time
                
                if processed > 0 and elapsed > 0:
                    rate = processed / elapsed
                    remaining = (total - processed) / rate if rate > 0 else 0
                    percent = (processed / total) * 100
                    
                    logger.info(
                        f"[{self.label}] Progress: {processed}/{total} ({percent:.1f}%) "
                        f"- {rate:.1f} docs/sec - ETA: {remaining:.0f}s"
                    )
        
        total_time = time.time() - start_time
        logger.info(
            f"[{self.label}] ✓ Completed {total} embeddings in {total_time:.1f}s "
            f"({total_time/total:.2f}s per doc)"
        )
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query (no progress needed)
        
        Args:
            text: Query text
            
        Returns:
            Embedding vector
        """
        return self.base_embeddings.embed_query(text)
    
    def __call__(self, text: str) -> List[float]:
        """
        Make the object callable for compatibility with FAISS
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        return self.embed_query(text)
    
    def __getattr__(self, name):
        """
        Delegate any unknown attributes to the base embeddings object
        This ensures full compatibility with OllamaEmbeddings interface
        """
        return getattr(self.base_embeddings, name)
