"""
Context Expander Module

Dynamically expands context around retrieved chunks based on confidence scores.
Higher confidence = smaller expansion, lower confidence = larger expansion.
"""

import logging
from typing import List, Dict, Any
from langchain_core.documents import Document
from system_api.layer2_vectorstore import Layer2VectorStore

logger = logging.getLogger(__name__)


class ContextExpander:
    """
    Expand context around retrieved chunks based on confidence
    
    Expansion strategy:
    - High confidence (≥0.85): ±1 chunk (minimal expansion)
    - Medium confidence (0.75-0.85): ±2 chunks (moderate expansion)
    - Low confidence (<0.75): ±3 chunks (maximum expansion)
    """
    
    def __init__(self, layer2_vectorstore: Layer2VectorStore):
        """
        Initialize Context Expander
        
        Args:
            layer2_vectorstore: Layer 2 vectorstore instance for retrieving surrounding chunks
        """
        self.vectorstore = layer2_vectorstore
        
    def expand(
        self,
        chunks: List[Document],
        confidence: float
    ) -> List[str]:
        """
        Expand context dynamically based on confidence score
        
        Args:
            chunks: Retrieved chunks from Layer 2
            confidence: Confidence score from evaluation (0-1)
            
        Returns:
            List of expanded context strings (one per input chunk)
        """
        if not chunks:
            logger.warning("No chunks provided for expansion")
            return []
        
        try:
            # Determine expansion range based on confidence
            expand_range = self._determine_expansion_range(confidence)
            
            logger.info(f"Expanding {len(chunks)} chunks with range ±{expand_range} "
                       f"(confidence: {confidence:.2f})")
            
            # Expand each chunk
            expanded_contexts = []
            for chunk in chunks:
                expanded = self._expand_single_chunk(chunk, expand_range)
                if expanded:
                    expanded_contexts.append(expanded)
            
            logger.info(f"Successfully expanded {len(expanded_contexts)} contexts")
            
            return expanded_contexts
            
        except Exception as e:
            logger.error(f"Context expansion failed: {e}", exc_info=True)
            # Fallback: return original chunk content
            return [chunk.page_content for chunk in chunks]
    
    def _determine_expansion_range(self, confidence: float) -> int:
        """
        Determine expansion range based on confidence
        
        Args:
            confidence: Confidence score (0-1)
            
        Returns:
            Number of chunks to expand before and after (1, 2, or 3)
        """
        if confidence >= 0.85:
            # High confidence: minimal expansion
            return 1
        elif confidence >= 0.75:
            # Medium confidence: moderate expansion
            return 2
        else:
            # Low confidence: maximum expansion
            return 3
    
    def _expand_single_chunk(
        self,
        chunk: Document,
        expand_range: int
    ) -> str:
        """
        Expand a single chunk by retrieving surrounding chunks
        
        Args:
            chunk: Document to expand
            expand_range: Number of chunks to retrieve before and after
            
        Returns:
            Concatenated context with proper formatting
        """
        try:
            paper_id = chunk.metadata.get('paper_id')
            chunk_id = chunk.metadata.get('chunk_id')
            
            if not paper_id or not chunk_id:
                logger.warning("Chunk missing paper_id or chunk_id, returning original content")
                return chunk.page_content
            
            # Get surrounding chunks from vectorstore
            expanded_context = self.vectorstore.get_surrounding_chunks(
                paper_id=paper_id,
                chunk_id=chunk_id,
                before=expand_range,
                after=expand_range
            )
            
            if not expanded_context:
                logger.warning(f"Failed to get surrounding chunks for {chunk_id}, "
                             "using original content")
                return chunk.page_content
            
            return expanded_context
            
        except Exception as e:
            logger.error(f"Error expanding single chunk: {e}")
            return chunk.page_content
    
    def expand_with_metadata(
        self,
        chunks: List[Document],
        confidence: float
    ) -> List[Dict[str, Any]]:
        """
        Expand context and return with metadata
        
        Args:
            chunks: Retrieved chunks
            confidence: Confidence score
            
        Returns:
            List of dictionaries with expanded context and metadata
        """
        expand_range = self._determine_expansion_range(confidence)
        
        results = []
        for chunk in chunks:
            expanded_text = self._expand_single_chunk(chunk, expand_range)
            
            result = {
                'context': expanded_text,
                'paper_id': chunk.metadata.get('paper_id'),
                'chunk_id': chunk.metadata.get('chunk_id'),
                'expansion_range': expand_range,
                'confidence': confidence,
                'metadata': chunk.metadata
            }
            results.append(result)
        
        return results
    
    def get_expansion_stats(self, confidence: float) -> Dict[str, Any]:
        """
        Get statistics about expansion for given confidence
        
        Args:
            confidence: Confidence score
            
        Returns:
            Dictionary with expansion statistics
        """
        expand_range = self._determine_expansion_range(confidence)
        
        # Estimate context size (assuming ~800 chars per chunk)
        estimated_size_per_chunk = (2 * expand_range + 1) * 800
        
        return {
            'confidence': confidence,
            'expansion_range': expand_range,
            'total_chunks_per_context': 2 * expand_range + 1,
            'estimated_size_per_chunk': estimated_size_per_chunk,
            'strategy': self._get_strategy_name(expand_range)
        }
    
    def _get_strategy_name(self, expand_range: int) -> str:
        """Get human-readable strategy name"""
        if expand_range == 1:
            return "minimal"
        elif expand_range == 2:
            return "moderate"
        else:
            return "maximum"
