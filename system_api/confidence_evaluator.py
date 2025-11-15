"""
Confidence Evaluator Module

LLM-based confidence evaluation for determining whether retrieved documents
are sufficient to answer a query. Enables early termination in hierarchical RAG.
"""

import json
import logging
import re
from typing import List, Dict, Any
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# Evaluation prompts
QUICK_CONFIDENCE_PROMPT = """你是一個嚴格的學術查詢評估系統。請仔細評估檢索結果的質量。

用戶問題：{query}

檢索到的論文摘要：
{abstracts_summary}

請仔細評估這些摘要能否回答問題，使用 0.0 到 1.0 的連續評分：

**評估標準**：

1. **相關性 (Relevance)** - 0.0 到 1.0：
   - 0.0-0.3: 幾乎不相關，主題完全不同
   - 0.4-0.6: 部分相關，有一些共同關鍵詞但焦點不同
   - 0.7-0.8: 高度相關，主題一致但可能角度不同
   - 0.9-1.0: 完全相關，直接針對問題主題

2. **完整性 (Completeness)** - 0.0 到 1.0：
   - 0.0-0.3: 信息非常不足，只有概述
   - 0.4-0.6: 包含部分信息，但缺少關鍵細節
   - 0.7-0.8: 包含大部分信息，可能缺少一些細節
   - 0.9-1.0: 信息完整，足以全面回答問題

3. **總信心度 (Confidence)** - 建議計算方式：
   - 如果只是提到相關主題但沒有具體內容：< 0.5
   - 如果有基本信息但不夠詳細：0.5-0.7
   - 如果信息充分但可能需要更多細節：0.7-0.85
   - 如果信息非常完整和詳細：> 0.85

**嚴格評分要求**：
- 不要給予過高的分數
- 摘要通常只包含概述，很少能得到 > 0.8 的完整性
- 只有當摘要包含具體方法、數據、結果時才給高分
- 默認傾向於保守評分

請返回 JSON 格式（只返回 JSON，不要其他內容）：
{{
    "relevance": <0.0-1.0的浮點數>,
    "completeness": <0.0-1.0的浮點數>,
    "confidence": <0.0-1.0的浮點數>,
    "reasoning": "詳細說明評分理由"
}}"""


DETAILED_CONFIDENCE_PROMPT = """你是一個非常嚴格的學術查詢評估系統。請仔細評估文檔區塊的質量。

用戶問題：{query}

檢索到的文檔區塊：
{chunks_content}

請詳細評估這些區塊是否足以回答問題，使用 0.0 到 1.0 的連續評分：

**評估標準**：

1. **相關性 (Relevance)** - 0.0 到 1.0：
   - 0.0-0.3: 區塊與問題幾乎無關
   - 0.4-0.6: 區塊部分相關，包含一些相關術語
   - 0.7-0.8: 區塊高度相關，直接涉及問題主題
   - 0.9-1.0: 區塊完全針對問題，包含核心概念

2. **完整性 (Completeness)** - 0.0 到 1.0：
   - 0.0-0.3: 信息片段化，無法回答問題
   - 0.4-0.6: 包含部分答案，但缺少關鍵信息
   - 0.7-0.8: 包含大部分答案，可能缺少一些細節
   - 0.9-1.0: 信息完整，可以全面回答問題

3. **深度 (Depth)** - 0.0 到 1.0：
   - 0.0-0.3: 只有表面概述，缺少細節
   - 0.4-0.6: 有一些細節，但不夠深入
   - 0.7-0.8: 包含較詳細的信息（方法、數據等）
   - 0.9-1.0: 非常詳細，包含完整的實驗、結果、分析

4. **總信心度 (Confidence)** - 建議計算方式：
   - 如果區塊只是概述或背景：< 0.6
   - 如果包含部分方法或結果但不完整：0.6-0.75
   - 如果信息較充分但缺少一些關鍵細節：0.75-0.85
   - 如果信息非常完整和詳細：> 0.85

**嚴格評分要求**：
- 不要輕易給予高分（> 0.8）
- 只有當區塊包含具體的方法、完整的實驗數據和詳細結果時才給高分
- 如果缺少任何關鍵信息，分數應該 < 0.8
- 默認傾向於保守評分

請返回 JSON 格式（只返回 JSON，不要其他內容）：
{{
    "relevance": <0.0-1.0的浮點數>,
    "completeness": <0.0-1.0的浮點數>,
    "depth": <0.0-1.0的浮點數>,
    "confidence": <0.0-1.0的浮點數>,
    "reasoning": "詳細說明評分理由和缺少什麼信息",
    "missing_info": ["缺少的信息1", "缺少的信息2"]
}}"""


class ConfidenceEvaluator:
    """
    Evaluate whether retrieved documents are sufficient to answer query
    
    Supports two evaluation modes:
    - Quick evaluation for Layer 1 (fast, simplified)
    - Detailed evaluation for Layer 2 (thorough, comprehensive)
    """
    
    def __init__(self, llm: OllamaLLM):
        """
        Initialize Confidence Evaluator
        
        Args:
            llm: LLM instance for evaluation
        """
        self.llm = llm
        
    def evaluate(
        self,
        query: str,
        retrieved_docs: List[Document],
        layer: int,
        threshold: float = None
    ) -> Dict[str, Any]:
        """
        Evaluate confidence for given layer
        
        Args:
            query: User query string
            retrieved_docs: List of retrieved documents
            layer: Layer number (1 or 2)
            threshold: Optional custom threshold (uses layer default if None)
            
        Returns:
            {
                'confidence': float (0-1),
                'relevance': float (0-1),
                'completeness': float (0-1),
                'depth': float (0-1, Layer 2 only),
                'should_continue': bool,
                'reasoning': str,
                'missing_info': List[str] (Layer 2 only),
                'layer': int,
                'threshold': float
            }
        """
        try:
            if not retrieved_docs:
                logger.warning("No documents provided for evaluation")
                return self._empty_evaluation(layer)
            
            # Use appropriate evaluation based on layer
            if layer == 1:
                # Quick evaluation for Layer 1
                threshold = threshold if threshold is not None else 0.7
                result = self._quick_evaluation(query, retrieved_docs)
            elif layer == 2:
                # Detailed evaluation for Layer 2
                threshold = threshold if threshold is not None else 0.8
                result = self._detailed_evaluation(query, retrieved_docs)
            else:
                logger.error(f"Invalid layer: {layer}")
                return self._empty_evaluation(layer)
            
            # Add threshold and decision
            result['threshold'] = threshold
            result['should_continue'] = result['confidence'] < threshold
            result['layer'] = layer
            
            logger.info(f"Layer {layer} evaluation: confidence={result['confidence']:.2f}, "
                       f"threshold={threshold:.2f}, continue={result['should_continue']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}", exc_info=True)
            return self._empty_evaluation(layer)
    
    def _quick_evaluation(
        self,
        query: str,
        docs: List[Document]
    ) -> Dict[str, Any]:
        """
        Quick evaluation for Layer 1 (abstracts)
        
        Uses simplified prompt for faster response.
        
        Args:
            query: User query
            docs: Retrieved documents (abstracts)
            
        Returns:
            Evaluation result dictionary
        """
        try:
            logger.debug(f"Quick evaluation with {len(docs)} abstracts")
            
            # Prepare abstracts summary
            abstracts_summary = self._format_abstracts(docs)
            
            # Generate prompt
            prompt = QUICK_CONFIDENCE_PROMPT.format(
                query=query,
                abstracts_summary=abstracts_summary
            )
            
            # Call LLM (use model defaults, no per-call temperature)
            response = self.llm.invoke(prompt)
            
            # Parse JSON response
            result = self._parse_json_response(response)
            
            # Validate and fill defaults
            result = self._validate_quick_result(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Quick evaluation failed: {e}")
            return self._default_quick_result(low_confidence=True)
    
    def _detailed_evaluation(
        self,
        query: str,
        docs: List[Document]
    ) -> Dict[str, Any]:
        """
        Detailed evaluation for Layer 2 (chunks)
        
        Uses comprehensive prompt for thorough analysis.
        
        Args:
            query: User query
            docs: Retrieved documents (chunks)
            
        Returns:
            Evaluation result dictionary
        """
        try:
            logger.debug(f"Detailed evaluation with {len(docs)} chunks")
            
            # Prepare chunks content
            chunks_content = self._format_chunks(docs)
            
            # Generate prompt
            prompt = DETAILED_CONFIDENCE_PROMPT.format(
                query=query,
                chunks_content=chunks_content
            )
            
            # Call LLM (use model defaults, no per-call temperature)
            response = self.llm.invoke(prompt)
            
            # Parse JSON response
            result = self._parse_json_response(response)
            
            # Validate and fill defaults
            result = self._validate_detailed_result(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Detailed evaluation failed: {e}")
            return self._default_detailed_result(low_confidence=True)
    
    def _format_abstracts(self, docs: List[Document], max_length: int = 2000) -> str:
        """
        Format abstracts for evaluation prompt
        
        Args:
            docs: List of documents
            max_length: Maximum total length
            
        Returns:
            Formatted string
        """
        lines = []
        total_length = 0
        
        for i, doc in enumerate(docs[:10], 1):  # Limit to 10 abstracts
            title = doc.metadata.get('title', f'Paper {i}')
            abstract = doc.page_content[:300]  # Limit each abstract
            
            line = f"{i}. {title}\n   {abstract}..."
            
            if total_length + len(line) > max_length:
                lines.append("... (更多論文省略)")
                break
            
            lines.append(line)
            total_length += len(line)
        
        return "\n\n".join(lines)
    
    def _format_chunks(self, docs: List[Document], max_length: int = 3000) -> str:
        """
        Format chunks for evaluation prompt
        
        Args:
            docs: List of documents
            max_length: Maximum total length
            
        Returns:
            Formatted string
        """
        lines = []
        total_length = 0
        
        for i, doc in enumerate(docs[:8], 1):  # Limit to 8 chunks
            paper_id = doc.metadata.get('paper_id', 'unknown')
            content = doc.page_content[:500]  # Limit each chunk
            
            line = f"[Chunk {i} from {paper_id}]\n{content}..."
            
            if total_length + len(line) > max_length:
                lines.append("... (更多區塊省略)")
                break
            
            lines.append(line)
            total_length += len(line)
        
        return "\n\n".join(lines)
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON from LLM response
        
        Args:
            response: LLM response string
            
        Returns:
            Parsed dictionary
        """
        try:
            # Try to extract JSON from response
            # Sometimes LLM adds extra text before/after JSON
            
            # Method 1: Direct parse
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                pass
            
            # Method 2: Extract JSON block
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            
            # Method 3: Try to clean response
            cleaned = response.strip()
            if cleaned.startswith('```'):
                # Remove code block markers
                cleaned = re.sub(r'```json?\n?', '', cleaned)
                cleaned = re.sub(r'```\n?$', '', cleaned)
                cleaned = cleaned.strip()
                return json.loads(cleaned)
            
            logger.warning(f"Could not parse JSON from response: {response[:100]}")
            return {}
            
        except Exception as e:
            logger.error(f"JSON parsing error: {e}")
            return {}
    
    def _validate_quick_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and fill defaults for quick evaluation result"""
        validated = {
            'relevance': self._clamp(result.get('relevance', 0.5), 0.0, 1.0),
            'completeness': self._clamp(result.get('completeness', 0.5), 0.0, 1.0),
            'confidence': self._clamp(result.get('confidence', 0.5), 0.0, 1.0),
            'reasoning': result.get('reasoning', 'No reasoning provided'),
        }
        
        # If confidence not provided, calculate from relevance and completeness
        if 'confidence' not in result or result['confidence'] == 0:
            validated['confidence'] = (validated['relevance'] + validated['completeness']) / 2
        
        return validated
    
    def _validate_detailed_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and fill defaults for detailed evaluation result"""
        validated = {
            'relevance': self._clamp(result.get('relevance', 0.5), 0.0, 1.0),
            'completeness': self._clamp(result.get('completeness', 0.5), 0.0, 1.0),
            'depth': self._clamp(result.get('depth', 0.5), 0.0, 1.0),
            'confidence': self._clamp(result.get('confidence', 0.5), 0.0, 1.0),
            'reasoning': result.get('reasoning', 'No reasoning provided'),
            'missing_info': result.get('missing_info', []),
        }
        
        # If confidence not provided, calculate from other metrics
        if 'confidence' not in result or result['confidence'] == 0:
            validated['confidence'] = (
                validated['relevance'] + 
                validated['completeness'] + 
                validated['depth']
            ) / 3
        
        return validated
    
    def _clamp(self, value: float, min_val: float, max_val: float) -> float:
        """Clamp value between min and max"""
        return max(min_val, min(max_val, value))
    
    def _default_quick_result(self, low_confidence: bool = False) -> Dict[str, Any]:
        """Return default quick evaluation result"""
        confidence = 0.3 if low_confidence else 0.5
        return {
            'relevance': confidence,
            'completeness': confidence,
            'confidence': confidence,
            'reasoning': 'Evaluation failed, using default values',
        }
    
    def _default_detailed_result(self, low_confidence: bool = False) -> Dict[str, Any]:
        """Return default detailed evaluation result"""
        confidence = 0.3 if low_confidence else 0.5
        return {
            'relevance': confidence,
            'completeness': confidence,
            'depth': confidence,
            'confidence': confidence,
            'reasoning': 'Evaluation failed, using default values',
            'missing_info': [],
        }
    
    def _empty_evaluation(self, layer: int) -> Dict[str, Any]:
        """Return evaluation result for empty document list"""
        if layer == 1:
            result = self._default_quick_result(low_confidence=True)
        else:
            result = self._default_detailed_result(low_confidence=True)
        
        result['layer'] = layer
        result['threshold'] = 0.7 if layer == 1 else 0.8
        result['should_continue'] = True
        result['reasoning'] = 'No documents provided for evaluation'
        
        return result
