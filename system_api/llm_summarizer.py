"""
LLM Summarizer Module

Generates high-density summaries for academic paper sections using local LLM (Ollama).
Supports Map-Reduce strategy for long sections.

Key Features:
- Direct summarization for short sections (<1500 chars)
- Map-Reduce for long sections (≥1500 chars)
- Preserves technical terms, omits low-value content
- Configurable summary length
- Error handling with fallback to extractive summarization
"""

import logging
import time
from typing import List, Optional

try:
    from langchain_ollama import OllamaLLM
except ImportError:
    raise ImportError("langchain-ollama is required. Install with: pip install langchain-ollama")

logger = logging.getLogger(__name__)


class LLMSummarizer:
    """
    Generate high-density summaries for paper sections using LLM
    
    Features:
    - Map-Reduce for long sections
    - Prompt engineering to preserve technical terms
    - Retry logic and fallback
    - Sentence boundary-aware truncation
    """
    
    def __init__(
        self,
        model_name: str = "jcai/llama-3-taiwan-8b-instruct:q4_k_m",
        ollama_base_url: str = "http://localhost:11434",
        map_reduce_threshold: int = 1500,
        target_summary_length: int = 300,
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize LLM summarizer
        
        Args:
            model_name: Ollama model name (default: jcai/llama-3-taiwan-8b-instruct:q4_k_m)
            ollama_base_url: Ollama service URL
            map_reduce_threshold: Char count threshold for Map-Reduce
            target_summary_length: Target summary length in chars (base value, will be adjusted)
            timeout: Timeout per LLM call in seconds
            max_retries: Max retry attempts on failure
        """
        self.model_name = model_name
        self.ollama_base_url = ollama_base_url
        self.map_reduce_threshold = map_reduce_threshold
        self.target_summary_length = target_summary_length
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Dynamic summary length constraints
        self.min_summary_length = 200  # 最少 200 字
        self.max_summary_length = 1500  # 最多 1500 字
        
        # Initialize LLM client
        try:
            self.llm = OllamaLLM(
                model=model_name,
                base_url=ollama_base_url,
                timeout=timeout
            )
            logger.info(f"LLMSummarizer initialized (model={model_name})")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama LLM: {e}")
            raise RuntimeError(f"Cannot initialize LLM: {e}")
    
    def _calculate_target_length(self, text_length: int) -> int:
        """
        根據原文長度動態計算目標摘要長度
        
        策略：
        - 短文本（<1000 字）：200-300 字
        - 中等文本（1000-5000 字）：300-800 字
        - 長文本（>5000 字）：800-1500 字
        
        Args:
            text_length: 原文字數
            
        Returns:
            目標摘要長度（200-1500 字）
        """
        if text_length < 1000:
            # 短章節：壓縮比 30-40%
            target = int(text_length * 0.35)
        elif text_length < 3000:
            # 中等章節：壓縮比 20-30%
            target = int(text_length * 0.25)
        elif text_length < 5000:
            # 較長章節：壓縮比 15-20%
            target = int(text_length * 0.18)
        else:
            # 超長章節：壓縮比 10-15%
            target = int(text_length * 0.12)
        
        # 限制在 200-1500 字範圍內
        target = max(self.min_summary_length, min(target, self.max_summary_length))
        
        return target
    
    def summarize_section(
        self,
        section_text: str,
        section_name: str,
        paper_context: Optional[str] = None
    ) -> str:
        """
        Summarize a section with automatic Map-Reduce if needed
        
        Args:
            section_text: Full section text
            section_name: Section name (e.g., "Methodology")
            paper_context: Optional paper title for context
            
        Returns:
            Summary string (200-1500 chars, dynamically adjusted)
        """
        if not section_text or len(section_text) < 50:
            logger.warning(f"Section text too short ({len(section_text)} chars), skipping")
            return section_text
        
        # 動態計算目標摘要長度（根據原文長度）
        text_length = len(section_text)
        dynamic_target_length = self._calculate_target_length(text_length)
        
        logger.info(f"Section '{section_name}': {text_length} chars → target {dynamic_target_length} chars")
        
        # Choose strategy based on length
        if text_length < self.map_reduce_threshold:
            summary = self._summarize_direct(
                section_text, 
                section_name, 
                paper_context, 
                target_length=dynamic_target_length
            )
        else:
            summary = self._summarize_map_reduce(
                section_text, 
                section_name, 
                paper_context,
                target_length=dynamic_target_length
            )
        
        # Truncate to dynamic target length (with 10% tolerance)
        max_length = int(dynamic_target_length * 1.1)
        summary = self._truncate(summary, max_length)
        
        logger.debug(f"Summarized {section_name}: {len(section_text)} → {len(summary)} chars")
        return summary
    
    def _summarize_direct(
        self,
        text: str,
        section_name: str,
        paper_context: Optional[str],
        target_length: int = None
    ) -> str:
        """Direct summarization for short sections"""
        if target_length is None:
            target_length = self.target_summary_length
        
        prompt = self._build_prompt(
            text,
            section_name,
            paper_context,
            target_length=target_length
        )
        
        # Try with retries
        for attempt in range(self.max_retries):
            try:
                summary = self.llm.invoke(prompt)
                
                # Validate
                if summary and len(summary.strip()) >= 50:
                    return summary.strip()
                else:
                    logger.warning(f"LLM returned short summary ({len(summary)} chars), retry {attempt+1}/{self.max_retries}")
                    
            except Exception as e:
                logger.warning(f"LLM error on attempt {attempt+1}/{self.max_retries}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                continue
        
        # Fallback: Extractive summarization
        logger.warning(f"All LLM attempts failed, using extractive summary")
        return self._extractive_summary(text, self.target_summary_length)
    
    def _summarize_map_reduce(
        self,
        text: str,
        section_name: str,
        paper_context: Optional[str],
        target_length: int = None
    ) -> str:
        """
        Map-Reduce summarization for long sections
        
        Strategy:
        1. Map: Split into 1000-char sub-chunks, summarize each (~200 chars per sub-chunk)
        2. Reduce: Combine sub-summaries into final summary (dynamic length)
        """
        if target_length is None:
            target_length = self.target_summary_length
        
        # 計算每個子塊的目標長度（總目標長度的 60%，平均分配給各子塊）
        # 例如：總目標 900 字，5 個子塊 → 每塊約 108 字
        sub_chunks = self._split_text(text, chunk_size=1000, overlap=100)
        num_chunks = len(sub_chunks)
        sub_target_length = max(150, int(target_length * 0.6 / num_chunks))  # 最少 150 字/塊
        
        sub_summaries = []
        
        logger.info(f"Map-Reduce: Processing {num_chunks} sub-chunks for {section_name} (target: {target_length} chars)")
        
        for i, chunk in enumerate(sub_chunks):
            prompt = self._build_prompt(
                chunk,
                f"{section_name} (Part {i+1}/{num_chunks})",
                paper_context,
                target_length=sub_target_length
            )
            
            try:
                summary = self.llm.invoke(prompt)
                if summary and len(summary.strip()) >= 30:
                    sub_summaries.append(self._truncate(summary.strip(), sub_target_length))
                else:
                    # Fallback for this sub-chunk
                    sub_summaries.append(self._extractive_summary(chunk, sub_target_length))
            except Exception as e:
                logger.warning(f"Failed to summarize sub-chunk {i+1}: {e}")
                sub_summaries.append(self._extractive_summary(chunk, sub_target_length))
        
        # Reduce phase: Combine sub-summaries
        combined_text = "\n\n".join(sub_summaries)
        final_prompt = self._build_reduce_prompt(
            combined_text,
            section_name,
            paper_context,
            target_length=target_length
        )
        
        try:
            final_summary = self.llm.invoke(final_prompt)
            if final_summary and len(final_summary.strip()) >= 100:
                return final_summary.strip()
            else:
                # Fallback: Just concatenate first 2 sub-summaries
                return " ".join(sub_summaries[:2])
        except Exception as e:
            logger.warning(f"Failed to generate final summary: {e}")
            return " ".join(sub_summaries[:2])
    
    def _build_prompt(
        self,
        text: str,
        section_name: str,
        paper_context: Optional[str],
        target_length: int = 300
    ) -> str:
        """
        Build summarization prompt with instructions
        
        Emphasizes:
        - Preserve technical terms
        - Omit references, figures, acknowledgments
        - Natural, fluent language
        """
        context_line = f"\n**論文標題**：{paper_context}" if paper_context else ""
        
        return f"""你是一個專業的學術助理。請閱讀以下論文章節，並生成簡潔、精確的摘要。

**任務要求**：
1. 專注於核心內容：主要方法、關鍵技術、重要結果
2. 保留專業術語（例如：YOLOv8、LSTM、Transformer、CNN）
3. 省略不必要的細節（例如：圖表編號、參考文獻引用、致謝）
4. 使用自然、流暢的語言（不要列點）
5. 目標長度：約 {target_length} 字

**章節類型**：{section_name}{context_line}

**原文內容**：
{text}

**摘要**："""
    
    def _build_reduce_prompt(
        self,
        combined_summaries: str,
        section_name: str,
        paper_context: Optional[str],
        target_length: int = None
    ) -> str:
        """Build prompt for Reduce phase"""
        if target_length is None:
            target_length = self.target_summary_length
        
        context_line = f"\n**論文標題**：{paper_context}" if paper_context else ""
        
        return f"""你是一個專業的學術助理。以下是一個論文章節的多個子摘要，請將它們整合成一個連貫、完整的最終摘要。

**任務要求**：
1. 整合所有子摘要的核心資訊
2. 消除重複內容
3. 保持邏輯連貫性
4. 目標長度：約 {target_length} 字

**章節類型**：{section_name}{context_line}

**子摘要列表**：
{combined_summaries}

**最終摘要**："""
    
    def _split_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap
        return chunks
    
    def _truncate(self, text: str, max_length: int) -> str:
        """
        Truncate text to max_length, preserving sentence boundaries
        
        Finds last sentence boundary within limit to avoid mid-sentence cuts
        """
        if len(text) <= max_length:
            return text
        
        # Find last sentence boundary within limit
        truncated = text[:max_length]
        
        # Look for sentence endings (。. ! ? ！ ？)
        sentence_endings = [
            truncated.rfind('。'),
            truncated.rfind('.'),
            truncated.rfind('！'),
            truncated.rfind('？'),
            truncated.rfind('!'),
            truncated.rfind('?')
        ]
        
        last_period = max(sentence_endings)
        
        # Only use sentence boundary if it's not too early (>70% of target)
        if last_period > max_length * 0.7:
            return truncated[:last_period + 1]
        else:
            # No good sentence boundary, just add ellipsis
            return truncated.rstrip() + "..."
    
    def _extractive_summary(self, text: str, max_length: int) -> str:
        """
        Fallback extractive summarization
        
        Simply takes first max_length chars, preferring sentence boundaries
        """
        if len(text) <= max_length:
            return text
        
        # Take first max_length chars
        truncated = text[:max_length]
        
        # Find last sentence boundary
        sentence_endings = [
            truncated.rfind('。'),
            truncated.rfind('.'),
            truncated.rfind('！'),
            truncated.rfind('？')
        ]
        last_period = max(sentence_endings)
        
        if last_period > max_length * 0.5:
            return truncated[:last_period + 1]
        else:
            return truncated.rstrip() + "..."


# Utility function for quick usage
def summarize_text(
    text: str,
    section_name: str = "Section",
    paper_title: Optional[str] = None,
    model: str = "llama3:8b"
) -> str:
    """
    Convenience function to summarize text
    
    Args:
        text: Text to summarize
        section_name: Section name for context
        paper_title: Paper title for context
        model: Ollama model name
        
    Returns:
        Summary string
    """
    summarizer = LLMSummarizer(model_name=model)
    return summarizer.summarize_section(text, section_name, paper_title)
