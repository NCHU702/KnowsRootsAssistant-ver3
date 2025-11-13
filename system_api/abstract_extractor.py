"""
Abstract Extractor Module

Extracts or generates paper abstracts using hybrid strategy:
1. Try regex-based extraction from PDF text
2. If fails, use LLM to generate abstract from first few pages
"""

import re
import logging
from typing import Dict, Optional, Any
from langchain_ollama import OllamaLLM

logger = logging.getLogger(__name__)


class AbstractExtractor:
    """
    Extract abstracts from PDFs using hybrid strategy
    
    Supports:
    - Regex-based extraction (fast, works for well-formatted papers)
    - LLM-based generation (slower, works as fallback)
    """
    
    # Regex patterns for abstract extraction
    EXTRACTION_PATTERNS = [
        # English patterns
        r"Abstract[:\s\n]+(.*?)(?=\n\n[A-Z]|\nIntroduction|\n1\.|\nKeywords|\nIndex Terms)",
        r"ABSTRACT[:\s\n]+(.*?)(?=\n\n[A-Z]|\nINTRODUCTION|\n1\.|\nKEYWORDS|\nINDEX TERMS)",
        
        # Chinese patterns
        r"摘要[:：\s\n]+(.*?)(?=\n\n|\n前言|\n緒論|\n1\.|\n關鍵詞|\n关键词|\n關鍵字)",
        r"摘\s*要[:：\s\n]+(.*?)(?=\n\n|\n前言|\n緒論|\n1\.|\n關鍵詞|\n关键词|\n關鍵字)",
        # Common case: 摘要 on its own line and the abstract follows on subsequent lines
        r"摘要\s*\n(.*?)(?=\n\n|\n關鍵詞|\n关键词|\n關鍵字|\n前言|\n緒論)",
        r"摘\s*要\s*\n(.*?)(?=\n\n|\n關鍵詞|\n关键词|\n關鍵字|\n前言|\n緒論)",
        
        # Alternative patterns
        r"(?i)abstract\s*[:\-–—]\s*(.*?)(?=\n\n|\nintroduction|\n1\.)",
    ]
    
    # LLM prompt for abstract generation
    GENERATION_PROMPT = """請為以下學術論文生成一個簡潔的摘要（200-300字）。

論文開頭內容：
{text}

摘要應包含：
1. 研究目的和問題
2. 主要方法
3. 核心發現或貢獻

請只返回摘要文本，不要包含其他內容。"""

    def __init__(self, llm: OllamaLLM):
        """
        Initialize Abstract Extractor
        
        Args:
            llm: LLM instance for generating abstracts when extraction fails
        """
        self.llm = llm
        
    def extract(
        self,
        pdf_path: str,
        pdf_text: str,
        paper_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract or generate abstract using hybrid strategy
        """
        logger.info(f"Extracting abstract from {pdf_path}")
        
        # Step 1: Try regex extraction
        extracted_abstract = self._extract_with_regex(pdf_text)
        
        if extracted_abstract and self._validate_abstract(extracted_abstract):
            logger.info(f"Successfully extracted abstract using regex ({len(extracted_abstract)} chars)")
            return {
                'abstract': extracted_abstract,
                'source': 'extracted',
                'confidence': 0.9,
                'paper_id': paper_id or self._generate_paper_id(pdf_path),
                'method': 'regex',
                'metadata': {
                    'pdf_path': pdf_path,
                    'length': len(extracted_abstract)
                }
            }

        # Step 2: Fallback to LLM generation
        logger.info("Regex extraction failed, generating abstract with LLM")
        generated_abstract = self._generate_with_llm(pdf_text, pdf_path)
        
        if generated_abstract and self._validate_abstract(generated_abstract):
            logger.info(f"Successfully generated abstract with LLM ({len(generated_abstract)} chars)")
            return {
                'abstract': generated_abstract,
                'source': 'generated',
                'confidence': 0.7,
                'paper_id': paper_id or self._generate_paper_id(pdf_path),
                'method': 'llm',
                'metadata': {
                    'pdf_path': pdf_path,
                    'length': len(generated_abstract)
                }
            }

        # Step 3: Final fallback - use first N characters and include title metadata
        logger.warning("Both extraction and generation failed, using first 500 chars as fallback")
        fallback_abstract = self._fallback_abstract(pdf_text, pdf_path)
        return {
            'abstract': fallback_abstract,
            'source': 'fallback',
            'confidence': 0.3,
            'paper_id': paper_id or self._generate_paper_id(pdf_path),
            'method': 'fallback',
            'metadata': {
                'pdf_path': pdf_path,
                'length': len(fallback_abstract),
                'title': self._extract_title_from_path(pdf_path),
                'warning': 'Used fallback method due to extraction/generation failure'
            }
        }

    def _extract_with_regex(self, text: str) -> Optional[str]:
        """
        Try to extract abstract using regex patterns
        """
        # Preserve paragraph and line breaks so regex patterns using newlines can match
        # Normalize CRLF to LF
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        # Collapse more than 2 consecutive newlines to exactly two
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Trim excessive spaces but keep newlines
        text = re.sub(r'[ \t]+', ' ', text)

        for i, pattern in enumerate(self.EXTRACTION_PATTERNS):
            try:
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    abstract = match.group(1).strip()

                    # Clean up the extracted text
                    abstract = self._clean_extracted_text(abstract)

                    # Validate length
                    if 50 < len(abstract) < 2000:
                        logger.debug(f"Pattern {i} matched, extracted {len(abstract)} chars")
                        return abstract
                    else:
                        logger.debug(f"Pattern {i} matched but length invalid: {len(abstract)} chars")

            except Exception as e:
                logger.debug(f"Pattern {i} failed: {e}")
                continue

        return None

    def _generate_with_llm(self, text: str, pdf_path: Optional[str] = None) -> Optional[str]:
        """
        Generate abstract using LLM if extraction fails
        """
        try:
            # Use first 3000 characters (approximately first 3 pages)
            excerpt = text[:3000]

            # Clean excerpt but preserve some newlines for context
            excerpt = excerpt.replace('\r\n', '\n').replace('\r', '\n')
            excerpt = re.sub(r'\n{3,}', '\n\n', excerpt)
            excerpt = re.sub(r'[ \t]+', ' ', excerpt)

            # If we have a filename/title hint, include it to help LLM
            title_hint = self._extract_title_from_path(pdf_path) if pdf_path else None
            prompt_text = excerpt
            if title_hint:
                prompt_text = f"Title: {title_hint}\n\n{excerpt}"

            # Generate prompt
            prompt = self.GENERATION_PROMPT.format(text=prompt_text)

            # Call LLM with a small retry loop to handle transient errors
            logger.debug("Calling LLM to generate abstract (with retry)...")
            abstract = None
            for attempt in range(2):
                try:
                    abstract = self.llm.invoke(prompt)
                    if abstract:
                        break
                except Exception as inner_e:
                    logger.warning(f"LLM attempt {attempt+1} failed: {inner_e}")
                    abstract = None

            if not abstract:
                logger.error("LLM generation failed after retries")
                return None

            # Clean generated text
            abstract = abstract.strip()

            # Remove common prefixes that LLM might add
            prefixes_to_remove = [
                "摘要：", "摘要:", "Abstract:", "ABSTRACT:",
                "這篇論文", "本研究", "本文",
                "Here is the abstract:", "Here's the abstract:"
            ]

            for prefix in prefixes_to_remove:
                if abstract.startswith(prefix):
                    abstract = abstract[len(prefix):].strip()

            return abstract

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return None

    def _clean_extracted_text(self, text: str) -> str:
        """
        Clean extracted text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove page numbers
        text = re.sub(r'\b\d+\b\s*$', '', text)

        # Remove common artifacts
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)

        # Trim
        text = text.strip()

        return text

    def _validate_abstract(self, abstract: str) -> bool:
        """
        Validate abstract length and quality
        """
        if not abstract:
            return False

        # Check length (200-2000 characters is reasonable for abstracts)
        length = len(abstract)
        if length < 100 or length > 3000:
            logger.debug(f"Abstract length invalid: {length} chars")
            return False

        # Check if it contains actual words (not just numbers/symbols)
        word_count = len(re.findall(r'\b\w+\b', abstract))
        # Allow somewhat shorter abstracts from some conferences/languages
        if word_count < 20:
            logger.debug(f"Abstract has too few words: {word_count}")
            return False

        # Check for common nonsensical patterns
        if re.search(r'(.)\1{20,}', abstract):  # Repeated characters
            logger.debug("Abstract contains repeated characters")
            return False

        return True

    def _fallback_abstract(self, text: str, pdf_path: Optional[str] = None) -> str:
        """
        Generate fallback abstract and include title when possible
        """
        # Get first 500 characters
        fallback = text[:500].strip()

        # Try to end at sentence boundary
        last_period = fallback.rfind('.')
        if last_period > 200:  # At least 200 chars before period
            fallback = fallback[:last_period + 1]

        # If there's a title available, prefix it to make fallback more informative
        try:
            title = self._extract_title_from_path(pdf_path)
            if title:
                fallback = f"{title} — " + fallback
        except Exception:
            pass

        return fallback

    def _extract_title_from_path(self, pdf_path: Optional[str]) -> Optional[str]:
        """
        Derive a simple title from the PDF filename when no metadata is available
        """
        if not pdf_path:
            return None
        import os
        filename = os.path.splitext(os.path.basename(pdf_path))[0]
        # Replace underscores/dashes with spaces and trim
        title = re.sub(r'[_\-]+', ' ', filename).strip()
        # If title is very short return None
        if len(title) < 3:
            return None
        return title

    def _generate_paper_id(self, pdf_path: str) -> str:
        """
        Generate paper ID from PDF path
        """
        import os
        import hashlib

        # Use filename without extension
        filename = os.path.splitext(os.path.basename(pdf_path))[0]

        # Clean filename
        filename = re.sub(r'[^a-zA-Z0-9_-]', '_', filename)

        # Add hash for uniqueness
        hash_suffix = hashlib.md5(pdf_path.encode()).hexdigest()[:8]

        return f"paper_{filename}_{hash_suffix}"


# Convenience function for standalone usage
def extract_abstract(
    pdf_path: str,
    pdf_text: str,
    llm: OllamaLLM,
    paper_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to extract abstract
    """
    extractor = AbstractExtractor(llm)
    return extractor.extract(pdf_path, pdf_text, paper_id)
