"""
PDF Section Parser Module

Extracts semantic sections (Abstract, Introduction, Methodology, Results, Conclusion)
from academic PDF papers using PyMuPDF + regex patterns.

Key Features:
- Section detection for English and Chinese papers
- Multi-column layout handling
- Fallback to paragraph-based chunking for unstructured PDFs
- Support for numbered sections (1. Introduction, 2.1 Background)
"""

import os
import re
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field

try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError("PyMuPDF (fitz) is required. Install with: pip install PyMuPDF")

logger = logging.getLogger(__name__)


@dataclass
class Section:
    """Represents a section extracted from a PDF paper"""
    name: str                    # e.g., "Abstract", "Methodology"
    text: str                    # Full section text
    start_page: int             # Starting page number (0-indexed)
    end_page: int               # Ending page number (0-indexed)
    char_count: int             # Character count
    keywords: List[str] = field(default_factory=list)  # Extracted keywords (optional)


class PDFSectionParser:
    """
    Parse academic PDFs into semantic sections
    
    Supports:
    - English and Chinese section names
    - Numbered sections (1. Introduction, 2.1 Background)
    - Multi-column layouts
    - Fallback for unstructured PDFs
    """
    
    def __init__(self, method: str = 'pymupdf_regex', min_sections: int = 3):
        """
        Initialize PDF section parser
        
        Args:
            method: 'pymupdf_regex' (only supported method currently)
            min_sections: Minimum sections required to consider PDF structured
        """
        self.method = method
        self.min_sections = min_sections
        self._section_patterns = self._load_patterns()
        logger.info(f"PDFSectionParser initialized (method={method}, min_sections={min_sections})")
    
    def parse(self, pdf_path: str) -> List[Section]:
        """
        Extract sections from PDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of Section objects
            
        Raises:
            FileNotFoundError: If PDF file not found
            ValueError: If PDF cannot be opened
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            sections = self._parse_with_pymupdf(pdf_path)
            
            if len(sections) < self.min_sections:
                logger.warning(f"Only {len(sections)} sections found, using fallback chunking")
                sections = self._fallback_paragraph_chunking(pdf_path)
            
            logger.info(f"Parsed {len(sections)} sections from {os.path.basename(pdf_path)}")
            return sections
            
        except Exception as e:
            logger.error(f"Failed to parse PDF {pdf_path}: {e}", exc_info=True)
            raise ValueError(f"Failed to parse PDF: {e}")
    
    def _parse_with_pymupdf(self, pdf_path: str) -> List[Section]:
        """
        Parse PDF using PyMuPDF + regex patterns
        
        Strategy:
        1. Extract text blocks with page metadata
        2. Sort blocks by position (handle multi-column)
        3. Detect section headers using regex
        4. Split text at section boundaries
        5. Return Section objects
        """
        doc = fitz.open(pdf_path)
        
        # Extract text blocks with position metadata
        all_blocks = []
        for page_num, page in enumerate(doc):
            blocks = page.get_text("blocks")  # Returns (x0, y0, x1, y1, text, block_no, block_type)
            for block in blocks:
                if block[6] == 0:  # Text block (not image)
                    all_blocks.append({
                        'page': page_num,
                        'x0': block[0],
                        'y0': block[1],
                        'x1': block[2],
                        'y1': block[3],
                        'text': block[4].strip()
                    })
        
        doc.close()
        
        # Sort blocks by position (top-to-bottom, left-to-right for multi-column)
        all_blocks.sort(key=lambda b: (b['page'], b['y0'], b['x0']))
        
        # Concatenate text
        full_text = "\n\n".join([b['text'] for b in all_blocks if b['text']])
        
        # Detect section boundaries
        section_matches = []
        for section_type, patterns in self._section_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, full_text, re.MULTILINE | re.IGNORECASE):
                    section_matches.append({
                        'start': match.start(),
                        'end': match.end(),
                        'name': section_type.title(),
                        'matched_text': match.group()
                    })
        
        # Sort by position in text
        section_matches.sort(key=lambda m: m['start'])
        
        # Remove duplicates (keep first match at each position)
        unique_matches = []
        last_end = -1
        for match in section_matches:
            if match['start'] >= last_end + 10:  # At least 10 chars apart
                unique_matches.append(match)
                last_end = match['end']
        
        # Build sections
        sections = []
        for i, match in enumerate(unique_matches):
            # Determine section text
            section_start = match['end']
            section_end = unique_matches[i + 1]['start'] if i + 1 < len(unique_matches) else len(full_text)
            section_text = full_text[section_start:section_end].strip()
            
            if not section_text or len(section_text) < 50:  # Skip empty sections
                continue
            
            # Estimate pages (approximate)
            start_page = self._estimate_page(all_blocks, match['start'])
            end_page = self._estimate_page(all_blocks, section_end)
            
            sections.append(Section(
                name=match['name'],
                text=section_text,
                start_page=start_page,
                end_page=end_page,
                char_count=len(section_text)
            ))
        
        return sections
    
    def _fallback_paragraph_chunking(self, pdf_path: str) -> List[Section]:
        """
        Fallback strategy for unstructured PDFs
        
        Groups text into ~1500-char chunks based on paragraph boundaries
        """
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text() + "\n\n"
        doc.close()
        
        # Split by paragraphs
        paragraphs = re.split(r'\n\n+', full_text)
        
        # Group into chunks of ~1500 chars
        sections = []
        current_chunk = ""
        chunk_num = 1
        start_page = 0
        
        for para in paragraphs:
            if len(current_chunk) + len(para) > 1500 and current_chunk:
                # Create section
                sections.append(Section(
                    name=f"Section {chunk_num}",
                    text=current_chunk.strip(),
                    start_page=start_page,
                    end_page=start_page,  # Approximate
                    char_count=len(current_chunk)
                ))
                current_chunk = para
                chunk_num += 1
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        
        # Add last chunk
        if current_chunk.strip():
            sections.append(Section(
                name=f"Section {chunk_num}",
                text=current_chunk.strip(),
                start_page=start_page,
                end_page=start_page,
                char_count=len(current_chunk)
            ))
        
        return sections
    
    def _estimate_page(self, blocks: List[Dict], char_position: int) -> int:
        """Estimate page number from character position in concatenated text"""
        cumulative_length = 0
        for block in blocks:
            cumulative_length += len(block['text']) + 2  # +2 for "\n\n"
            if cumulative_length >= char_position:
                return block['page']
        return blocks[-1]['page'] if blocks else 0
    
    def _load_patterns(self) -> Dict[str, List[str]]:
        """
        Load regex patterns for section detection
        
        基於實際論文分析結果的完整 pattern（2025/01/18 更新）
        
        Returns:
            Dict mapping section types to regex patterns
        """
        return {
            'abstract': [
                # English
                r'^Abstract\s*$',
                r'^ABSTRACT\s*$',
                # Chinese
                r'^摘要\s*$',
                r'^中文摘要\s*$',
                r'^摘\s*要\s*$',
            ],
            'introduction': [
                # English with numbering
                r'^Introduction\s*$',
                r'^INTRODUCTION\s*$',
                r'^1\.?\s*Introduction',
                r'^1\.?\s*INTRODUCTION',
                r'^Chapter\s+1\s+Introduction',
                r'^Chapter\s+1\s*[:\-]\s*Introduction',
                r'^I\.?\s+Introduction',
                # Chinese
                r'^第一章\s*緒論',
                r'^第一章\s+緒論',
                r'^第一章、緒論',
                r'^第一章\s*導論',
                r'^一、?導論',
                r'^一、?緒論',
                r'^導論\s*$',
                r'^緒論\s*$',
            ],
            'related_work': [
                # English
                r'^Related Work\s*$',
                r'^RELATED WORK\s*$',
                r'^2\.?\s*Related Work',
                r'^Chapter\s+2\s+Related Work',
                r'^Literature Review\s*$',
                r'^Background\s*$',
                r'^[12]\.1\s+Background',  # 1.1 Background
                r'^II\.?\s+Related Work',
                # Chinese  
                r'^第二章\s*文獻回顧',
                r'^第二章\s+文獻回顧',
                r'^第二章\s*相關研究',
                r'^相關研究',
                r'^相關文獻',
                r'^相關文獻回顧',
                r'^文獻探討',
                r'^文獻回顧',
                r'^二、?文獻',
            ],
            'methodology': [
                # English
                r'^Methodology\s*$',
                r'^Method\s*$',
                r'^Methods\s*$',
                r'^METHODOLOGY\s*$',
                r'^METHODS?\s*$',
                r'^[234]\.?\s*Methodology',
                r'^[234]\.?\s*Method(s)?',
                r'^Chapter\s+[234]\s+Method',
                r'^Proposed Method',
                r'^Approach\s*$',
                r'^III\.?\s+Method',
                r'^IV\.?\s+Method',
                # Chinese
                r'^第[三四]章\s*研究方法',
                r'^第[三四]章\s+研究方法',
                r'^第四章\s*研究方法',
                r'^研究方法\s*$',
                r'^方法論\s*$',
                r'^方法\s*$',
                r'^三、?方法',
                r'^四、?方法',
            ],
            'results': [
                # English
                r'^Results\s*$',
                r'^RESULTS\s*$',
                r'^[345]\.?\s*Results',
                r'^Chapter\s+[345]\s+Results',
                r'^Experiments\s*$',
                r'^Experimental Results',
                r'^Experiment\s*$',
                r'^IV\.?\s+Results',
                r'^V\.?\s+Results',
                # Chinese
                r'^第[四五]章\s*實驗',
                r'^第[四五]章\s+實驗',
                r'^第[四五]章\s*實驗模擬',
                r'^第[四五]章\s+實驗模擬',
                r'^第四章\s*實驗模擬',
                r'^第五章\s*實驗模擬',
                r'^實驗結果',
                r'^實驗模擬',
                r'^結果\s*$',
                r'^四、?實驗',
                r'^五、?實驗',
            ],
            'discussion': [
                # English
                r'^Discussion\s*$',
                r'^DISCUSSION\s*$',
                r'^[45]\.?\s*Discussion',
                r'^Chapter\s+[45]\s+Discussion',
                r'^Results and Discussion',
                r'^V\.?\s+Discussion',
                # Chinese
                r'^討論\s*$',
                r'^結果與討論',
                r'^結論與探討',
            ],
            'conclusion': [
                # English
                r'^Conclusion\s*$',
                r'^Conclusions\s*$',
                r'^CONCLUSION\s*$',
                r'^[4567]\.?\s*Conclusion(s)?',
                r'^Chapter\s+[4567]\s+Conclusion',
                r'^Chapter\s+[4567]\s*[:\-]\s*Conclusion',
                r'^Summary\s*$',
                r'^VI\.?\s+Conclusion',
                r'^VII\.?\s+Conclusion',
                # Chinese
                r'^第[五六七]章\s*結論',
                r'^第[五六七]章\s+結論',
                r'^第五章\s*結論與未來研究',
                r'^第六章\s*結論與未來研究',
                r'^第七章\s*結論與未來研究',
                r'^第[五六七]章\s*結論與未來展望',
                r'^第[五六七]章\s*結論與未來研究建議',
                r'^第六章、結論',
                r'^結論\s*$',
                r'^結論與未來',
                r'^五、?結論',
                r'^六、?結論',
                r'^七、?結論',
            ],
            'references': [
                # English
                r'^References\s*$',
                r'^REFERENCES\s*$',
                r'^Bibliography\s*$',
                # Chinese
                r'^參考文獻\s*$',
                r'^参考文献\s*$',
            ],
        }


# Utility function for quick usage
def parse_pdf_sections(pdf_path: str, min_sections: int = 3) -> List[Section]:
    """
    Convenience function to parse PDF sections
    
    Args:
        pdf_path: Path to PDF file
        min_sections: Minimum sections to consider structured
        
    Returns:
        List of Section objects
    """
    parser = PDFSectionParser(min_sections=min_sections)
    return parser.parse(pdf_path)
