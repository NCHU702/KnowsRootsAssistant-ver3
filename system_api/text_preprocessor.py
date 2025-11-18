"""
Text Preprocessor Module

Removes unwanted sections (references, appendix) from academic papers before chunking.
"""

import re
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Preprocessor to clean academic paper text before embedding
    
    Removes:
    - References section (參考文獻, References, Bibliography)
    - Appendix section (附錄, Appendix, Appendices)
    - Acknowledgements (致謝, 誌謝, Acknowledgements)
    - Table of Contents (目錄, Contents) - front matter
    - List of Tables (表目錄, List of Tables) - front matter
    - List of Figures (圖目錄, List of Figures) - front matter
    
    Strategy:
    - Front matter (ToC, LoT, LoF): Remove from start to end of section
    - Back matter (References, Appendix, Acknowledgements): Remove from start to end of document
    """
    
    # Patterns to detect section starts (case-insensitive)
    SECTION_PATTERNS = {
        'references': [
            # English patterns
            r'\n\s*References\s*\n',
            r'\n\s*REFERENCES\s*\n',
            r'\n\s*Bibliography\s*\n',
            r'\n\s*BIBLIOGRAPHY\s*\n',
            r'\n\s*\[?\d+\]?\s*References\s*\n',
            
            # Chinese patterns
            r'\n\s*參考文獻\s*\n',
            r'\n\s*参考文献\s*\n',
            r'\n\s*參\s*考\s*文\s*獻\s*\n',
            r'\n\s*引用文獻\s*\n',
            r'\n\s*文獻\s*\n',
        ],
        'appendix': [
            # English patterns
            r'\n\s*Appendix\s*\n',
            r'\n\s*APPENDIX\s*\n',
            r'\n\s*Appendices\s*\n',
            r'\n\s*APPENDICES\s*\n',
            r'\n\s*Appendix\s+[A-Z]\s*\n',
            r'\n\s*Appendix\s+\d+\s*\n',
            
            # Chinese patterns
            r'\n\s*附錄\s*\n',
            r'\n\s*附录\s*\n',
            r'\n\s*附\s*錄\s*\n',
        ],
        'acknowledgements': [
            # English patterns
            r'\n\s*Acknowledgements?\s*\n',
            r'\n\s*ACKNOWLEDGEMENTS?\s*\n',
            
            # Chinese patterns (may have page number before, like "iii 誌謝")
            r'\n[^\n]{0,10}致謝\s*\n',  # Allow up to 10 chars before 致謝
            r'\n[^\n]{0,10}誌謝\s*\n',  # Allow up to 10 chars before 誌謝
            r'\n[^\n]{0,10}謝辭\s*\n',  # Allow up to 10 chars before 謝辭
            r'\n\s*謝\s*辭\s*\n',
        ],
        'table_of_contents': [
            # English patterns
            r'\n\s*Table\s+of\s+Contents\s*\n',
            r'\n\s*TABLE\s+OF\s+CONTENTS\s*\n',
            r'\n\s*Contents\s*\n',
            r'\n\s*CONTENTS\s*\n',
            
            # Chinese patterns (may have page number before, like "iv 目錄")
            r'\n[^\n]{0,10}目錄\s*\n',  # Allow up to 10 chars (page number, etc.) before 目錄
            r'\n\s*目\s*錄\s*\n',
            r'\n\s*目\s+錄\s*\n',
        ],
        'list_of_tables': [
            # English patterns
            r'\n\s*List\s+of\s+Tables\s*\n',
            r'\n\s*LIST\s+OF\s+TABLES\s*\n',
            
            # Chinese patterns (may have page number before)
            r'\n[^\n]{0,10}表目錄\s*\n',  # Allow up to 10 chars before 表目錄
            r'\n\s*表\s*目\s*錄\s*\n',
            r'\n\s*表\s+目\s+錄\s*\n',
        ],
        'list_of_figures': [
            # English patterns
            r'\n\s*List\s+of\s+Figures\s*\n',
            r'\n\s*LIST\s+OF\s+FIGURES\s*\n',
            
            # Chinese patterns (may have page number before)
            r'\n[^\n]{0,10}圖目錄\s*\n',  # Allow up to 10 chars before 圖目錄
            r'\n\s*图目录\s*\n',
            r'\n\s*圖\s*目\s*錄\s*\n',
            r'\n\s*圖\s+目\s+錄\s*\n',
        ]
    }
    
    # Section classification
    # Note: acknowledgements usually appears in front matter (before ToC) in Chinese thesis
    FRONT_MATTER_SECTIONS = ['acknowledgements', 'table_of_contents', 'list_of_tables', 'list_of_figures']
    BACK_MATTER_SECTIONS = ['references', 'appendix']
    
    # Patterns to detect content sections (usually mark the end of front matter)
    CONTENT_START_PATTERNS = [
        # Chinese - Chapter 1
        r'\n\s*第一章',
        r'\n\s*第\s*一\s*章',
        r'\n\s*1\.\s*緒論',
        r'\n\s*1\s+緒論',
        r'\n\s*1\.\s*前言',
        
        # English - Chapter 1
        r'\n\s*Chapter\s+1',
        r'\n\s*CHAPTER\s+1',
        r'\n\s*1\.\s*Introduction',
        r'\n\s*1\s+Introduction',
        
        # Fallback patterns
        r'\n\s*緒論\s*\n',
        r'\n\s*Introduction\s*\n',
    ]
    
    def __init__(self):
        """
        Initialize Text Preprocessor
        
        All unwanted sections will be removed by default.
        """
        pass
        
    def preprocess(self, text: str) -> Tuple[str, dict]:
        """
        Remove unwanted sections from text
        
        Args:
            text: Raw PDF text
            
        Returns:
            Tuple of (cleaned_text, stats_dict)
            stats_dict contains information about what was removed
        """
        original_length = len(text)
        stats = {
            'original_length': original_length,
            'removed_sections': [],
            'removed_length': 0,
            'final_length': 0
        }
        
        cleaned_text = text
        
        # Step 1: Remove front matter (ToC, LoT, LoF)
        cleaned_text, front_matter_removed = self._remove_front_matter(cleaned_text)
        if front_matter_removed:
            stats['removed_sections'].extend(front_matter_removed)
        
        # Step 2: Remove back matter (References, Appendix, Acknowledgements)
        cleaned_text, back_matter_removed = self._remove_back_matter(cleaned_text)
        if back_matter_removed:
            stats['removed_sections'].extend(back_matter_removed)
        
        # Calculate statistics
        stats['removed_length'] = original_length - len(cleaned_text)
        stats['removed_percentage'] = (stats['removed_length'] / original_length) * 100 if original_length > 0 else 0
        stats['final_length'] = len(cleaned_text)
        
        if stats['removed_sections']:
            logger.info(
                f"Removed sections {stats['removed_sections']}: "
                f"{stats['removed_length']} chars ({stats['removed_percentage']:.1f}%)"
            )
        else:
            logger.debug("No unwanted sections found")
        
        return cleaned_text, stats
    
    def _remove_front_matter(self, text: str) -> Tuple[str, list]:
        """
        Remove front matter sections (ToC, LoT, LoF)
        Strategy: Find section start, then find where main content begins
        
        Args:
            text: Text to process
            
        Returns:
            Tuple of (cleaned_text, removed_sections_list)
        """
        removed_sections = []
        
        # Find all front matter sections
        front_matter_positions = []
        for section_type in self.FRONT_MATTER_SECTIONS:
            pos, _ = self._find_section_start(text, section_type)
            if pos is not None:
                front_matter_positions.append((pos, section_type))
        
        if not front_matter_positions:
            return text, removed_sections
        
        # Sort by position
        front_matter_positions.sort(key=lambda x: x[0])
        
        # Find where main content starts (after front matter)
        earliest_front_matter_pos = front_matter_positions[0][0]
        content_start_pos = self._find_content_start(text, earliest_front_matter_pos)
        
        if content_start_pos is not None and content_start_pos > earliest_front_matter_pos:
            # Remove from earliest front matter to start of content
            cleaned_text = text[:earliest_front_matter_pos] + text[content_start_pos:]
            removed_sections = [section for _, section in front_matter_positions]
            logger.debug(f"Removed front matter from pos {earliest_front_matter_pos} to {content_start_pos}")
        else:
            # Fallback: just remove a reasonable chunk after front matter start
            # (e.g., next 3000 chars or until we find content markers)
            search_end = min(earliest_front_matter_pos + 5000, len(text))
            chunk_to_search = text[earliest_front_matter_pos:search_end]
            
            # Look for double newline (often marks end of list)
            double_newline = chunk_to_search.find('\n\n\n')
            if double_newline != -1:
                end_pos = earliest_front_matter_pos + double_newline + 3
                cleaned_text = text[:earliest_front_matter_pos] + text[end_pos:]
                removed_sections = [section for _, section in front_matter_positions]
            else:
                # Can't find clear boundary, keep original
                cleaned_text = text
                logger.debug("Could not find clear end of front matter, keeping original")
        
        return cleaned_text, removed_sections
    
    def _remove_back_matter(self, text: str) -> Tuple[str, list]:
        """
        Remove back matter sections (References, Appendix, Acknowledgements)
        Strategy: Find earliest occurrence and truncate from there
        
        Args:
            text: Text to process
            
        Returns:
            Tuple of (cleaned_text, removed_sections_list)
        """
        removed_sections = []
        earliest_pos = len(text)
        text_length = len(text)
        
        # Find earliest back matter section with safety checks
        for section_type in self.BACK_MATTER_SECTIONS:
            pos, _ = self._find_section_start(text, section_type)
            
            if pos is not None:
                # Safety check: acknowledgements/致謝 shouldn't appear too early
                # In thesis format, it usually appears BEFORE table of contents (not after)
                # So we should ignore it in back matter and handle it in front matter
                if section_type == 'acknowledgements':
                    position_percentage = (pos / text_length) * 100 if text_length > 0 else 0
                    
                    if position_percentage < 60:
                        logger.debug(
                            f"Ignoring acknowledgements at {position_percentage:.1f}% "
                            "(too early, treating as front matter)"
                        )
                        continue
                
                if pos < earliest_pos:
                    earliest_pos = pos
                    if section_type not in removed_sections:
                        removed_sections.append(section_type)
        
        if earliest_pos < len(text):
            cleaned_text = text[:earliest_pos].strip()
        else:
            cleaned_text = text
        
        return cleaned_text, removed_sections
    
    def _find_content_start(self, text: str, after_position: int) -> Optional[int]:
        """
        Find where main content starts (typically Abstract or Introduction)
        
        Args:
            text: Text to search
            after_position: Start searching after this position
            
        Returns:
            Position where content starts, or None if not found
        """
        search_text = text[after_position:]
        
        for pattern in self.CONTENT_START_PATTERNS:
            match = re.search(pattern, search_text, re.IGNORECASE)
            if match:
                return after_position + match.start()
        
        return None
    
    def _find_section_start(self, text: str, section_type: str) -> Tuple[Optional[int], Optional[str]]:
        """
        Find the starting position of a section type
        
        Args:
            text: Text to search in
            section_type: Type of section ('references', 'appendix', 'acknowledgements')
            
        Returns:
            Tuple of (position, matched_pattern) or (None, None) if not found
        """
        patterns = self.SECTION_PATTERNS.get(section_type, [])
        
        earliest_pos = None
        matched_pattern = None
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                pos = match.start()
                if earliest_pos is None or pos < earliest_pos:
                    earliest_pos = pos
                    matched_pattern = pattern
        
        return earliest_pos, matched_pattern
    
    def validate_preprocessing(self, original_text: str, cleaned_text: str) -> bool:
        """
        Validate that preprocessing didn't remove too much content
        
        Args:
            original_text: Original text
            cleaned_text: Cleaned text
            
        Returns:
            True if validation passes, False otherwise
        """
        original_len = len(original_text)
        cleaned_len = len(cleaned_text)
        
        # Check if we removed more than 60% (likely something went wrong)
        if cleaned_len < original_len * 0.4:
            logger.warning(
                f"Preprocessing removed {100 - (cleaned_len/original_len)*100:.1f}% of text. "
                f"This might be too aggressive."
            )
            return False
        
        # Check if cleaned text is too short
        if cleaned_len < 500:
            logger.warning(
                f"Cleaned text is very short ({cleaned_len} chars). "
                f"Might have removed important content."
            )
            return False
        
        return True
    
    def get_statistics(self, text: str) -> dict:
        """
        Get statistics about sections in text without removing them
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with section positions and stats
        """
        stats = {
            'total_length': len(text),
            'sections_found': {}
        }
        
        section_types = [
            'references',
            'appendix',
            'acknowledgements',
            'table_of_contents',
            'list_of_tables',
            'list_of_figures'
        ]
        
        for section_type in section_types:
            pos, pattern = self._find_section_start(text, section_type)
            if pos is not None:
                stats['sections_found'][section_type] = {
                    'position': pos,
                    'pattern': pattern,
                    'position_percentage': (pos / len(text)) * 100
                }
        
        return stats
