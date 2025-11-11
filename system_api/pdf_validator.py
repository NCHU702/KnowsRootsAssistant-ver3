"""
PDF Validator Module

Validates uploaded PDF files for:
- File type and size
- Text extractability
- Required academic sections (bilingual support)
"""

import os
import re
from typing import Dict, List, Tuple, Optional
import PyPDF2
import logging

logger = logging.getLogger(__name__)

# Maximum file size: 50MB
MAX_FILE_SIZE = 50 * 1024 * 1024

# Minimum characters for text extractability
MIN_TEXT_LENGTH = 100

# Required fields with bilingual keywords
REQUIRED_FIELDS = {
    "論文標題": ["標題", "title", "paper title", "題目"],
    "年份": ["年份", "year", "published", "發表"],
    "作者": ["作者", "author", "authors", "撰寫人"],
    "摘要": ["摘要", "abstract", "summary"],
    "研究目的": ["研究目的", "objective", "purpose", "research objective", "目標", "目的"],
    "相關研究": ["相關研究", "related work", "literature review", "背景", "background", "文獻探討"],
    "資料集": ["資料集", "dataset", "data", "數據集"],
    "資料前處理": ["前處理", "preprocessing", "data preprocessing", "資料處理", "預處理"],
    "建模方式": ["模型", "model", "方法", "method", "methodology", "建模", "模式"],
    "評估指標": ["評估", "evaluation", "metrics", "指標", "實驗結果", "results", "performance"]
}


class ValidationError(Exception):
    """Custom exception for validation errors"""
    def __init__(self, message: str, missing_fields: List[str] = None):
        super().__init__(message)
        self.missing_fields = missing_fields or []


class PDFValidator:
    """Validates PDF files for paper upload"""
    
    def __init__(self, llm=None):
        """
        Initialize PDF validator
        
        Args:
            llm: Optional LLM instance for assisted field detection
        """
        self.llm = llm
    
    def validate_file_type(self, file_path: str) -> bool:
        """
        Validate that file is a PDF
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if valid PDF
            
        Raises:
            ValidationError: If file is not a PDF
        """
        # Check extension
        if not file_path.lower().endswith('.pdf'):
            raise ValidationError("Invalid file type. Only PDF files are accepted.")
        
        # Check PDF magic bytes
        try:
            with open(file_path, 'rb') as f:
                header = f.read(4)
                if header != b'%PDF':
                    raise ValidationError("Invalid file type. Only PDF files are accepted.")
        except Exception as e:
            raise ValidationError(f"Error reading file: {str(e)}")
        
        return True
    
    def validate_file_size(self, file_path: str) -> bool:
        """
        Validate file size is within limits
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if size is acceptable
            
        Raises:
            ValidationError: If file exceeds size limit
        """
        file_size = os.path.getsize(file_path)
        if file_size > MAX_FILE_SIZE:
            size_mb = file_size / (1024 * 1024)
            raise ValidationError(f"File size ({size_mb:.1f}MB) exceeds maximum limit of 50MB")
        
        return True
    
    def extract_text(self, file_path: str) -> str:
        """
        Extract text from PDF
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text
            
        Raises:
            ValidationError: If text extraction fails
        """
        try:
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                text = ""
                
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                
                return text.strip()
        except Exception as e:
            raise ValidationError(f"Error extracting text from PDF: {str(e)}")
    
    def validate_text_extractability(self, text: str) -> bool:
        """
        Validate that sufficient text was extracted
        
        Args:
            text: Extracted text
            
        Returns:
            True if text is sufficient
            
        Raises:
            ValidationError: If insufficient text
        """
        if len(text) < MIN_TEXT_LENGTH:
            raise ValidationError(
                "PDF contains insufficient text. Please upload a searchable PDF, not a scanned image."
            )
        
        return True
    
    def detect_field_by_keywords(self, text: str, field_name: str, keywords: List[str]) -> bool:
        """
        Detect if a field exists using keyword matching (case-insensitive)
        
        Args:
            text: PDF text
            field_name: Name of the field
            keywords: List of keywords to search for
            
        Returns:
            True if field is detected
        """
        text_lower = text.lower()
        
        for keyword in keywords:
            # Create pattern to match keyword as section header
            # Match keyword followed by colon, newline, or Chinese colon
            patterns = [
                rf'\b{re.escape(keyword.lower())}\b[\s:：]',
                rf'^{re.escape(keyword.lower())}[\s:：]',
                rf'\n{re.escape(keyword.lower())}[\s:：]'
            ]
            
            for pattern in patterns:
                if re.search(pattern, text_lower, re.MULTILINE | re.IGNORECASE):
                    logger.info(f"Field '{field_name}' detected with keyword '{keyword}'")
                    return True
        
        return False
    
    def detect_fields_with_llm(self, text: str, undetected_fields: List[str]) -> Dict[str, bool]:
        """
        Use LLM to detect fields when keyword matching has low confidence
        
        Args:
            text: PDF text
            undetected_fields: List of field names that weren't detected by keywords
            
        Returns:
            Dictionary mapping field names to detection status
        """
        if not self.llm or not undetected_fields:
            return {field: False for field in undetected_fields}
        
        # Truncate text if too long (keep first 8000 characters)
        truncated_text = text[:8000] if len(text) > 8000 else text
        
        prompt = f"""以下是一篇論文的文本（可能為中文、英文或中英混合）。請判斷是否包含以下部分，並返回JSON格式結果：
The following is a paper text (may be in Chinese, English, or mixed). Please determine if it contains the following sections and return JSON format results:

必需欄位 / Required fields:
{', '.join(undetected_fields)}

論文內容 / Paper content:
{truncated_text}

請只返回JSON格式（不要其他文字），格式如下 / Return only JSON format:
{{
  "論文標題": true/false,
  "年份": true/false,
  "作者": true/false,
  "摘要": true/false,
  "研究目的": true/false,
  "相關研究": true/false,
  "資料集": true/false,
  "資料前處理": true/false,
  "建模方式": true/false,
  "評估指標": true/false
}}
"""
        
        try:
            response = self.llm.invoke(prompt)
            
            # Try to parse JSON from response
            import json
            # Extract JSON from response (remove markdown code blocks if present)
            json_text = response.strip()
            if json_text.startswith('```'):
                json_text = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', json_text, re.DOTALL)
                if json_text:
                    json_text = json_text.group(1)
            
            result = json.loads(json_text)
            
            return {field: result.get(field, False) for field in undetected_fields}
        
        except Exception as e:
            logger.warning(f"LLM field detection failed: {e}")
            return {field: False for field in undetected_fields}
    
    def validate_required_fields(self, text: str) -> Tuple[bool, List[str]]:
        """
        Validate that all required fields are present
        
        Args:
            text: Extracted PDF text
            
        Returns:
            Tuple of (success, list of missing fields)
        """
        missing_fields = []
        detected_fields = {}
        
        # First pass: keyword-based detection
        for field_name, keywords in REQUIRED_FIELDS.items():
            detected = self.detect_field_by_keywords(text, field_name, keywords)
            detected_fields[field_name] = detected
            if not detected:
                missing_fields.append(field_name)
        
        # Second pass: LLM-assisted detection for undetected fields
        if missing_fields and self.llm:
            logger.info(f"Using LLM to detect {len(missing_fields)} undetected fields")
            llm_results = self.detect_fields_with_llm(text, missing_fields)
            
            # Update detection status
            newly_detected = []
            for field in missing_fields[:]:
                if llm_results.get(field, False):
                    detected_fields[field] = True
                    missing_fields.remove(field)
                    newly_detected.append(field)
            
            if newly_detected:
                logger.info(f"LLM detected additional fields: {newly_detected}")
        
        if missing_fields:
            logger.warning(f"Missing required fields: {missing_fields}")
            return False, missing_fields
        
        return True, []
    
    def validate(self, file_path: str) -> Tuple[bool, str, Optional[str]]:
        """
        Perform complete validation on a PDF file
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Tuple of (success: bool, message: str, extracted_text: Optional[str])
        """
        try:
            # Step 1: Validate file type
            self.validate_file_type(file_path)
            logger.info(f"File type validation passed: {file_path}")
            
            # Step 2: Validate file size
            self.validate_file_size(file_path)
            logger.info(f"File size validation passed: {file_path}")
            
            # Step 3: Extract text
            text = self.extract_text(file_path)
            logger.info(f"Text extraction completed: {len(text)} characters")
            
            # Step 4: Validate text extractability
            self.validate_text_extractability(text)
            logger.info("Text extractability validation passed")
            
            # Step 5: Validate required fields
            success, missing_fields = self.validate_required_fields(text)
            
            if not success:
                error_msg = f"Missing required fields: {', '.join(missing_fields)}"
                raise ValidationError(error_msg, missing_fields)
            
            logger.info("All required fields detected")
            return True, "Validation successful", text
        
        except ValidationError as e:
            logger.error(f"Validation failed: {e}")
            return False, str(e), None
        except Exception as e:
            logger.error(f"Unexpected validation error: {e}")
            return False, f"Validation error: {str(e)}", None


def validate_pdf(file_path: str, llm=None) -> Tuple[bool, str, Optional[str]]:
    """
    Convenience function to validate a PDF file
    
    Args:
        file_path: Path to the PDF file
        llm: Optional LLM instance for assisted field detection
        
    Returns:
        Tuple of (success, message, extracted_text)
    """
    validator = PDFValidator(llm=llm)
    return validator.validate(file_path)
