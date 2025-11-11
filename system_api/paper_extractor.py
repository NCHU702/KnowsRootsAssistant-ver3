"""
Paper Extraction Module

Extracts structured metadata from research papers using LLM (Ollama gemma3:12b).
Supports Chinese, English, and mixed-language content.
"""

import json
import re
import logging
from typing import Dict, Optional, List
from langchain_ollama import OllamaLLM

logger = logging.getLogger(__name__)


class ExtractionError(Exception):
    """Custom exception for extraction errors"""
    pass


class PaperExtractor:
    """Extracts structured paper metadata using LLM"""
    
    def __init__(self, model: str = "gemma3:12b", base_url: str = "http://localhost:11434"):
        """
        Initialize paper extractor
        
        Args:
            model: Ollama model name
            base_url: Ollama server URL
        """
        self.model = model
        self.base_url = base_url
        self.llm = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize Ollama LLM connection"""
        try:
            self.llm = OllamaLLM(
                model=self.model,
                temperature=0.3,
                base_url=self.base_url
            )
            # Test connection
            test_response = self.llm.invoke("Hello")
            logger.info(f"Paper extractor LLM initialized successfully with model {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise ExtractionError(f"Failed to connect to Ollama: {e}")
    
    def _create_extraction_prompt(self, pdf_text: str) -> str:
        """
        Create bilingual extraction prompt
        
        Args:
            pdf_text: Extracted text from PDF
            
        Returns:
            Formatted prompt string
        """
        # Truncate text if too long (keep first 12000 characters for context)
        truncated_text = pdf_text[:12000] if len(pdf_text) > 12000 else pdf_text
        
        prompt = f"""你是一個學術論文分析助手。請從以下論文內容中提取結構化資訊。論文可能是中文、英文或中英混合。
You are an academic paper analysis assistant. Please extract structured information from the following paper content. The paper may be in Chinese, English, or mixed.

論文內容 / Paper Content:
{truncated_text}

請提取以下資訊並以JSON格式返回 / Please extract the following information and return in JSON format:
1. 論文標題 / Paper Title: 論文的完整標題（保持原語言）/ Complete title (keep original language)
2. 年份 / Year: 發表或完成年份（陣列格式，如["2024"]）/ Publication or completion year (array format, e.g., ["2024"])
3. 作者 / Authors: 作者姓名（陣列格式）/ Author names (array format)
4. 研究目的 / Research Purpose: 研究的主要目標和目的（陣列格式，每個目的一項）/ Main research objectives (array format, one item per objective)
5. 資料集 / Datasets: 使用的資料集名稱（陣列格式）/ Dataset names used (array format)
6. 資料前處理 / Data Preprocessing: 資料前處理方法（陣列格式）/ Data preprocessing methods (array format)
7. 建模 / Modeling: 使用的模型或方法（陣列格式，如["lstm", "cnn"]）/ Models or methods used (array format, e.g., ["lstm", "cnn"])
8. 評估指標 / Evaluation Metrics: 使用的評估指標（陣列格式，如["accuracy", "f1-score"]）/ Evaluation metrics used (array format)
9. 其他 / Other: 其他重要資訊（陣列格式，可選）/ Other important information (array format, optional)

重要提示 / Important notes:
- 所有列表欄位必須是陣列格式，即使只有一個項目 / All list fields must be arrays, even with one item
- 標題保持原語言，不要翻譯 / Keep title in original language, do not translate
- 如果無法確定某個欄位，使用空陣列 [] / Use empty array [] if field cannot be determined
- 只返回JSON，不要其他文字或解釋 / Return only JSON, no other text or explanations

請返回以下格式的JSON / Please return JSON in the following format:
{{
  "論文標題": "完整標題",
  "年份": ["2024"],
  "作者": ["作者1", "作者2"],
  "研究目的": ["目的1", "目的2"],
  "資料集": ["資料集1", "資料集2"],
  "資料前處理": ["方法1", "方法2"],
  "建模": ["模型1", "模型2"],
  "評估指標": ["指標1", "指標2"],
  "其他": []
}}

只返回JSON，不要包含```json```標記 / Return only JSON, do not include ```json``` markers:"""
        
        return prompt
    
    def _parse_llm_response(self, response: str) -> Dict:
        """
        Parse LLM response and extract JSON
        
        Args:
            response: Raw LLM response
            
        Returns:
            Parsed JSON dict
            
        Raises:
            ExtractionError: If parsing fails
        """
        try:
            # Remove markdown code blocks if present
            json_text = response.strip()
            
            # Try to find JSON in markdown code blocks
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', json_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
            
            # Try to find JSON object directly
            if not json_text.startswith('{'):
                json_match = re.search(r'\{.*\}', json_text, re.DOTALL)
                if json_match:
                    json_text = json_match.group(0)
            
            # Parse JSON
            data = json.loads(json_text)
            
            # Validate required fields
            required_fields = [
                "論文標題", "年份", "作者", "研究目的", 
                "資料集", "資料前處理", "建模", "評估指標"
            ]
            
            for field in required_fields:
                if field not in data:
                    logger.warning(f"Missing field in extraction: {field}")
                    # Set default empty value
                    if field == "論文標題":
                        data[field] = "未提取"
                    else:
                        data[field] = []
            
            # Ensure "其他" field exists
            if "其他" not in data:
                data["其他"] = []
            
            # Ensure 論文標題 is a string, not a list
            if isinstance(data.get("論文標題"), list):
                data["論文標題"] = data["論文標題"][0] if data["論文標題"] else "未提取"
            
            # Ensure all array fields are actually arrays
            array_fields = ["年份", "作者", "研究目的", "資料集", "資料前處理", "建模", "評估指標", "其他"]
            for field in array_fields:
                if not isinstance(data[field], list):
                    data[field] = [data[field]] if data[field] else []
            
            return data
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}")
            logger.error(f"Response was: {response[:500]}")
            raise ExtractionError(f"Failed to parse JSON from LLM response: {e}")
        except Exception as e:
            logger.error(f"Unexpected error parsing LLM response: {e}")
            raise ExtractionError(f"Error parsing extraction results: {e}")
    
    def extract(self, pdf_text: str, max_retries: int = 2) -> Dict:
        """
        Extract paper metadata from PDF text
        
        Args:
            pdf_text: Extracted text from PDF
            max_retries: Maximum number of retry attempts
            
        Returns:
            Dictionary with extracted metadata
            
        Raises:
            ExtractionError: If extraction fails
        """
        if not self.llm:
            raise ExtractionError("LLM not initialized")
        
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(f"Retry attempt {attempt} for paper extraction")
                
                # Create prompt
                prompt = self._create_extraction_prompt(pdf_text)
                
                # Invoke LLM
                logger.info("Invoking LLM for paper extraction...")
                response = self.llm.invoke(prompt)
                logger.info("LLM extraction completed")
                
                # Parse response
                extracted_data = self._parse_llm_response(response)
                
                # Log extracted title for verification
                logger.info(f"Successfully extracted paper: {extracted_data.get('論文標題', 'Unknown')}")
                
                return extracted_data
            
            except ExtractionError as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(f"Extraction attempt {attempt + 1} failed: {e}")
                    continue
                else:
                    logger.error(f"All extraction attempts failed")
                    raise
            except Exception as e:
                last_error = ExtractionError(f"Unexpected extraction error: {e}")
                if attempt < max_retries:
                    logger.warning(f"Extraction attempt {attempt + 1} failed: {e}")
                    continue
                else:
                    raise last_error
        
        # Should not reach here, but just in case
        if last_error:
            raise last_error
        raise ExtractionError("Extraction failed for unknown reason")
    
    def validate_extraction(self, extracted_data: Dict) -> bool:
        """
        Validate that extraction contains minimum required data
        
        Args:
            extracted_data: Extracted metadata dictionary
            
        Returns:
            True if validation passes
            
        Raises:
            ExtractionError: If validation fails
        """
        # Check title
        if not extracted_data.get("論文標題") or extracted_data["論文標題"] == "未提取":
            raise ExtractionError("Failed to extract paper title")
        
        # Check if at least some fields have data
        non_empty_fields = sum(
            1 for field in ["年份", "作者", "研究目的", "資料集", "建模", "評估指標"]
            if extracted_data.get(field) and len(extracted_data[field]) > 0
        )
        
        if non_empty_fields < 2:
            raise ExtractionError(
                f"Insufficient data extracted. Only {non_empty_fields} out of 6 key fields have data."
            )
        
        logger.info(f"Extraction validation passed: {non_empty_fields}/6 key fields extracted")
        return True


def extract_paper_metadata(pdf_text: str, model: str = "gemma3:12b") -> Dict:
    """
    Convenience function to extract paper metadata
    
    Args:
        pdf_text: Extracted text from PDF
        model: Ollama model name
        
    Returns:
        Dictionary with extracted metadata
    """
    extractor = PaperExtractor(model=model)
    extracted_data = extractor.extract(pdf_text)
    extractor.validate_extraction(extracted_data)
    return extracted_data
