"""
Graph Data Extractor Module

Uses LLM to extract structured graph metadata (Goal, Methods, Datasets) 
from paper text for Neo4j ingestion.
"""

import json
import logging
import re
from typing import Dict, List, Any, Optional
from langchain_ollama import OllamaLLM

logger = logging.getLogger(__name__)

class GraphDataExtractor:
    """
    從論文文本中提取圖譜所需的結構化資訊
    """
    
    # 定義提取用的 Prompt
    # 這裡使用 Few-shot 提示來確保 LLM 輸出正確的 JSON 格式
    EXTRACTION_PROMPT = """
    你是一個學術論文分析專家。請從以下論文文本中提取關鍵的結構化資訊。
    
    任務目標：
    1. **research_goal**: 用一句簡潔的話總結論文的主要研究目標或解決的問題 (請用繁體中文)。
    
    2. **methods**: 列出論文使用的核心技術、模型或演算法。
       - ⚠️ **重要規則：必須統一使用「英文」或「標準英文縮寫」**。
       - 若原文是中文，請翻譯成英文術語。
       - 範例轉換：
         * "卷積神經網路" -> "CNN"
         * "生成對抗網路" -> "GAN"
         * "類神經網路" -> "ANN" or "Neural Network"
         * "集成學習" -> "Ensemble Learning"
       - 移除括號和解釋性文字，只保留核心名詞 (如 "YOLOv4" 而非 "基於YOLOv4的物件偵測")。

    3. **datasets**: 列出論文使用或評估的數據集。
       - 優先使用標準英文名稱 (如 "COCO", "ImageNet")。
       - 若為私有數據，請用英文描述其類型 (如 "Endoscopic Images", "Traffic Flow Data")。

    輸入文本：
    {text}

    請嚴格按照以下 JSON 格式回傳 (不要包含 Markdown 或其他文字)：
    {{
        "research_goal": "改進...的準確度 / 解決...的問題",
        "methods": ["CNN", "Transformer", "YOLOv4"],
        "datasets": ["COCO", "Traffic Flow Data"]
    }}
    """

    def __init__(self, llm: OllamaLLM):
        self.llm = llm

    def extract(self, text: str) -> Dict[str, Any]:
        """
        執行提取邏輯
        
        Args:
            text: 論文文本 (建議傳入摘要或前 3000 字)
            
        Returns:
            Dict 包含 research_goal, methods, datasets
        """
        # 截取前 4000 字以避免超過 Context Window，通常摘要和前言包含主要資訊
        input_text = text[:4000]
        
        try:
            prompt = self.EXTRACTION_PROMPT.format(text=input_text)
            response = self.llm.invoke(prompt)
            
            # 解析 JSON
            data = self._parse_json_response(response)
            
            # 簡單驗證與清理
            data['methods'] = [m.strip() for m in data.get('methods', []) if isinstance(m, str)]
            data['datasets'] = [d.strip() for d in data.get('datasets', []) if isinstance(d, str)]
            if not data.get('research_goal'):
                data['research_goal'] = "Unknown Goal"
                
            logger.info(f"Graph extraction successful: found {len(data['methods'])} methods")
            return data
            
        except Exception as e:
            logger.error(f"Graph extraction failed: {e}")
            # 回傳空結構以避免流程崩潰
            return {
                "research_goal": "Extraction Failed",
                "methods": [],
                "datasets": []
            }

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """解析 LLM 回傳的 JSON 字串"""
        try:
            # 嘗試直接解析
            return json.loads(response)
        except json.JSONDecodeError:
            # 清理 Markdown 標記 (```json ... ```)
            cleaned = re.sub(r'```json\s*|\s*```', '', response).strip()
            # 嘗試尋找 JSON區塊
            match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            else:
                logger.warning(f"Could not parse JSON from response: {response[:100]}...")
                return {}