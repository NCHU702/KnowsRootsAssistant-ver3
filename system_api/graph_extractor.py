"""
Graph Data Extractor Module (Final Optimized Version)

Uses LLM to extract structured graph metadata with strict standardization.
Optimized for:
1. Domain Classification (Reducing Unknowns)
2. Metric Extraction (Catching hidden values in tables)
3. Method Cleaning (Removing redundant suffixes)
"""

import json
import logging
import re
from typing import Dict, List, Any
from langchain_ollama import OllamaLLM

logger = logging.getLogger(__name__)

class GraphDataExtractor:
    """
    從論文文本中提取圖譜所需的結構化資訊
    針對 Domain 和 Metric 進行了嚴格標準化，並大幅增強了指標的關鍵字搜尋能力
    """
    
    # 定義標準領域清單 (Taxonomy)
    ALLOWED_DOMAINS = [
        "Smart Transportation",      # 智慧交通 (捷運, 公車, 車流, YouBike)
        "Healthcare",                # 智慧醫療 (疾病辨識, 內視鏡, 醫院人流)
        "Smart Manufacturing",       # 智慧製造 (刀具磨耗, 瑕疵檢測, CNC, 工廠)
        "Agriculture",               # 智慧農業 (芒果, 水資源, 灌溉)
        "Environmental Monitoring",  # 環境監測 (PM2.5, 空氣, 溫度)
        "Finance",                   # 金融科技 (股價, 交易)
        "Computer Vision",           # 電腦視覺 (通用影像, 標註, 追蹤)
        "Sports Analytics",          # 運動分析 (NBA, 戰術)
        "Tourism",                   # 觀光旅遊 (景點, 旅程推薦)
        "Disaster Management",       # 災害管理 (人流疏散, 防災)
        "Smart Building"             # [新增] 智慧建築 (空調, 室內溫度, 節能)
    ]

    # Prompt 優化：加入具體的映射規則與清理指令
    EXTRACTION_PROMPT = """
    你是一個學術論文分析專家。我會提供你一篇論文的「摘要」以及「包含關鍵字的片段」。
    請根據這些文字提取結構化資訊。
    
    任務目標：
    1. **research_goal**: 用一句簡潔的話總結論文的主要研究目標或解決的問題 (請用繁體中文)。
    
    2. **methods**: 列出論文使用的核心技術、模型或演算法。
       - ⚠️ **規則：只保留核心技術名詞，並盡量轉為標準英文縮寫**。
       - **拆解複合詞**：如 "Autoencoder based on RNN" -> ["Autoencoder", "RNN"]。
       - **移除贅字**：移除 "based on", "framework", "model", "system", "approach", "algorithm", "proposed"。
       - 範例："mRBF-LSTM Framework" -> "mRBF-LSTM"。

    3. **datasets**: 列出論文使用或評估的數據集。
       - 公開數據集用英文名稱 (如 "COCO", "YouBike Data", "PeMSD7")。
       - 私有數據用英文描述類型 (如 "Traffic Flow Data", "Endoscopic Images")。

    4. **domain**: 請將論文歸類到以下**唯一**一個最合適的領域：
       {allowed_domains}
       - ⚠️ **映射規則 (請嚴格遵守)**：
         * "刀具磨耗", "CNC", "機台" -> **Smart Manufacturing**
         * "旅程推薦", "景點", "遊客" -> **Tourism**
         * "PM2.5", "空氣品質", "氣象" -> **Environmental Monitoring**
         * "空調", "室內溫度", "冷氣房" -> **Smart Building**
         * "捷運", "YouBike", "車流", "路網" -> **Smart Transportation**
         * "股票", "股價" -> **Finance**
         * "芒果", "灌溉", "水資源" -> **Agriculture**

    5. **metrics**: 列出論文使用的量化評估指標。
       - ⚠️ **規則：必須轉換為標準英文大寫縮寫**。
       - 預測類 (迴歸)：RMSE, MAE, MAPE, MSE, R2 (R-Squared)。
       - 分類/偵測類：Accuracy, F1, Precision, Recall, IoU, mAP。
       - 不要列出非量化指標 (如 "Efficiency", "Cost")。

    輸入文本：
    {text}

    請嚴格按照以下 JSON 格式回傳 (不要包含 Markdown 或其他文字)：
    {{
        "research_goal": "改進...的準確度 / 解決...的問題",
        "methods": ["CNN", "LSTM"],
        "datasets": ["Data A"],
        "domain": "Smart Transportation",
        "metrics": ["RMSE", "MAE"]
    }}
    """

    def __init__(self, llm: OllamaLLM):
        self.llm = llm

    def extract(self, text: str) -> Dict[str, Any]:
        """
        執行提取邏輯並進行標準化
        """
        # 使用增強版截斷邏輯
        input_text = self._smart_truncate(text, max_chars=6000)
        
        try:
            # 將允許的領域清單格式化並注入 Prompt
            domains_str = "\\n       - ".join(self.ALLOWED_DOMAINS)
            prompt = self.EXTRACTION_PROMPT.format(
                text=input_text,
                allowed_domains="- " + domains_str
            )
            
            response = self.llm.invoke(prompt)
            data = self._parse_json_response(response)
            
            # --- 資料清理與標準化 ---
            
            data['methods'] = [self._clean_term(m) for m in data.get('methods', []) if isinstance(m, str)]
            data['datasets'] = [self._clean_term(d) for d in data.get('datasets', []) if isinstance(d, str)]
            data['metrics'] = [self._normalize_metric(m) for m in data.get('metrics', []) if isinstance(m, str)]
            
            # Domain 強制驗證與模糊比對
            raw_domain = str(data.get('domain', '')).strip()
            domain_map = {d.lower(): d for d in self.ALLOWED_DOMAINS}
            
            # 1. 直接比對
            matched_domain = domain_map.get(raw_domain.lower())
            
            if matched_domain:
                data['domain'] = matched_domain
            else:
                # 2. 模糊搜尋 (例如 "Transportation System" -> "Smart Transportation")
                found = False
                for key, val in domain_map.items():
                    if key in raw_domain.lower() or raw_domain.lower() in key:
                        data['domain'] = val
                        found = True
                        break
                # 3. 若都不符合，設為 Unknown
                if not found:
                    data['domain'] = "Unknown"

            # 移除空值
            data['methods'] = [m for m in data['methods'] if m]
            data['datasets'] = [d for d in data['datasets'] if d]
            data['metrics'] = [m for m in data['metrics'] if m]
            
            if not data.get('research_goal'):
                data['research_goal'] = "Unknown Goal"
                
            logger.info(f"Graph extraction: {len(data['methods'])} methods, Domain: {data['domain']}, Metrics: {len(data['metrics'])}")
            return data
            
        except Exception as e:
            logger.error(f"Graph extraction failed: {e}")
            return {
                "research_goal": "Extraction Failed",
                "methods": [],
                "datasets": [],
                "domain": "Unknown",
                "metrics": []
            }

    def _clean_term(self, term: str) -> str:
        """基本清理：移除前後空白和標點"""
        return term.strip().strip(".,;").strip()

    def _normalize_metric(self, metric: str) -> str:
        """指標標準化邏輯"""
        m = metric.strip().upper()
        mapping = {
            "ACC": "ACCURACY",
            "AVE. PRECISION": "mAP",
            "MEAN AVERAGE PRECISION": "mAP",
            "ROOT MEAN SQUARED ERROR": "RMSE",
            "MEAN ABSOLUTE ERROR": "MAE",
            "MEAN ABSOLUTE PERCENTAGE ERROR": "MAPE",
            "PRECISION": "PRECISION",
            "RECALL": "RECALL",
            "IOU": "IoU",
            "F1": "F1-SCORE",
            "F1-SCORE": "F1-SCORE",
            "AUC": "AUC",
            "R2": "R-SQUARED",
            "R-SQUARED": "R-SQUARED",
            "COEFFICIENT OF DETERMINATION": "R-SQUARED"
        }
        # 移除中文贅字以便比對
        m = m.replace("準確率", "").replace("誤差", "").replace("分數", "").strip()
        return mapping.get(m, m)

    def _smart_truncate(self, text: str, max_chars: int = 6000) -> str:
        """
        智慧截斷文本：大幅增強對評估指標 (Metrics) 的搜尋能力
        """
        if len(text) <= max_chars:
            return text

        # 1. 保留前 1500 字 (摘要+前言)
        head_chars = 1500
        head_text = text[:head_chars]
        remaining_text = text[head_chars:]
        
        # 2. 定義關鍵詞 (中英對照)
        keywords = [
            "資料集", "數據", "資料來源", "dataset", "data source",
            "研究方法", "系統架構", "模型設計", "演算法", "methodology", "architecture",
            "實驗設置", "實驗環境", "實驗結果", "評估指標", "績效評估", "experiment", "evaluation", "metric"
        ]
        
        # 3. 優先關鍵詞 (High Priority) - 只要包含這些，權重加倍
        # 加入了 "Table", "表", "Performance" 以捕獲表格標題
        priority_keywords = [
            # 中文指標
            "準確率", "精確率", "召回率", "均方根誤差", "平均絕對誤差", "F1分數", "交集聯集比",
            # 英文指標 (常見縮寫)
            "RMSE", "MSE", "MAE", "MAPE", "F1", "AUC", "mAP", "IoU", "R2", "R-squared", "Accuracy", "Precision", "Recall",
            # 表格與比較 (Metrics 常出現的地方)
            "Table", "表", "Performance", "Comparison", "比較",
            # 資料集相關
            "資料集", "數據", "dataset"
        ]
        
        extracted_chunks = []
        current_length = len(head_text)
        chunk_size = 800 
        step = 400
        
        for i in range(0, len(remaining_text), step):
            if current_length >= max_chars:
                break
                
            chunk = remaining_text[i : i + chunk_size]
            chunk_lower = chunk.lower()
            
            score = 0
            for kw in keywords:
                if kw.lower() in chunk_lower:
                    score += 1
            
            for kw in priority_keywords:
                # 使用簡單字串匹配確保穩健性
                if kw.lower() in chunk_lower:
                    score += 3 # 權重設為 3，確保重要段落被選中
            
            if score > 0:
                formatted_chunk = f"\n\n--- 關鍵片段 ---\n{chunk}"
                extracted_chunks.append(formatted_chunk)
                current_length += len(chunk)
        
        final_text = head_text + "\n\n... (略過中間無關內容) ...\n" + "".join(extracted_chunks)
        return final_text[:max_chars + 500] 

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """解析 LLM 回傳的 JSON 字串"""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            cleaned = re.sub(r'```json\s*|\s*```', '', response).strip()
            match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except:
                    pass
            logger.warning(f"Could not parse JSON from response: {response[:100]}...")
            return {}