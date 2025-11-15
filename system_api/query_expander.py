"""
Query Expander Module

擴展短查詢為更詳細的描述，提高檢索效果
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class QueryExpander:
    """
    查詢擴展器
    
    功能：
    1. 檢測短查詢（詞數少於閾值）
    2. 使用 LLM 擴展查詢為更詳細的學術描述
    3. 保留原始查詢意圖，增加相關技術細節
    
    使用場景：
    - 用戶輸入: "人流預測"
    - 擴展後: "使用深度學習方法進行城市公共空間的人流量預測與分析，包括時空特徵建模和預測模型研究"
    """
    
    def __init__(self, llm, min_word_count: int = 5):
        """
        初始化查詢擴展器
        
        Args:
            llm: LLM 實例 (OllamaLLM)
            min_word_count: 最小詞數閾值，少於此數才擴展
        """
        self.llm = llm
        self.min_word_count = min_word_count
        logger.info(f"QueryExpander 初始化: 詞數閾值 = {min_word_count}")
    
    def should_expand(self, query: str) -> bool:
        """
        判斷是否需要擴展查詢
        
        Args:
            query: 原始查詢
            
        Returns:
            True 如果需要擴展，False 否則
        """
        # 使用 jieba 分詞來計算中文詞數
        try:
            import jieba
            words = list(jieba.cut(query.strip()))
            # 過濾掉單字元和空白
            meaningful_words = [w for w in words if len(w.strip()) > 1]
            word_count = len(meaningful_words)
        except:
            # 如果 jieba 不可用，使用簡單的空格分割
            word_count = len(query.split())
        
        should_expand = word_count < self.min_word_count
        
        logger.debug(
            f"查詢 '{query}' 詞數: {word_count}, "
            f"閾值: {self.min_word_count}, "
            f"需要擴展: {should_expand}"
        )
        
        return should_expand
    
    def expand(self, query: str) -> str:
        """
        擴展查詢
        
        Args:
            query: 原始查詢
            
        Returns:
            擴展後的查詢（如果不需要擴展則返回原查詢）
        """
        # 檢查是否需要擴展
        if not self.should_expand(query):
            logger.info(f"查詢長度足夠，不需擴展: '{query}'")
            return query
        
        logger.info(f"檢測到短查詢，準備擴展: '{query}'")
        
        try:
            expanded_query = self._expand_with_llm(query)
            
            # 驗證擴展結果
            if not expanded_query or len(expanded_query.strip()) < len(query):
                logger.warning("擴展結果無效，使用原始查詢")
                return query
            
            logger.info(f"查詢擴展成功:")
            logger.info(f"  原始: {query}")
            logger.info(f"  擴展: {expanded_query}")
            
            return expanded_query
            
        except Exception as e:
            logger.error(f"查詢擴展失敗: {e}", exc_info=True)
            logger.info("回退到原始查詢")
            return query
    
    def _expand_with_llm(self, query: str) -> str:
        """
        使用 LLM 擴展查詢
        
        Args:
            query: 原始查詢
            
        Returns:
            擴展後的查詢
        """
        prompt = self._build_expansion_prompt(query)
        
        logger.debug("調用 LLM 進行查詢擴展...")
        response = self.llm.invoke(prompt)
        
        # 提取擴展結果
        expanded = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        
        # 清理結果（移除引號、多餘空白等）
        expanded = expanded.strip('"\'「」『』')
        expanded = ' '.join(expanded.split())
        
        return expanded
    
    def _build_expansion_prompt(self, query: str) -> str:
        """
        構建 LLM 擴展 prompt
        
        Args:
            query: 原始查詢
            
        Returns:
            Prompt 字串
        """
        prompt = f"""你是一個專業的學術搜尋助手。用戶輸入了一個簡短的查詢詞，你需要將其擴展為更完整、更有利於學術論文檢索的描述。

**重要規則**:
1. 保留原始查詢的核心概念，不要改變主要意圖
2. 添加相關的技術術語、研究方法、應用場景
3. 使用學術論文常見的表達方式
4. 擴展後長度控制在 20-40 個中文字
5. 如果原查詢包含專有名詞（如 CNN、LSTM），務必保留
6. 不要添加與原查詢無關的概念

**擴展範例**:

原始查詢: "人流預測"
擴展結果: 使用深度學習方法進行城市公共空間的人流量預測與分析，包括時空特徵建模和預測模型研究

原始查詢: "CNN"
擴展結果: 卷積神經網路 CNN 的架構設計、訓練方法及其在圖像識別、電腦視覺領域的應用研究

原始查詢: "時間序列"
擴展結果: 時間序列數據分析與預測方法，包括統計模型、機器學習模型及其在各領域的應用

原始查詢: "深度學習"
擴展結果: 深度學習模型的理論基礎、網路架構設計、訓練優化方法及其在各類任務中的應用研究

**現在請擴展以下查詢**:

原始查詢: {query}

**要求**: 只輸出擴展後的查詢文本，不要加任何解釋、標點符號或其他內容。"""

        return prompt
    
    def expand_with_fallback(self, query: str, fallback_patterns: dict = None) -> str:
        """
        擴展查詢，並支援基於規則的 fallback
        
        Args:
            query: 原始查詢
            fallback_patterns: 規則映射字典 {關鍵詞: 擴展模板}
            
        Returns:
            擴展後的查詢
        """
        if not self.should_expand(query):
            return query
        
        # 先嘗試 LLM 擴展
        try:
            expanded = self._expand_with_llm(query)
            if expanded and len(expanded) > len(query):
                return expanded
        except Exception as e:
            logger.warning(f"LLM 擴展失敗: {e}")
        
        # Fallback: 使用規則擴展
        if fallback_patterns:
            for keyword, template in fallback_patterns.items():
                if keyword in query:
                    expanded = template.format(query=query)
                    logger.info(f"使用規則擴展: '{query}' → '{expanded}'")
                    return expanded
        
        # 最後的 fallback: 添加通用後綴
        generic_suffix = "的相關研究方法、應用場景和技術實現"
        expanded = f"{query}{generic_suffix}"
        logger.info(f"使用通用擴展: '{query}' → '{expanded}'")
        
        return expanded


# 預定義的擴展規則（作為 LLM 失敗時的 fallback）
DEFAULT_EXPANSION_PATTERNS = {
    "人流": "{query}分析與預測方法，包括數據採集、特徵提取、模型建立和應用研究",
    "預測": "{query}模型的建立方法、評估指標及其在實際場景中的應用",
    "深度學習": "{query}模型的架構設計、訓練優化和各領域應用研究",
    "CNN": "卷積神經網路 {query} 的結構原理、訓練方法和應用案例",
    "LSTM": "長短期記憶網路 {query} 的原理、改進方法和序列建模應用",
    "RNN": "循環神經網路 {query} 的架構設計和時序數據處理應用",
    "時間序列": "{query}數據的分析、預測方法和模型評估研究",
    "圖像": "{query}處理與分析技術，包括特徵提取、模型訓練和應用",
}
