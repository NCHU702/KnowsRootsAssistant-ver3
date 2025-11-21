"""
Query Router - 智能查詢路由器

根據用戶查詢的類型，自動判斷應該使用哪種檢索策略：
1. Graph-first: 跨論文比較、統計、結構化查詢（方法、數據集、領域等）
2. RAG (AssistantCall): 單一論文的詳細內容、總結、解釋

設計理念：
- Graph 適合回答「哪些論文...」、「比較...」、「統計...」等宏觀問題
- RAG 適合回答「這篇論文的方法是...」、「解釋論文內容」等微觀問題
"""

import logging
from typing import Dict, Literal
from langchain_ollama import OllamaLLM
import re

logger = logging.getLogger(__name__)

RouteType = Literal["graph", "rag", "web"]


class QueryRouter:
    """
    智能查詢路由器
    
    使用 LLM 判斷查詢類型，決定使用 Graph-first 還是 RAG 檢索
    """
    
    def __init__(self, llm: OllamaLLM = None):
        """
        初始化路由器
        
        Args:
            llm: Ollama LLM 實例（用於智能判斷）
        """
        self.llm = llm
        
        # 關鍵詞模式（作為 fallback）
        self.graph_keywords = [
            # 中文
            r'哪些論文', r'有哪些', r'列出.*論文', r'比較.*論文', 
            r'多少.*論文', r'幾篇.*論文', r'統計',
            r'使用.*方法', r'採用.*方法', r'應用.*方法',
            r'使用.*數據集', r'採用.*數據集',
            r'.*領域.*研究', r'.*領域.*論文',
            r'.*指標', r'.*評估標準',
            r'趨勢', r'發展', r'演變',
            # 英文
            r'which papers', r'list.*papers', r'how many', r'compare',
            r'papers.*use', r'papers.*apply', r'papers.*adopt',
            r'papers.*in.*domain', r'papers.*about',
            r'statistics', r'trends', r'evolution',
        ]
        
        self.rag_keywords = [
            # 中文
            r'總結', r'摘要', r'概述', r'說明',
            r'這篇論文', r'該論文', r'此論文',
            r'詳細.*方法', r'詳細.*內容', r'具體.*步驟',
            r'解釋', r'闡述', r'介紹',
            r'研究目標', r'研究問題', r'研究方法',
            r'實驗.*設計', r'實驗.*結果', r'實驗.*分析',
            # 英文
            r'summarize', r'summary', r'abstract', r'overview',
            r'this paper', r'the paper', r'explain',
            r'describe', r'introduce', r'detail',
            r'methodology', r'experiment', r'result',
        ]
        
        self.web_keywords = [
            r'search.*internet', r'search.*online', r'web.*search',
            r'find.*internet', r'google', r'搜尋.*網路', r'網路.*搜尋'
        ]
        
        logger.info("QueryRouter initialized")
    
    def route(self, query: str) -> Dict[str, any]:
        """
        判斷查詢應該路由到哪個系統
        
        Args:
            query: 用戶查詢
            
        Returns:
            {
                'route': 'graph' | 'rag' | 'web',
                'confidence': float (0-1),
                'reasoning': str
            }
        """
        logger.info(f"Routing query: {query[:100]}...")
        
        # 1. 首先檢查 Web 搜尋（明確意圖）
        if self._match_keywords(query, self.web_keywords):
            return {
                'route': 'web',
                'confidence': 1.0,
                'reasoning': '用戶明確要求網路搜尋'
            }
        
        # 2. 使用 LLM 進行智能判斷
        if self.llm:
            try:
                result = self._llm_route(query)
                if result['confidence'] >= 0.7:  # 高信心度直接採用
                    return result
                else:
                    logger.info(f"LLM confidence low ({result['confidence']}), using keyword fallback")
            except Exception as e:
                logger.warning(f"LLM routing failed: {e}, falling back to keyword matching")
        
        # 3. Fallback 到關鍵詞匹配
        return self._keyword_route(query)
    
    def _llm_route(self, query: str) -> Dict[str, any]:
        """
        使用 LLM 進行智能路由判斷
        """
        prompt = f"""你是一個查詢路由專家。請判斷下面的查詢應該使用哪種檢索策略。

查詢類型定義：

1. **Graph** (跨論文結構化查詢):
   - 適用於：比較多篇論文、統計分析、查找使用特定方法/數據集的論文
   - 關鍵特徵：「哪些論文」、「有多少」、「列出」、「比較」、「使用XX方法」
   - 例子：「哪些論文使用 LSTM?」、「比較交通領域的研究」、「使用了哪些數據集?」

2. **RAG** (單一論文詳細內容):
   - 適用於：總結特定論文、解釋論文內容、查詢論文細節
   - 關鍵特徵：「總結」、「解釋」、「這篇論文」、「詳細說明」、「方法論」
   - 例子：「總結這篇論文」、「詳細說明澳門公車軌跡辨識的方法論」

用戶查詢: "{query}"

請以 JSON 格式回答（不要包含其他文字）:
{{
    "route": "graph" 或 "rag",
    "confidence": 0.0-1.0 之間的數字,
    "reasoning": "簡短的判斷理由（一句話）"
}}"""

        try:
            response = self.llm.invoke(prompt)
            
            # 提取 JSON（處理可能的格式問題）
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.startswith('```'):
                response = response[3:]
            if response.endswith('```'):
                response = response[:-3]
            response = response.strip()
            
            import json
            result = json.loads(response)
            
            # 驗證結果
            if result['route'] not in ['graph', 'rag']:
                raise ValueError(f"Invalid route: {result['route']}")
            if not (0 <= result['confidence'] <= 1):
                raise ValueError(f"Invalid confidence: {result['confidence']}")
            
            logger.info(f"LLM routing: {result['route']} (confidence: {result['confidence']:.2f})")
            logger.info(f"Reasoning: {result['reasoning']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to parse LLM routing response: {e}")
            raise
    
    def _keyword_route(self, query: str) -> Dict[str, any]:
        """
        基於關鍵詞的 fallback 路由
        """
        graph_score = sum(1 for pattern in self.graph_keywords if re.search(pattern, query, re.IGNORECASE))
        rag_score = sum(1 for pattern in self.rag_keywords if re.search(pattern, query, re.IGNORECASE))
        
        if graph_score > rag_score:
            confidence = min(0.8, 0.5 + graph_score * 0.1)
            return {
                'route': 'graph',
                'confidence': confidence,
                'reasoning': f'關鍵詞匹配 (Graph 指標: {graph_score}, RAG 指標: {rag_score})'
            }
        elif rag_score > graph_score:
            confidence = min(0.8, 0.5 + rag_score * 0.1)
            return {
                'route': 'rag',
                'confidence': confidence,
                'reasoning': f'關鍵詞匹配 (Graph 指標: {graph_score}, RAG 指標: {rag_score})'
            }
        else:
            # 預設使用 RAG（更安全）
            return {
                'route': 'rag',
                'confidence': 0.5,
                'reasoning': '無法明確判斷，預設使用 RAG'
            }
    
    def _match_keywords(self, query: str, keywords: list) -> bool:
        """
        檢查查詢是否匹配任何關鍵詞模式
        """
        return any(re.search(pattern, query, re.IGNORECASE) for pattern in keywords)


# 測試代碼
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 初始化 LLM（可選）
    try:
        llm = OllamaLLM(
            model="jcai/llama-3-taiwan-8b-instruct:q4_k_m",
            temperature=0,
            base_url="http://localhost:11434"
        )
        print("✓ LLM initialized")
    except Exception as e:
        print(f"⚠️  LLM not available: {e}")
        llm = None
    
    # 創建路由器
    router = QueryRouter(llm=llm)
    
    # 測試案例
    test_queries = [
        # Graph 查詢（跨論文）
        "哪些論文使用 CNN?",
        "列出所有使用 LSTM 的研究",
        "比較交通領域和醫療領域的研究方法",
        "使用了哪些數據集?",
        "有多少論文研究智慧交通?",
        "Which papers use deep learning?",
        
        # RAG 查詢（單一論文詳細內容）
        "總結澳門公車軌跡辨識這篇論文",
        "詳細說明該論文的方法論",
        "解釋這篇論文的研究目標",
        "這篇論文的實驗結果如何?",
        "介紹論文中使用的評估指標",
        "Summarize the methodology of this paper",
        
        # 邊界案例
        "論文中使用了哪些數據集?",  # 可能是 Graph 或 RAG
        "CNN 方法的詳細實現步驟",   # RAG
        "比較不同論文使用的 CNN 架構",  # Graph
    ]
    
    print("\n" + "="*80)
    print("🧪 Query Router Test Cases")
    print("="*80 + "\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Test {i}: {query}")
        print("-" * 80)
        
        result = router.route(query)
        
        # 顯示結果
        route_emoji = "🔗" if result['route'] == 'graph' else "📄"
        confidence_bar = "█" * int(result['confidence'] * 10) + "░" * (10 - int(result['confidence'] * 10))
        
        print(f"{route_emoji} Route: {result['route'].upper()}")
        print(f"📊 Confidence: {confidence_bar} {result['confidence']:.2f}")
        print(f"💭 Reasoning: {result['reasoning']}")
    
    print("\n" + "="*80)
    print("✅ Test complete!")
    print("="*80)
