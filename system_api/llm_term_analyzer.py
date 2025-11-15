"""
LLM-based Smart Term Analyzer - 基於 LLM 的智能詞彙分析器

使用 LLM 自動分析查詢，判斷：
1. 哪些是核心/重點詞彙（需要精準匹配）
2. 哪些是廣泛/泛用詞彙（可以降低權重）
3. 領域類別和查詢意圖
"""

import logging
import json
from typing import List, Dict, Tuple, Optional
from langchain_ollama import OllamaLLM

logger = logging.getLogger(__name__)


class LLMTermAnalyzer:
    """
    基於 LLM 的詞彙分析器
    
    使用 LLM 智能判斷查詢中的詞彙重要性
    """
    
    def __init__(self, llm: OllamaLLM, cache_enabled: bool = True):
        """
        初始化 LLM 詞彙分析器
        
        Args:
            llm: LLM 實例
            cache_enabled: 是否啟用快取（避免重複分析相同查詢）
        """
        self.llm = llm
        self.cache_enabled = cache_enabled
        self.cache: Dict[str, Dict] = {}  # 查詢快取
        
        logger.info("LLMTermAnalyzer 初始化完成")
    
    def _build_analysis_prompt(self, query: str, query_tokens: List[str]) -> str:
        """
        構建詞彙分析提示詞
        
        Args:
            query: 原始查詢
            query_tokens: 分詞列表
            
        Returns:
            提示詞字串
        """
        tokens_str = ', '.join(query_tokens)
        
        prompt = f"""你是一個學術檢索系統的智能分析助手。請分析以下查詢中的詞彙重要性。

查詢: {query}
分詞結果: {tokens_str}

請判斷每個詞彙的類型：

1. **核心詞彙（CORE）**: 查詢的關鍵概念，必須精準匹配
   - 例如: 具體領域（醫療、人流、交通）、專有名詞（CNN、LSTM）、特定疾病（鼻咽癌）
   - 這些詞彙決定了查詢的核心主題

2. **廣泛詞彙（GENERIC）**: 常見的學術用語，區分度低
   - 例如: 深度學習、機器學習、應用、方法、研究、分析、預測、模型
   - 這些詞彙在很多論文中都會出現

3. **輔助詞彙（AUXILIARY）**: 問句詞、連接詞等
   - 例如: 什麼、如何、是、的、在、等

請以 JSON 格式回答，包含：
1. core_terms: 核心詞彙列表
2. generic_terms: 廣泛詞彙列表
3. auxiliary_terms: 輔助詞彙列表
4. domain: 查詢所屬領域（如：醫療、人流、交通、AI模型等）
5. intent: 查詢意圖（如：尋找應用、方法比較、概念解釋等）
6. reasoning: 判斷理由（簡短說明）

範例 1:
查詢: "深度學習在醫療的應用是什麼？"
分詞: 深度, 學習, 醫療, 應用, 什麼
回答:
{{
  "core_terms": ["醫療"],
  "generic_terms": ["深度", "學習", "應用"],
  "auxiliary_terms": ["什麼"],
  "domain": "醫療",
  "intent": "尋找深度學習在醫療領域的應用案例",
  "reasoning": "「醫療」是核心領域詞，決定查詢範圍；「深度學習」和「應用」是廣泛的技術詞彙，很多論文都會包含"
}}

範例 2:
查詢: "人流預測"
分詞: 人流, 預測
回答:
{{
  "core_terms": ["人流"],
  "generic_terms": ["預測"],
  "auxiliary_terms": [],
  "domain": "人流分析",
  "intent": "尋找人流預測相關研究",
  "reasoning": "「人流」是具體的研究對象，是核心詞；「預測」是常見的研究方法詞彙"
}}

範例 3:
查詢: "使用CNN進行圖像分類"
分詞: 使用, CNN, 進行, 圖像, 分類
回答:
{{
  "core_terms": ["CNN", "圖像"],
  "generic_terms": ["分類"],
  "auxiliary_terms": ["使用", "進行"],
  "domain": "計算機視覺",
  "intent": "尋找CNN在圖像分類中的應用",
  "reasoning": "「CNN」是特定模型，「圖像」是具體領域，這兩個是核心；「分類」是常見任務詞彙"
}}

現在請分析以下查詢:
查詢: {query}
分詞: {tokens_str}

請直接返回 JSON，不要包含其他文字說明。
"""
        return prompt
    
    def analyze_query(
        self, 
        query: str, 
        query_tokens: List[str]
    ) -> Dict[str, any]:
        """
        使用 LLM 分析查詢詞彙
        
        Args:
            query: 原始查詢
            query_tokens: 分詞列表
            
        Returns:
            分析結果字典
        """
        # 檢查快取
        if self.cache_enabled and query in self.cache:
            logger.debug(f"使用快取結果: {query}")
            return self.cache[query]
        
        logger.info(f"LLM 分析查詢: {query}")
        
        try:
            # 構建提示詞
            prompt = self._build_analysis_prompt(query, query_tokens)
            
            # 呼叫 LLM
            response = self.llm.invoke(prompt)
            
            logger.debug(f"LLM 原始回應: {response[:200]}...")
            
            # 解析 JSON 回應
            analysis = self._parse_llm_response(response, query_tokens)
            
            # 快取結果
            if self.cache_enabled:
                self.cache[query] = analysis
            
            logger.info(
                f"✓ LLM 分析完成: "
                f"核心詞={len(analysis['core_terms'])}, "
                f"廣泛詞={len(analysis['generic_terms'])}, "
                f"領域={analysis.get('domain', 'Unknown')}"
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"LLM 分析失敗: {e}", exc_info=True)
            # 返回保守的預設分析
            return self._fallback_analysis(query_tokens)
    
    def _parse_llm_response(
        self, 
        response: str, 
        query_tokens: List[str]
    ) -> Dict[str, any]:
        """
        解析 LLM 的 JSON 回應
        
        Args:
            response: LLM 回應文字
            query_tokens: 原始分詞（用於 fallback）
            
        Returns:
            解析後的分析結果
        """
        try:
            # 嘗試找到 JSON 部分
            response = response.strip()
            
            # 移除可能的 markdown 標記
            if response.startswith('```json'):
                response = response[7:]
            if response.startswith('```'):
                response = response[3:]
            if response.endswith('```'):
                response = response[:-3]
            
            response = response.strip()
            
            # 找到第一個 { 和最後一個 }
            start = response.find('{')
            end = response.rfind('}')
            
            if start != -1 and end != -1:
                json_str = response[start:end+1]
                
                # 嘗試修復常見的 JSON 錯誤
                # 1. 處理尾隨逗號
                json_str = json_str.replace(',]', ']').replace(',}', '}')
                
                # 2. 嘗試解析
                try:
                    analysis = json.loads(json_str)
                except json.JSONDecodeError as json_err:
                    # 如果失敗，嘗試逐行清理
                    logger.warning(f"首次 JSON 解析失敗: {json_err}")
                    lines = json_str.split('\n')
                    cleaned_lines = []
                    for line in lines:
                        # 移除註釋
                        if '//' in line:
                            line = line[:line.index('//')]
                        cleaned_lines.append(line)
                    json_str = '\n'.join(cleaned_lines)
                    analysis = json.loads(json_str)
                
                # 驗證必要欄位
                required_fields = ['core_terms', 'generic_terms', 'auxiliary_terms']
                for field in required_fields:
                    if field not in analysis:
                        analysis[field] = []
                
                # 確保都是列表
                for field in required_fields:
                    if not isinstance(analysis[field], list):
                        analysis[field] = []
                
                return analysis
            else:
                raise ValueError("無法找到 JSON 結構")
                
        except Exception as e:
            logger.warning(f"解析 LLM 回應失敗: {e}")
            return self._fallback_analysis(query_tokens)
    
    def _fallback_analysis(self, query_tokens: List[str]) -> Dict[str, any]:
        """
        當 LLM 分析失敗時的後備方案
        
        使用簡單規則進行分類
        
        Args:
            query_tokens: 分詞列表
            
        Returns:
            基本分析結果
        """
        logger.warning("使用 Fallback 分析")
        
        # 預定義的廣泛詞列表
        common_generic_terms = {
            '深度', '學習', '機器', '應用', '方法', '模型', '系統',
            '研究', '分析', '預測', '使用', '基於', '利用', '探討',
            '提出', '設計', '實現', '建構', '開發', '優化', '改進'
        }
        
        # 預定義的輔助詞
        auxiliary_words = {
            '什麼', '如何', '為什麼', '是', '的', '在', '與', '和',
            '或', '等', '及', '以', '於', '中', '嗎', '呢'
        }
        
        core_terms = []
        generic_terms = []
        auxiliary_terms = []
        
        for token in query_tokens:
            if token in auxiliary_words:
                auxiliary_terms.append(token)
            elif token in common_generic_terms:
                generic_terms.append(token)
            else:
                # 預設為核心詞
                core_terms.append(token)
        
        return {
            'core_terms': core_terms,
            'generic_terms': generic_terms,
            'auxiliary_terms': auxiliary_terms,
            'domain': 'Unknown',
            'intent': 'Unknown',
            'reasoning': 'Fallback analysis - LLM 不可用',
            'is_fallback': True
        }
    
    def compute_smart_weights(
        self,
        query: str,
        query_tokens: List[str],
        default_semantic: float = 0.5,
        default_keyword: float = 0.5
    ) -> Tuple[float, float, str]:
        """
        基於 LLM 分析結果計算智能權重
        
        Args:
            query: 原始查詢
            query_tokens: 分詞列表
            default_semantic: 預設語義權重
            default_keyword: 預設關鍵詞權重
            
        Returns:
            (semantic_weight, keyword_weight, reason)
        """
        analysis = self.analyze_query(query, query_tokens)
        
        core_count = len(analysis['core_terms'])
        generic_count = len(analysis['generic_terms'])
        total_meaningful = core_count + generic_count
        
        if total_meaningful == 0:
            # 沒有有意義的詞，使用預設
            return default_semantic, default_keyword, "無有意義詞彙"
        
        # 計算核心詞比例
        core_ratio = core_count / total_meaningful
        generic_ratio = generic_count / total_meaningful
        
        semantic_w = default_semantic
        keyword_w = default_keyword
        reasons = []
        
        # ========== 新策略: 純語義理解，不依賴 BM25 ==========
        # 核心理念: 廣泛詞權重低 → 依靠語義理解
        #          核心詞權重高 → 也依靠語義理解（可能有同義詞、相關詞）
        
        # 規則 1: 有廣泛詞存在 → 大幅提高語義權重
        if generic_count > 0:
            # 廣泛詞比例越高，越需要語義理解來過濾
            adjustment = 0.3 + (generic_ratio * 0.2)  # 0.3 ~ 0.5
            semantic_w += adjustment
            keyword_w -= adjustment
            
            generic_list = ', '.join(analysis['generic_terms'][:3])
            reasons.append(f"含廣泛詞({generic_count}個: {generic_list})，提高語義理解")
        
        # 規則 2: 核心詞比例高 → 提高語義權重
        if core_ratio > 0.4:
            # 核心詞可能以同義詞、相關詞出現在論文中
            adjustment = 0.2 + (core_ratio * 0.1)  # 0.2 ~ 0.3
            semantic_w += adjustment
            keyword_w -= adjustment
            
            core_list = ', '.join(analysis['core_terms'][:3])
            reasons.append(f"核心詞比例高({core_ratio:.0%}: {core_list})，提高語義理解")
        
        # 規則 3: 有明確領域 → 大幅提高語義權重
        domain = analysis.get('domain', '')
        if domain and domain != 'Unknown':
            # 領域概念需要語義理解（如「醫療」→「鼻咽癌」）
            adjustment = 0.2
            semantic_w += adjustment
            keyword_w -= adjustment
            reasons.append(f"明確領域({domain})，提高語義理解")
        
        # 規則 4: 查詢較短 → 提高語義權重
        if total_meaningful <= 3:
            adjustment = 0.15
            semantic_w += adjustment
            keyword_w -= adjustment
            reasons.append("查詢簡短，提高語義理解")
        
        # ========== 標準化權重 ==========
        total = semantic_w + keyword_w
        if total > 0:
            semantic_w /= total
            keyword_w /= total
        
        # 限制範圍（語義為主）
        semantic_w = max(0.7, min(0.98, semantic_w))  # 至少 70%
        keyword_w = max(0.02, min(0.3, keyword_w))    # 最多 30%
        
        # 再次標準化
        total = semantic_w + keyword_w
        semantic_w /= total
        keyword_w /= total
        
        reason = "; ".join(reasons) if reasons else "使用預設權重"
        
        logger.info(
            f"LLM 智能權重: 核心詞={core_count}, 廣泛詞={generic_count} → "
            f"語義={semantic_w:.2f}, 關鍵詞={keyword_w:.2f}"
        )
        
        return semantic_w, keyword_w, reason
    
    def boost_terms(
        self,
        query_tokens: List[str],
        analysis: Optional[Dict] = None,
        core_boost: int = 3,
        generic_reduction: bool = True
    ) -> List[str]:
        """
        根據 LLM 分析增強核心詞
        
        Args:
            query_tokens: 原始分詞
            analysis: LLM 分析結果（如為 None 則重新分析）
            core_boost: 核心詞重複次數
            generic_reduction: 是否移除廣泛詞
            
        Returns:
            增強後的分詞列表
        """
        if analysis is None:
            # 需要查詢字串，這裡簡化處理
            analysis = {'core_terms': [], 'generic_terms': query_tokens, 'auxiliary_terms': []}
        
        core_terms_set = set(t.lower() for t in analysis['core_terms'])
        generic_terms_set = set(t.lower() for t in analysis['generic_terms'])
        auxiliary_terms_set = set(t.lower() for t in analysis['auxiliary_terms'])
        
        boosted_tokens = []
        
        for token in query_tokens:
            token_lower = token.lower()
            
            if token_lower in core_terms_set:
                # 核心詞重複多次
                boosted_tokens.extend([token] * core_boost)
            elif token_lower in generic_terms_set:
                if not generic_reduction:
                    # 保留廣泛詞（但不增強）
                    boosted_tokens.append(token)
                # 否則移除廣泛詞
            elif token_lower not in auxiliary_terms_set:
                # 非輔助詞的其他詞彙保留
                boosted_tokens.append(token)
            # 輔助詞直接跳過
        
        return boosted_tokens
    
    def explain_analysis(self, query: str, query_tokens: List[str]) -> str:
        """
        生成詳細的分析說明
        
        Args:
            query: 查詢字串
            query_tokens: 分詞列表
            
        Returns:
            說明文字
        """
        analysis = self.analyze_query(query, query_tokens)
        
        explanation = f"""
LLM 查詢分析結果:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

查詢: {query}
分詞: {', '.join(query_tokens)}

核心詞彙 ({len(analysis['core_terms'])}個):
  → {', '.join(analysis['core_terms']) if analysis['core_terms'] else '無'}
  說明: 這些是查詢的關鍵概念，必須精準匹配

廣泛詞彙 ({len(analysis['generic_terms'])}個):
  → {', '.join(analysis['generic_terms']) if analysis['generic_terms'] else '無'}
  說明: 常見學術用語，區分度低，會降低權重

輔助詞彙 ({len(analysis['auxiliary_terms'])}個):
  → {', '.join(analysis['auxiliary_terms']) if analysis['auxiliary_terms'] else '無'}
  說明: 問句詞、連接詞等，不影響檢索

領域: {analysis.get('domain', 'Unknown')}
意圖: {analysis.get('intent', 'Unknown')}

判斷理由:
{analysis.get('reasoning', '無')}
"""
        
        if analysis.get('is_fallback'):
            explanation += "\n⚠️ 注意: 這是 Fallback 分析結果（LLM 不可用）\n"
        
        return explanation.strip()


# 便捷函數
def create_llm_analyzer(llm: OllamaLLM) -> LLMTermAnalyzer:
    """
    創建 LLM 詞彙分析器的便捷函數
    
    Args:
        llm: LLM 實例
        
    Returns:
        LLMTermAnalyzer 實例
    """
    return LLMTermAnalyzer(llm=llm, cache_enabled=True)
