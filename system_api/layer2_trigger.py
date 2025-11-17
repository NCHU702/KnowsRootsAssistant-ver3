"""
Layer 2 觸發決策器

智能判斷是否需要進入 Layer 2 詳細檢索
根據查詢類型、Layer 1 結果質量等因素進行決策
"""

import logging
from typing import Dict, List, Any
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM

logger = logging.getLogger(__name__)


class Layer2TriggerDecision:
    """Layer 2 觸發決策器"""
    
    def __init__(self, llm: OllamaLLM = None):
        """
        初始化決策器
        
        Args:
            llm: LLM 實例（可選，用於更複雜的查詢分析）
        """
        self.llm = llm
        logger.info("Layer2TriggerDecision 初始化完成")
    
    def should_trigger_layer2(
        self,
        query: str,
        layer1_evaluation: Dict[str, Any],
        layer1_docs: List[Document],
        base_threshold: float = 0.6
    ) -> Dict[str, Any]:
        """
        智能判斷是否需要觸發 Layer 2
        
        Args:
            query: 使用者查詢
            layer1_evaluation: Layer 1 的信心評估結果
            layer1_docs: Layer 1 檢索到的文檔
            base_threshold: 基礎閾值（預設 0.6）
        
        Returns:
            {
                'should_trigger': bool,
                'reason': str,
                'adjusted_threshold': float,
                'decision_factors': Dict
            }
        """
        confidence = layer1_evaluation.get('confidence', 0.0)
        reasoning = layer1_evaluation.get('reasoning', '')
        
        # 決策因子
        factors = {
            'confidence': confidence,
            'base_threshold': base_threshold,
            'query_type': None,
            'doc_count': len(layer1_docs),
            'query_complexity': None,
        }
        
        # 1. 分析查詢類型和複雜度
        query_analysis = self._analyze_query_intent(query)
        factors['query_type'] = query_analysis['type']
        factors['query_complexity'] = query_analysis['complexity']
        
        logger.info(
            f"查詢分析: 類型={query_analysis['type']}, "
            f"複雜度={query_analysis['complexity']}"
        )
        
        # 2. 根據查詢類型調整閾值
        adjusted_threshold = self._adjust_threshold_by_query_type(
            base_threshold,
            query_analysis
        )
        factors['adjusted_threshold'] = adjusted_threshold
        
        # 3. 特殊情況判斷（優化版）
        
        # ===== 優先級 1：綜觀性/大方向問題 → 只需 Layer 1 =====
        # 情況 A：概覽、列表、比較、定義、應用類查詢 → Layer 1 摘要足夠
        if query_analysis['type'] in ['overview', 'list', 'comparison', 'definition', 'application']:
            # 只要有結果且信心度不是太低（>= 0.45），就不需要 Layer 2
            if confidence >= 0.45 and len(layer1_docs) >= 1:
                return {
                    'should_trigger': False,
                    'reason': (
                        f"查詢類型 '{query_analysis['type']}' 屬於綜觀性問題，"
                        f"Layer 1 摘要層已足夠（找到 {len(layer1_docs)} 篇論文，"
                        f"信心度: {confidence:.2f}）"
                    ),
                    'adjusted_threshold': adjusted_threshold,
                    'decision_factors': factors
                }
        
        # ===== 優先級 2：詳細內容需求 → 必須進 Layer 2 =====
        # 情況 B：詳細解釋、方法論、實作細節 → 必須進入 Layer 2
        if query_analysis['type'] in ['detailed_explanation', 'methodology', 'implementation']:
            if len(layer1_docs) > 0:  # 只要 Layer 1 有結果
                return {
                    'should_trigger': True,
                    'reason': (
                        f"查詢類型 '{query_analysis['type']}' 需要詳細內容或方法論，"
                        f"必須進入 Layer 2 獲取具體資訊（當前信心度: {confidence:.2f}）"
                    ),
                    'adjusted_threshold': adjusted_threshold,
                    'decision_factors': factors
                }
        
        # ===== 優先級 3：特定論文查詢 → 需要 Layer 2 =====
        # 情況 C：查詢提到「論文」、「這篇」、「該研究」→ 需要具體內容
        specific_paper_keywords = [
            '論文', '這篇', '該篇', '該研究', '這個研究',
            '文章', '該論文', '此論文', '這份研究'
        ]
        if any(kw in query for kw in specific_paper_keywords):
            if len(layer1_docs) > 0:
                return {
                    'should_trigger': True,
                    'reason': (
                        f"查詢提到特定論文或研究，需要進入 Layer 2 "
                        f"獲取詳細內容以精確回答"
                    ),
                    'adjusted_threshold': adjusted_threshold,
                    'decision_factors': factors
                }
        
        # ===== 優先級 4：Layer 1 沒有結果 → 無法進 Layer 2 =====
        # 情況 D：Layer 1 結果為空
        if len(layer1_docs) == 0:
            return {
                'should_trigger': False,
                'reason': "Layer 1 沒有找到相關論文，無法進入 Layer 2",
                'adjusted_threshold': adjusted_threshold,
                'decision_factors': factors
            }
        
        # ===== 優先級 5：Layer 1 結果太少 → 謹慎判斷 =====
        # 情況 E：結果很少（1-2 篇）→ 根據查詢類型決定
        if len(layer1_docs) <= 2:
            # 如果是一般查詢且信心度還可以，進 Layer 2 補充資訊
            if query_analysis['type'] == 'general' and confidence >= 0.4:
                return {
                    'should_trigger': True,
                    'reason': (
                        f"Layer 1 僅找到 {len(layer1_docs)} 篇論文（較少），"
                        f"進入 Layer 2 以獲取更完整資訊"
                    ),
                    'adjusted_threshold': adjusted_threshold,
                    'decision_factors': factors
                }
        
        # ===== 優先級 6：信心評估明確指出需要細節 =====
        # 情況 F：LLM 評估認為需要更多資訊
        need_details_keywords = [
            '缺少', '不足', '需要更多', '細節', '具體',
            '不夠', '需補充', '資訊有限', '不完整', '太簡略'
        ]
        if any(keyword in reasoning for keyword in need_details_keywords):
            return {
                'should_trigger': True,
                'reason': f"Layer 1 評估指出需要更多細節: {reasoning[:80]}...",
                'adjusted_threshold': adjusted_threshold,
                'decision_factors': factors
            }
        
        # ===== 優先級 7：Layer 1 結果充足 → 可能不需要 Layer 2 =====
        # 情況 G：結果很多（>= 5 篇）→ 優先使用 Layer 1
        if len(layer1_docs) >= 5:
            # 對於一般查詢或綜觀性查詢，結果多就不需要 Layer 2
            if query_analysis['type'] in ['general', 'overview', 'list', 'comparison', 'application']:
                if confidence >= 0.40:  # 降低閾值，因為結果已經很多
                    return {
                        'should_trigger': False,
                        'reason': (
                            f"Layer 1 找到 {len(layer1_docs)} 篇論文（充足），"
                            f"信心度 {confidence:.2f}，無需進入 Layer 2"
                        ),
                        'adjusted_threshold': adjusted_threshold,
                        'decision_factors': factors
                    }
        
        # ===== 優先級 8：查詢太寬泛 + 結果很多 → 不需要 Layer 2 =====
        # 情況 H：查詢詞數很少（<= 3）且結果多（>= 10）
        query_words = len([w for w in query if w.strip()])
        if query_words <= 10 and len(layer1_docs) >= 10:
            # 寬泛查詢，Layer 1 足夠
            if query_analysis['type'] == 'general':
                return {
                    'should_trigger': False,
                    'reason': (
                        f"查詢較寬泛（{query_words} 字）且 Layer 1 找到 "
                        f"{len(layer1_docs)} 篇論文，摘要已足夠提供概覽"
                    ),
                    'adjusted_threshold': adjusted_threshold,
                    'decision_factors': factors
                }
        
        # 4. 預設判斷：使用調整後的閾值
        should_trigger = confidence < adjusted_threshold
        
        if should_trigger:
            reason = (
                f"信心度 {confidence:.2f} 低於調整後閾值 {adjusted_threshold:.2f}，"
                f"需要 Layer 2 提供更詳細資訊"
            )
        else:
            reason = (
                f"信心度 {confidence:.2f} 達到要求（>= {adjusted_threshold:.2f}），"
                f"Layer 1 結果已足夠"
            )
        
        return {
            'should_trigger': should_trigger,
            'reason': reason,
            'adjusted_threshold': adjusted_threshold,
            'decision_factors': factors
        }
    
    def _analyze_query_intent(self, query: str) -> Dict[str, str]:
        """
        分析查詢意圖和複雜度
        
        Args:
            query: 查詢字串
        
        Returns:
            {
                'type': str,  # 查詢類型
                'complexity': str  # 複雜度
            }
        """
        query_lower = query.lower()
        
        # 簡單規則式判斷
        # 未來可以升級為 LLM 判斷（如果有 self.llm）
        
        # ===== 優先級 1：方法論關鍵詞（如何、怎麼）→ 必須優先檢查 =====
        # 「如何應用」、「如何實現」等明確要求方法的查詢
        how_keywords = ['如何', '怎麼', '怎樣', 'how', 'how to']
        if any(kw in query_lower for kw in how_keywords):
            return {'type': 'methodology', 'complexity': 'complex'}
        
        # ===== 應用類查詢 =====
        # 「XX在XX的應用」、「XX應用於XX」這類大方向問題（不含「如何」）
        application_keywords = [
            '應用', 'application', 'apply', '用於', '用在',
            '使用於', '應用於', '運用'
        ]
        if any(kw in query_lower for kw in application_keywords):
            return {'type': 'application', 'complexity': 'simple'}
        
        # ===== 概覽類查詢 =====
        overview_keywords = [
            '有哪些', '列出', '總結', '概述', '介紹', '簡介',
            'overview', 'summary', 'introduce', '摘要'
        ]
        if any(kw in query_lower for kw in overview_keywords):
            return {'type': 'overview', 'complexity': 'simple'}
        
        # ===== 列表類查詢 =====
        list_keywords = [
            '列表', 'list', '所有', '全部', '哪幾', '幾個',
            '多少', '統計'
        ]
        if any(kw in query_lower for kw in list_keywords):
            return {'type': 'list', 'complexity': 'simple'}
        
        # ===== 比較類查詢 =====
        comparison_keywords = [
            '比較', '差異', '不同', '優缺點', 'vs', '比對',
            '對比', 'compare', 'difference', 'versus'
        ]
        if any(kw in query_lower for kw in comparison_keywords):
            return {'type': 'comparison', 'complexity': 'medium'}
        
        # ===== 模型/架構查詢（新增）=====
        # 這類查詢通常需要詳細的模型結構、參數等資訊
        model_keywords = [
            '模型', '架構', '結構', 'model', 'architecture',
            '網路', 'network', '層', 'layer', '參數', 'parameter',
            '設計', 'design', '框架', 'framework'
        ]
        if any(kw in query_lower for kw in model_keywords):
            # 如果同時提到「什麼是」則是定義查詢，不需要太詳細
            if not any(def_kw in query_lower for def_kw in ['什麼是', '是什麼', 'what is']):
                return {'type': 'methodology', 'complexity': 'complex'}
        
        # ===== 詳細解釋類查詢 =====
        detailed_keywords = [
            '詳細', '解釋', '說明', '為什麼', '為何',
            '如何運作', '原理', '機制', 'explain', 'how does',
            'why', 'detail', '深入', '詳述', '具體'
        ]
        if any(kw in query_lower for kw in detailed_keywords):
            return {'type': 'detailed_explanation', 'complexity': 'complex'}
        
        # ===== 方法學查詢 =====
        methodology_keywords = [
            '方法', '步驟', '流程', '做法', '過程', '技術',
            'method', 'approach', 'procedure', 'process',
            'methodology', '怎麼做', '如何實現', '演算法',
            'algorithm', '策略', 'strategy'
        ]
        if any(kw in query_lower for kw in methodology_keywords):
            return {'type': 'methodology', 'complexity': 'complex'}
        
        # ===== 實作類查詢 =====
        implementation_keywords = [
            '實作', '實現', '實施', '程式', '代碼', '範例',
            'code', 'implement', 'example', 'sample',
            '程式碼', '寫法', '實驗', 'experiment', '驗證',
            'validation', '測試', 'test'
        ]
        if any(kw in query_lower for kw in implementation_keywords):
            return {'type': 'implementation', 'complexity': 'complex'}
        
        # ===== 定義/概念類查詢 =====
        definition_keywords = [
            '什麼是', '是什麼', '定義', 'what is', 'define',
            'definition', '意思'
        ]
        if any(kw in query_lower for kw in definition_keywords):
            return {'type': 'definition', 'complexity': 'simple'}
        
        # ===== 預設：一般查詢 =====
        return {'type': 'general', 'complexity': 'medium'}
    
    def _adjust_threshold_by_query_type(
        self,
        base_threshold: float,
        query_analysis: Dict[str, str]
    ) -> float:
        """
        根據查詢類型調整閾值
        
        概覽/列表查詢 → 降低閾值（更容易滿足，Layer 1 通常足夠）
        詳細解釋/方法學查詢 → 提高閾值（更嚴格要求，更可能需要 Layer 2）
        
        Args:
            base_threshold: 基礎閾值
            query_analysis: 查詢分析結果
        
        Returns:
            調整後的閾值
        """
        query_type = query_analysis['type']
        
        # 調整映射
        # 負值 = 降低閾值（更容易滿足）
        # 正值 = 提高閾值（更嚴格）
        threshold_adjustments = {
            'overview': -0.10,        # 0.6 → 0.5  (Layer 1 摘要通常足夠)
            'list': -0.10,            # 0.6 → 0.5  (Layer 1 標題列表足夠)
            'application': -0.10,     # 0.6 → 0.5  (應用概覽，Layer 1 足夠)
            'definition': -0.05,      # 0.6 → 0.55 (簡單定義 Layer 1 可滿足)
            'comparison': 0.0,        # 0.6 → 0.6  (中等難度，標準閾值)
            'general': 0.0,           # 0.6 → 0.6  (一般查詢，標準閾值)
            'detailed_explanation': +0.10,  # 0.6 → 0.7  (需要詳細內容)
            'methodology': +0.15,     # 0.6 → 0.75 (需要具體方法步驟)
            'implementation': +0.20,  # 0.6 → 0.8  (需要實作細節)
        }
        
        adjustment = threshold_adjustments.get(query_type, 0.0)
        adjusted = base_threshold + adjustment
        
        # 限制在合理範圍內 [0.4, 0.85]
        adjusted = max(0.4, min(0.85, adjusted))
        
        if adjustment != 0:
            logger.info(
                f"閾值調整: {base_threshold:.2f} → {adjusted:.2f} "
                f"(查詢類型: {query_type}, 調整: {adjustment:+.2f})"
            )
        
        return adjusted


# ============================================================================
# 測試用例
# ============================================================================
if __name__ == '__main__':
    # 初始化
    trigger = Layer2TriggerDecision()
    
    # 測試查詢
    test_cases = [
        {
            'query': '列出所有關於深度學習的論文',
            'confidence': 0.55,
            'doc_count': 10,
            'expected': False,  # 列表查詢，Layer 1 足夠
        },
        {
            'query': '有哪些人流預測的研究',
            'confidence': 0.52,
            'doc_count': 8,
            'expected': False,  # 概覽查詢，Layer 1 足夠
        },
        {
            'query': '詳細解釋 CNN 的卷積運算原理',
            'confidence': 0.65,
            'doc_count': 5,
            'expected': True,  # 詳細解釋，需要 Layer 2
        },
        {
            'query': 'LSTM 的具體實作方法',
            'confidence': 0.70,
            'doc_count': 6,
            'expected': True,  # 方法學查詢，需要 Layer 2
        },
        {
            'query': '深度學習在醫療的應用',
            'confidence': 0.45,
            'doc_count': 7,
            'expected': True,  # 信心低，需要 Layer 2
        },
        {
            'query': '比較 RNN 和 LSTM 的差異',
            'confidence': 0.62,
            'doc_count': 8,
            'expected': False,  # 比較查詢，信心足夠
        },
    ]
    
    print("="*70)
    print("Layer 2 觸發決策測試")
    print("="*70)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n測試 {i}:")
        print(f"  查詢: {case['query']}")
        print(f"  信心度: {case['confidence']:.2f}")
        print(f"  文檔數: {case['doc_count']}")
        
        # 模擬 Layer 1 評估結果
        layer1_eval = {
            'confidence': case['confidence'],
            'reasoning': 'Test evaluation'
        }
        
        # 模擬文檔列表
        from langchain_core.documents import Document
        layer1_docs = [
            Document(page_content=f"Doc {i}", metadata={'title': f'Paper {i}'})
            for i in range(case['doc_count'])
        ]
        
        # 決策
        decision = trigger.should_trigger_layer2(
            query=case['query'],
            layer1_evaluation=layer1_eval,
            layer1_docs=layer1_docs,
            base_threshold=0.6
        )
        
        print(f"  決策: {'觸發 Layer 2' if decision['should_trigger'] else '不觸發'}")
        print(f"  原因: {decision['reason']}")
        print(f"  調整後閾值: {decision['adjusted_threshold']:.2f}")
        print(f"  查詢類型: {decision['decision_factors']['query_type']}")
        
        # 檢查是否符合預期
        if decision['should_trigger'] == case['expected']:
            print(f"  ✅ 符合預期")
        else:
            print(f"  ❌ 不符合預期（預期: {'觸發' if case['expected'] else '不觸發'}）")
    
    print("\n" + "="*70)
