"""
Adaptive Weight Adjuster Module

根據查詢特性動態調整語義/關鍵詞檢索權重
"""

import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)


class AdaptiveWeightAdjuster:
    """
    自適應權重調整器
    
    功能：
    1. 分析查詢特徵（長度、專有名詞、問題形式等）
    2. 動態調整語義搜尋和關鍵詞搜尋的權重
    3. 優化不同類型查詢的檢索效果
    
    調整策略：
    - 短查詢（≤3詞）: 提高關鍵詞權重（確保精準匹配）
    - 長查詢（≥10詞）: 提高語義權重（理解複雜意圖）
    - 包含專有名詞: 提高關鍵詞權重（精準匹配技術術語）
    - 問題形式: 提高語義權重（理解問題意圖）
    """
    
    # 預定義的專有名詞列表（技術術語）
    TECH_TERMS = [
        # 深度學習模型
        'CNN', 'RNN', 'LSTM', 'GRU', 'Transformer', 'Attention',
        'ResNet', 'VGG', 'AlexNet', 'GoogLeNet', 'DenseNet',
        'BERT', 'GPT', 'YOLO', 'SSD', 'Faster-RCNN', 'Mask-RCNN',
        'GAN', 'VAE', 'Autoencoder', 'U-Net', 'SegNet',
        
        # 機器學習方法
        'SVM', 'Random Forest', 'XGBoost', 'LightGBM',
        'K-means', 'DBSCAN', 'PCA', 'LDA',
        
        # 優化算法
        'SGD', 'Adam', 'RMSprop', 'AdaGrad',
        
        # 其他技術術語
        'API', 'REST', 'GraphQL', 'SQL', 'NoSQL',
        'Kubernetes', 'Docker', 'TensorFlow', 'PyTorch',
    ]
    
    # 問題關鍵詞
    QUESTION_KEYWORDS = [
        '什麼', '如何', '為什麼', '為何', '怎麼', '怎樣',
        '哪些', '哪個', '是否', '能否', '可以', '應該',
        '什么', '怎么',  # 簡體中文
    ]
    
    def __init__(
        self,
        default_semantic_weight: float = 0.5,
        default_keyword_weight: float = 0.5
    ):
        """
        初始化自適應權重調整器
        
        Args:
            default_semantic_weight: 預設語義權重
            default_keyword_weight: 預設關鍵詞權重
        """
        self.default_semantic_weight = default_semantic_weight
        self.default_keyword_weight = default_keyword_weight
        
        logger.info(
            f"AdaptiveWeightAdjuster 初始化: "
            f"預設權重 = (語義: {default_semantic_weight:.2f}, "
            f"關鍵詞: {default_keyword_weight:.2f})"
        )
    
    def adjust_weights(self, query: str) -> Tuple[float, float, str]:
        """
        根據查詢特性調整權重
        
        Args:
            query: 查詢字串
            
        Returns:
            (semantic_weight, keyword_weight, reason)
            - semantic_weight: 語義搜尋權重 (0-1)
            - keyword_weight: 關鍵詞搜尋權重 (0-1)
            - reason: 調整原因說明
        """
        # 初始化為預設權重
        semantic_w = self.default_semantic_weight
        keyword_w = self.default_keyword_weight
        reasons = []
        
        # 分析查詢特徵
        word_count = self._count_words(query)
        has_tech_terms = self._has_tech_terms(query)
        is_question = self._is_question_form(query)
        
        # ========== 規則 1: 查詢長度 ==========
        if word_count <= 3:
            # 短查詢 → 關鍵詞權重高
            adjustment = 0.2
            keyword_w += adjustment
            semantic_w -= adjustment
            reasons.append(f"短查詢({word_count}詞)")
            
        elif word_count <= 6:
            # 中短查詢 → 稍微提高關鍵詞權重
            adjustment = 0.1
            keyword_w += adjustment
            semantic_w -= adjustment
            reasons.append(f"中短查詢({word_count}詞)")
            
        elif word_count <= 10:
            # 中等查詢 → 保持平衡
            reasons.append(f"中等查詢({word_count}詞)")
            
        else:
            # 長查詢 → 語義權重高
            adjustment = 0.2
            semantic_w += adjustment
            keyword_w -= adjustment
            reasons.append(f"長查詢({word_count}詞)")
        
        # ========== 規則 2: 專有名詞 ==========
        if has_tech_terms:
            # 包含技術術語 → 提高關鍵詞權重（確保精準匹配）
            adjustment = 0.15
            keyword_w += adjustment
            semantic_w -= adjustment
            
            # 找出包含的專有名詞
            found_terms = [
                term for term in self.TECH_TERMS
                if term.lower() in query.lower()
            ]
            reasons.append(f"包含專有名詞({', '.join(found_terms[:3])})")
        
        # ========== 規則 3: 問題形式 ==========
        if is_question:
            # 問題形式 → 提高語義權重（理解問題意圖）
            adjustment = 0.1
            semantic_w += adjustment
            keyword_w -= adjustment
            reasons.append("問題形式")
        
        # ========== 規則 4: 包含否定詞 ==========
        if self._has_negation(query):
            # 包含否定 → 提高語義權重（理解否定語義）
            adjustment = 0.1
            semantic_w += adjustment
            keyword_w -= adjustment
            reasons.append("包含否定詞")
        
        # ========== 標準化權重 ==========
        # 確保權重和為 1
        total = semantic_w + keyword_w
        if total > 0:
            semantic_w /= total
            keyword_w /= total
        
        # 限制權重範圍在 [0.1, 0.9]
        semantic_w = max(0.1, min(0.9, semantic_w))
        keyword_w = max(0.1, min(0.9, keyword_w))
        
        # 再次標準化
        total = semantic_w + keyword_w
        semantic_w /= total
        keyword_w /= total
        
        # 組合原因說明
        reason = "; ".join(reasons) if reasons else "使用預設權重"
        
        logger.info(
            f"權重調整: 查詢='{query}' | "
            f"語義={semantic_w:.2f}, 關鍵詞={keyword_w:.2f} | "
            f"原因: {reason}"
        )
        
        return semantic_w, keyword_w, reason
    
    def _count_words(self, query: str) -> int:
        """
        計算查詢詞數（使用 jieba 分詞）
        
        Args:
            query: 查詢字串
            
        Returns:
            詞數
        """
        try:
            import jieba
            words = list(jieba.cut(query.strip()))
            # 過濾單字元和空白
            meaningful_words = [w for w in words if len(w.strip()) > 1]
            return len(meaningful_words)
        except:
            # Fallback: 使用空格分割
            return len(query.split())
    
    def _has_tech_terms(self, query: str) -> bool:
        """
        檢查是否包含專有名詞
        
        Args:
            query: 查詢字串
            
        Returns:
            True 如果包含專有名詞
        """
        query_lower = query.lower()
        return any(term.lower() in query_lower for term in self.TECH_TERMS)
    
    def _is_question_form(self, query: str) -> bool:
        """
        檢查是否為問題形式
        
        Args:
            query: 查詢字串
            
        Returns:
            True 如果是問題
        """
        return any(keyword in query for keyword in self.QUESTION_KEYWORDS)
    
    def _has_negation(self, query: str) -> bool:
        """
        檢查是否包含否定詞
        
        Args:
            query: 查詢字串
            
        Returns:
            True 如果包含否定
        """
        negation_words = ['不', '沒', '無', '非', '未', '別', '莫']
        return any(neg in query for neg in negation_words)
    
    def get_weight_explanation(
        self,
        query: str,
        semantic_weight: float,
        keyword_weight: float,
        reason: str
    ) -> str:
        """
        生成權重調整的詳細說明
        
        Args:
            query: 查詢字串
            semantic_weight: 語義權重
            keyword_weight: 關鍵詞權重
            reason: 調整原因
            
        Returns:
            詳細說明文字
        """
        word_count = self._count_words(query)
        has_tech = self._has_tech_terms(query)
        is_question = self._is_question_form(query)
        
        explanation = f"""
查詢分析:
  原始查詢: {query}
  詞數: {word_count}
  包含專有名詞: {'是' if has_tech else '否'}
  問題形式: {'是' if is_question else '否'}

權重調整:
  語義搜尋權重: {semantic_weight:.2%}
  關鍵詞搜尋權重: {keyword_weight:.2%}
  
調整原因: {reason}

權重說明:
  - 語義權重高: 更重視概念理解和語義相關性
  - 關鍵詞權重高: 更重視關鍵詞精準匹配
"""
        return explanation.strip()


class WeightPresets:
    """
    預設權重方案
    
    提供常見查詢類型的權重配置
    """
    
    # 短查詢/關鍵詞查詢
    KEYWORD_FOCUSED = (0.3, 0.7)  # (semantic, keyword)
    
    # 平衡查詢
    BALANCED = (0.5, 0.5)
    
    # 長查詢/概念查詢
    SEMANTIC_FOCUSED = (0.7, 0.3)
    
    # 極短查詢（1-2詞）
    ULTRA_SHORT = (0.2, 0.8)
    
    # 極長查詢（>15詞）
    ULTRA_LONG = (0.8, 0.2)
    
    @classmethod
    def get_preset_for_query(cls, query: str) -> Tuple[float, float]:
        """
        根據查詢選擇合適的預設權重
        
        Args:
            query: 查詢字串
            
        Returns:
            (semantic_weight, keyword_weight)
        """
        try:
            import jieba
            words = list(jieba.cut(query.strip()))
            word_count = len([w for w in words if len(w.strip()) > 1])
        except:
            word_count = len(query.split())
        
        if word_count <= 2:
            return cls.ULTRA_SHORT
        elif word_count <= 4:
            return cls.KEYWORD_FOCUSED
        elif word_count <= 10:
            return cls.BALANCED
        elif word_count <= 15:
            return cls.SEMANTIC_FOCUSED
        else:
            return cls.ULTRA_LONG
