"""
Smart Term Weighting Module - 智能詞彙權重系統

自動識別並區分：
1. 領域特定詞（Domain-specific terms）：醫療、人流、CNN 等
2. 泛用詞（Generic terms）：深度學習、應用、方法等

根據詞彙類型動態調整檢索策略
"""

import logging
from typing import List, Dict, Set, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class SmartTermWeighting:
    """
    智能詞彙權重系統
    
    功能：
    1. 自動分類詞彙為「領域詞」或「泛用詞」
    2. 根據語料庫統計計算詞彙重要性
    3. 動態調整混合搜尋權重
    """
    
    # 預定義領域詞典（可擴充）
    DOMAIN_KEYWORDS = {
        # 醫療領域
        '醫療': ['醫療', '醫學', '臨床', '診斷', '治療', '病患', '疾病', '癌症', 
                '腫瘤', '鼻咽癌', '檢測', '辨識', '健康', '醫院', '護理'],
        
        # 人流/交通領域  
        '人流': ['人流', '人群', '行人', '客流', '人潮', '人數', '流量', '擁擠',
                '移動', '軌跡', '出行', '通勤', '交通', '運輸', '道路'],
        
        # 模型/算法（技術特定）
        'AI模型': ['CNN', 'RNN', 'LSTM', 'GRU', 'Transformer', 'BERT', 'GPT',
                  'YOLO', 'ResNet', 'VGG', 'GAN', 'VAE', 'Attention',
                  'Faster-RCNN', 'Mask-RCNN', 'U-Net', 'SegNet'],
        
        # 時空數據
        '時空': ['時空', '空間', '時序', '時間序列', '地理', '位置', 'GPS',
                '軌跡', '路徑', '區域', 'OD矩陣'],
        
        # 環境/能源
        '環境': ['PM2.5', 'PM10', '空氣品質', '污染', '環境', '氣象', '溫度',
                '濕度', '風速', '能源', '電力'],
        
        # 交通工具
        '交通工具': ['公車', '捷運', 'YouBike', 'Ubike', '共享單車', '計程車',
                    '汽車', '機車', '電動車', '巴士', '復康巴士'],
    }
    
    # 泛用詞（在學術論文中常見但缺乏區分度）
    GENERIC_TERMS = {
        # 研究相關
        '研究': ['研究', '探討', '分析', '方法', '模型', '系統', '框架', 
                '架構', '設計', '開發', '建構', '實現', '驗證'],
        
        # 深度學習通用詞
        '深度學習': ['深度學習', '機器學習', '人工智慧', '神經網路', '類神經',
                    '訓練', '學習', '特徵', '卷積', '池化', '激活'],
        
        # 應用相關
        '應用': ['應用', '使用', '運用', '利用', '基於', '結合', '融合',
                '整合', '導入', '採用'],
        
        # 預測/分類
        '預測': ['預測', '預報', '預估', '估計', '分類', '辨識', '識別',
                '檢測', '偵測', '判斷'],
        
        # 改進/優化
        '優化': ['優化', '改善', '改進', '提升', '增強', '強化', '提高',
                '降低', '減少', '最佳化'],
        
        # 數據/資料
        '資料': ['資料', '數據', '資訊', '訊息', '樣本', '集合', '庫',
                '來源', '收集', '處理'],
    }
    
    def __init__(self, documents: List = None, use_statistics: bool = True):
        """
        初始化智能詞彙權重系統
        
        Args:
            documents: 文檔列表（用於統計分析）
            use_statistics: 是否使用統計方法自動識別泛用詞
        """
        self.use_statistics = use_statistics
        self.documents = documents or []
        
        # 展開詞典為集合（加速查找）
        self.domain_terms: Set[str] = set()
        self.domain_categories: Dict[str, str] = {}  # term -> category
        
        for category, terms in self.DOMAIN_KEYWORDS.items():
            for term in terms:
                self.domain_terms.add(term.lower())
                self.domain_categories[term.lower()] = category
        
        self.generic_terms: Set[str] = set()
        for category, terms in self.GENERIC_TERMS.items():
            for term in terms:
                self.generic_terms.add(term.lower())
        
        # 統計資訊
        self.term_doc_freq: Dict[str, int] = defaultdict(int)  # 詞彙在多少文檔中出現
        self.total_docs: int = 0
        
        if self.documents and use_statistics:
            self._compute_statistics()
        
        logger.info(
            f"SmartTermWeighting 初始化完成: "
            f"{len(self.domain_terms)} 個領域詞, "
            f"{len(self.generic_terms)} 個泛用詞"
        )
    
    def _compute_statistics(self):
        """計算詞彙統計資訊"""
        from collections import Counter
        import jieba
        
        logger.info("計算詞彙統計資訊...")
        self.total_docs = len(self.documents)
        
        for doc in self.documents:
            # 獲取文檔文本
            if hasattr(doc, 'page_content'):
                text = doc.page_content
            else:
                text = str(doc)
            
            # 分詞
            tokens = set(jieba.cut(text.lower()))
            
            # 統計每個詞出現在多少文檔中
            for token in tokens:
                if len(token.strip()) > 1:
                    self.term_doc_freq[token] += 1
        
        logger.info(f"✓ 統計完成: {len(self.term_doc_freq)} 個不同詞彙")
    
    def classify_term(self, term: str) -> Tuple[str, float]:
        """
        分類詞彙並返回重要性分數
        
        Args:
            term: 詞彙
            
        Returns:
            (category, importance_score)
            - category: 'domain' / 'generic' / 'unknown'
            - importance_score: 0-1，越高越重要
        """
        term_lower = term.lower()
        
        # 1. 檢查是否為領域詞
        if term_lower in self.domain_terms:
            # 領域詞非常重要
            importance = 1.0
            return 'domain', importance
        
        # 2. 檢查是否為泛用詞
        if term_lower in self.generic_terms:
            # 泛用詞重要性低
            importance = 0.3
            return 'generic', importance
        
        # 3. 使用統計方法判斷
        if self.use_statistics and self.total_docs > 0:
            doc_freq = self.term_doc_freq.get(term_lower, 0)
            doc_ratio = doc_freq / self.total_docs
            
            # IDF-based importance
            # 出現在很多文檔中 → 可能是泛用詞
            # 出現在少數文檔中 → 可能是領域詞
            if doc_ratio > 0.7:
                # 超過70%文檔都有 → 泛用詞
                importance = 0.3
                return 'generic_auto', importance
            elif doc_ratio < 0.3 and doc_freq > 0:
                # 少於30%文檔有，但確實出現過 → 可能是領域詞
                importance = 0.8
                return 'domain_auto', importance
            else:
                # 中等頻率
                importance = 0.5
                return 'neutral', importance
        
        # 4. 未知詞彙，給予中等重要性
        return 'unknown', 0.5
    
    def analyze_query(
        self, 
        query_tokens: List[str]
    ) -> Dict[str, any]:
        """
        分析查詢詞彙組成
        
        Args:
            query_tokens: 查詢分詞列表
            
        Returns:
            分析結果字典
        """
        domain_terms = []
        generic_terms = []
        neutral_terms = []
        
        term_importance = {}
        
        for token in query_tokens:
            category, importance = self.classify_term(token)
            term_importance[token] = importance
            
            if category in ['domain', 'domain_auto']:
                domain_terms.append(token)
            elif category in ['generic', 'generic_auto']:
                generic_terms.append(token)
            else:
                neutral_terms.append(token)
        
        # 計算領域詞比例
        total_terms = len(query_tokens)
        domain_ratio = len(domain_terms) / total_terms if total_terms > 0 else 0
        generic_ratio = len(generic_terms) / total_terms if total_terms > 0 else 0
        
        return {
            'domain_terms': domain_terms,
            'generic_terms': generic_terms,
            'neutral_terms': neutral_terms,
            'domain_ratio': domain_ratio,
            'generic_ratio': generic_ratio,
            'term_importance': term_importance,
            'has_domain_focus': domain_ratio > 0.3  # 超過30%是領域詞
        }
    
    def compute_smart_weights(
        self,
        query_tokens: List[str],
        default_semantic: float = 0.5,
        default_keyword: float = 0.5
    ) -> Tuple[float, float, str]:
        """
        根據查詢詞彙組成計算智能權重
        
        Args:
            query_tokens: 查詢分詞列表
            default_semantic: 預設語義權重
            default_keyword: 預設關鍵詞權重
            
        Returns:
            (semantic_weight, keyword_weight, reason)
        """
        analysis = self.analyze_query(query_tokens)
        
        semantic_w = default_semantic
        keyword_w = default_keyword
        reasons = []
        
        # ========== 規則 1: 領域詞比例高 ==========
        if analysis['has_domain_focus']:
            # 有明確領域詞 → 提高語義權重
            # 原因: 領域詞可能在標題中以不同形式出現（如"鼻咽癌"表示"醫療"）
            adjustment = 0.2
            semantic_w += adjustment
            keyword_w -= adjustment
            
            domain_list = ', '.join(analysis['domain_terms'][:3])
            reasons.append(f"包含領域詞({domain_list})")
        
        # ========== 規則 2: 泛用詞比例高 ==========
        if analysis['generic_ratio'] > 0.5:
            # 超過一半是泛用詞 → 降低關鍵詞權重
            # 原因: 泛用詞會干擾 BM25，讓不相關的論文得高分
            adjustment = 0.15
            keyword_w -= adjustment
            semantic_w += adjustment
            
            generic_list = ', '.join(analysis['generic_terms'][:3])
            reasons.append(f"泛用詞過多({generic_list})")
        
        # ========== 規則 3: 同時有領域詞和泛用詞 ==========
        if len(analysis['domain_terms']) > 0 and len(analysis['generic_terms']) > 0:
            # 混合查詢 → 平衡但稍微偏向語義
            # 讓語義理解領域詞，減少泛用詞干擾
            adjustment = 0.1
            semantic_w += adjustment
            keyword_w -= adjustment
            reasons.append("混合查詢(領域+泛用)")
        
        # ========== 規則 4: 全是泛用詞 ==========
        if analysis['generic_ratio'] > 0.8:
            # 幾乎全是泛用詞 → 大幅提高語義權重
            adjustment = 0.2
            semantic_w += adjustment
            keyword_w -= adjustment
            reasons.append("查詢過於泛用")
        
        # ========== 標準化權重 ==========
        total = semantic_w + keyword_w
        if total > 0:
            semantic_w /= total
            keyword_w /= total
        
        # 限制範圍
        semantic_w = max(0.2, min(0.9, semantic_w))
        keyword_w = max(0.1, min(0.8, keyword_w))
        
        # 再次標準化
        total = semantic_w + keyword_w
        semantic_w /= total
        keyword_w /= total
        
        reason = "; ".join(reasons) if reasons else "使用預設權重"
        
        logger.info(
            f"智能權重計算: 領域詞={len(analysis['domain_terms'])}, "
            f"泛用詞={len(analysis['generic_terms'])} → "
            f"語義={semantic_w:.2f}, 關鍵詞={keyword_w:.2f}"
        )
        
        return semantic_w, keyword_w, reason
    
    def boost_domain_terms(
        self,
        query_tokens: List[str],
        boost_factor: int = 2
    ) -> List[str]:
        """
        在 BM25 分詞中增強領域詞權重
        
        方法: 重複領域詞 N 次
        
        Args:
            query_tokens: 原始分詞
            boost_factor: 領域詞重複次數
            
        Returns:
            增強後的分詞列表
        """
        boosted_tokens = []
        
        for token in query_tokens:
            category, importance = self.classify_term(token)
            
            if category in ['domain', 'domain_auto']:
                # 領域詞重複多次
                boosted_tokens.extend([token] * boost_factor)
            elif category in ['generic', 'generic_auto']:
                # 泛用詞只出現一次或跳過
                if importance > 0.3:  # 只保留稍微重要的泛用詞
                    boosted_tokens.append(token)
                # 否則跳過（完全移除）
            else:
                # 中性詞正常保留
                boosted_tokens.append(token)
        
        return boosted_tokens
    
    def compute_term_match_score(
        self,
        query_tokens: List[str],
        doc_tokens: List[str]
    ) -> float:
        """
        計算查詢與文檔的核心詞匹配分數
        
        重視領域詞匹配，忽略泛用詞匹配
        
        Args:
            query_tokens: 查詢分詞
            doc_tokens: 文檔分詞
            
        Returns:
            匹配分數 (0-1)
        """
        query_analysis = self.analyze_query(query_tokens)
        domain_terms_in_query = set(t.lower() for t in query_analysis['domain_terms'])
        
        if not domain_terms_in_query:
            # 查詢中沒有領域詞，返回中等分數
            return 0.5
        
        # 檢查文檔中有多少領域詞被匹配
        doc_tokens_set = set(t.lower() for t in doc_tokens)
        matched_domain_terms = domain_terms_in_query & doc_tokens_set
        
        # 計算領域詞匹配率
        match_ratio = len(matched_domain_terms) / len(domain_terms_in_query)
        
        return match_ratio
    
    def get_term_categories(self, term: str) -> List[str]:
        """
        獲取詞彙所屬的領域類別
        
        Args:
            term: 詞彙
            
        Returns:
            類別列表
        """
        term_lower = term.lower()
        categories = []
        
        for category, terms in self.DOMAIN_KEYWORDS.items():
            if term_lower in [t.lower() for t in terms]:
                categories.append(category)
        
        return categories
    
    def explain_analysis(self, query_tokens: List[str]) -> str:
        """
        生成查詢分析的詳細說明
        
        Args:
            query_tokens: 查詢分詞
            
        Returns:
            說明文字
        """
        analysis = self.analyze_query(query_tokens)
        
        explanation = f"""
查詢詞彙分析:
  總詞數: {len(query_tokens)}
  領域詞: {len(analysis['domain_terms'])} ({analysis['domain_ratio']:.1%})
    → {', '.join(analysis['domain_terms']) if analysis['domain_terms'] else '無'}
  泛用詞: {len(analysis['generic_terms'])} ({analysis['generic_ratio']:.1%})
    → {', '.join(analysis['generic_terms']) if analysis['generic_terms'] else '無'}
  中性詞: {len(analysis['neutral_terms'])}
    → {', '.join(analysis['neutral_terms']) if analysis['neutral_terms'] else '無'}

判斷: {'✓ 有明確領域焦點' if analysis['has_domain_focus'] else '⚠ 缺乏領域焦點'}

詞彙重要性分數:
"""
        for term, score in analysis['term_importance'].items():
            explanation += f"  {term}: {score:.2f}\n"
        
        return explanation.strip()


# 便捷函數
def create_smart_weighting(documents: List = None) -> SmartTermWeighting:
    """
    創建智能詞彙權重系統的便捷函數
    
    Args:
        documents: 文檔列表
        
    Returns:
        SmartTermWeighting 實例
    """
    return SmartTermWeighting(documents=documents, use_statistics=True)
