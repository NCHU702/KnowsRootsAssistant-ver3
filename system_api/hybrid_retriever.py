"""
Hybrid Retriever: 混合檢索器（語義 + 關鍵詞）

結合向量相似度搜尋（語義理解）和 BM25 關鍵詞搜尋（精準匹配）
解決單一向量搜尋在多概念查詢時的問題。
"""

import logging
import jieba
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM
from system_api.smart_term_weighting import SmartTermWeighting
from system_api.llm_term_analyzer import LLMTermAnalyzer

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    混合檢索器：結合語義搜尋和關鍵詞搜尋
    
    解決問題：
    - 當查詢包含多個概念時（如「深度學習在人流的應用」）
    - 純向量搜尋可能只匹配主要概念（深度學習）
    - 忽略次要但重要的概念（人流）
    
    解決方案：
    - 語義搜尋：捕捉整體語義和概念理解
    - BM25 搜尋：確保關鍵詞精準匹配
    - 加權組合：兼顧語義和精準度
    """
    
    def __init__(
        self,
        layer1_vectorstore,
        use_jieba: bool = True,
        build_index_on_init: bool = True,
        enable_smart_weighting: bool = True,
        llm: Optional[OllamaLLM] = None,
        use_llm_analyzer: bool = True
    ):
        """
        初始化混合檢索器
        
        Args:
            layer1_vectorstore: Layer 1 向量存儲
            use_jieba: 是否使用 jieba 分詞（中文）
            build_index_on_init: 是否在初始化時建立 BM25 索引
            enable_smart_weighting: 是否啟用智能詞彙權重（舊系統，基於預定義字典）
            llm: LLM 實例（用於 LLM-based 分析）
            use_llm_analyzer: 是否使用 LLM 自動分析詞彙（優先於 smart_weighting）
        """
        self.layer1 = layer1_vectorstore
        self.use_jieba = use_jieba
        self.enable_smart_weighting = enable_smart_weighting
        self.use_llm_analyzer = use_llm_analyzer
        
        # BM25 相關
        self.bm25: Optional[BM25Okapi] = None
        self.documents: List[Document] = []
        self.paper_id_to_idx: Dict[str, int] = {}
        
        # 智能詞彙權重系統（舊系統）
        self.smart_weighting: Optional[SmartTermWeighting] = None
        
        # LLM 詞彙分析器（新系統）
        self.llm_analyzer: Optional[LLMTermAnalyzer] = None
        if use_llm_analyzer and llm is not None:
            logger.info("初始化 LLM 詞彙分析器...")
            self.llm_analyzer = LLMTermAnalyzer(llm=llm, cache_enabled=True)
            logger.info("✓ LLM 詞彙分析器初始化完成")
        elif use_llm_analyzer and llm is None:
            logger.warning("use_llm_analyzer=True 但未提供 LLM，將使用字典式智能權重")
            self.use_llm_analyzer = False
        
        if build_index_on_init and self.layer1.is_initialized:
            self.build_bm25_index()
    
    def _tokenize(self, text: str) -> List[str]:
        """
        文本分詞
        
        Args:
            text: 輸入文本
            
        Returns:
            分詞列表
        """
        if self.use_jieba:
            # 使用 jieba 分詞（適合中文）
            tokens = list(jieba.cut(text.lower()))
            # 過濾停用詞和短詞
            tokens = [t for t in tokens if len(t.strip()) > 1]
        else:
            # 簡單空格分詞（適合英文）
            tokens = text.lower().split()
        
        return tokens
    
    def build_bm25_index(self) -> bool:
        """
        建立 BM25 索引
        
        Returns:
            是否成功建立
        """
        try:
            if not self.layer1.vectorstore:
                logger.warning("Layer 1 vectorstore 未初始化，無法建立 BM25 索引")
                return False
            
            logger.info("開始建立 BM25 索引...")
            
            # 獲取所有文檔
            self.documents = list(self.layer1.vectorstore.docstore._dict.values())
            
            if not self.documents:
                logger.warning("沒有文檔可建立 BM25 索引")
                return False
            
            # 建立 paper_id → index 映射
            self.paper_id_to_idx = {
                doc.metadata['paper_id']: idx
                for idx, doc in enumerate(self.documents)
            }
            
            # 準備文本並分詞
            logger.info(f"對 {len(self.documents)} 篇論文進行分詞...")
            tokenized_corpus = []
            
            for doc in self.documents:
                # 組合標題和摘要
                title = doc.metadata.get('title', '')
                abstract = doc.page_content
                text = f"{title} {abstract}"
                
                # 分詞
                tokens = self._tokenize(text)
                tokenized_corpus.append(tokens)
            
            # 建立 BM25 索引
            self.bm25 = BM25Okapi(tokenized_corpus)
            
            # 初始化智能詞彙權重系統
            if self.enable_smart_weighting:
                logger.info("初始化智能詞彙權重系統...")
                self.smart_weighting = SmartTermWeighting(
                    documents=self.documents,
                    use_statistics=True
                )
            
            logger.info(f"✓ BM25 索引建立成功：{len(self.documents)} 篇論文")
            return True
            
        except Exception as e:
            logger.error(f"建立 BM25 索引失敗: {e}", exc_info=True)
            return False
    
    def hybrid_search(
        self,
        query: str,
        k: int = 10,
        semantic_weight: float = 0.5,
        keyword_weight: float = 0.5,
        score_threshold: Optional[float] = None,
        return_scores_breakdown: bool = False,
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """
        混合搜尋：結合語義和關鍵詞
        
        Args:
            query: 查詢字串
            k: 返回結果數量
            semantic_weight: 語義分數權重（0-1）
            keyword_weight: 關鍵詞分數權重（0-1）
            score_threshold: 最終分數閾值
            return_scores_breakdown: 是否返回分數細節
            **kwargs: 傳遞給 Layer 1 search 的額外參數
            
        Returns:
            List of (Document, combined_score) 或 
            List of (Document, (combined_score, semantic_score, keyword_score))
        """
        if not self.layer1.is_initialized:
            logger.error("Layer 1 未初始化")
            return []
        
        if not self.bm25:
            logger.warning("BM25 索引未建立，嘗試建立...")
            if not self.build_bm25_index():
                logger.error("無法建立 BM25 索引，回退到純語義搜尋")
                return self.layer1.search_with_scores(
                    query=query,
                    k=k,
                    score_threshold=score_threshold,
                    **kwargs
                )
        
        # ========== 0. 智能權重調整 ==========
        original_semantic_weight = semantic_weight
        original_keyword_weight = keyword_weight
        weight_adjustment_reason = ""
        query_tokens = self._tokenize(query)
        llm_analysis = None  # 用於後續 boosting
        
        # 優先使用 LLM 分析器
        if self.use_llm_analyzer and self.llm_analyzer:
            logger.info("使用 LLM 自動分析查詢詞彙...")
            
            # LLM 分析查詢
            smart_semantic_w, smart_keyword_w, reason = self.llm_analyzer.compute_smart_weights(
                query=query,
                query_tokens=query_tokens,
                default_semantic=semantic_weight,
                default_keyword=keyword_weight
            )
            
            # 獲取分析結果供後續使用
            llm_analysis = self.llm_analyzer.analyze_query(query, query_tokens)
            
            # 應用 LLM 智能權重
            semantic_weight = smart_semantic_w
            keyword_weight = smart_keyword_w
            weight_adjustment_reason = f"LLM: {reason}"
            
            logger.info(
                f"LLM 權重調整: {original_semantic_weight:.2f}/{original_keyword_weight:.2f} "
                f"→ {semantic_weight:.2f}/{keyword_weight:.2f} ({reason})"
            )
            
        # Fallback: 使用字典式智能權重
        elif self.smart_weighting:
            logger.info("使用字典式智能權重分析...")
            
            # 使用智能系統分析查詢並調整權重
            smart_semantic_w, smart_keyword_w, reason = self.smart_weighting.compute_smart_weights(
                query_tokens=query_tokens,
                default_semantic=semantic_weight,
                default_keyword=keyword_weight
            )
            
            # 應用智能權重
            semantic_weight = smart_semantic_w
            keyword_weight = smart_keyword_w
            weight_adjustment_reason = f"字典: {reason}"
            
            logger.info(
                f"字典權重調整: {original_semantic_weight:.2f}/{original_keyword_weight:.2f} "
                f"→ {semantic_weight:.2f}/{keyword_weight:.2f} ({reason})"
            )
        
        logger.info(
            f"混合搜尋: query='{query[:50]}...', "
            f"語義權重={semantic_weight:.2f}, 關鍵詞權重={keyword_weight:.2f}"
        )
        
        # ========== 1. 語義搜尋 ==========
        semantic_results = self.layer1.search_with_scores(
            query=query,
            k=min(50, len(self.documents)),  # 取更多候選
            score_threshold=None,  # 先不過濾
            **kwargs
        )
        
        # 建立 paper_id → semantic_score 映射
        semantic_scores = {
            doc.metadata['paper_id']: score
            for doc, score in semantic_results
        }
        
        logger.debug(f"語義搜尋找到 {len(semantic_results)} 篇論文")
        
        # ========== 2. BM25 關鍵詞搜尋 ==========
        # query_tokens 已在前面計算
        
        # 使用 LLM 分析或智能系統增強核心詞
        boosted_tokens = query_tokens  # 預設不增強
        
        if self.use_llm_analyzer and llm_analysis:
            # 使用 LLM 分析結果增強核心詞
            boosted_tokens = self.llm_analyzer.boost_terms(
                query_tokens=query_tokens,
                analysis=llm_analysis,
                core_boost=3,  # 核心詞重複3次
                generic_reduction=True  # 移除廣泛詞
            )
            logger.debug(
                f"LLM 詞彙增強: {query_tokens} → {boosted_tokens}"
            )
            bm25_scores = self.bm25.get_scores(boosted_tokens)
            
        elif self.smart_weighting:
            # Fallback: 使用字典式增強
            boosted_tokens = self.smart_weighting.boost_domain_terms(
                query_tokens=query_tokens,
                boost_factor=3  # 領域詞重複3次
            )
            logger.debug(
                f"字典詞彙增強: {query_tokens} → {boosted_tokens}"
            )
            bm25_scores = self.bm25.get_scores(boosted_tokens)
            
        else:
            # 無增強
            bm25_scores = self.bm25.get_scores(query_tokens)
        
        # 正規化 BM25 分數到 0-1 範圍
        max_bm25_score = max(bm25_scores) if len(bm25_scores) > 0 else 1.0
        if max_bm25_score > 0:
            normalized_bm25_scores = bm25_scores / max_bm25_score
        else:
            normalized_bm25_scores = bm25_scores
        
        # 建立 paper_id → keyword_score 映射
        keyword_scores = {
            doc.metadata['paper_id']: score
            for doc, score in zip(self.documents, normalized_bm25_scores)
            if score > 0  # 只保留有分數的
        }
        
        logger.debug(f"關鍵詞搜尋找到 {len(keyword_scores)} 篇相關論文")
        
        # ========== 3. 組合分數 + 核心詞匹配獎勵 ==========
        all_paper_ids = set(semantic_scores.keys()) | set(keyword_scores.keys())
        combined_results = []
        
        for paper_id in all_paper_ids:
            sem_score = semantic_scores.get(paper_id, 0.0)
            key_score = keyword_scores.get(paper_id, 0.0)
            
            # 基礎加權組合
            combined_score = (
                semantic_weight * sem_score +
                keyword_weight * key_score
            )
            
            # 智能系統：核心詞匹配獎勵
            if self.smart_weighting:
                doc = self.layer1.get_paper_by_id(paper_id)
                if doc:
                    # 獲取文檔分詞
                    doc_text = f"{doc.metadata.get('title', '')} {doc.page_content}"
                    doc_tokens = self._tokenize(doc_text)
                    
                    # 計算核心詞匹配分數
                    match_score = self.smart_weighting.compute_term_match_score(
                        query_tokens=query_tokens,
                        doc_tokens=doc_tokens
                    )
                    
                    # 如果核心詞匹配好，給予獎勵（最多 +0.2）
                    if match_score > 0.5:
                        bonus = 0.2 * match_score
                        combined_score += bonus
                        logger.debug(
                            f"核心詞匹配獎勵: {doc.metadata.get('title', '')[:40]} "
                            f"+{bonus:.3f} (匹配率={match_score:.2f})"
                        )
            
            # 應用閾值過濾
            if score_threshold is None or combined_score >= score_threshold:
                doc = self.layer1.get_paper_by_id(paper_id)
                if doc:
                    if return_scores_breakdown:
                        combined_results.append(
                            (doc, (combined_score, sem_score, key_score))
                        )
                    else:
                        combined_results.append((doc, combined_score))
        
        # ========== 4. 排序並返回 top-k ==========
        # 根據組合分數排序
        if return_scores_breakdown:
            combined_results.sort(key=lambda x: x[1][0], reverse=True)
        else:
            combined_results.sort(key=lambda x: x[1], reverse=True)
        
        final_results = combined_results[:k]
        
        logger.info(
            f"混合搜尋結果: {len(semantic_results)} 語義 + "
            f"{len(keyword_scores)} 關鍵詞 → {len(combined_results)} 組合 → "
            f"{len(final_results)} 最終 (top-{k})"
        )
        
        # 顯示前幾筆的分數細節
        if final_results and not return_scores_breakdown:
            logger.info("前 5 名論文分數:")
            for idx, (doc, score) in enumerate(final_results[:5], 1):
                paper_id = doc.metadata['paper_id']
                sem = semantic_scores.get(paper_id, 0.0)
                key = keyword_scores.get(paper_id, 0.0)
                title = doc.metadata.get('title', 'Unknown')[:50]
                logger.info(
                    f"  {idx}. [{score:.4f}] (語義:{sem:.4f} + 關鍵詞:{key:.4f}) "
                    f"{title}"
                )
        
        return final_results
    
    def search(
        self,
        query: str,
        k: int = 10,
        score_threshold: Optional[float] = None,
        **kwargs
    ) -> List[Document]:
        """
        搜尋接口（兼容 Layer1VectorStore.search）
        
        Returns:
            List of Document objects (without scores)
        """
        results_with_scores = self.hybrid_search(
            query=query,
            k=k,
            score_threshold=score_threshold,
            **kwargs
        )
        return [doc for doc, score in results_with_scores]
    
    def search_with_scores(
        self,
        query: str,
        k: int = 10,
        score_threshold: Optional[float] = None,
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """
        搜尋接口（兼容 Layer1VectorStore.search_with_scores）
        
        Returns:
            List of (Document, score) tuples
        """
        return self.hybrid_search(
            query=query,
            k=k,
            score_threshold=score_threshold,
            **kwargs
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        獲取統計資訊
        
        Returns:
            統計資訊字典
        """
        layer1_stats = self.layer1.get_stats()
        
        return {
            **layer1_stats,
            'retriever_type': 'hybrid',
            'bm25_enabled': self.bm25 is not None,
            'bm25_documents': len(self.documents) if self.bm25 else 0,
            'use_jieba': self.use_jieba,
        }
    
    def tune_weights(
        self,
        query: str,
        expected_paper_ids: List[str],
        k: int = 10,
        test_weights: List[Tuple[float, float]] = None
    ) -> Tuple[float, float]:
        """
        自動調整權重以優化特定查詢
        
        Args:
            query: 測試查詢
            expected_paper_ids: 期望返回的論文 ID
            k: 檢索數量
            test_weights: 測試的權重組合 [(sem_w, key_w), ...]
            
        Returns:
            最佳的 (semantic_weight, keyword_weight)
        """
        if test_weights is None:
            # 預設測試權重組合
            test_weights = [
                (1.0, 0.0),  # 純語義
                (0.8, 0.2),
                (0.7, 0.3),
                (0.6, 0.4),
                (0.5, 0.5),  # 平衡
                (0.4, 0.6),
                (0.3, 0.7),
                (0.2, 0.8),
                (0.0, 1.0),  # 純關鍵詞
            ]
        
        best_score = 0
        best_weights = (0.5, 0.5)
        
        logger.info(f"開始權重調優: 測試 {len(test_weights)} 組權重...")
        
        for sem_w, key_w in test_weights:
            # 執行搜尋
            results = self.hybrid_search(
                query=query,
                k=k,
                semantic_weight=sem_w,
                keyword_weight=key_w,
                score_threshold=None
            )
            
            # 計算命中率
            retrieved_ids = [doc.metadata['paper_id'] for doc, _ in results]
            hits = sum(1 for pid in expected_paper_ids if pid in retrieved_ids)
            hit_rate = hits / len(expected_paper_ids) if expected_paper_ids else 0
            
            logger.debug(
                f"  權重({sem_w:.1f}, {key_w:.1f}): "
                f"命中 {hits}/{len(expected_paper_ids)} = {hit_rate:.2%}"
            )
            
            if hit_rate > best_score:
                best_score = hit_rate
                best_weights = (sem_w, key_w)
        
        logger.info(
            f"✓ 最佳權重: 語義={best_weights[0]:.1f}, "
            f"關鍵詞={best_weights[1]:.1f} (命中率={best_score:.2%})"
        )
        
        return best_weights
