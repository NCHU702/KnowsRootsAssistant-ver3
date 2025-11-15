# 短查詢問題與解決方案

## 🔍 問題分析

### 問題描述
當用戶輸入簡短查詢時（例如："人流預測"），所有相關論文的相似度都很接近，難以區分哪篇最相關。

### 實際範例

**查詢**: "人流預測"

**問題結果**:
```
1. 基於CNN的人流預測 (0.652)
2. LSTM人流分析 (0.648)
3. 人群密度估計 (0.645)
4. 捷運站人流 (0.642)
5. 時間序列預測 (0.638)
```

**問題**: 分數太接近（0.64-0.65），無法有效區分優先級

### 根本原因

1. **查詢向量信息量不足**
   ```
   "人流預測" → 只有 2 個詞 → 向量表達能力有限
   ```

2. **語義空間收縮**
   ```
   短查詢 → 概念模糊 → 許多論文都「相關」
   ```

3. **關鍵詞過於通用**
   ```
   "人流" "預測" → 太多論文包含這些詞
   ```

---

## 💡 解決方案

### 方案 1: 查詢擴展（Query Expansion）⭐⭐⭐⭐⭐

**概念**: 使用 LLM 將短查詢擴展為更詳細的描述

#### 實現

```python
class QueryExpander:
    """查詢擴展器"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def expand_query(self, query: str, min_length: int = 10) -> str:
        """
        擴展短查詢為更詳細的描述
        
        Args:
            query: 原始查詢
            min_length: 最小詞數，少於此數才擴展
            
        Returns:
            擴展後的查詢
        """
        # 檢查查詢長度
        word_count = len(query.split())
        
        if word_count >= min_length:
            logger.info(f"查詢長度足夠 ({word_count} 詞)，不需擴展")
            return query
        
        logger.info(f"檢測到短查詢 ({word_count} 詞)，進行擴展...")
        
        prompt = f"""你是一個學術搜尋助手。用戶輸入了一個簡短的查詢，請將其擴展為更詳細的學術問題描述。

原始查詢: {query}

要求:
1. 保留原始查詢的核心概念
2. 添加相關的技術細節、應用場景或研究方向
3. 使用學術論文常見的表達方式
4. 擴展後長度約 20-30 個中文字
5. 不要偏離原始意圖

範例:
- 輸入: "人流預測"
- 輸出: "使用深度學習方法進行城市公共空間的人流量預測與分析，包括時空特徵建模和預測模型研究"

- 輸入: "CNN"
- 輸出: "卷積神經網路 (CNN) 的架構設計、訓練方法及其在圖像識別、電腦視覺領域的應用研究"

現在請擴展查詢: {query}

直接輸出擴展後的查詢，不要解釋:"""

        response = self.llm.invoke(prompt)
        expanded_query = response.content.strip()
        
        logger.info(f"查詢擴展: '{query}' → '{expanded_query}'")
        
        return expanded_query
    
    def expand_with_context(self, query: str, context: Dict[str, Any]) -> str:
        """
        根據對話上下文擴展查詢
        
        Args:
            query: 原始查詢
            context: 對話上下文（如前一個查詢、用戶意圖等）
        """
        if not context:
            return self.expand_query(query)
        
        previous_query = context.get('previous_query', '')
        
        prompt = f"""基於對話上下文擴展查詢。

對話歷史:
{previous_query}

當前查詢: {query}

請將當前查詢擴展為完整的學術問題，考慮對話上下文。
直接輸出擴展結果:"""

        response = self.llm.invoke(prompt)
        return response.content.strip()
```

#### 集成到系統

```python
# 在 hierarchical_rag_system.py

def _hierarchical_retrieval(self, query: str) -> Dict[str, Any]:
    """層級檢索"""
    
    # 添加: 查詢擴展
    original_query = query
    if self.config.get('query_expansion', {}).get('enabled', False):
        query_expander = QueryExpander(self.llm)
        query = query_expander.expand_query(query, min_length=5)
        
        logger.info(f"原始查詢: '{original_query}'")
        logger.info(f"擴展查詢: '{query}'")
    
    # 繼續原有的檢索流程...
    layer1_results = self.hybrid_retriever.hybrid_search(query, ...)
```

**效果**:
```
輸入: "人流預測"
擴展: "使用深度學習方法進行城市公共空間的人流量預測與分析"

結果: 向量更豐富，分數區分度提高
```

---

### 方案 2: 相關性重排序（Reranking）⭐⭐⭐⭐⭐

**概念**: 先用混合檢索找出候選，再用更精細的方法重新排序

#### 實現

```python
class SemanticReranker:
    """語義重排序器"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def rerank(
        self,
        query: str,
        candidates: List[Tuple[Document, float]],
        top_k: int = 10
    ) -> List[Tuple[Document, float]]:
        """
        使用 LLM 對候選結果進行精細化重排序
        
        Args:
            query: 原始查詢
            candidates: 候選結果 [(Document, score), ...]
            top_k: 最終返回數量
            
        Returns:
            重排序後的結果
        """
        logger.info(f"對 {len(candidates)} 個候選進行重排序...")
        
        # 只對前 N 個候選重排（節省時間）
        rerank_count = min(20, len(candidates))
        to_rerank = candidates[:rerank_count]
        rest = candidates[rerank_count:]
        
        # 構建重排序 prompt
        papers_text = ""
        for idx, (doc, score) in enumerate(to_rerank, 1):
            title = doc.metadata.get('title', 'Unknown')
            abstract = doc.page_content[:200]  # 只取前 200 字
            papers_text += f"\n{idx}. 【{title}】\n摘要: {abstract}...\n"
        
        prompt = f"""你是學術論文檢索專家。請根據查詢評估每篇論文的相關性。

查詢: {query}

候選論文:
{papers_text}

任務:
1. 為每篇論文評估相關性分數 (0-100)
2. 考慮標題和摘要與查詢的匹配度
3. 短查詢時，優先考慮核心概念的直接匹配

請以 JSON 格式返回，例如:
{{"1": 85, "2": 72, "3": 90, ...}}

只輸出 JSON，不要其他文字:"""

        response = self.llm.invoke(prompt)
        
        try:
            import json
            import re
            
            # 提取 JSON
            content = response.content.strip()
            json_match = re.search(r'\{[^}]+\}', content)
            if json_match:
                scores_dict = json.loads(json_match.group())
            else:
                logger.warning("無法解析 LLM 回應，使用原始分數")
                return candidates[:top_k]
            
            # 更新分數
            reranked = []
            for idx, (doc, original_score) in enumerate(to_rerank, 1):
                llm_score = scores_dict.get(str(idx), 50) / 100.0  # 轉換到 0-1
                
                # 組合原始分數和 LLM 分數
                combined_score = 0.5 * original_score + 0.5 * llm_score
                reranked.append((doc, combined_score))
            
            # 排序
            reranked.sort(key=lambda x: x[1], reverse=True)
            
            # 合併未重排的部分
            final_results = reranked + rest
            
            logger.info(f"重排序完成，返回 top-{top_k}")
            return final_results[:top_k]
            
        except Exception as e:
            logger.error(f"重排序失敗: {e}")
            return candidates[:top_k]


class CrossEncoderReranker:
    """Cross-Encoder 重排序器（更精確但較慢）"""
    
    def __init__(self):
        # 需要安裝: pip install sentence-transformers
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            self.available = True
        except:
            logger.warning("CrossEncoder 模型不可用")
            self.available = False
    
    def rerank(
        self,
        query: str,
        candidates: List[Tuple[Document, float]],
        top_k: int = 10
    ) -> List[Tuple[Document, float]]:
        """
        使用 Cross-Encoder 重排序
        
        Cross-Encoder 會同時考慮查詢和文檔，更準確
        """
        if not self.available:
            return candidates[:top_k]
        
        # 準備查詢-文檔對
        pairs = []
        for doc, _ in candidates:
            text = doc.metadata.get('title', '') + ' ' + doc.page_content[:500]
            pairs.append([query, text])
        
        # 計算相關性分數
        scores = self.model.predict(pairs)
        
        # 重新排序
        reranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 返回 top-k
        results = [(doc, float(score)) for (doc, _), score in reranked[:top_k]]
        
        logger.info(f"Cross-Encoder 重排序完成")
        return results
```

#### 集成

```python
# 在 hierarchical_rag_system.py

def _hierarchical_retrieval(self, query: str):
    # 原有的混合檢索
    layer1_results = self.hybrid_retriever.hybrid_search(
        query=query,
        k=20,  # 取更多候選
        ...
    )
    
    # 添加: 重排序
    if self.config.get('reranking', {}).get('enabled', False):
        reranker = SemanticReranker(self.llm)
        layer1_results = reranker.rerank(
            query=query,
            candidates=layer1_results,
            top_k=10  # 最終只要 10 個
        )
```

**效果**:
```
混合檢索候選 (20個):
1. 論文A (0.652)
2. 論文B (0.648)
...

重排序後 (10個):
1. 論文B (0.823) ← LLM 認為最相關
2. 論文A (0.751)
3. 論文E (0.689) ← 從第 8 名提升
```

---

### 方案 3: 對比學習分數調整 ⭐⭐⭐⭐

**概念**: 增強分數的區分度，拉大差距

#### 實現

```python
class ScoreNormalizer:
    """分數正規化與增強"""
    
    @staticmethod
    def enhance_contrast(
        results: List[Tuple[Document, float]],
        method: str = 'power',
        power: float = 2.0
    ) -> List[Tuple[Document, float]]:
        """
        增強分數對比度
        
        Args:
            results: 原始結果
            method: 'power' | 'sigmoid' | 'minmax'
            power: 冪次方（method='power' 時使用）
        """
        if not results:
            return results
        
        scores = [score for _, score in results]
        
        if method == 'power':
            # 冪次方增強：高分更高，低分更低
            # score^2 會放大差異
            enhanced_scores = [s ** power for s in scores]
            
        elif method == 'sigmoid':
            # Sigmoid 增強：壓縮中間，拉開兩端
            import numpy as np
            mean_score = np.mean(scores)
            enhanced_scores = [
                1 / (1 + np.exp(-10 * (s - mean_score)))
                for s in scores
            ]
            
        elif method == 'minmax':
            # Min-Max 拉伸到 0-1
            min_score = min(scores)
            max_score = max(scores)
            if max_score > min_score:
                enhanced_scores = [
                    (s - min_score) / (max_score - min_score)
                    for s in scores
                ]
            else:
                enhanced_scores = scores
        
        else:
            return results
        
        # 重新正規化到原始範圍
        if enhanced_scores:
            max_enhanced = max(enhanced_scores)
            if max_enhanced > 0:
                normalized = [s / max_enhanced for s in enhanced_scores]
            else:
                normalized = enhanced_scores
        else:
            normalized = enhanced_scores
        
        # 組合回 Document
        enhanced_results = [
            (doc, norm_score)
            for (doc, _), norm_score in zip(results, normalized)
        ]
        
        logger.info(
            f"分數增強: 原始範圍 [{min(scores):.3f}, {max(scores):.3f}] → "
            f"增強後 [{min(normalized):.3f}, {max(normalized):.3f}]"
        )
        
        return enhanced_results
```

**效果**:
```
原始分數 (區分度低):
1. 0.652
2. 0.648
3. 0.645
4. 0.642

冪次方增強 (power=2):
1. 0.425 (0.652²)
2. 0.420 (0.648²)
3. 0.416 (0.645²)
4. 0.412 (0.642²)

正規化到 0-1:
1. 1.000 ← 最高
2. 0.988
3. 0.978
4. 0.969 ← 差距拉大！
```

---

### 方案 4: 個性化權重調整 ⭐⭐⭐

**概念**: 根據查詢特性動態調整語義/關鍵詞權重

#### 實現

```python
class AdaptiveWeightAdjuster:
    """自適應權重調整器"""
    
    @staticmethod
    def adjust_weights(query: str) -> Tuple[float, float]:
        """
        根據查詢特性調整權重
        
        Returns:
            (semantic_weight, keyword_weight)
        """
        import jieba
        
        # 分詞
        tokens = list(jieba.cut(query))
        word_count = len(tokens)
        
        # 規則 1: 查詢長度
        if word_count <= 3:
            # 短查詢 → 關鍵詞權重高（確保精準匹配）
            semantic_w, keyword_w = 0.3, 0.7
            reason = "短查詢，提高關鍵詞權重"
            
        elif word_count <= 10:
            # 中等查詢 → 平衡
            semantic_w, keyword_w = 0.5, 0.5
            reason = "中等查詢，平衡權重"
            
        else:
            # 長查詢 → 語義權重高（理解複雜意圖）
            semantic_w, keyword_w = 0.7, 0.3
            reason = "長查詢，提高語義權重"
        
        # 規則 2: 是否包含專有名詞
        # 如果包含 CNN, LSTM, YOLO 等，提高關鍵詞權重
        tech_terms = ['CNN', 'LSTM', 'RNN', 'GAN', 'BERT', 'GPT', 'YOLO', 
                      'ResNet', 'Transformer', 'Attention']
        
        if any(term.lower() in query.lower() for term in tech_terms):
            keyword_w += 0.2
            semantic_w -= 0.2
            reason += "；包含專有名詞"
        
        # 規則 3: 是否是問題形式
        if any(q in query for q in ['什麼', '如何', '為什麼', '哪些']):
            semantic_w += 0.1
            keyword_w -= 0.1
            reason += "；問題形式"
        
        # 確保權重和為 1
        total = semantic_w + keyword_w
        semantic_w /= total
        keyword_w /= total
        
        logger.info(
            f"自適應權重: 語義={semantic_w:.2f}, 關鍵詞={keyword_w:.2f} "
            f"({reason})"
        )
        
        return semantic_w, keyword_w
```

#### 集成

```python
# 在 hierarchical_rag_system.py

def _hierarchical_retrieval(self, query: str):
    # 動態調整權重
    if self.config.get('adaptive_weights', {}).get('enabled', False):
        semantic_w, keyword_w = AdaptiveWeightAdjuster.adjust_weights(query)
    else:
        semantic_w = self.config['hybrid_search']['semantic_weight']
        keyword_w = self.config['hybrid_search']['keyword_weight']
    
    # 使用調整後的權重
    layer1_results = self.hybrid_retriever.hybrid_search(
        query=query,
        semantic_weight=semantic_w,
        keyword_weight=keyword_w,
        ...
    )
```

**效果**:
```
查詢: "人流" (2詞)
→ 權重: 語義 0.3, 關鍵詞 0.7 (短查詢)

查詢: "CNN 物件檢測" (3詞 + 專有名詞)
→ 權重: 語義 0.2, 關鍵詞 0.8 (確保 CNN 精準匹配)

查詢: "深度學習在人流預測的應用研究進展" (10詞)
→ 權重: 語義 0.7, 關鍵詞 0.3 (長查詢理解意圖)
```

---

## 📊 方案比較

| 方案 | 實施難度 | 效果 | 成本 | 推薦指數 |
|-----|---------|------|------|---------|
| **查詢擴展** | ⭐⭐ 簡單 | ⭐⭐⭐⭐⭐ | 每次查詢 +0.5s | ⭐⭐⭐⭐⭐ |
| **LLM 重排序** | ⭐⭐⭐ 中等 | ⭐⭐⭐⭐⭐ | 每次查詢 +2s | ⭐⭐⭐⭐ |
| **CrossEncoder** | ⭐⭐⭐⭐ 較難 | ⭐⭐⭐⭐⭐ | 每次查詢 +1s | ⭐⭐⭐⭐ |
| **分數增強** | ⭐ 很簡單 | ⭐⭐⭐ | 可忽略 | ⭐⭐⭐ |
| **自適應權重** | ⭐⭐ 簡單 | ⭐⭐⭐⭐ | 可忽略 | ⭐⭐⭐⭐⭐ |

---

## 🎯 推薦組合方案

### 組合 A: 快速改善（10分鐘）

```python
# 1. 查詢擴展
# 2. 自適應權重調整
# 3. 分數增強

config = {
    'query_expansion': {
        'enabled': True,
        'min_length': 5,  # 少於5詞才擴展
    },
    'adaptive_weights': {
        'enabled': True,
    },
    'score_enhancement': {
        'enabled': True,
        'method': 'power',
        'power': 1.5,
    }
}
```

### 組合 B: 最佳效果（1小時）

```python
# 1. 查詢擴展
# 2. 混合檢索
# 3. LLM 重排序
# 4. 自適應權重

config = {
    'query_expansion': {'enabled': True},
    'hybrid_search': {'enabled': True},
    'reranking': {
        'enabled': True,
        'method': 'llm',  # or 'cross_encoder'
        'rerank_top_n': 20,
    },
    'adaptive_weights': {'enabled': True},
}
```

---

## 🚀 立即實施

我可以為您實現哪個方案？

1. **方案 1: 查詢擴展** - 最快見效，10分鐘完成
2. **方案 2: LLM 重排序** - 效果最好，30分鐘完成
3. **方案 4: 自適應權重** - 簡單有效，15分鐘完成
4. **組合方案** - 最全面，1小時完成

選擇一個方案，我立即為您實現！
