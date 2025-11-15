# Layer 1 搜索流程詳解（混合檢索模式）

## 📋 目錄
1. [完整搜索流程](#完整搜索流程)
2. [混合檢索詳細步驟](#混合檢索詳細步驟)
3. [代碼追蹤](#代碼追蹤)
4. [實際運行範例](#實際運行範例)

---

## 完整搜索流程

### 高層視圖

```
用戶查詢: "深度學習在人流的應用是什麼"
    ↓
┌─────────────────────────────────────────────────────┐
│ HierarchicalRAGSystem._hierarchical_retrieval()      │
│ (system_api/hierarchical_rag_system.py:390)         │
└──────────────────┬──────────────────────────────────┘
                   ↓
    檢查配置: hybrid_search.enabled = True?
                   ↓
            ┌──────┴──────┐
            │  是         │  否
            ↓             ↓
┌────────────────────┐  ┌─────────────────────┐
│ 混合檢索模式       │  │ 純語義搜尋模式      │
│ (Hybrid Search)    │  │ (Semantic Search)   │
└──────┬─────────────┘  └──────┬──────────────┘
       ↓                        ↓
┌────────────────────┐  ┌─────────────────────┐
│ HybridRetriever    │  │ Layer1VectorStore   │
│ .hybrid_search()   │  │ .search_with_scores()│
└────────────────────┘  └─────────────────────┘
```

### 當前配置

根據您的 `agent2.py`：

```python
config = {
    'hybrid_search': {
        'enabled': True,           # ✓ 啟用混合檢索
        'semantic_weight': 0.5,    # 語義權重 50%
        'keyword_weight': 0.5,     # 關鍵詞權重 50%
        'use_jieba': True,         # 使用 jieba 中文分詞
    }
}
```

**結論**: 目前系統使用 **混合檢索模式**

---

## 混合檢索詳細步驟

### 步驟流程圖

```
┌──────────────────────────────────────────────────────────────┐
│ 步驟 1: 接收查詢                                              │
│ HybridRetriever.hybrid_search()                              │
│ (system_api/hybrid_retriever.py:121)                         │
└────────────────────────┬─────────────────────────────────────┘
                         ↓
         query = "深度學習在人流的應用是什麼"
         k = 10
         semantic_weight = 0.5
         keyword_weight = 0.5
                         ↓
┌──────────────────────────────────────────────────────────────┐
│ 步驟 2A: 語義搜尋路徑                                         │
└────────────────────────┬─────────────────────────────────────┘
                         ↓
    self.layer1.search_with_scores(query, k=50)
                         ↓
         ┌───────────────────────────┐
         │ 查詢向量化                │
         │ embeddings.embed_query()  │
         └─────────┬─────────────────┘
                   ↓
    query_vector = [0.12, -0.45, 0.78, ..., 0.33]  (768維)
                   ↓
         ┌───────────────────────────┐
         │ FAISS 相似度搜尋          │
         │ vectorstore.similarity_   │
         │ search_with_score()       │
         └─────────┬─────────────────┘
                   ↓
    返回: [(論文A, 距離1.35), (論文B, 距離1.42), ...]
                   ↓
         ┌───────────────────────────┐
         │ 距離 → 相似度轉換         │
         │ sim = 1/(1+sqrt(dist))    │
         └─────────┬─────────────────┘
                   ↓
    semantic_scores = {
        'P001': 0.4618,  # 論文A
        'P002': 0.4556,  # 論文B
        'P003': 0.6234,  # 論文C
        ...
    }
                         ↓
┌──────────────────────────────────────────────────────────────┐
│ 步驟 2B: 關鍵詞搜尋路徑（同時進行）                           │
└────────────────────────┬─────────────────────────────────────┘
                         ↓
    query_tokens = self._tokenize(query)
                         ↓
         ┌───────────────────────────┐
         │ jieba 中文分詞            │
         │ jieba.cut(query)          │
         └─────────┬─────────────────┘
                   ↓
    tokens = ["深度", "學習", "人流", "應用", "什麼"]
                   ↓
         ┌───────────────────────────┐
         │ BM25 評分                 │
         │ bm25.get_scores(tokens)   │
         └─────────┬─────────────────┘
                   ↓
    計算每篇論文與查詢的詞頻匹配度
    考慮: TF (詞頻), IDF (逆文檔頻率), 文檔長度
                   ↓
    bm25_raw_scores = [12.5, 8.3, 15.2, ...]  (35篇論文)
                   ↓
         ┌───────────────────────────┐
         │ 正規化到 0-1              │
         │ scores / max(scores)      │
         └─────────┬─────────────────┘
                   ↓
    keyword_scores = {
        'P001': 0.6543,  # 論文A (包含「深度學習」和「人流」)
        'P002': 0.3210,  # 論文B (只包含「深度學習」)
        'P003': 0.7891,  # 論文C (包含所有關鍵詞)
        ...
    }
                         ↓
┌──────────────────────────────────────────────────────────────┐
│ 步驟 3: 分數組合                                              │
│ (hybrid_retriever.py:177)                                    │
└────────────────────────┬─────────────────────────────────────┘
                         ↓
    for paper_id in all_paper_ids:
        sem_score = semantic_scores.get(paper_id, 0.0)
        key_score = keyword_scores.get(paper_id, 0.0)
        
        combined_score = (
            0.5 × sem_score +    # 語義權重 × 語義分數
            0.5 × key_score      # 關鍵詞權重 × 關鍵詞分數
        )
                         ↓
    範例計算:
    
    論文A (包含「深度學習」和「人流」):
      - 語義分數: 0.4618
      - 關鍵詞分數: 0.6543
      - 組合分數: 0.5×0.4618 + 0.5×0.6543 = 0.5581  ← 較高！
    
    論文B (只包含「深度學習」):
      - 語義分數: 0.7200  (語義高！)
      - 關鍵詞分數: 0.1234  (缺少「人流」關鍵詞)
      - 組合分數: 0.5×0.7200 + 0.5×0.1234 = 0.4217  ← 較低
    
    論文C (包含所有概念):
      - 語義分數: 0.6234
      - 關鍵詞分數: 0.7891
      - 組合分數: 0.5×0.6234 + 0.5×0.7891 = 0.7063  ← 最高！
                         ↓
    combined_results = [
        (論文C, 0.7063),
        (論文A, 0.5581),
        (論文B, 0.4217),
        ...
    ]
                         ↓
┌──────────────────────────────────────────────────────────────┐
│ 步驟 4: 排序與閾值過濾                                        │
│ (hybrid_retriever.py:194)                                    │
└────────────────────────┬─────────────────────────────────────┘
                         ↓
    # 按組合分數排序（高到低）
    combined_results.sort(key=lambda x: x[1], reverse=True)
                         ↓
    # 應用閾值過濾（如果設定）
    if score_threshold:
        filtered = [
            (doc, score) for (doc, score) in combined_results
            if score >= score_threshold
        ]
                         ↓
    # 取 top-k
    final_results = combined_results[:k]
                         ↓
┌──────────────────────────────────────────────────────────────┐
│ 步驟 5: 返回結果                                              │
└────────────────────────┬─────────────────────────────────────┘
                         ↓
    返回給 HierarchicalRAGSystem:
    
    [
        (Document(論文C), 0.7063),
        (Document(論文A), 0.5581),
        (Document(論文D), 0.5234),
        ...
        (Document(論文J), 0.4123),  # 第 10 名
    ]
                         ↓
┌──────────────────────────────────────────────────────────────┐
│ 步驟 6: 後續處理                                              │
│ (hierarchical_rag_system.py:446)                             │
└────────────────────────┬─────────────────────────────────────┘
                         ↓
    - 提取 Document 物件
    - 記錄分數和時間
    - 進行信心度評估
    - 決定是否進入 Layer 2
```

---

## 代碼追蹤

### 1. 入口點：`hierarchical_rag_system.py`

```python
# Line 390-453
def _hierarchical_retrieval(self, query: str):
    """層級檢索的主要邏輯"""
    
    # 步驟 1: 讀取配置
    k1 = self.config['layer1']['k_documents']  # 10
    similarity_threshold = self.config['layer1'].get('similarity_threshold')  # 0.48
    
    # 步驟 2: 判斷使用哪種檢索模式
    if self.hybrid_retriever and self.config['hybrid_search']['enabled']:
        # ✓ 進入這個分支（混合檢索）
        logger.info("使用混合檢索 (Hybrid Search: 語義 + 關鍵詞)")
        
        hybrid_config = self.config['hybrid_search']
        
        # 步驟 3: 調用混合檢索
        layer1_results_with_scores = self.hybrid_retriever.hybrid_search(
            query=query,
            k=k1,                                               # 10
            semantic_weight=hybrid_config['semantic_weight'],  # 0.5
            keyword_weight=hybrid_config['keyword_weight'],    # 0.5
            score_threshold=similarity_threshold,              # 0.48
            return_scores_breakdown=False
        )
    else:
        # 純語義搜尋（目前不會執行）
        layer1_results_with_scores = self.layer1.search_with_scores(...)
    
    # 步驟 4: 提取結果
    layer1_docs = [doc for doc, score in layer1_results_with_scores]
```

### 2. 混合檢索核心：`hybrid_retriever.py`

```python
# Line 121-213
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
    """混合搜尋主函數"""
    
    # ===== 語義搜尋部分 =====
    # Line 146-162
    semantic_results = self.layer1.search_with_scores(
        query=query,
        k=min(50, len(self.documents)),  # 取更多候選
        score_threshold=None,             # 先不過濾
        **kwargs
    )
    
    # 建立 paper_id → semantic_score 映射
    semantic_scores = {
        doc.metadata['paper_id']: score
        for doc, score in semantic_results
    }
    
    # ===== 關鍵詞搜尋部分 =====
    # Line 168-178
    # 分詞
    query_tokens = self._tokenize(query)  # 使用 jieba
    
    # BM25 評分
    bm25_scores = self.bm25.get_scores(query_tokens)
    
    # 正規化到 0-1
    max_bm25_score = max(bm25_scores)
    if max_bm25_score > 0:
        normalized_bm25_scores = bm25_scores / max_bm25_score
    
    # 建立 paper_id → keyword_score 映射
    keyword_scores = {
        doc.metadata['paper_id']: score
        for doc, score in zip(self.documents, normalized_bm25_scores)
        if score > 0
    }
    
    # ===== 組合分數 =====
    # Line 185-197
    all_paper_ids = set(semantic_scores.keys()) | set(keyword_scores.keys())
    combined_results = []
    
    for paper_id in all_paper_ids:
        sem_score = semantic_scores.get(paper_id, 0.0)
        key_score = keyword_scores.get(paper_id, 0.0)
        
        # 加權組合
        combined_score = (
            semantic_weight * sem_score +
            keyword_weight * key_score
        )
        
        # 閾值過濾
        if score_threshold is None or combined_score >= score_threshold:
            doc = self.layer1.get_paper_by_id(paper_id)
            if doc:
                combined_results.append((doc, combined_score))
    
    # ===== 排序並返回 =====
    # Line 200-203
    combined_results.sort(key=lambda x: x[1], reverse=True)
    final_results = combined_results[:k]
    
    return final_results
```

### 3. 分詞函數：`hybrid_retriever.py`

```python
# Line 63-78
def _tokenize(self, text: str) -> List[str]:
    """文本分詞"""
    if self.use_jieba:
        # 使用 jieba 分詞（中文）
        tokens = list(jieba.cut(text.lower()))
        # 過濾停用詞和短詞
        tokens = [t for t in tokens if len(t.strip()) > 1]
    else:
        # 簡單空格分詞（英文）
        tokens = text.lower().split()
    
    return tokens

# 範例:
# Input:  "深度學習在人流的應用是什麼"
# Output: ["深度", "學習", "人流", "應用", "什麼"]
```

### 4. BM25 索引建立：`hybrid_retriever.py`

```python
# Line 80-119
def build_bm25_index(self) -> bool:
    """建立 BM25 索引"""
    # 獲取所有文檔
    self.documents = list(self.layer1.vectorstore.docstore._dict.values())
    
    # 對每篇論文進行分詞
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
    
    logger.info(f"✓ BM25 索引建立成功：{len(self.documents)} 篇論文")
    return True

# 範例:
# 論文A: ["深度", "學習", "圖像", "識別", "CNN", ...]
# 論文B: ["深度", "學習", "人流", "預測", "LSTM", ...]
# 論文C: ["時間", "序列", "分析", "統計", "方法", ...]
```

---

## 實際運行範例

### 啟動日誌分析

從您的啟動日誌可以看到：

```
INFO:system_api.hierarchical_rag_system:初始化混合檢索器 (Hybrid Retriever)...
INFO:system_api.hierarchical_rag_system:  語義權重: 0.50
INFO:system_api.hierarchical_rag_system:  關鍵詞權重: 0.50
INFO:system_api.hierarchical_rag_system:  中文分詞: 啟用
INFO:system_api.hybrid_retriever:開始建立 BM25 索引...
INFO:system_api.hybrid_retriever:對 35 篇論文進行分詞...
Building prefix dict from the default dictionary ...
Loading model cost 0.195 seconds.
Prefix dict has been built successfully.
INFO:system_api.hybrid_retriever:✓ BM25 索引建立成功：35 篇論文
INFO:system_api.hierarchical_rag_system:✓ 混合檢索器初始化成功
```

**解讀**:
1. ✅ 混合檢索器成功初始化
2. ✅ BM25 索引建立完成（35 篇論文）
3. ✅ jieba 分詞已載入
4. ✅ 系統準備就緒

### 模擬查詢流程

假設用戶查詢：**"深度學習在人流的應用是什麼"**

#### 1. 語義搜尋結果（假設）

```python
semantic_scores = {
    'P001': 0.7200,  # "深度學習在圖像識別的應用" (只匹配「深度學習」)
    'P005': 0.4618,  # "基於CNN的人流預測" (匹配兩者但語義分數中等)
    'P012': 0.6543,  # "神經網路在智慧城市的應用"
    'P018': 0.4556,  # "LSTM在人流分析的應用" (匹配兩者)
    'P023': 0.3891,  # "人群密度估計方法研究"
    ...
}
```

#### 2. 關鍵詞搜尋結果（假設）

```python
# 查詢分詞: ["深度", "學習", "人流", "應用", "什麼"]

keyword_scores = {
    'P001': 0.3210,  # 包含「深度」「學習」「應用」，缺「人流」
    'P005': 0.7891,  # 包含「深度」「學習」「人流」「應用」 ✓✓✓
    'P012': 0.4567,  # 包含「應用」
    'P018': 0.8234,  # 包含「人流」「應用」「學習」 ✓✓
    'P023': 0.6123,  # 包含「人流」
    ...
}
```

#### 3. 組合分數計算

```python
# P001: 深度學習在圖像識別的應用
combined = 0.5 × 0.7200 + 0.5 × 0.3210 = 0.5205

# P005: 基於CNN的人流預測
combined = 0.5 × 0.4618 + 0.5 × 0.7891 = 0.6255  ← 排名上升！

# P012: 神經網路在智慧城市的應用
combined = 0.5 × 0.6543 + 0.5 × 0.4567 = 0.5555

# P018: LSTM在人流分析的應用
combined = 0.5 × 0.4556 + 0.5 × 0.8234 = 0.6395  ← 排名最高！

# P023: 人群密度估計方法研究
combined = 0.5 × 0.3891 + 0.5 × 0.6123 = 0.5007
```

#### 4. 最終排序

```
排名 | 論文 | 組合分數 | 語義 | 關鍵詞 | 說明
-----|------|---------|------|--------|------
1    | P018 | 0.6395  | 0.46 | 0.82   | 包含所有關鍵概念 ✓
2    | P005 | 0.6255  | 0.46 | 0.79   | CNN+人流預測 ✓
3    | P012 | 0.5555  | 0.65 | 0.46   | 應用相關
4    | P001 | 0.5205  | 0.72 | 0.32   | 只有深度學習
5    | P023 | 0.5007  | 0.39 | 0.61   | 人流相關
```

**關鍵觀察**:
- 純語義搜尋中排第 1 的 P001（分數 0.72）降到第 4 名
- 同時匹配「深度學習」和「人流」的論文（P018, P005）排名上升
- **混合檢索成功解決了多概念查詢問題！**

---

## 關鍵特性

### 1. 互補性

| 搜尋方式 | 優勢 | 劣勢 |
|---------|------|------|
| **語義搜尋** | 理解概念、同義詞、語義關係 | 可能被主導概念綁架 |
| **關鍵詞搜尋** | 精準詞彙匹配、多概念AND邏輯 | 無法理解同義詞和語義 |
| **混合檢索** | 兼具兩者優勢 | 需要調整權重 |

### 2. 權重影響

```python
# 語義權重高 (0.7/0.3) - 適合概念查詢
"深度學習的發展趨勢" → 更重視語義理解

# 平衡權重 (0.5/0.5) - 適合多概念查詢 ⭐
"深度學習在人流的應用" → 兩者並重

# 關鍵詞權重高 (0.3/0.7) - 適合精準查詢
"YOLOv8 物件檢測" → 確保專有名詞匹配
```

### 3. BM25 演算法

```
BM25 分數 = IDF × (TF × (k1 + 1)) / (TF + k1 × (1 - b + b × (文檔長度/平均文檔長度)))

其中:
- TF: 詞頻（詞在文檔中出現次數）
- IDF: 逆文檔頻率 log((N - n + 0.5) / (n + 0.5))
- N: 總文檔數
- n: 包含該詞的文檔數
- k1, b: 調整參數
```

**效果**: 常見詞（如「方法」）權重低，罕見詞（如「人流」）權重高

---

## 對比圖表

### 純語義 vs 混合檢索

```
查詢: "深度學習在人流的應用是什麼"

純語義搜尋:
    深度學習論文 ████████████████ 72%
    人流論文     ████████░░░░░░░░ 46%  ← 排名低
    綜合論文     ███████████░░░░░ 65%

混合檢索:
    深度學習論文 █████████░░░░░░░ 52%  ← 降低
    人流論文     █████████████░░░ 63%  ← 提升！
    綜合論文     ██████████████░░ 64%

結果: 相關論文從第 8-12 名提升到第 1-3 名
```

---

## 總結

### 當前 Layer 1 搜索方式

✅ **使用混合檢索（Hybrid Search）**

**流程**: 
1. 接收查詢
2. **並行執行**:
   - 語義搜尋（FAISS 向量相似度）
   - 關鍵詞搜尋（BM25 詞頻匹配）
3. 加權組合分數（0.5 × 語義 + 0.5 × 關鍵詞）
4. 排序並返回 top-k

**優勢**:
- ✓ 解決多概念查詢問題
- ✓ 確保次要概念不被忽略
- ✓ 提高檢索精準度
- ✓ 適合中文查詢（jieba 分詞）

**代碼路徑**:
```
agent2.py
  → HierarchicalRAGSystem._hierarchical_retrieval()
    → HybridRetriever.hybrid_search()
      → [Layer1VectorStore.search_with_scores() + BM25.get_scores()]
        → 組合分數 → 排序 → 返回
```

需要更詳細解釋哪個部分嗎？
