# Layer 1 資料結構與搜尋機制詳解

## 📚 目錄
1. [資料結構概覽](#資料結構概覽)
2. [Document 物件結構](#document-物件結構)
3. [FAISS 索引結構](#faiss-索引結構)
4. [搜尋流程詳解](#搜尋流程詳解)
5. [距離與相似度轉換](#距離與相似度轉換)
6. [實際範例](#實際範例)

---

## 資料結構概覽

### Layer 1 的定位
```
用戶查詢
    ↓
【Layer 1: 論文級檢索】← 您在這裡！
    ↓ (篩選出相關論文)
【Layer 2: 區塊級檢索】
    ↓
回答生成
```

### 核心組件
```python
class Layer1VectorStore:
    embeddings: OllamaEmbeddings      # embedding 模型
    vectorstore: FAISS                 # FAISS 索引 (核心)
    vectorstore_path: str              # 儲存路徑
    _paper_count: int                  # 論文數量
    
    # 檔案結構
    ./vectorstore/layer1/
        ├── abstract.faiss             # FAISS 索引檔案
        ├── abstract.pkl               # Document 物件
        └── metadata.pkl               # 統計資訊
```

---

## Document 物件結構

### 每篇論文的 Document
```python
Document(
    page_content="論文摘要全文...",     # 向量化的文本
    metadata={
        # === 核心識別 ===
        'paper_id': 'ABC123',           # 唯一 ID (用於 Layer 2 過濾)
        'layer': 1,                      # 標記為 Layer 1 文件
        
        # === 論文資訊 ===
        'title': '論文標題',
        'authors': ['作者1', '作者2'],
        'year': 2024,
        'pdf_path': '/path/to/paper.pdf',
        
        # === 摘要品質 ===
        'abstract_source': 'pdf',       # 來源: pdf/grobid/arxiv
        'abstract_confidence': 0.95,    # 提取信心度 (0-1)
    }
)
```

### 範例：實際的 Document
```python
Document(
    page_content="""
    本研究提出一種基於集成式生成對抗網路的人流異常預測方法。
    通過結合時空卷積神經網路與注意力機制，能夠有效捕捉人流
    的時間與空間依賴性。實驗結果顯示，該方法在準確率和召回
    率上均優於現有基準模型...
    """,
    metadata={
        'paper_id': 'P001',
        'title': '基於集成式生成對抗網路進行人流異常預測',
        'authors': ['張三', '李四'],
        'year': 2023,
        'pdf_path': '/data/papers/P001.pdf',
        'abstract_source': 'pdf',
        'abstract_confidence': 0.92,
        'layer': 1
    }
)
```

---

## FAISS 索引結構

### FAISS 的內部結構
```python
FAISS(
    # === 向量索引 (高維空間) ===
    index: faiss.Index                  # C++ 層的索引結構
        ├── ntotal: int                 # 向量數量 (= 論文數)
        ├── d: int                      # 向量維度 (例如 768)
        └── metric_type: L2             # 距離度量: L2 歐氏距離
    
    # === Document 儲存 ===
    docstore: InMemoryDocstore          # 儲存 Document 物件
        └── _dict: Dict[str, Document]  # {doc_id: Document}
    
    # === ID 映射 ===
    index_to_docstore_id: Dict[int, str]  # {FAISS_idx: doc_id}
)
```

### 向量空間示意圖
```
高維向量空間 (例如 768 維)
    
    📄 P003 (距離=0.85)
        ↗
    📄 P001 (距離=1.35)  
        ↗
    🔍 查詢向量
        ↘
    📄 P002 (距離=1.42)
        ↘
    📄 P004 (距離=2.10)

距離越小 = 越相似
```

### 資料流
```
建立索引:
abstracts (原始資料) 
    → [embed_documents()] 
    → vectors (768 維) 
    → FAISS.add()
    → index 檔案

搜尋:
query (用戶問題)
    → [embed_query()]
    → query_vector (768 維)
    → FAISS.search()
    → [(doc, distance), ...]
```

---

## 搜尋流程詳解

### 1. 完整搜尋流程圖
```
┌─────────────────────────────────────────────────────────┐
│ 1. 用戶查詢                                              │
│    query = "人流預測相關論文"                            │
└────────────┬────────────────────────────────────────────┘
             ↓
┌─────────────────────────────────────────────────────────┐
│ 2. 查詢向量化                                            │
│    query_vector = embeddings.embed_query(query)          │
│    → [0.12, -0.45, 0.78, ..., 0.33]  (768 維)           │
└────────────┬────────────────────────────────────────────┘
             ↓
┌─────────────────────────────────────────────────────────┐
│ 3. FAISS 近似最近鄰搜尋                                  │
│    results = index.search(query_vector, k=fetch_k)       │
│                                                          │
│    返回: [(doc_1, dist_1), (doc_2, dist_2), ...]        │
│                                                          │
│    例如:                                                 │
│    - (P001, 1.3584)  ← 最近                              │
│    - (P002, 1.3617)                                      │
│    - (P003, 1.3946)                                      │
│    - ...                                                 │
│    - (P035, 2.8901)  ← 最遠                              │
└────────────┬────────────────────────────────────────────┘
             ↓
┌─────────────────────────────────────────────────────────┐
│ 4. 距離 → 相似度轉換                                     │
│    for (doc, distance) in results:                       │
│        similarity = 1 / (1 + sqrt(distance))             │
│                                                          │
│    例如:                                                 │
│    - P001: dist=1.3584 → sim=0.4618                      │
│    - P002: dist=1.3617 → sim=0.4615                      │
│    - P003: dist=1.3946 → sim=0.4585                      │
└────────────┬────────────────────────────────────────────┘
             ↓
┌─────────────────────────────────────────────────────────┐
│ 5. 閾值過濾 (如果有設定 score_threshold)                 │
│    filtered = [                                          │
│        (doc, sim) for (doc, sim) in results              │
│        if sim >= score_threshold                         │
│    ]                                                     │
│                                                          │
│    例如閾值 0.3:                                         │
│    - P001: 0.4618 ≥ 0.3  ✓ PASS                          │
│    - P002: 0.4615 ≥ 0.3  ✓ PASS                          │
│                                                          │
│    例如閾值 0.5:                                         │
│    - P001: 0.4618 < 0.5  ✗ FAIL (被過濾!)               │
│    - P002: 0.4615 < 0.5  ✗ FAIL (被過濾!)               │
└────────────┬────────────────────────────────────────────┘
             ↓
┌─────────────────────────────────────────────────────────┐
│ 6. 結果排序 & 限制數量                                   │
│    sorted(filtered, key=lambda x: x[1], reverse=True)    │
│    return results[:k]                                    │
└─────────────────────────────────────────────────────────┘
```

### 2. 程式碼對應

#### search_with_scores() 函數拆解
```python
def search_with_scores(
    self,
    query: str,                      # 用戶問題
    k: int = 15,                     # 最多返回幾篇論文
    score_threshold: Optional[float] = None,  # 相似度閾值
    **kwargs
) -> List[tuple]:                   # 返回 [(Document, similarity), ...]
    
    # ========== 步驟 1: 決定取多少論文 ==========
    if score_threshold is not None:
        # 有閾值 → 取所有論文來過濾
        fetch_k = self._paper_count  # 例如 35 篇
        logger.info(f"使用閾值 {score_threshold:.2f}，取出 {fetch_k} 篇論文")
    else:
        # 無閾值 → 只取 top-k
        fetch_k = k  # 例如 15 篇
    
    # ========== 步驟 2: FAISS 搜尋 ==========
    results = self.vectorstore.similarity_search_with_score(
        query=query,
        k=fetch_k,
        **kwargs
    )
    # 返回: [(Document, L2_distance), ...]
    
    # ========== 步驟 3: 有閾值的情況 ==========
    if score_threshold is not None:
        filtered_results = []
        
        for idx, (doc, distance) in enumerate(results):
            # 步驟 3.1: 距離 → 相似度轉換
            import math
            similarity = 1 / (1 + math.sqrt(distance))
            
            # 步驟 3.2: 記錄前幾篇的詳細資訊
            if idx < 5:
                logger.info(f"  論文 {idx+1}: {doc.metadata.get('title')[:50]}")
                logger.info(f"    距離={distance:.4f}, 相似度={similarity:.4f}")
                logger.info(f"    閾值={score_threshold:.2f}, 通過={similarity >= score_threshold}")
            
            # 步驟 3.3: 閾值過濾
            if similarity >= score_threshold:
                filtered_results.append((doc, similarity))
        
        # 步驟 3.4: 按相似度排序 (高 → 低)
        filtered_results.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(
            f"Layer 1: {len(results)} 篇論文 → "
            f"{len(filtered_results)} 篇通過閾值 {score_threshold:.2f}"
        )
        
        return filtered_results
    
    # ========== 步驟 4: 無閾值的情況 ==========
    else:
        converted_results = []
        for doc, distance in results[:k]:
            similarity = 1 / (1 + math.sqrt(distance))
            converted_results.append((doc, similarity))
        
        return converted_results
```

### 3. 關鍵參數說明

| 參數 | 說明 | 範例 | 影響 |
|------|------|------|------|
| `query` | 用戶問題 | "人流預測" | 影響搜尋結果的相關性 |
| `k` | 最多返回幾篇 | 15 | 無閾值時的上限 |
| `score_threshold` | 相似度閾值 | 0.3 或 0.5 | **決定哪些論文會被過濾** |
| `fetch_k` | 實際搜尋數量 | 35 (有閾值) / 15 (無閾值) | 越大越全面但越慢 |

---

## 距離與相似度轉換

### 核心公式
```python
similarity = 1 / (1 + sqrt(distance))
```

### 數學原理
```
FAISS 返回: L2 歐氏距離 (distance)
- 範圍: [0, +∞)
- 越小越相似 (0 = 完全相同)

轉換成: 相似度分數 (similarity)
- 範圍: (0, 1]
- 越大越相似 (1 = 完全相同)

為什麼用 sqrt()?
- 距離是平方和的平方根: d = sqrt(Σ(xi - yi)²)
- 取 sqrt() 可以減緩距離增長速度
- 使分數分佈更均勻
```

### 轉換對照表
```
Distance  |  sqrt(dist)  |  Similarity  |  說明
----------|--------------|--------------|------------------
  0.00    |    0.00      |    1.0000    |  完全相同
  0.25    |    0.50      |    0.6667    |  非常相似
  1.00    |    1.00      |    0.5000    |  中等相似 (閾值常設在這)
  1.35    |    1.16      |    0.4618    |  略低於閾值 ← 您的論文!
  2.00    |    1.41      |    0.4142    |  較不相似
  4.00    |    2.00      |    0.3333    |  不太相似
  9.00    |    3.00      |    0.2500    |  很不相似
```

### 視覺化
```
相似度分數分佈:

1.0 ┤ 完全相同
    │
0.8 ┤ 非常相關
    │
0.6 ┤
    ├─── 0.5 閾值 ───────────  ← 常用閾值
0.4 ┤       ↓ 您的論文 (0.46)   ← 被過濾了!
    │
0.2 ┤ 不相關
    │
0.0 ┴
```

### 為什麼您的論文被過濾?
```python
# 您的人流論文實際距離
papers = [
    ('人流異常預測', 1.3584),
    ('人群分布生成', 1.3617),
    ('捷運人數預測', 1.3946),
    ('人流強健預測', 1.4240),
    ('大區域人流',   1.4279),
]

# 轉換成相似度
for title, dist in papers:
    sim = 1 / (1 + math.sqrt(dist))
    print(f"{title}: dist={dist:.2f} → sim={sim:.4f}")

# 輸出:
# 人流異常預測: dist=1.36 → sim=0.4618
# 人群分布生成: dist=1.36 → sim=0.4615
# 捷運人數預測: dist=1.39 → sim=0.4585
# 人流強健預測: dist=1.42 → sim=0.4559
# 大區域人流:   dist=1.43 → sim=0.4556

# 問題: 所有論文都在 0.45-0.46 之間
# 如果閾值 = 0.5  →  全部被過濾 ✗
# 如果閾值 = 0.3  →  全部通過   ✓
```

---

## 實際範例

### 範例 1: 無閾值搜尋
```python
layer1 = Layer1VectorStore(embeddings)
layer1.load()

results = layer1.search(
    query="人流預測",
    k=5  # 只要 top-5
    # 沒有 score_threshold
)

# 返回: [Document, Document, ...]
# 不做過濾，直接返回前 5 篇最相似的論文
```

### 範例 2: 有閾值搜尋 (正確配置)
```python
results_with_scores = layer1.search_with_scores(
    query="人流預測",
    k=10,
    score_threshold=0.3  # ← 正確的閾值!
)

# 過程:
# 1. 取出所有 35 篇論文
# 2. 計算每篇的相似度
# 3. 過濾: 只保留 similarity ≥ 0.3
# 4. 排序: 從高到低
# 5. 返回: [(Document, 0.4618), (Document, 0.4615), ...]

# 結果: 5 篇人流論文都會通過! ✓
```

### 範例 3: 有閾值搜尋 (問題配置)
```python
results_with_scores = layer1.search_with_scores(
    query="人流預測",
    k=10,
    score_threshold=0.5  # ← 閾值太高!
)

# 過程相同，但:
# 3. 過濾: 只保留 similarity ≥ 0.5
#    → 人流論文 (0.46) < 0.5  ✗ 被過濾!

# 結果: 返回 [] 空列表
```

### 範例 4: 檢視實際分數
```python
# 取得所有論文來診斷
all_results = layer1.search_with_scores(
    query="人流預測",
    k=35,  # 取所有
    score_threshold=None  # 不過濾
)

# 列印每篇論文的分數
for idx, (doc, similarity) in enumerate(all_results, 1):
    title = doc.metadata.get('title', 'Unknown')
    print(f"{idx}. [{similarity:.4f}] {title}")

# 輸出範例:
# 1. [0.4618] 基於集成式生成對抗網路進行人流異常預測
# 2. [0.4615] 基於3D-RCL與條件式生成對抗網路...
# 3. [0.4585] 基於人流資料、土地使用分區圖...
# ...
# 35. [0.2891] 某個不太相關的論文
```

---

## 🔧 常見問題與調整

### Q1: 為什麼我的相關論文被過濾掉?
```python
# 檢查步驟:
1. 確認論文確實存在於 Layer 1
   stats = layer1.get_stats()
   print(f"共有 {stats['paper_count']} 篇論文")

2. 檢查實際相似度分數
   results = layer1.search_with_scores(query, k=50, score_threshold=None)
   # 找到您的論文，看它的分數

3. 對比閾值設定
   config['layer1']['similarity_threshold']  # 應該 ≤ 論文分數
```

### Q2: 如何調整閾值?
```python
# 在 agent2.py
config = {
    'layer1': {
        'similarity_threshold': 0.3,  # 降低到 0.3 或更低
    }
}

# 或者不用閾值，改用 top-k
config = {
    'layer1': {
        'k_documents': 15,  # 直接取前 15 篇
        # 不設 similarity_threshold
    }
}
```

### Q3: 如何提升相似度分數?
```python
# 方法 1: 改善摘要品質
# - 確保摘要完整提取
# - 使用 GROBID 而非簡單 PDF 解析

# 方法 2: 使用更好的 embedding 模型
embeddings = OllamaEmbeddings(
    model="bge-large-zh-v1.5"  # 中文優化模型
)

# 方法 3: 增加摘要長度 (保留更多資訊)
# 在 abstract_extractor.py 調整
```

### Q4: 搜尋太慢怎麼辦?
```python
# 如果有閾值，會搜尋所有論文 (fetch_k=總數)
# 解決方案:

# 方案 1: 使用 top-k 而非閾值
results = layer1.search(query, k=15)  # 只搜尋 15 篇

# 方案 2: 使用 HNSW 索引 (更快的近似搜尋)
# 在建立索引時改用 FAISS.IndexHNSWFlat

# 方案 3: 減少論文數量 (分類索引)
# 例如: 按領域建立多個 Layer 1 索引
```

---

## 📊 效能統計

### 搜尋時間 (35 篇論文)
```
無閾值 (k=15):           ~50ms   (只計算 15 篇)
有閾值 (threshold=0.3):  ~120ms  (計算所有 35 篇)
```

### 記憶體使用
```
FAISS 索引:     ~5MB  (35 papers × 768 dims × 4 bytes)
Document 物件:  ~2MB  (摘要文本 + metadata)
總計:           ~7MB
```

### 向量維度影響
```
模型              | 維度  | 索引大小 | 搜尋速度
------------------|-------|----------|----------
bge-base-zh       | 768   | 標準     | 標準
bge-large-zh      | 1024  | 較大     | 較慢
bge-small-zh      | 512   | 較小     | 較快
all-MiniLM-L6-v2  | 384   | 最小     | 最快
```

---

## 🎯 最佳實踐建議

### 1. 閾值設定策略
```python
# 根據論文數量調整
if paper_count < 50:
    threshold = 0.3  # 論文少，閾值可低一點
elif paper_count < 200:
    threshold = 0.4  # 中等數量
else:
    threshold = 0.5  # 論文多，提高精確度

# 或者根據用途
if use_case == "查全":
    threshold = 0.3  # 寧可多找，不要遺漏
elif use_case == "查準":
    threshold = 0.5  # 只要高度相關的
```

### 2. 診斷工具
```python
# 建立診斷腳本
def diagnose_layer1():
    layer1 = Layer1VectorStore(embeddings)
    layer1.load()
    
    queries = ["人流預測", "深度學習", "時間序列"]
    
    for query in queries:
        print(f"\n查詢: {query}")
        results = layer1.search_with_scores(query, k=10, score_threshold=None)
        
        for idx, (doc, sim) in enumerate(results, 1):
            print(f"  {idx}. [{sim:.4f}] {doc.metadata['title'][:50]}")
```

### 3. 監控與日誌
```python
# 記錄每次搜尋的統計
logger.info(f"查詢: {query[:50]}")
logger.info(f"找到: {len(results)} 篇論文")
logger.info(f"平均相似度: {np.mean([s for _, s in results]):.4f}")
logger.info(f"最高相似度: {max([s for _, s in results]):.4f}")
logger.info(f"最低相似度: {min([s for _, s in results]):.4f}")
```

---

## 📝 總結

### Layer 1 的本質
- **一篇論文 = 一個向量**
- **向量由摘要產生**
- **FAISS 用距離找最近的論文**
- **距離需轉換成相似度**
- **閾值過濾決定哪些論文進入 Layer 2**

### 關鍵理解
1. **FAISS 返回距離**，不是相似度 (需要轉換)
2. **閾值是相似度**，不是距離 (0-1 範圍)
3. **有閾值會搜尋全部論文** (確保不遺漏)
4. **無閾值只搜尋 top-k** (更快)
5. **距離 1.0 ≈ 相似度 0.5** (常用閾值點)

### 您的問題根源
```
配置顯示: threshold = 0.3  ✓
實際使用: threshold = 0.5  ✗ (需修正)
論文分數: 0.45-0.46
結果:     0.46 < 0.5 → 被過濾
```

### 解決方案
1. 確保 `hierarchical_rag_system.py` 正確讀取 `0.3`
2. 檢查 `HIERARCHICAL_RAG_CONFIG` 預設值
3. 驗證配置合併邏輯 (`config.update()`)
4. 重啟系統並查看日誌確認 `"Using threshold 0.30"`
