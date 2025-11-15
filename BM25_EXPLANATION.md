# BM25 在系統中的運作方式

## 什麼是 BM25？

BM25 (Best Matching 25) 是一種**基於統計的關鍵詞檢索算法**，用於評估查詢與文檔之間的相關性。

### BM25 vs 向量相似度

| 特性 | BM25（關鍵詞） | 向量相似度（語義） |
|-----|--------------|-----------------|
| **原理** | 詞頻統計 + IDF | 向量空間距離 |
| **優勢** | 精準匹配專有名詞 | 理解語義和概念 |
| **劣勢** | 不理解語義 | 可能忽略關鍵詞 |
| **適用** | 術語查詢（CNN、LSTM） | 概念查詢（深度學習應用） |

## BM25 數學原理

### 公式

```
BM25(D, Q) = Σ IDF(qi) × [ f(qi, D) × (k1 + 1) / (f(qi, D) + k1 × (1 - b + b × |D|/avgdl)) ]
```

### 參數說明

- **D**: 文檔（論文摘要）
- **Q**: 查詢（使用者輸入）
- **qi**: 查詢中的第 i 個詞
- **f(qi, D)**: 詞 qi 在文檔 D 中的詞頻
- **|D|**: 文檔 D 的長度（詞數）
- **avgdl**: 語料庫中文檔的平均長度
- **IDF(qi)**: 逆文檔頻率（Inverse Document Frequency）
- **k1**: 詞頻飽和參數（預設 1.5）
- **b**: 長度標準化參數（預設 0.75）

### IDF 計算

```
IDF(qi) = log( (N - n(qi) + 0.5) / (n(qi) + 0.5) )
```

- **N**: 語料庫中的文檔總數
- **n(qi)**: 包含詞 qi 的文檔數量

**直觀理解**: 
- 如果一個詞在很多文檔中出現 → IDF 低（常見詞，如「的」、「是」）
- 如果一個詞只在少數文檔中出現 → IDF 高（重要詞，如「LSTM」）

## 在系統中的實現

### 1. 初始化階段

```python
# system_api/hybrid_retriever.py
class HybridRetriever:
    def __init__(self, layer1_vectorstore, use_jieba=True):
        self.layer1 = layer1_vectorstore
        self.use_jieba = use_jieba  # 使用 jieba 中文分詞
        self.bm25 = None
        self.documents = []
        
        # 自動建立 BM25 索引
        if layer1_vectorstore.is_initialized:
            self.build_bm25_index()
```

### 2. 建立 BM25 索引

```python
def build_bm25_index(self):
    """建立 BM25 索引的步驟"""
    
    # Step 1: 獲取所有論文文檔
    self.documents = list(self.layer1.vectorstore.docstore._dict.values())
    # 結果: 35 篇論文的 Document 物件
    
    # Step 2: 準備文本語料庫
    tokenized_corpus = []
    for doc in self.documents:
        # 組合標題和摘要
        title = doc.metadata.get('title', '')
        abstract = doc.page_content
        text = f"{title} {abstract}"
        
        # Step 3: 中文分詞
        tokens = self._tokenize(text)
        # 範例: "深度學習在人流預測中的應用"
        # → ["深度學習", "在", "人流", "預測", "中", "的", "應用"]
        # → 過濾短詞後: ["深度學習", "人流", "預測", "應用"]
        
        tokenized_corpus.append(tokens)
    
    # Step 4: 建立 BM25 索引
    from rank_bm25 import BM25Okapi
    self.bm25 = BM25Okapi(tokenized_corpus)
    # BM25Okapi 內部會計算:
    # - 每個詞的 IDF
    # - 文檔平均長度 (avgdl)
    # - 詞頻統計
```

### 3. 中文分詞 (_tokenize)

```python
def _tokenize(self, text: str) -> List[str]:
    """使用 jieba 進行中文分詞"""
    
    if self.use_jieba:
        # 使用 jieba 精確分詞
        tokens = list(jieba.cut(text.lower()))
        # 範例: "使用CNN進行人流預測"
        # → ["使用", "cnn", "進行", "人流", "預測"]
        
        # 過濾短詞和停用詞
        tokens = [t for t in tokens if len(t.strip()) > 1]
        # → ["使用", "cnn", "進行", "人流", "預測"]
    else:
        # 簡單空格分詞（英文）
        tokens = text.lower().split()
    
    return tokens
```

### 4. 混合檢索流程

```python
def hybrid_search(self, query, k=10, semantic_weight=0.5, keyword_weight=0.5):
    """混合搜尋的完整流程"""
    
    # ========== Step 1: 語義搜尋 (FAISS) ==========
    semantic_results = self.layer1.search_with_scores(query, k=50)
    # 使用向量 embedding 計算相似度
    # 結果: [(doc1, 0.65), (doc2, 0.62), ...]
    
    semantic_scores = {
        doc.metadata['paper_id']: score
        for doc, score in semantic_results
    }
    
    # ========== Step 2: BM25 關鍵詞搜尋 ==========
    query_tokens = self._tokenize(query)
    # 範例查詢: "深度學習在人流的應用"
    # → ["深度學習", "人流", "應用"]
    
    bm25_scores = self.bm25.get_scores(query_tokens)
    # BM25 為每篇論文計算分數
    # 計算過程:
    #   1. 查詢詞 "深度學習" 在論文1出現2次 → 計算 f(qi, D)
    #   2. "深度學習" 在35篇中有20篇包含 → 計算 IDF
    #   3. 論文1長度150詞，平均長度120詞 → 長度標準化
    #   4. 套用 BM25 公式 → 得到分數 8.5
    # 結果: [8.5, 3.2, 0.0, 5.1, ...]  (35個分數)
    
    # 正規化 BM25 分數到 [0, 1] 範圍
    max_bm25 = max(bm25_scores)  # 假設 = 12.0
    normalized_bm25_scores = bm25_scores / max_bm25
    # [0.71, 0.27, 0.0, 0.43, ...]
    
    keyword_scores = {
        doc.metadata['paper_id']: score
        for doc, score in zip(self.documents, normalized_bm25_scores)
        if score > 0  # 只保留有分數的
    }
    
    # ========== Step 3: 組合分數 ==========
    all_paper_ids = set(semantic_scores.keys()) | set(keyword_scores.keys())
    # 取聯集：任一方法找到的論文都考慮
    
    combined_results = []
    for paper_id in all_paper_ids:
        sem_score = semantic_scores.get(paper_id, 0.0)  # 語義分數
        key_score = keyword_scores.get(paper_id, 0.0)   # 關鍵詞分數
        
        # 加權組合
        combined_score = (
            semantic_weight * sem_score +
            keyword_weight * key_score
        )
        # 範例: 0.5 * 0.65 + 0.5 * 0.71 = 0.68
        
        combined_results.append((doc, combined_score))
    
    # ========== Step 4: 排序並返回 ==========
    combined_results.sort(key=lambda x: x[1], reverse=True)
    return combined_results[:k]
```

## 實際運作範例

### 範例 1: 短查詢「人流預測」

```
查詢: "人流預測"

1. 分詞: ["人流", "預測"]

2. BM25 計算（論文A）:
   - "人流" 在論文A出現 3 次
   - "預測" 在論文A出現 5 次
   - IDF("人流") = log((35-8+0.5)/(8+0.5)) ≈ 1.17
   - IDF("預測") = log((35-15+0.5)/(15+0.5)) ≈ 0.74
   - 套用公式計算...
   - BM25 分數 = 6.8
   - 正規化後 = 6.8/12.0 = 0.57

3. 語義搜尋:
   - embedding 相似度 = 0.64

4. 組合（權重 0.5/0.5）:
   - 最終分數 = 0.5 × 0.64 + 0.5 × 0.57 = 0.605

5. 與其他論文比較後排序
```

### 範例 2: 專有名詞查詢「CNN應用」

```
查詢: "CNN應用"

論文B（包含CNN）:
- BM25: 0.85 (高，因為精準匹配 "CNN")
- 語義: 0.60 (中等)
- 組合: 0.5 × 0.60 + 0.5 × 0.85 = 0.725 ✓ 排名高

論文C（不包含CNN但語義相關）:
- BM25: 0.0 (無匹配)
- 語義: 0.70 (高)
- 組合: 0.5 × 0.70 + 0.5 × 0.0 = 0.35 ✓ 排名低

結果: 包含 "CNN" 的論文排名更高
```

### 範例 3: 多概念查詢「深度學習在人流的應用」

```
查詢: "深度學習在人流的應用"

論文D（包含兩個概念）:
- 標題: "基於深度學習的城市人流預測研究"
- BM25: 0.75 (同時匹配 "深度學習" 和 "人流")
- 語義: 0.68
- 組合: 0.5 × 0.68 + 0.5 × 0.75 = 0.715 ✓ 高分

論文E（只包含深度學習）:
- 標題: "深度學習在圖像識別中的應用"
- BM25: 0.40 (只匹配 "深度學習")
- 語義: 0.70
- 組合: 0.5 × 0.70 + 0.5 × 0.40 = 0.55 ✓ 中等分數

結果: 同時包含兩個概念的論文得分更高
```

## BM25 的優勢

### 1. 精準匹配專有名詞
```
查詢: "LSTM"
- 純語義: 可能找到 RNN、GRU 等相關但不同的模型
- BM25: 精準找到包含 "LSTM" 的論文
```

### 2. 多概念查詢
```
查詢: "深度學習在人流的應用"
- 純語義: 可能被 "深度學習" 主導，忽略 "人流"
- BM25: 確保兩個關鍵詞都被考慮
```

### 3. 詞頻重要性
```
如果 "人流" 在論文中出現多次
→ BM25 分數更高
→ 表示該論文真的專注於人流研究
```

### 4. 罕見詞權重
```
IDF 機制:
- 常見詞（"方法"、"研究"）→ IDF 低 → 對分數貢獻小
- 罕見詞（"YOLO"、"時空圖"）→ IDF 高 → 對分數貢獻大
```

## BM25 的限制

### 1. 不理解語義
```
查詢: "神經網絡"
- 論文包含: "neural network"
- BM25: 無法匹配（不同語言）
- 語義: 可以匹配（概念相同）
```

### 2. 同義詞問題
```
查詢: "人流"
- 論文用詞: "人群流動"、"行人流量"
- BM25: 無法匹配（詞彙不同）
- 語義: 可以匹配（意思相同）
```

### 3. 依賴分詞品質
```
分詞錯誤:
"深度學習模型" → ["深度", "學習", "模型"] ✓ 正確
"深度學習模型" → ["深", "度學", "習模型"] ✗ 錯誤

→ 使用 jieba 提高中文分詞準確度
```

## 權重調整策略

### 何時提高 BM25 權重？

```python
# 情況 1: 短查詢
query = "CNN"
→ semantic_weight = 0.3, keyword_weight = 0.7
原因: 短查詢語義模糊，需要精準匹配

# 情況 2: 包含專有名詞
query = "使用 LSTM 進行預測"
→ semantic_weight = 0.4, keyword_weight = 0.6
原因: "LSTM" 是專有名詞，需要精準匹配

# 情況 3: 多個關鍵技術
query = "ResNet vs VGG"
→ semantic_weight = 0.3, keyword_weight = 0.7
原因: 需要同時匹配兩個模型名稱
```

### 何時提高語義權重？

```python
# 情況 1: 長查詢
query = "如何使用深度學習方法進行城市人流預測和分析"
→ semantic_weight = 0.7, keyword_weight = 0.3
原因: 長查詢語義豐富，理解意圖更重要

# 情況 2: 概念性查詢
query = "深度學習在計算機視覺的應用"
→ semantic_weight = 0.6, keyword_weight = 0.4
原因: 關注概念關係，不是特定術語

# 情況 3: 問題形式
query = "什麼是遷移學習"
→ semantic_weight = 0.6, keyword_weight = 0.4
原因: 需要理解問題意圖
```

## 系統日誌解讀

### 初始化階段
```
[INFO] 開始建立 BM25 索引...
[INFO] 對 35 篇論文進行分詞...
[INFO] ✓ BM25 索引建立成功：35 篇論文
```
→ 系統已載入 35 篇論文並完成分詞索引

### 檢索階段
```
[INFO] 混合搜尋: query='人流預測', 語義權重=0.70, 關鍵詞權重=0.30
[DEBUG] 語義搜尋找到 35 篇論文
[DEBUG] 關鍵詞搜尋找到 12 篇相關論文
[INFO] 混合搜尋結果: 35 語義 + 12 關鍵詞 → 35 組合 → 10 最終 (top-10)
```

解讀:
- 35 語義: FAISS 返回所有論文（未過濾）
- 12 關鍵詞: BM25 找到 12 篇包含查詢詞的論文
- 35 組合: 對所有論文計算組合分數
- 10 最終: 返回前 10 名

### 分數細節
```
[INFO] 前 5 名論文分數:
  1. [0.7150] (語義:0.6800 + 關鍵詞:0.7500) 基於深度學習的城市人流預測研究
  2. [0.6820] (語義:0.7200 + 關鍵詞:0.6440) 時空特徵建模在人流分析中的應用
  3. [0.6550] (語義:0.6900 + 關鍵詞:0.6200) LSTM 模型在人流預測中的應用
  4. [0.5980] (語義:0.6500 + 關鍵詞:0.5460) 深度學習方法綜述
  5. [0.5720] (語義:0.6200 + 關鍵詞:0.5240) 卷積神經網絡在圖像處理中的應用
```

解讀:
- 第1名: 兩個分數都高 → 最相關
- 第2名: 語義分數最高，但關鍵詞稍低
- 第4名: 語義中等但關鍵詞較低 → 可能只包含 "深度學習"
- 第5名: 關鍵詞分數低 → 可能與 "人流" 無關

## 總結

### BM25 在系統中的角色

```
使用者查詢
    ↓
┌─────────────────────────────────┐
│   混合檢索 (Hybrid Retriever)    │
├─────────────────────────────────┤
│  語義搜尋    │      BM25 搜尋    │
│  (FAISS)    │   (關鍵詞統計)    │
│             │                  │
│  理解概念    │    精準匹配       │
│  語義相關    │    術語匹配       │
│  同義詞     │    詞頻統計       │
└──────┬──────────────┬──────────┘
       │              │
       ↓              ↓
   語義分數       關鍵詞分數
       │              │
       └──────┬───────┘
              ↓
        加權組合分數
              ↓
        排序返回結果
```

### 關鍵要點

1. **BM25 = 統計方法**: 基於詞頻和 IDF，不理解語義
2. **補充語義搜尋**: 解決純向量搜尋的關鍵詞遺漏問題
3. **中文分詞**: 使用 jieba 提高分詞品質
4. **加權組合**: 可根據查詢類型調整權重比例
5. **正規化**: BM25 分數正規化到 [0,1] 以便組合

### 何時最有效？

✅ **短查詢**: "CNN"、"LSTM"  
✅ **專有名詞**: "YOLO"、"ResNet"  
✅ **多概念**: "深度學習在人流的應用"  
✅ **精準匹配**: 需要特定技術/方法的論文  

❌ **語義查詢**: "如何提高模型準確度"（語義權重應更高）  
❌ **同義詞**: "神經網絡" vs "neural network"（語義搜尋更好）  
❌ **概念理解**: "遷移學習的優勢"（語義搜尋更好）
