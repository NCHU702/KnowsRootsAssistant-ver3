# 混合檢索（Hybrid Search）使用指南

## 🎯 什麼是混合檢索？

混合檢索結合了兩種互補的搜尋方式：

1. **語義搜尋（Semantic Search）**
   - 使用向量 embeddings
   - 理解查詢的「意義」
   - 可以找到語義相關但用詞不同的內容

2. **關鍵詞搜尋（Keyword Search - BM25）**
   - 基於詞頻和逆文檔頻率
   - 確保關鍵詞精準匹配
   - 適合多概念查詢

## 🔧 已完成的實現

### 1. 核心模組
- ✅ `system_api/hybrid_retriever.py` - 混合檢索器
- ✅ `system_api/hierarchical_rag_system.py` - 集成混合檢索
- ✅ `agent2.py` - 配置更新

### 2. 依賴庫
- ✅ `rank-bm25` - BM25 演算法
- ✅ `jieba` - 中文分詞

### 3. 配置參數
```python
'hybrid_search': {
    'enabled': True,           # 啟用/停用
    'semantic_weight': 0.5,    # 語義權重 (0-1)
    'keyword_weight': 0.5,     # 關鍵詞權重 (0-1)
    'use_jieba': True,         # 中文分詞
}
```

## 🚀 使用方式

### 方法 1: 自動使用（推薦）

系統已經配置為自動使用混合檢索，不需要修改代碼：

```bash
# 直接啟動，系統會自動使用混合檢索
python agent2.py
```

系統會在啟動時：
1. 載入 Layer 1 索引
2. 建立 BM25 索引（自動）
3. 在查詢時使用混合檢索

### 方法 2: 手動控制

如果需要停用混合檢索：

```python
# 在 agent2.py 中修改
config={
    'hybrid_search': {
        'enabled': False,  # 設為 False 停用
    }
}
```

### 方法 3: 調整權重

根據需求調整語義和關鍵詞的權重：

```python
config={
    'hybrid_search': {
        'enabled': True,
        'semantic_weight': 0.6,  # 語義權重高 → 更重視概念理解
        'keyword_weight': 0.4,   # 關鍵詞權重低
    }
}

# 或者

config={
    'hybrid_search': {
        'enabled': True,
        'semantic_weight': 0.3,  # 語義權重低
        'keyword_weight': 0.7,   # 關鍵詞權重高 → 更精準匹配
    }
}
```

## 📊 測試效果

### 測試工具

```bash
# 測試混合檢索效果
python test_hybrid_search.py
```

這會測試多個查詢，比較：
- 純語義搜尋結果
- 混合檢索結果
- 兩者差異

### 測試查詢範例

```python
test_queries = [
    "深度學習在人流的應用是什麼",      # 多概念查詢
    "如何使用神經網路預測人群密度",     # 方法 + 應用
    "LSTM 模型在時間序列預測的優勢",   # 模型 + 任務
    "捷運站的人流分析方法",            # 場景 + 方法
]
```

### 預期改善

**問題查詢**: "深度學習在人流的應用是什麼"

**純語義搜尋**:
```
1. [0.72] 深度學習在圖像識別的應用  ← 只匹配「深度學習」
2. [0.68] 卷積神經網路綜述
3. [0.65] 強化學習方法研究
4. [0.58] 基於CNN的人流預測  ← 相關論文排名低！
5. [0.55] LSTM在時間序列的應用
```

**混合檢索**:
```
1. [0.68] 基於CNN的人流預測  ← 同時匹配兩個概念！
2. [0.65] 深度學習用於人群密度估計
3. [0.62] 神經網路在人流分析中的應用
4. [0.58] 深度學習在圖像識別的應用
5. [0.55] 人流預測方法綜述
```

## 🎛️ 權重調優

### 自動調優

使用內建的調優功能：

```python
from system_api.hybrid_retriever import HybridRetriever

# 初始化
hybrid = HybridRetriever(layer1)

# 自動找最佳權重
best_weights = hybrid.tune_weights(
    query="深度學習在人流的應用是什麼",
    expected_paper_ids=["P001", "P005", "P012"],  # 期望找到的論文
    k=10
)

print(f"最佳權重: 語義={best_weights[0]}, 關鍵詞={best_weights[1]}")
```

### 手動調優指南

| 場景 | 語義權重 | 關鍵詞權重 | 說明 |
|------|---------|-----------|------|
| **概念查詢** | 0.7 | 0.3 | "深度學習的發展趨勢" |
| **多概念查詢** | 0.5 | 0.5 | "深度學習在人流的應用" ⭐ |
| **精準匹配** | 0.3 | 0.7 | "LSTM-Attention 模型" |
| **專有名詞** | 0.2 | 0.8 | "YOLOv8 物件檢測" |
| **模糊查詢** | 0.8 | 0.2 | "機器學習相關研究" |

## 🔍 工作原理

### 流程圖

```
用戶查詢: "深度學習在人流的應用"
    │
    ├─────────────────┬─────────────────┐
    │                 │                 │
    ↓                 ↓                 ↓
語義搜尋          關鍵詞搜尋         分詞
embed_query()     BM25              jieba.cut()
    │                 │                 │
    ↓                 ↓                 ↓
向量相似度        詞頻匹配        ["深度學習", "人流", "應用"]
FAISS search      BM25 scores         │
    │                 │                 │
    ↓                 ↓                 ↓
論文A: 0.72       論文A: 0.45         │
論文B: 0.58       論文B: 0.82    ←────┘
論文C: 0.65       論文C: 0.60
    │                 │
    └────────┬────────┘
             ↓
      加權組合 (0.5 × 語義 + 0.5 × 關鍵詞)
             │
             ↓
    論文A: 0.585 (0.5×0.72 + 0.5×0.45)
    論文B: 0.700 (0.5×0.58 + 0.5×0.82)  ← 最高分！
    論文C: 0.625 (0.5×0.65 + 0.5×0.60)
             │
             ↓
       按分數排序返回
```

### 關鍵組件

#### 1. BM25 索引建立
```python
# 在初始化時自動執行
def build_bm25_index(self):
    for doc in documents:
        # 組合標題和摘要
        text = doc.metadata['title'] + ' ' + doc.page_content
        
        # jieba 分詞
        tokens = jieba.cut(text)  # ["深度", "學習", "人流", ...]
        
        # 建立 BM25 索引
        bm25.add_document(tokens)
```

#### 2. 查詢處理
```python
def hybrid_search(query):
    # 語義搜尋
    semantic_scores = layer1.search_with_scores(query)
    
    # 關鍵詞搜尋
    query_tokens = jieba.cut(query)
    keyword_scores = bm25.get_scores(query_tokens)
    
    # 組合分數
    for paper_id in all_papers:
        combined = (
            semantic_weight * semantic_scores[paper_id] +
            keyword_weight * keyword_scores[paper_id]
        )
```

## 📈 效能影響

### 額外開銷

| 項目 | 初始化 | 每次查詢 | 記憶體 |
|-----|--------|---------|--------|
| **BM25 索引** | ~2-3 秒 | 0 秒 | ~5-10 MB |
| **jieba 分詞** | ~0.5 秒 | ~0.01 秒 | ~20 MB |
| **混合搜尋** | 0 秒 | +0.05 秒 | 0 MB |
| **總額外開銷** | ~3 秒 | ~0.06 秒 | ~30 MB |

### 優化建議

如果記憶體不足：
```python
config={
    'hybrid_search': {
        'use_jieba': False,  # 停用 jieba，使用簡單空格分詞
    }
}
```

如果速度太慢：
```python
# 降低 Layer 1 檢索數量
config={
    'layer1': {
        'k_documents': 5,  # 從 10 降到 5
    }
}
```

## 🐛 故障排除

### 問題 1: BM25 索引建立失敗
```
錯誤: 混合檢索器初始化失敗
```

**解決**:
```bash
# 確認依賴已安裝
pip install rank-bm25 jieba

# 重新啟動
python agent2.py
```

### 問題 2: jieba 分詞效果不佳
```
問題: 中文分詞不準確
```

**解決**:
```python
# 添加自定義詞典
import jieba
jieba.load_userdict("custom_dict.txt")

# 格式: 每行一個詞
# 例如: custom_dict.txt
# 深度學習
# 人流預測
# 捷運站
```

### 問題 3: 記憶體不足
```
錯誤: MemoryError
```

**解決**:
```python
# 停用 jieba
config={
    'hybrid_search': {
        'use_jieba': False,
    }
}

# 或完全停用混合檢索
config={
    'hybrid_search': {
        'enabled': False,
    }
}
```

### 問題 4: 結果不如預期
```
問題: 混合檢索結果不好
```

**解決**:
```python
# 調整權重
# 如果關鍵詞不重要，提高語義權重
config={
    'hybrid_search': {
        'semantic_weight': 0.7,
        'keyword_weight': 0.3,
    }
}

# 如果需要精準匹配，提高關鍵詞權重
config={
    'hybrid_search': {
        'semantic_weight': 0.3,
        'keyword_weight': 0.7,
    }
}
```

## 📚 進階功能

### 查看詳細分數

```python
results = hybrid.hybrid_search(
    query="深度學習在人流的應用",
    return_scores_breakdown=True  # 返回詳細分數
)

for doc, (combined, semantic, keyword) in results:
    print(f"{doc.metadata['title']}")
    print(f"  組合分數: {combined:.4f}")
    print(f"  語義分數: {semantic:.4f}")
    print(f"  關鍵詞分數: {keyword:.4f}")
```

### 動態調整權重

```python
# 根據查詢長度調整
query_length = len(query.split())

if query_length <= 3:
    # 短查詢，關鍵詞重要
    semantic_weight, keyword_weight = 0.3, 0.7
elif query_length >= 10:
    # 長查詢，語義理解重要
    semantic_weight, keyword_weight = 0.7, 0.3
else:
    # 中等長度，平衡
    semantic_weight, keyword_weight = 0.5, 0.5
```

### 統計資訊

```python
stats = hybrid.get_stats()
print(f"檢索器類型: {stats['retriever_type']}")
print(f"BM25 啟用: {stats['bm25_enabled']}")
print(f"索引文檔數: {stats['bm25_documents']}")
print(f"使用 jieba: {stats['use_jieba']}")
```

## 🎓 最佳實踐

1. **預設使用 0.5/0.5 權重**
   - 大多數情況下效果最好
   
2. **多概念查詢必用混合檢索**
   - 例如: "A 在 B 的應用"
   
3. **定期檢視日誌**
   - 查看哪些論文被檢索到
   - 調整權重優化結果
   
4. **測試不同權重**
   - 使用 `test_hybrid_search.py`
   - 找到最適合您資料的權重

## 🔗 相關文件

- `MULTI_CONCEPT_QUERY_SOLUTION.md` - 多概念查詢問題分析
- `LAYER1_STRUCTURE_DETAIL.md` - Layer 1 資料結構
- `system_api/hybrid_retriever.py` - 實現代碼

## ✅ 檢查清單

確認混合檢索已正確啟用：

- [ ] `rank-bm25` 和 `jieba` 已安裝
- [ ] `agent2.py` 中 `hybrid_search.enabled = True`
- [ ] 啟動時看到 "初始化混合檢索器" 日誌
- [ ] 查詢時看到 "使用混合檢索" 日誌
- [ ] 測試腳本運行正常

---

**準備好了嗎？** 現在就啟動系統測試混合檢索的效果吧！

```bash
python agent2.py
```
