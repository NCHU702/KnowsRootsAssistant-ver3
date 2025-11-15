# 論文參考與語言支援功能

## 📚 新增功能

### 1. 顯示參考論文列表

系統現在會在回答前顯示 LLM 參考了哪些論文（來自 Layer 1 檢索結果）。

**格式**：
- 英文查詢：`📚 Referenced Papers:`
- 中文查詢：`📚 參考論文：`

**顯示內容**：
```
📚 參考論文：
- 論文標題 (作者, 年份)
- Deep Learning Methods for Remote Sensing (Zhang et al., 2023)
- Transformer Networks in Computer Vision (Liu et al., 2024)
...

============================================================

[LLM 回答內容]
```

### 2. 語言自動匹配

系統會自動偵測查詢語言，並使用相同語言回答。

**偵測邏輯**：
- 如果文字中中文字符 > 30%，判定為中文查詢
- 否則判定為英文查詢

**Prompt 調整**：
- 中文查詢 → "請根據以下研究論文的內容回答問題。請使用繁體中文回答。"
- 英文查詢 → "Based on the following research paper contexts, please answer the question."

## 🔧 實作細節

### 修改檔案
- `system_api/hierarchical_rag_system.py`

### 修改方法

#### 1. `_hierarchical_retrieval()`
```python
# 保存 Layer 1 檢索結果供後續顯示
result['layer1_docs'] = layer1_docs
```

#### 2. `_generate_answer()`
```python
# 語言偵測
is_chinese_query = is_chinese(query)

# 提取論文資訊
referenced_papers = []
for doc in layer1_docs[:10]:
    title = doc.metadata.get('title', 'Unknown')
    author = doc.metadata.get('author', 'Unknown')
    year = doc.metadata.get('year', 'N/A')
    referenced_papers.append(f"- {title} ({author}, {year})")

# 根據語言生成 prompt
if is_chinese_query:
    prompt = f"""請根據以下研究論文的內容回答問題。請使用繁體中文回答。

問題：{query}
...
```

#### 3. `_generate_answer_stream()`
```python
# 先串流輸出參考論文
if referenced_papers:
    if is_chinese_query:
        yield "📚 參考論文：\n"
    else:
        yield "📚 Referenced Papers:\n"
    
    for paper in referenced_papers:
        yield paper + "\n"
    
    yield "\n" + "="*60 + "\n\n"

# 再串流 LLM 回答
for chunk in self.llm.stream(prompt):
    yield chunk
```

## 📋 使用範例

### 範例 1：英文查詢

**輸入**：
```
What are the main challenges in deep learning?
```

**輸出**：
```
📚 Referenced Papers:
- Deep Learning: A Comprehensive Survey (LeCun et al., 2015)
- Challenges and Opportunities in Deep Learning (Bengio, 2019)
- Optimization Methods for Deep Neural Networks (Kingma et al., 2020)

============================================================

The main challenges in deep learning include:

1. **Data Requirements**: Deep learning models typically require large amounts of labeled data...
2. **Computational Resources**: Training deep neural networks demands significant computational power...
3. **Interpretability**: Understanding why a model makes certain predictions remains difficult...
...
```

### 範例 2：中文查詢

**輸入**：
```
深度學習的主要挑戰是什麼？
```

**輸出**：
```
📚 參考論文：
- Deep Learning: A Comprehensive Survey (LeCun et al., 2015)
- Challenges and Opportunities in Deep Learning (Bengio, 2019)
- Optimization Methods for Deep Neural Networks (Kingma et al., 2020)

============================================================

深度學習的主要挑戰包括：

1. **資料需求**：深度學習模型通常需要大量的標註資料...
2. **運算資源**：訓練深度神經網路需要大量的運算能力...
3. **可解釋性**：理解模型為何做出特定預測仍然困難...
...
```

### 範例 3：串流模式

使用 `/query_stream` API 時，參考論文會先串流輸出，然後再串流 LLM 回答。

## 🧪 測試

運行測試腳本：
```bash
python test_paper_references.py
```

測試項目：
1. ✅ 語言偵測邏輯
2. ✅ 英文查詢 + 論文參考顯示
3. ✅ 中文查詢 + 論文參考顯示
4. ✅ 串流模式 + 論文參考顯示

## 🎯 功能特性

### 優點
- ✅ 提高透明度：用戶知道 LLM 參考了哪些論文
- ✅ 便於驗證：可以檢查引用的論文是否相關
- ✅ 語言一致性：中文問中文答，英文問英文答
- ✅ 支援串流：參考論文先顯示，不影響串流體驗

### 限制
- ⚠️ 顯示的論文來自 Layer 1（摘要層），最多 10 篇
- ⚠️ 如果早期終止在 Layer 1，可能只有摘要資訊
- ⚠️ Layer 2 檢索的論文區塊不單獨列出（已包含在 Layer 1 論文中）

## 🔮 未來改進

可能的增強功能：
- [ ] 顯示每篇論文的相似度分數
- [ ] 標示哪些論文的內容被實際引用在回答中
- [ ] 支援點擊論文標題查看完整 PDF
- [ ] Layer 2 區塊級別的引用標註（例如：[1]、[2]）
- [ ] 混合語言查詢的智能處理

## 📝 配置

無需額外配置，功能已自動啟用。

如需調整顯示的論文數量，修改 `hierarchical_rag_system.py`：
```python
# 預設顯示最多 10 篇
for doc in layer1_docs[:10]:  # 改為 [:5] 或其他數字
```

## 🌐 API 端點

功能適用於所有查詢端點：
- `POST /query` - 非串流查詢
- `POST /query_stream` - 串流查詢（推薦）

使用方式不變：
```bash
curl -X POST http://localhost:4000/query_stream \
  -H "Content-Type: application/json" \
  -d '{"input": "深度學習的應用有哪些？"}'
```

---

**實作完成日期**：2025-01-14  
**功能狀態**：✅ 已測試並可用
