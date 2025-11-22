# 單一論文查詢模式 - 修改總結

## 概述

根據需求，我們修改了系統架構，讓 `AssistantCall` 專門負責單一論文的詳細查詢，優化了查詢流程。

## 修改的邏輯流程

### 1. Router 判斷（agent2.py）
- **修改前**：Router 沒有明確區分單論文和多論文查詢
- **修改後**：Router 明確區分：
  - `AssistantCall`：處理**單一特定論文**的詳細內容查詢
  - `GraphAnalysisCall`：處理**跨論文**的查詢、比較、聚合

### 2. Layer1 論文識別（hierarchical_rag_system.py）
- **修改前**：Layer1 返回多篇相關論文（top-k）
- **修改後**：在單論文模式下，Layer1 只返回**最相關的單篇論文**（top-1）

### 3. 摘要充分性判斷（hierarchical_rag_system.py）
- **新增功能**：找到目標論文後，先判斷摘要是否足夠回答問題
- 使用 `ConfidenceEvaluator` 評估摘要的充分性
- 如果摘要足夠（confidence ≥ 0.7），直接基於摘要生成答案
- 如果不夠，進入 Layer2 獲取詳細 chunks

### 4. Layer2 屬性分類檢索（layer2_vectorstore_reranking.py）
- **修改前**：Layer2 檢索所有類型的 chunks
- **修改後**：類似 GraphRAG 的細節查詢方式
  - 使用 `GraphIntegrator.determine_chunk_types_for_query()` 判斷問題相關的屬性
  - 添加 `filter_chunk_types` 參數到 `search_with_scores()`
  - 只檢索相關屬性的 chunks（如 methods、dataset、results 等）

## 主要修改的文件

### 1. agent2.py
- **位置**：Router prompt template（line ~410-520）
- **修改內容**：
  - 更新 `AssistantCall` 的描述和關鍵詞
  - 更新 `GraphAnalysisCall` 的描述
  - 更新所有示例，明確區分單論文和跨論文查詢
- **關鍵變化**：
  ```python
  # 修改前：模糊的 Micro-View/Macro-View
  # 修改後：明確的 Single Paper Detail Query / Cross-Paper Query
  ```

### 2. system_api/hierarchical_rag_system.py
- **新增方法**：
  - `query_single_paper(query, return_metadata)` - 單論文查詢（阻塞模式）
  - `query_single_paper_stream(query)` - 單論文查詢（串流模式）

- **查詢流程**：
  ```python
  STEP 1: Layer1 識別目標論文（top-1）
    └─> 返回最相關的單篇論文
  
  STEP 2: 檢查摘要充分性
    └─> 使用 ConfidenceEvaluator 評估
    └─> 如果足夠 → 直接返回基於摘要的答案
    └─> 如果不足 → 繼續到 STEP 3
  
  STEP 3: Layer2 檢索詳細 chunks（帶屬性過濾）
    └─> 判斷相關的 chunk 類型（methods, dataset, results 等）
    └─> 從目標論文的相關類型 chunks 中檢索
    └─> 返回最相關的 5 個 chunks
  
  STEP 4: 生成最終答案
  ```

### 3. system_api/layer2_vectorstore_reranking.py
- **修改方法**：
  - `search_with_scores()` - 添加 `filter_chunk_types` 參數
  - `search()` - 添加 `filter_chunk_types` 參數

- **過濾邏輯**：
  ```python
  if filter_chunk_types:
      candidate_chunks = [
          chunk for chunk in candidate_chunks
          if chunk.get('metadata', {}).get('chunk_type') in filter_chunk_types
      ]
  ```

### 4. agent2.py（streaming 調用）
- **位置**：line ~703
- **修改內容**：
  ```python
  # 修改前
  rag_stream = rag_system.query_stream(action_input)
  
  # 修改後
  rag_stream = rag_system.query_single_paper_stream(action_input)
  ```

## 新增測試文件

### test_single_paper_query.py
- 測試單論文查詢的完整流程
- 測試阻塞模式和串流模式
- 包含多個測試查詢範例

## 使用方式

### 1. 單論文查詢（通過 Router）
```python
# 用戶問題
query = "芒果分類那篇論文用什麼方法？"

# Router 會識別為 AssistantCall
# → 調用 rag_system.query_single_paper_stream(query)
# → Layer1 識別目標論文
# → 檢查摘要是否足夠
# → 如需要，Layer2 檢索相關 chunks（帶屬性過濾）
# → 返回答案
```

### 2. 直接調用（測試）
```python
# 阻塞模式
result = rag_system.query_single_paper(query, return_metadata=True)
print(result['answer'])

# 串流模式
for chunk in rag_system.query_single_paper_stream(query):
    print(chunk, end='', flush=True)
```

## 測試方法

```bash
# 1. 測試單論文查詢功能
python test_single_paper_query.py

# 2. 通過完整系統測試（需要啟動 Flask）
python agent2.py
# 然後在前端輸入單論文查詢問題
```

## 關鍵特性

1. **智能論文識別**：Layer1 自動識別用戶要查詢的論文
2. **摘要優先**：先嘗試用摘要回答，節省時間
3. **屬性分類檢索**：Layer2 只檢索相關屬性的 chunks，提高精確度
4. **語義理解**：使用 LLM 判斷問題與哪些屬性相關
5. **支持串流**：提供流暢的用戶體驗

## 與原系統的兼容性

- ✅ 不影響 `GraphAnalysisCall` 的跨論文查詢功能
- ✅ 不影響直接使用 `query()` 和 `query_stream()` 的傳統模式
- ✅ 保持所有現有 API 的向後兼容
- ✅ 只在 `AssistantCall` 調用時使用新的單論文模式

## 注意事項

1. **Chunk 分類依賴**：需要 chunks 已經被正確分類（metadata 中有 `chunk_type`）
2. **Graph Integrator 依賴**：需要 `graph_integrator` 來判斷相關的 chunk 類型
3. **評估閾值**：摘要充分性閾值設為 0.7，可根據需要調整

## 後續優化建議

1. 可以添加 fallback 機制：如果 chunk type 過濾後沒有結果，自動重試不帶過濾
2. 可以記錄統計數據：摘要足夠的比例、Layer2 觸發率等
3. 可以添加快取機制：常見單論文查詢的結果
