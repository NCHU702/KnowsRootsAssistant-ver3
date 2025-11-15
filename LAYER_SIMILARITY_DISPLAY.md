# Layer 搜尋相似度顯示功能

## 📊 新增功能

現在在進行 Layer 1 和 Layer 2 搜尋時，會在終端機清楚顯示**前 5 名相似度最高的結果**！

---

## 🎯 顯示內容

### Layer 1 (論文層級)

```
📊 前 5 名相似度最高的論文:
────────────────────────────────────────────────────────────────────────────────
  1. [0.5288] 標準12_基於R-tree與SPACE-MDL-LSTM提升大區域人流預測之效率
  2. [0.5148] 標準11_應用補償式遷移學習模型於區域人流之強健性預測
  3. [0.4948] 標準9_應用三維高斯函數結合深度學習之區域 PM2.5 預測
  4. [0.4930] 標準14_基於集成式生成對抗網路進行人流異常預測
  5. [0.4927] 標準10_基於3D-RCL與條件式生成對抗網路產生未來時刻人群分布之可能性探討
────────────────────────────────────────────────────────────────────────────────
```

### Layer 2 (文檔片段)

```
📊 前 5 名相似度最高的文檔片段:
────────────────────────────────────────────────────────────────────────────────
  1. [0.8523] 標準12_基於R-tree與SPACE-MDL-LSTM提升大區域人流預測之效率
      Chunk: chunk_3 | Preview: 本研究提出使用 LSTM 模型進行人流預測，結合空間索引技術...
  2. [0.8102] 標準11_應用補償式遷移學習模型於區域人流之強健性預測
      Chunk: chunk_1 | Preview: 遷移學習在人流預測中的應用能夠有效提升模型泛化能力...
  3. [0.7856] 標準12_基於R-tree與SPACE-MDL-LSTM提升大區域人流預測之效率
      Chunk: chunk_5 | Preview: 實驗結果顯示，SPACE-MDL-LSTM 相較於傳統 LSTM 提升了...
  4. [0.7621] 標準14_基於集成式生成對抗網路進行人流異常預測
      Chunk: chunk_2 | Preview: 生成對抗網路 (GAN) 在異常檢測中能夠學習正常人流模式...
  5. [0.7412] 標準11_應用補償式遷移學習模型於區域人流之強健性預測
      Chunk: chunk_4 | Preview: 補償式遷移學習透過動態調整源域和目標域之間的權重...
────────────────────────────────────────────────────────────────────────────────
```

---

## 📝 相似度分數說明

### 分數範圍
- **0.0 - 1.0**: 相似度分數（越高越相關）
- **> 0.8**: 非常相關
- **0.6 - 0.8**: 相關
- **0.4 - 0.6**: 中度相關
- **< 0.4**: 弱相關

### 分數計算
- **Layer 1**: 混合檢索分數 = 語義分數 × 權重 + 關鍵詞分數 × 權重
- **Layer 2**: 純語義相似度分數（FAISS 距離轉換）

---

## 🔧 修改的檔案

### 1. `system_api/hierarchical_rag_system.py`

**Layer 1 顯示**:
```python
# 顯示前 5 名相似度最高的論文
if layer1_results_with_scores:
    logger.info("\n📊 前 5 名相似度最高的論文:")
    logger.info("─" * 80)
    for i, (doc, score) in enumerate(layer1_results_with_scores[:5], 1):
        title = doc.metadata.get('title', 'Unknown')
        logger.info(f"  {i}. [{score:.4f}] {title}")
    logger.info("─" * 80)
```

**Layer 2 顯示**:
```python
# 顯示前 5 名相似度最高的文檔片段
if layer2_results_with_scores:
    logger.info("\n📊 前 5 名相似度最高的文檔片段:")
    logger.info("─" * 80)
    for i, (doc, score) in enumerate(layer2_results_with_scores[:5], 1):
        chunk_id = doc.metadata.get('chunk_id', 'Unknown')
        paper_title = doc.metadata.get('title', 'Unknown')
        content_preview = doc.page_content[:60].replace('\n', ' ')
        logger.info(f"  {i}. [{score:.4f}] {paper_title}")
        logger.info(f"      Chunk: {chunk_id} | Preview: {content_preview}...")
    logger.info("─" * 80)
```

### 2. `system_api/layer2_vectorstore.py`

新增方法：
- `search_with_scores()`: 返回帶分數的搜尋結果
- `_search_filtered_with_scores()`: 過濾搜尋並返回分數

---

## 🧪 測試

### 執行測試腳本
```bash
python test_layer_similarity_display.py
```

### 或使用 agent2.py
```bash
python agent2.py
```

然後透過網頁或 API 發送查詢，終端機會顯示相似度資訊。

---

## 📊 範例輸出

```
INFO - ============================================================
INFO - LAYER 1: Paper-level Retrieval
INFO - ============================================================
INFO - 使用混合檢索 (Hybrid Search: 語義 + 關鍵詞)
INFO -   當前權重: 語義=0.98, 關鍵詞=0.02
INFO - Retrieved 10 papers (threshold: 0.48)

INFO - 📊 前 5 名相似度最高的論文:
INFO - ────────────────────────────────────────────────────────────────────────────────
INFO -   1. [0.5288] 標準12_基於R-tree與SPACE-MDL-LSTM提升大區域人流預測之效率
INFO -   2. [0.5148] 標準11_應用補償式遷移學習模型於區域人流之強健性預測
INFO -   3. [0.4948] 標準9_應用三維高斯函數結合深度學習之區域 PM2.5 預測
INFO -   4. [0.4930] 標準14_基於集成式生成對抗網路進行人流異常預測
INFO -   5. [0.4927] 標準10_基於3D-RCL與條件式生成對抗網路產生未來時刻人群分布之可能性探討
INFO - ────────────────────────────────────────────────────────────────────────────────

INFO - Layer 1 Evaluation:
INFO -   Confidence: 0.60 (threshold: 0.6)
INFO -   Should continue: False
INFO -   Reasoning: 相關性高，因為都在討論人流預測模型。完整性中缺少具體方法和數據...

INFO - ✓ Early termination: Layer 1 results sufficient
```

---

## 💡 使用建議

1. **相似度閾值**: 如果前 5 名的分數都很低（< 0.4），可能需要：
   - 調整查詢詞彙
   - 檢查論文庫是否包含相關主題

2. **Layer 2 觸發**: 如果想看到 Layer 2 的相似度，可以：
   - 降低 `layer1.confidence_threshold`（預設 0.6）
   - 或使用更複雜的查詢

3. **除錯**: 相似度顯示能幫助你了解：
   - 哪些論文/片段被檢索到
   - 為什麼某些結果排名較高
   - 系統是否正確理解查詢

---

## 🎉 效果

✅ **清楚看到相似度分數**  
✅ **快速評估檢索品質**  
✅ **便於調優和除錯**  
✅ **提供 4 位小數精度**  
✅ **包含論文標題和內容預覽**  

---

**更新日期**: 2025-11-15  
**版本**: v3.3  
