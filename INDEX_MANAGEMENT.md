# 智能索引管理系統使用指南

## 🎯 功能概述

系統現在具備**智能索引管理**功能，能自動檢測並處理：

1. ✅ **索引不存在** → 自動建立
2. ✅ **索引損壞** → 自動重建
3. ✅ **新增 PDF** → 自動更新
4. ✅ **PDF 被刪除** → 提示重建
5. ✅ **手動觸發** → API 端點支援

## 🚀 自動檢查（啟動時）

### 啟動行為

```bash
python agent2.py
```

**系統會自動執行**：

1. **載入現有索引**（如果存在）
2. **智能檢查**：
   - 檢查索引是否存在
   - 檢查索引是否損壞
   - 比對 PDF 數量（索引 vs 實際）
   - 檢測新增的 PDF
3. **自動處理**：
   - 如需要，自動建立/更新索引
   - 輸出詳細日誌

### 場景 1: 首次啟動（無索引）

```
Initializing Hierarchical RAG System...
Loading existing indices...
ℹ️  No existing indices found, need to build from scratch
✓ Hierarchical RAG System initialized successfully

Checking index status...
================================================================================
Checking Index Status
================================================================================
Indices not ready
================================================================================
Building Hierarchical Indices
================================================================================
Processing 32 PDF files...
[1/32] Processing: paper1.pdf
  ✓ Abstract: llm_generated (confidence: 0.85)
  ✓ Chunks: 15
...
✓ Indices built successfully!
✓ Action taken: build
✓ Processed 32 papers
```

### 場景 2: 索引已存在且最新

```
Loading existing indices...
✓ Both indices loaded successfully
  Layer 1: 32 papers
  Layer 2: 517 chunks from 32 papers

Checking index status...
Current PDF files: 32
Indexed papers: 32
✓ Indices are up-to-date and healthy
✓ Indices are healthy and up-to-date
```

### 場景 3: 偵測到新增 PDF

假設您在 `./data` 目錄新增了 3 個 PDF：

```
Loading existing indices...
✓ Both indices loaded successfully
  Layer 1: 32 papers
  Layer 2: 517 chunks from 32 papers

Checking index status...
Current PDF files: 35
Indexed papers: 32
⚠️  PDF count mismatch! Current: 35, Indexed: 32
Detected 3 new PDF(s)

================================================================================
Building Hierarchical Indices
================================================================================
Processing 35 PDF files...
[1/35] Processing: paper1.pdf
...
[33/35] Processing: new_paper1.pdf  ← 新增
[34/35] Processing: new_paper2.pdf  ← 新增
[35/35] Processing: new_paper3.pdf  ← 新增

✓ Indices rebuilt successfully!
✓ Action taken: incremental_update
✓ Processed 35 papers
```

### 場景 4: 索引損壞

```
Loading existing indices...
✗ Failed to load Layer 1: [Error details]
⚠️  Only Layer 2 loaded, Layer 1 needs building

Checking index status...
Indices appear corrupted
Attempting to rebuild indices...

================================================================================
Building Hierarchical Indices
================================================================================
...
✓ Indices rebuilt successfully!
✓ Action taken: rebuild
```

## 🔧 手動觸發索引更新

### API 端點：POST /rag/rebuild

#### 用法 1: 智能更新（推薦）

檢查並僅在需要時更新：

```bash
curl -X POST http://localhost:4000/rag/rebuild
```

回應：
```json
{
  "status": "success",
  "action_taken": "incremental_update",
  "details": {
    "papers_processed": 35,
    "abstracts_extracted": 35,
    "chunks_created": 565,
    "duration": 127.5
  }
}
```

#### 用法 2: 強制完全重建

無論狀態如何都重建：

```bash
curl -X POST "http://localhost:4000/rag/rebuild?force=true"
```

回應：
```json
{
  "status": "success",
  "action_taken": "rebuild",
  "details": {
    "papers_processed": 35,
    "abstracts_extracted": 35,
    "chunks_created": 565,
    "duration": 145.2
  }
}
```

#### 用法 3: 檢查而不更新

如果索引已是最新：

```bash
curl -X POST http://localhost:4000/rag/rebuild
```

回應：
```json
{
  "status": "up_to_date",
  "action_taken": null,
  "details": {
    "layer1": {
      "paper_count": 35,
      "is_initialized": true
    },
    "layer2": {
      "chunk_count": 565,
      "paper_count": 35,
      "is_initialized": true
    },
    "pdf_count": 35
  }
}
```

## 📊 檢查邏輯

系統執行以下檢查：

### 1. 索引就緒檢查
```
Layer 1 initialized? ✓/✗
Layer 2 initialized? ✓/✗
```

### 2. 數量比對
```
當前 PDF 數量: X
索引中的 PDF: Y

X > Y  → 有新增的 PDF，執行更新
X < Y  → 有刪除的 PDF，執行重建
X == Y → 繼續檢查
```

### 3. 完整性驗證
```
Layer 1 is_initialized: true/false
Layer 2 is_initialized: true/false

任一為 false → 索引損壞，執行重建
```

## 🔄 工作流程

### 新增 PDF 的完整流程

1. **放入 PDF**：
   ```bash
   cp new_paper.pdf ./data/
   ```

2. **重啟或觸發更新**：
   
   選項 A（自動）：
   ```bash
   # 重啟服務，自動檢測並更新
   python agent2.py
   ```
   
   選項 B（手動 API）：
   ```bash
   # 不重啟，使用 API 觸發
   curl -X POST http://localhost:4000/rag/rebuild
   ```

3. **驗證**：
   ```bash
   curl http://localhost:4000/rag/stats | jq .
   ```

## 🛡️ 錯誤處理

### 如果索引重建失敗

檢查日誌：
```
✗ Failed to process /path/to/corrupted.pdf: [Error]
```

**可能原因**：
- PDF 損壞或加密
- 磁碟空間不足
- Ollama 服務未運行
- 記憶體不足

**解決方案**：
1. 移除問題 PDF
2. 修復問題
3. 重新觸發：`curl -X POST http://localhost:4000/rag/rebuild?force=true`

### 如果檢測不到新 PDF

**檢查清單**：
- [ ] PDF 確實在 `./data` 目錄
- [ ] 檔案副檔名是 `.pdf`（小寫）
- [ ] 檔案沒有被隱藏（不是 `.file.pdf`）
- [ ] 有讀取權限

**手動強制重建**：
```bash
curl -X POST "http://localhost:4000/rag/rebuild?force=true"
```

## 📈 性能考量

### 索引建立時間估算

| PDF 數量 | 預估時間 |
|----------|----------|
| 1-10 篇  | 1-3 分鐘 |
| 10-30 篇 | 3-8 分鐘 |
| 30-50 篇 | 8-15 分鐘 |
| 50+ 篇   | 15+ 分鐘 |

*時間取決於 PDF 大小、硬體性能和 Ollama 模型*

### 優化建議

1. **批次添加**：一次添加多個 PDF，而不是逐個添加
2. **夜間更新**：使用 cron 在非高峰時段更新
3. **監控性能**：觀察 `/rag/stats` 的回應時間

## 🎯 最佳實踐

### 開發環境
- ✅ 允許自動檢測和更新
- ✅ 頻繁重啟沒問題（會快速檢查）
- ✅ 使用 API 端點手動觸發

### 生產環境
- ✅ 定期（如每天）檢查更新
- ✅ 使用監控工具追蹤 `/rag/stats`
- ✅ 設置告警（如 PDF 數量異常）
- ⚠️ 避免高峰時段重建索引

### 添加大量 PDF
```bash
# 1. 停止服務
Ctrl+C

# 2. 批次複製 PDF
cp batch/*.pdf ./data/

# 3. 啟動服務（自動更新）
python agent2.py
```

## 🔍 監控和診斷

### 檢查當前狀態

```bash
# 統計資訊
curl http://localhost:4000/rag/stats | jq .

# 健康狀態
curl http://localhost:4000/rag/health | jq .

# 模式資訊
curl http://localhost:4000/rag/mode | jq .
```

### 診斷索引問題

```bash
# 檢查索引目錄
ls -lh vectorstore/layer1/
ls -lh vectorstore/layer2/

# 檢查 PDF 數量
ls -1 data/*.pdf | wc -l

# 觸發重建並觀察日誌
curl -X POST http://localhost:4000/rag/rebuild?force=true
```

## ❓ 常見問題

### Q: 每次啟動都會重建索引嗎？
**A:** 不會！只有在檢測到問題時才重建：
- 索引不存在
- 索引損壞
- PDF 數量不匹配

### Q: 重建會保留舊數據嗎？
**A:** 重建會完全替換索引，但：
- 原始 PDF 不受影響
- 可以隨時重建
- 建議定期備份 `vectorstore/` 目錄

### Q: 可以在服務運行時添加 PDF 嗎？
**A:** 可以！添加後：
- 選項 1: 重啟服務（自動檢測）
- 選項 2: 調用 `/rag/rebuild` API

### Q: 強制重建和智能更新的區別？
**A:**
- **智能更新** (`/rag/rebuild`): 檢查後決定是否需要操作
- **強制重建** (`/rag/rebuild?force=true`): 無條件重建所有索引

### Q: 索引損壞怎麼辦？
**A:** 系統會自動偵測並重建。如果失敗：
```bash
# 手動刪除舊索引
rm -rf vectorstore/layer1 vectorstore/layer2

# 重啟服務（自動重建）
python agent2.py
```

## 📚 相關文檔

- **快速開始**: [QUICKSTART.md](QUICKSTART.md)
- **完整手冊**: [HIERARCHICAL_RAG_USAGE.md](HIERARCHICAL_RAG_USAGE.md)
- **API 參考**: 查看 agent2.py 的端點定義

---

**系統版本**: 1.1 (智能索引管理)  
**最後更新**: 2025-11-13
