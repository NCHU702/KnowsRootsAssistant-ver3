# Hierarchical RAG 系統 - 快速開始

## 🎯 系統簡介

本系統使用 **階層式 RAG（Hierarchical RAG）** 技術，提供高精確度的學術論文檢索和問答功能。

### 核心特性

✨ **兩層索引架構**
- Layer 1: 論文摘要索引（快速過濾）
- Layer 2: 論文區塊索引（詳細檢索）

🧠 **智能信心評估**
- LLM 評估答案信心度
- 自動決定是否需要深入檢索

📈 **動態上下文擴展**
- 根據需要擴展相鄰區塊
- 提供更完整的上下文

⚡ **性能優化**
- 早期終止機制（50-70% 查詢更快）
- 查詢結果快取

## 🚀 快速開始

### 步驟 1: 準備 PDF 文件

將你的 PDF 學術論文放入 `./data` 目錄：

```bash
mkdir -p data
cp your_papers/*.pdf data/
```

### 步驟 2: 建立索引

```bash
# 安裝依賴（如果還沒做）
pip install -r requirements.txt

# 建立階層式索引
python scripts/migrate_to_hierarchical_rag.py --execute
```

預期輸出：
```
================================================================================
Step 3: Executing Migration
================================================================================
Initializing HierarchicalRAGSystem...
✓ System initialized

Building hierarchical indices for X PDFs...

================================================================================
✓ Migration Completed Successfully!
================================================================================
Duration: XXs

New Index Statistics:
  Layer 1 (Abstracts): X papers
  Layer 2 (Chunks): XXX chunks
```

### 步驟 3: 啟動系統

```bash
python agent2.py
```

預期輸出：
```
======================================================================
🚀 AI Agent Router Starting...
======================================================================

📊 System Status:
  RAG Mode: HIERARCHICAL (HierarchicalRAGSystem)
  RAG System: ✓ Ready
  System Ready: ✓ Yes
  Layer 1: X papers
  Layer 2: XXX chunks

🌐 Monitoring Endpoints:
  GET  /rag/stats  - System statistics
  GET  /rag/health - Health check
  GET  /rag/mode   - Current RAG mode info

======================================================================
Server running on http://localhost:4000
======================================================================
```

### 步驟 4: 測試系統

在另一個終端運行：

```bash
# 自動化測試
python tests/test_agent2_integration.py

# 或手動測試
curl http://localhost:4000/rag/health
```

### 步驟 5: 開始使用

```bash
# 查詢範例
curl -X POST http://localhost:4000/query \
  -H "Content-Type: application/json" \
  -d '{"input": "What is deep learning?"}'
```

## 📊 監控系統

### 健康檢查

```bash
curl http://localhost:4000/rag/health | jq .
```

回應範例：
```json
{
  "status": "healthy",
  "components": {
    "rag_system": "up",
    "indices": "ready",
    "layer1": "up",
    "layer2": "up"
  },
  "issues": []
}
```

### 系統統計

```bash
curl http://localhost:4000/rag/stats | jq .
```

回應範例：
```json
{
  "mode": "hierarchical",
  "system_ready": true,
  "system_stats": {
    "layer1": {
      "paper_count": 2,
      "is_initialized": true
    },
    "layer2": {
      "chunk_count": 317,
      "paper_count": 2,
      "is_initialized": true
    }
  },
  "query_stats": {
    "total_queries": 10,
    "termination_distribution": {
      "layer1": 6,
      "layer2": 4
    }
  }
}
```

### 模式資訊

```bash
curl http://localhost:4000/rag/mode | jq .
```

## ⚙️ 配置選項

通過環境變數調整系統行為：

```bash
# 調整 Layer 1 閾值（預設：0.7）
export LAYER1_THRESHOLD=0.75

# 調整 Layer 2 閾值（預設：0.8）
export LAYER2_THRESHOLD=0.85

# 啟用/關閉上下文擴展（預設：true）
export ENABLE_EXPANSION=true

# 快取大小（預設：10）
export CACHE_SIZE=20

# 啟動系統
python agent2.py
```

### 閾值調整建議

| 場景 | Layer 1 | Layer 2 | 說明 |
|------|---------|---------|------|
| 高精確度 | 0.8 | 0.85 | 更嚴格的過濾，更少的誤報 |
| 平衡 | 0.7 | 0.8 | 預設設定，平衡精確度和召回率 |
| 高召回率 | 0.6 | 0.7 | 更寬鬆的過濾，減少遺漏 |

## 📁 文件結構

```
KnowsRootsAssistant-ver3/
├── agent2.py                          # 主應用（Flask API）
├── data/                              # PDF 文件目錄
├── vectorstore/                       # 階層式索引
│   ├── layer1_abstracts/             # Layer 1 索引
│   └── layer2_chunks/                # Layer 2 索引
├── system_api/                        # 核心模組
│   ├── hierarchical_rag_system.py    # 階層式 RAG 核心
│   ├── abstract_extractor.py         # 摘要提取
│   ├── layer1_vectorstore.py         # Layer 1 索引
│   ├── layer2_vectorstore.py         # Layer 2 索引
│   ├── confidence_evaluator.py       # 信心評估
│   ├── context_expander.py           # 上下文擴展
│   ├── index_manager.py              # 索引管理
│   └── query_logger.py               # 查詢日誌
├── scripts/
│   └── migrate_to_hierarchical_rag.py # 遷移/建立索引工具
└── tests/
    └── test_agent2_integration.py    # 整合測試
```

## 🔧 常見問題

### Q: 為什麼第一次查詢很慢？
**A:** 系統需要載入 LLM 和索引到記憶體。後續查詢會更快。

### Q: 如何添加新的 PDF？
**A:** 
1. 將 PDF 放入 `./data` 目錄
2. 重新運行：`python scripts/migrate_to_hierarchical_rag.py --execute`
3. 重啟 agent2.py

### Q: 索引建立失敗怎麼辦？
**A:** 
- 檢查 PDF 是否損壞或加密
- 確保 Ollama 服務正在運行
- 查看錯誤日誌獲取詳細資訊

### Q: 查詢結果不準確？
**A:** 
- 提高閾值（LAYER1_THRESHOLD, LAYER2_THRESHOLD）
- 確保 PDF 包含相關內容
- 嘗試重新表述問題

### Q: 如何備份索引？
**A:**
```bash
# 備份
tar -czf vectorstore_backup_$(date +%Y%m%d).tar.gz vectorstore/

# 還原
tar -xzf vectorstore_backup_YYYYMMDD.tar.gz
```

## 📖 API 端點

### 查詢端點

**POST /query**
```bash
curl -X POST http://localhost:4000/query \
  -H "Content-Type: application/json" \
  -d '{"input": "你的問題"}'
```

回應：
```json
{
  "answer": "...",
  "rag_mode": "hierarchical",
  "retrieval_stats": {
    "termination_layer": "layer1",
    "confidence": 0.95,
    "papers_used": ["paper1.pdf"],
    "retrieval_time": {
      "layer1": 2.5,
      "total": 2.5
    }
  }
}
```

**POST /query_stream**
- 流式輸出版本
- Server-Sent Events (SSE) 格式

### 監控端點

- **GET /rag/health** - 健康檢查
- **GET /rag/stats** - 系統統計
- **GET /rag/mode** - 模式資訊

## 🎯 性能基準

典型性能（基於實際測試）：

| 場景 | 耗時 | 說明 |
|------|------|------|
| Layer 1 早期終止 | 10-16s | 摘要層找到高信心答案 |
| Layer 2 完整檢索 | 60-75s | 需要深入區塊檢索 |
| 早期終止率 | 50-70% | 多數查詢可在 Layer 1 終止 |

## 📚 更多資源

- **完整文檔**: `HIERARCHICAL_RAG_USAGE.md`
- **測試指南**: `tests/test_agent2_integration.py`
- **索引工具**: `scripts/migrate_to_hierarchical_rag.py`

## ✅ 檢查清單

- [ ] PDF 文件已放入 `./data` 目錄
- [ ] 已執行索引建立：`python scripts/migrate_to_hierarchical_rag.py --execute`
- [ ] Ollama 服務正在運行
- [ ] agent2.py 成功啟動
- [ ] 健康檢查返回 `healthy`
- [ ] 整合測試全部通過

## 🆘 獲取幫助

遇到問題？

1. 查看日誌輸出
2. 運行健康檢查：`curl http://localhost:4000/rag/health`
3. 查閱完整文檔：`HIERARCHICAL_RAG_USAGE.md`
4. 運行測試診斷：`python tests/test_agent2_integration.py`

---

**祝使用愉快！** 🚀

**版本**: 1.0  
**最後更新**: 2025-11-13
