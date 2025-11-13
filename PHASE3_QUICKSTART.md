# Phase 3 完成 - 快速開始指南

## 🎉 恭喜！階層式 RAG 系統已整合完成

Phase 3 已經完成所有核心功能的整合。本文檔提供快速開始指南。

## 📋 已完成的工作

### ✅ Phase 3.1: 遷移策略文檔
- 創建 `PHASE3_INTEGRATION_PLAN.md` - 完整的整合計劃

### ✅ Phase 3.2: 雙模式支持
- `agent2.py` 支援 legacy/hierarchical 雙模式
- 通過 `RAG_MODE` 環境變數切換
- 完全向後兼容

### ✅ Phase 3.3: 遷移腳本
- `scripts/migrate_to_hierarchical_rag.py`
- 支援 backup/dry-run/execute/verify/rollback
- 完整的錯誤處理和回滾機制

### ✅ Phase 3.4: API 端點更新
- 現有 `/query` 和 `/query_stream` 端點兼容兩種模式
- 自動檢測並使用正確的 RAG 系統

### ✅ Phase 3.5: 監控端點
- `/rag/stats` - 系統統計（文檔數、查詢性能）
- `/rag/health` - 健康檢查
- `/rag/mode` - 模式資訊

### ✅ Phase 3.6: 整合測試
- `tests/test_agent2_integration.py`
- 7 項自動化測試
- 端到端驗證

### ✅ Phase 3.7: 文檔
- `HIERARCHICAL_RAG_USAGE.md` - 使用指南
- `PHASE3_INTEGRATION_PLAN.md` - 整合計劃
- 本快速開始指南

## 🚀 立即開始使用

### 方案 A：使用 Legacy 模式（預設）

```bash
# 不需要任何變更，直接啟動
python agent2.py
```

### 方案 B：使用 Hierarchical 模式（推薦）

#### 步驟 1：建立階層式索引

```bash
# 如果你已有 PDF 文件在 ./data 目錄
python scripts/migrate_to_hierarchical_rag.py --backup    # 備份舊索引（如果有）
python scripts/migrate_to_hierarchical_rag.py --dry-run   # 預覽
python scripts/migrate_to_hierarchical_rag.py --execute   # 執行遷移
```

預期輸出：
```
================================================================================
Step 3: Executing Migration
================================================================================
Initializing HierarchicalRAGSystem...
  Model: gemma3:12b
  Embedding: embeddinggemma:latest
✓ System initialized

Building hierarchical indices for 2 PDFs...

================================================================================
✓ Migration Completed Successfully!
================================================================================
Duration: 34.0s (0min 34s)

New Index Statistics:
  Layer 1 (Abstracts): 2 papers
  Layer 2 (Chunks): 317 chunks
  Index Size: 2.45 MB
```

#### 步驟 2：啟用 Hierarchical 模式

```bash
# 設定環境變數
export RAG_MODE=hierarchical

# 可選：調整配置
export LAYER1_THRESHOLD=0.7
export LAYER2_THRESHOLD=0.8

# 啟動應用
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
  Layer 1: 2 papers
  Layer 2: 317 chunks
  ResearchInheritanceAnalyzer: ✓ Ready
  Inheritance LLM: ✓ Ready
  Agent LLM: ✓ Ready

🌐 Monitoring Endpoints:
  GET  /rag/stats  - System statistics
  GET  /rag/health - Health check
  GET  /rag/mode   - Current RAG mode info

======================================================================
Server running on http://localhost:4000
======================================================================
```

#### 步驟 3：驗證系統

在另一個終端運行：

```bash
# 方式 A：使用測試腳本（推薦）
python tests/test_agent2_integration.py
```

預期輸出：
```
======================================================================
Agent2.py + HierarchicalRAGSystem 整合測試
======================================================================

...（7 項測試）...

======================================================================
  測試總結
======================================================================

✓ 伺服器連線
✓ RAG 模式
✓ 健康檢查
✓ 系統統計
✓ 查詢功能
✓ 流式查詢
✓ API 兼容性

通過率: 7/7 (100.0%)

🎉 所有測試通過！系統已準備好使用。
```

```bash
# 方式 B：手動測試
curl http://localhost:4000/rag/health
curl http://localhost:4000/rag/stats
curl http://localhost:4000/rag/mode
```

#### 步驟 4：開始使用

```bash
# 測試查詢
curl -X POST http://localhost:4000/query \
  -H "Content-Type: application/json" \
  -d '{"input": "What is deep learning?"}'
```

## 📊 監控系統

### 查看即時統計

```bash
# 系統健康
curl http://localhost:4000/rag/health | jq .

# 詳細統計
curl http://localhost:4000/rag/stats | jq .

# 當前模式
curl http://localhost:4000/rag/mode | jq .
```

### 查詢性能範例

**Hierarchical 模式典型性能**：
- Layer 1 早期終止：10-16s
- Layer 2 完整檢索：60-75s
- 早期終止率：50-70%

## 🔄 模式切換

### 從 Legacy 切換到 Hierarchical

```bash
# 1. 停止應用（Ctrl+C）

# 2. 建立索引（如果還沒做）
python scripts/migrate_to_hierarchical_rag.py --execute

# 3. 切換模式
export RAG_MODE=hierarchical

# 4. 重啟應用
python agent2.py
```

### 從 Hierarchical 切換回 Legacy

```bash
# 1. 停止應用

# 2. 切換模式
export RAG_MODE=legacy
# 或直接取消設定
unset RAG_MODE

# 3. 重啟應用
python agent2.py
```

## 📁 文件結構

```
KnowsRootsAssistant-ver3/
├── agent2.py                          # ✨ 已更新：支援雙模式
├── system_api/
│   ├── hierarchical_rag_system.py    # 階層式 RAG 核心
│   ├── abstract_extractor.py         # 改進的摘要提取
│   ├── layer1_vectorstore.py         # Layer 1 索引
│   ├── layer2_vectorstore.py         # Layer 2 索引
│   ├── confidence_evaluator.py       # 信心評估
│   ├── context_expander.py           # 上下文擴展
│   ├── index_manager.py              # 索引管理
│   └── query_logger.py               # 查詢日誌
├── scripts/
│   └── migrate_to_hierarchical_rag.py # 遷移腳本
├── tests/
│   ├── test_hierarchical_components.py
│   ├── test_hierarchical_integration.py
│   ├── test_abstract_extractor.py
│   └── test_agent2_integration.py    # ✨ 新增：整合測試
├── data/                              # PDF 文件
├── vectorstore/                       # Legacy 索引（或新索引）
├── vectorstore_hierarchical/          # Hierarchical 索引（可選）
└── 文檔/
    ├── PHASE3_INTEGRATION_PLAN.md
    ├── HIERARCHICAL_RAG_USAGE.md
    └── PHASE3_QUICKSTART.md          # 本文檔
```

## 🎯 功能對比

| 功能 | Legacy | Hierarchical |
|------|--------|--------------|
| 索引結構 | 單層 | 雙層（摘要+區塊） |
| 檢索策略 | Top-K | 階層式過濾 |
| 早期終止 | ❌ | ✅ |
| 信心評估 | ❌ | ✅ (LLM-based) |
| 上下文擴展 | ❌ | ✅ (動態±1-3) |
| 中文摘要提取 | N/A | ✅ (100%成功率) |
| 查詢監控 | ❌ | ✅ |
| API 兼容 | - | 完全兼容 |

## 💡 使用建議

### 什麼時候使用 Legacy 模式？
- 快速開發和測試
- 文檔數量很少（<10 篇）
- 需要最快的響應（不考慮精確度）

### 什麼時候使用 Hierarchical 模式？
- 生產環境
- 文檔數量中等以上（>10 篇）
- 需要更高的檢索精確度
- 想要早期終止優化（節省時間）
- 需要查詢監控和統計

## 🔧 配置優化

### 提高精確度
```bash
export LAYER1_THRESHOLD=0.8
export LAYER2_THRESHOLD=0.85
```

### 提高召回率
```bash
export LAYER1_THRESHOLD=0.6
export LAYER2_THRESHOLD=0.7
```

### 優化性能
```bash
export CACHE_SIZE=20  # 增加快取
```

## 📖 更多資源

- 完整使用指南：`HIERARCHICAL_RAG_USAGE.md`
- 整合計劃：`PHASE3_INTEGRATION_PLAN.md`
- API 測試：`tests/test_agent2_integration.py`
- 遷移腳本：`scripts/migrate_to_hierarchical_rag.py`

## ❓ 常見問題

### Q: 需要重新上傳 PDF 嗎？
**A:** 不需要！遷移腳本會自動處理現有 PDF。

### Q: 可以隨時切換模式嗎？
**A:** 可以！只需停止應用、切換 `RAG_MODE`、重啟即可。

### Q: Legacy 模式還會繼續支援嗎？
**A:** 是的，完全向後兼容並會持續維護。

### Q: 如何回滾到舊版本？
**A:** 使用 `python scripts/migrate_to_hierarchical_rag.py --rollback`

### Q: 為什麼查詢比 Legacy 慢？
**A:** 因為增加了 LLM 評估步驟。但早期終止時（50-70%情況）會更快。

## ✅ 檢查清單

使用本清單確保正確設置：

- [ ] PDF 文件已放在 `./data` 目錄
- [ ] 已執行遷移腳本建立索引
- [ ] 設定 `RAG_MODE=hierarchical`
- [ ] 啟動 `agent2.py` 成功
- [ ] 健康檢查返回 `healthy`
- [ ] 統計顯示正確的文檔數
- [ ] 測試查詢成功執行
- [ ] 整合測試全部通過

## 🚀 下一步

1. **生產部署**：參考 `HIERARCHICAL_RAG_USAGE.md` 的生產環境建議
2. **性能調優**：根據監控數據調整閾值
3. **持續監控**：定期檢查 `/rag/stats` 端點
4. **索引維護**：定期備份索引

---

**祝使用愉快！** 如有問題，請查閱完整文檔或提交 Issue。

**最後更新**: 2025-11-13  
**版本**: Phase 3.0 Complete
