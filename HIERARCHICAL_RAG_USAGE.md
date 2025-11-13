# Hierarchical RAG 使用指南

## 概述
本文檔說明如何使用新的階層式 RAG 系統，包括啟用、配置、監控和疑難排解。

## 快速開始

### 1. 啟用 Hierarchical RAG 模式

#### 方式 A：使用環境變數（推薦）
```bash
# 設定 RAG 模式為 hierarchical
export RAG_MODE=hierarchical

# 可選：調整閾值
export LAYER1_THRESHOLD=0.7  # Layer 1 信心閾值（預設 0.7）
export LAYER2_THRESHOLD=0.8  # Layer 2 信心閾值（預設 0.8）

# 可選：調整快取大小
export CACHE_SIZE=10  # 子索引快取大小（預設 10）

# 啟動應用
python agent2.py
```

#### 方式 B：修改程式碼
在 `agent2.py` 中直接修改：
```python
RAG_MODE = 'hierarchical'  # 改為 'hierarchical'
```

### 2. 首次使用：建立索引

如果你有現有的 PDF 文件但還沒有階層式索引：

```bash
# 方式 A：使用遷移腳本（推薦）
python scripts/migrate_to_hierarchical_rag.py --backup    # 備份舊索引
python scripts/migrate_to_hierarchical_rag.py --execute   # 建立新索引

# 方式 B：在應用中自動建立
# 啟動 agent2.py 後，系統會檢測到沒有索引並提示建立
```

### 3. 驗證系統狀態

```bash
# 檢查健康狀態
curl http://localhost:4000/rag/health

# 查看系統統計
curl http://localhost:4000/rag/stats

# 查看當前模式
curl http://localhost:4000/rag/mode
```

## 功能特性

### 🎯 階層式檢索
- **Layer 1（論文級）**: 基於摘要的快速篩選
- **Layer 2（區塊級）**: 詳細內容檢索
- **早期終止**: 當 Layer 1 結果足夠時，無需進入 Layer 2（節省時間）

### 🧠 信心評估
- LLM 評估檢索結果的相關性和完整性
- 動態決定是否需要更詳細的檢索
- 可調整的信心閾值

### 📈 動態上下文擴展
- 根據信心分數自動擴展上下文範圍
- 高信心（≥0.85）: ±1 chunks
- 中信心（0.75-0.85）: ±2 chunks
- 低信心（<0.75）: ±3 chunks

### ⚡ 性能優化
- 子索引快取（LRU）
- 過濾式 FAISS 搜尋
- 查詢日誌和性能統計

## API 使用

### 查詢接口

#### 標準查詢
```bash
curl -X POST http://localhost:4000/query \
  -H "Content-Type: application/json" \
  -d '{"input": "What is machine learning?"}'
```

**響應格式**（與 legacy 模式兼容）：
```json
{
  "answer": "...",
  "rag_mode": "hierarchical",  // 新增：模式資訊
  "retrieval_stats": {          // 新增：檢索統計（可選）
    "termination_layer": "layer1",
    "confidence": 0.95,
    "retrieval_time": {
      "layer1": 0.1,
      "layer1_evaluation": 8.5,
      "total": 8.6
    }
  }
}
```

#### 流式查詢
```bash
curl -X POST http://localhost:4000/query_stream \
  -H "Content-Type: application/json" \
  -d '{"input": "深度學習的應用"}'
```

### 監控接口

#### 系統統計
```bash
curl http://localhost:4000/rag/stats
```

**響應範例**：
```json
{
  "mode": "hierarchical",
  "system_ready": true,
  "system_stats": {
    "layer1": {
      "paper_count": 42,
      "is_initialized": true
    },
    "layer2": {
      "chunk_count": 3456,
      "paper_count": 42,
      "is_initialized": true
    }
  },
  "query_stats": {
    "total_queries": 150,
    "status_distribution": {
      "success": 148,
      "error": 2
    },
    "termination_distribution": {
      "layer1": 95,  // 63% 在 Layer 1 終止
      "layer2": 55
    }
  },
  "performance_summary": {
    "average_duration": 12.5,
    "average_confidence": 0.88,
    "timing_breakdown": {
      "layer1_retrieval": 0.15,
      "layer1_evaluation": 10.2,
      "layer2_retrieval": 25.3,
      "layer2_evaluation": 18.7
    }
  }
}
```

#### 健康檢查
```bash
curl http://localhost:4000/rag/health
```

**響應範例**：
```json
{
  "status": "healthy",  // healthy | degraded | unhealthy
  "components": {
    "rag_system": "up",
    "indices": "ready",
    "layer1": "up",
    "layer2": "up"
  },
  "issues": []
}
```

#### 模式資訊
```bash
curl http://localhost:4000/rag/mode
```

**響應範例**：
```json
{
  "mode": "hierarchical",
  "type": "HierarchicalRAGSystem",
  "features": ["two-layer", "confidence-evaluation", "context-expansion"],
  "config": {
    "layer1_threshold": 0.7,
    "layer2_threshold": 0.8,
    "expansion_enabled": true,
    "cache_size": 10
  }
}
```

## 配置選項

### 環境變數

| 變數名 | 預設值 | 說明 |
|--------|--------|------|
| `RAG_MODE` | `legacy` | RAG 模式：`legacy` 或 `hierarchical` |
| `LAYER1_THRESHOLD` | `0.7` | Layer 1 信心閾值（0.0-1.0） |
| `LAYER2_THRESHOLD` | `0.8` | Layer 2 信心閾值（0.0-1.0） |
| `ENABLE_EXPANSION` | `true` | 是否啟用上下文擴展 |
| `CACHE_SIZE` | `10` | 子索引快取大小 |

### 調整建議

#### 提高精確度（降低 false positives）
```bash
export LAYER1_THRESHOLD=0.8  # 提高 Layer 1 閾值
export LAYER2_THRESHOLD=0.85  # 提高 Layer 2 閾值
```

#### 提高召回率（降低 false negatives）
```bash
export LAYER1_THRESHOLD=0.6  # 降低 Layer 1 閾值
export LAYER2_THRESHOLD=0.7  # 降低 Layer 2 閾值
```

#### 優化性能
```bash
export CACHE_SIZE=20  # 增加快取（適合重複查詢相同論文）
```

## 索引管理

### 遷移現有索引

```bash
# 1. 備份現有索引
python scripts/migrate_to_hierarchical_rag.py --backup

# 2. 預覽遷移計劃
python scripts/migrate_to_hierarchical_rag.py --dry-run

# 3. 執行遷移
python scripts/migrate_to_hierarchical_rag.py --execute

# 4. 驗證結果
python scripts/migrate_to_hierarchical_rag.py --verify

# 如果遇到問題，回滾
python scripts/migrate_to_hierarchical_rag.py --rollback
```

### 上傳新論文

上傳 API 與 legacy 模式完全兼容：

```bash
curl -X POST http://localhost:4000/upload_paper \
  -F "file=@paper.pdf"
```

系統會自動：
1. 提取論文摘要（使用改進的 AbstractExtractor）
2. 更新 Layer 1 索引（摘要）
3. 更新 Layer 2 索引（區塊）
4. 清除相關快取

## 性能監控

### 查看查詢日誌

查詢日誌會自動記錄到 `./vectorstore/query_log.jsonl`（hierarchical 模式）

```bash
# 查看最近 10 條查詢
tail -n 10 ./vectorstore/query_log.jsonl | jq .

# 統計終止層分布
grep -o '"termination_layer":"[^"]*"' ./vectorstore/query_log.jsonl | sort | uniq -c
```

### 性能基準

**典型性能指標**（2 篇論文，317 區塊）：

| 指標 | Layer 1 終止 | Layer 2 終止 |
|------|--------------|--------------|
| 平均查詢時間 | 10-16s | 60-75s |
| Layer 1 檢索 | ~0.1s | ~0.1s |
| Layer 1 評估 | 10-15s | 10-15s |
| Layer 2 檢索 | - | 30-35s |
| Layer 2 評估 | - | 20-30s |
| 上下文擴展 | - | <0.1s |

**早期終止率**：通常 50-70% 查詢在 Layer 1 終止（節省大量時間）

## 疑難排解

### 問題 1：系統初始化失敗

**症狀**：啟動時報錯 "Failed to initialize RAG System"

**解決方案**：
```bash
# 檢查 Ollama 服務
curl http://localhost:11434/api/tags

# 檢查模型是否存在
ollama list | grep gemma3

# 如果缺少模型，下載
ollama pull gemma3:12b
ollama pull embeddinggemma:latest
```

### 問題 2：索引未就緒

**症狀**：健康檢查顯示 "indices not ready"

**解決方案**：
```bash
# 檢查索引文件是否存在
ls -la ./vectorstore/layer1/
ls -la ./vectorstore/layer2/

# 如果不存在，重新建立
python scripts/migrate_to_hierarchical_rag.py --execute
```

### 問題 3：查詢很慢

**可能原因**：
1. **LLM 評估延遲高**（最常見）
   - 解決：使用更快的模型或調整 LLM 設定
   
2. **Layer 2 檢索慢**
   - 解決：增加 `CACHE_SIZE`
   
3. **閾值設太低，很少早期終止**
   - 解決：提高 `LAYER1_THRESHOLD`

**診斷**：
```bash
# 查看性能分解
curl http://localhost:4000/rag/stats | jq '.performance_summary.timing_breakdown'

# 查看終止層分布
curl http://localhost:4000/rag/stats | jq '.query_stats.termination_distribution'
```

### 問題 4：信心分數總是很低

**可能原因**：
1. 文檔與查詢語言不匹配（中英文）
2. 摘要品質不佳
3. 評估 prompt 需要調整

**解決方案**：
```bash
# 檢查摘要提取品質
python -c "
from system_api.hierarchical_rag_system import HierarchicalRAGSystem
rag = HierarchicalRAGSystem(pdf_directory='./data', model_name='gemma3:12b', embedding_model='embeddinggemma:latest', vectorstore_path='./vectorstore')
stats = rag.get_stats()
print('Papers:', stats['layer1']['paper_count'])
"
```

## 最佳實踐

### 1. 開發環境建議

```bash
# 使用 legacy 模式進行快速測試
export RAG_MODE=legacy

# 切換到 hierarchical 進行功能驗證
export RAG_MODE=hierarchical
```

### 2. 生產環境建議

```bash
# 使用 hierarchical 模式獲得最佳效果
export RAG_MODE=hierarchical

# 根據監控調整閾值
export LAYER1_THRESHOLD=0.75  # 基於實際查詢調整
export LAYER2_THRESHOLD=0.82

# 定期備份
python scripts/migrate_to_hierarchical_rag.py --backup
```

### 3. 性能優化

- **增加快取**：如果經常查詢相同論文集
- **調整閾值**：根據 `/rag/stats` 的終止層分布調整
- **監控日誌**：定期檢查查詢日誌，識別慢查詢

### 4. 索引維護

```bash
# 定期重建索引（如果文檔更新頻繁）
python scripts/migrate_to_hierarchical_rag.py --backup
python scripts/migrate_to_hierarchical_rag.py --execute

# 清理舊備份（保留最近 3 個）
ls -t ./vectorstore_backup_* | tail -n +4 | xargs rm -rf
```

## 與 Legacy 模式的對比

| 特性 | Legacy (AcademicRAGSystem) | Hierarchical (HierarchicalRAGSystem) |
|------|---------------------------|--------------------------------------|
| 索引結構 | 單層 FAISS | 雙層 FAISS（摘要 + 區塊） |
| 檢索策略 | Top-K 向量相似度 | 階層式過濾 + 信心評估 |
| 上下文擴展 | 無 | 動態擴展（±1-3 chunks） |
| 早期終止 | 無 | 有（Layer 1 終止） |
| 查詢時間 | 一致 | 可變（早期終止更快） |
| 精確度 | 中等 | 更高（階層式過濾） |
| 摘要品質 | N/A | 改進的提取（中文支援） |
| API 兼容 | - | 完全兼容 |

## 更新日誌

### v3.0 (2025-11-13)
- ✅ 初始發布階層式 RAG 系統
- ✅ 雙模式支援（legacy/hierarchical）
- ✅ 監控端點（/rag/stats, /rag/health, /rag/mode）
- ✅ 遷移腳本
- ✅ 改進的摘要提取（中文支援 100%）

## 支援與反饋

如遇問題或需要支援，請：
1. 檢查本文檔的「疑難排解」部分
2. 查看 `/rag/health` 端點診斷
3. 檢查應用日誌（`agent2.py` 輸出）
4. 提交 Issue（包含錯誤日誌和系統狀態）

---

**文檔版本**: v1.0  
**更新日期**: 2025-11-13  
**適用系統**: agent2.py + HierarchicalRAGSystem v3.0
