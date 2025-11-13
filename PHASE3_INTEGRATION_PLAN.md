# Phase 3: Integration & Deployment Plan

## 概述
將階層式 RAG 系統（HierarchicalRAGSystem）整合到現有的 agent2.py 應用中，同時保持向後兼容性。

## 目標
1. ✅ 保留現有的 AcademicRAGSystem 功能（向後兼容）
2. ✅ 添加 HierarchicalRAGSystem 作為可選增強版本
3. ✅ 提供平滑的遷移路徑
4. ✅ 添加性能監控和統計
5. ✅ 完整的測試和文檔

## 架構變更

### 當前架構 (AcademicRAGSystem)
```
agent2.py
  ├── AcademicRAGSystem (單層 FAISS)
  │   ├── vectorstore: FAISS
  │   ├── query()
  │   ├── query_stream()
  │   └── add_document()
  └── API Endpoints
      ├── /query_stream
      ├── /query
      └── /upload_paper
```

### 新架構 (雙模式支持)
```
agent2.py
  ├── RAG System (可切換)
  │   ├── Mode 1: AcademicRAGSystem (legacy, 預設)
  │   │   └── 單層 FAISS 檢索
  │   └── Mode 2: HierarchicalRAGSystem (新版)
  │       ├── Layer 1: 論文摘要索引
  │       ├── Layer 2: 區塊級索引
  │       ├── Confidence Evaluator
  │       └── Context Expander
  └── API Endpoints (增強)
      ├── /query_stream (支援兩種模式)
      ├── /query (支援兩種模式)
      ├── /upload_paper (更新索引管理)
      └── /rag/stats (新增：監控統計)
```

## 實作策略

### Phase 3.1: 遷移策略文檔 ✅
- [x] 創建本文檔
- [x] 定義 API 兼容性
- [x] 規劃配置選項

### Phase 3.2: 添加 Hierarchical RAG 選項
**目標**: 在 agent2.py 中添加配置開關，支援雙模式

**實作步驟**:
1. 添加環境變數或配置文件控制 RAG 模式
2. 初始化時根據配置選擇 RAG 系統
3. 提供統一的接口（query/query_stream/add_document）

**配置選項**:
```python
# 環境變數或 config.json
RAG_MODE = "hierarchical"  # or "legacy"
HIERARCHICAL_RAG_CONFIG = {
    "layer1_threshold": 0.7,
    "layer2_threshold": 0.8,
    "enable_context_expansion": True,
    "cache_size": 10
}
```

**代碼變更**:
```python
# agent2.py 修改示例
RAG_MODE = os.getenv('RAG_MODE', 'legacy')  # 預設使用 legacy 保證兼容

if RAG_MODE == 'hierarchical':
    from system_api.hierarchical_rag_system import HierarchicalRAGSystem
    rag_system = HierarchicalRAGSystem(
        pdf_directory="./data",
        model_name=model_name,
        embedding_model="embeddinggemma:latest",
        vectorstore_path="./vectorstore",
        chunk_size=800,
        chunk_overlap=100
    )
else:
    from system_api.rag_system import AcademicRAGSystem
    rag_system = AcademicRAGSystem(...)
```

### Phase 3.3: 索引遷移腳本
**目標**: 從舊的單層 FAISS 遷移到雙層架構

**腳本功能**:
1. 讀取現有 vectorstore（如果存在）
2. 重新處理所有 PDF，提取摘要
3. 建立 Layer 1 和 Layer 2 索引
4. 備份舊索引
5. 驗證遷移結果

**腳本位置**: `scripts/migrate_to_hierarchical_rag.py`

**執行流程**:
```bash
# 1. 備份現有索引
python scripts/migrate_to_hierarchical_rag.py --backup

# 2. 執行遷移（dry-run）
python scripts/migrate_to_hierarchical_rag.py --dry-run

# 3. 正式遷移
python scripts/migrate_to_hierarchical_rag.py --execute

# 4. 驗證
python scripts/migrate_to_hierarchical_rag.py --verify
```

### Phase 3.4: 更新 API 端點
**目標**: 確保現有 API 在兩種模式下都能正常工作

**需要更新的端點**:
1. `/query_stream` - 支援階層式檢索的流式輸出
2. `/query` - 支援階層式檢索
3. `/upload_paper` - 使用新的 IndexManager
4. `/rag/stats` - 新增統計端點

**API 兼容性保證**:
- 請求/響應格式不變
- 添加可選的響應字段（用於監控）
- 錯誤處理向後兼容

**新增響應字段（可選）**:
```json
{
  "answer": "...",
  "rag_mode": "hierarchical",
  "retrieval_stats": {
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

### Phase 3.5: 監控儀表板端點
**目標**: 提供性能監控和系統健康檢查

**新增端點**:
```python
@app.route('/rag/stats', methods=['GET'])
def rag_stats():
    """獲取 RAG 系統統計"""
    
@app.route('/rag/health', methods=['GET'])
def rag_health():
    """健康檢查"""
    
@app.route('/rag/query_log', methods=['GET'])
def rag_query_log():
    """查詢日誌（最近 N 條）"""
```

**統計資訊**:
- 索引狀態（Layer 1/2 文檔數）
- 查詢性能（平均時間、信心度分布）
- 終止層分布（Layer 1 vs Layer 2）
- 快取命中率

### Phase 3.6: 整合測試
**目標**: 端到端驗證整合

**測試場景**:
1. Legacy 模式下的完整流程（確保無破壞）
2. Hierarchical 模式下的查詢流程
3. 模式切換測試
4. 上傳新論文並更新索引
5. 壓力測試（並發查詢）

**測試腳本**: `tests/test_agent2_integration.py`

### Phase 3.7: 文檔和部署
**目標**: 完整的部署指南和使用文檔

**文檔內容**:
1. **MIGRATION_GUIDE.md**: 遷移步驟指南
2. **API_REFERENCE.md**: 更新的 API 文檔
3. **HIERARCHICAL_RAG_USAGE.md**: 使用說明
4. **DEPLOYMENT.md**: 部署清單
5. **PERFORMANCE_TUNING.md**: 性能調優建議

## 配置文件設計

### config.json (新增)
```json
{
  "rag": {
    "mode": "hierarchical",
    "hierarchical_config": {
      "layer1_threshold": 0.7,
      "layer2_threshold": 0.8,
      "enable_context_expansion": true,
      "context_expansion_mapping": {
        "high": 1,
        "medium": 2,
        "low": 3
      },
      "cache_size": 10,
      "enable_query_logging": true,
      "query_log_path": "./logs/query_log.jsonl"
    },
    "legacy_config": {
      "k_documents": 4
    }
  },
  "models": {
    "llm": "gemma3:12b",
    "embedding": "embeddinggemma:latest"
  },
  "paths": {
    "pdf_directory": "./data",
    "vectorstore_path": "./vectorstore",
    "backup_path": "./backups"
  }
}
```

## 風險管理

### 風險 1: 索引遷移失敗
- **緩解**: 完整備份 + dry-run 模式
- **回退**: 保留舊索引，可隨時切回 legacy 模式

### 風險 2: 性能下降（評估延遲）
- **緩解**: 添加快取、調整閾值
- **監控**: 實時性能統計

### 風險 3: API 破壞性變更
- **緩解**: 嚴格向後兼容
- **測試**: 完整的回歸測試套件

## 成功指標

### 功能指標
- [x] Phase 1-2 所有測試通過（已完成）
- [ ] Legacy 模式測試通過
- [ ] Hierarchical 模式測試通過
- [ ] 索引遷移成功率 100%
- [ ] API 兼容性測試通過

### 性能指標
- [ ] Layer 1 早期終止率 > 50%（減少計算）
- [ ] 平均查詢時間 < 15s（Layer 1 終止）
- [ ] 摘要提取成功率 > 90%
- [ ] 信心評估準確性（人工驗證樣本）

### 用戶體驗指標
- [ ] 無需重新上傳文檔（平滑遷移）
- [ ] API 響應格式一致
- [ ] 錯誤提示清晰
- [ ] 部署文檔完整

## 時程規劃

### Week 1
- [x] Phase 1-2 完成（已完成）
- [ ] Phase 3.1-3.2: 添加配置和雙模式支持
- [ ] Phase 3.3: 遷移腳本初版

### Week 2
- [ ] Phase 3.4: API 端點更新
- [ ] Phase 3.5: 監控端點
- [ ] Phase 3.6: 整合測試

### Week 3
- [ ] Phase 3.7: 文檔編寫
- [ ] 用戶驗收測試
- [ ] 正式部署

## 下一步行動

1. **立即**: 開始 Phase 3.2 - 添加配置和雙模式支持到 agent2.py
2. **並行**: 編寫遷移腳本（Phase 3.3）
3. **後續**: 更新 API 端點並添加監控

---

**文檔創建時間**: 2025-11-13  
**狀態**: Phase 3.1 完成，開始 Phase 3.2
