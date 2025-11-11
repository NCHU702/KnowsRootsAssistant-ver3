# 決策點：向量存儲持久化實施策略

## 概述

在實施向量存儲持久化之前，有幾個關鍵決策需要確定。本文件列出各選項的利弊，幫助您做出明智選擇。

---

## 🔴 決策 1：PDF 上傳後如何處理？

### 背景
當前系統有 `/upload_paper` endpoint，新 PDF 會存到 `./data`。  
**問題**：如果向量存儲已經加載（從磁盤），新 PDF 不會被包含在查詢中。

### 選項比較

| 選項 | 描述 | 優點 | 缺點 | 實施複雜度 |
|------|------|------|------|-----------|
| **A. 立即完整重建** | Upload 完成後立即調用 `rag_system.reload(force_rebuild=True)` | • 新 PDF 立即可查詢<br>• 邏輯簡單 | • 阻塞 6-10 分鐘<br>• 用戶體驗差<br>• 浪費資源 | ⭐ 簡單 |
| **B. 標記 dirty（延遲重建）** | 刪除 metadata.pkl，下次啟動重建 | • Upload 立即完成<br>• 不阻塞用戶<br>• 簡單實施 | • 新 PDF 要等重啟<br>• 需要提示用戶 | ⭐⭐ 中等 |
| **C. 增量更新** | 只 embed 新 PDF，合併到現有存儲 | • Upload 快（10-30s）<br>• 最佳用戶體驗<br>• 新 PDF 立即可查 | • 實施複雜<br>• 需要額外方法 | ⭐⭐⭐ 複雜 |
| **D. 背景任務重建** | Upload 後觸發背景重建任務 | • Upload 立即完成<br>• 自動更新<br>• 不阻塞服務 | • 需要任務隊列<br>• 複雜度高<br>• 資源競爭 | ⭐⭐⭐⭐ 很複雜 |

### 📊 使用場景分析

#### 如果您的使用模式是：
- **偶爾上傳（每週 < 5 次）** → 推薦 **選項 B**（標記 dirty）
- **頻繁上傳（每天 10+ 次）** → 推薦 **選項 C**（增量更新）
- **立即查詢需求** → 必須 **選項 C** 或 **D**
- **資源受限環境** → 推薦 **選項 B**

### 💡 建議實施路徑

**階段 1 (MVP)**：選項 B - 標記 dirty
```python
# 在 /upload_paper endpoint 最後加上
storage = PDFStorage()
stored_filename = storage.store(temp_file_path, paper_title)

# 標記 RAG 需要重建
if rag_system:
    rag_system.mark_dirty()  # 新方法

return jsonify({
    'success': True,
    'message': '論文上傳成功。請重啟服務以更新搜尋索引。',  # 提示用戶
    'requires_restart': True  # 前端可顯示提示
})
```

**階段 2 (增強)**：選項 C - 增量更新
```python
# 同樣在 upload 後
if rag_system:
    rag_system.add_document(stored_pdf_path)  # 新方法，10-30秒

return jsonify({
    'success': True,
    'message': '論文上傳並已加入搜尋索引',
    'requires_restart': False
})
```

### ❓ 您的決策
請選擇：
- [ ] **選項 A** - 立即重建（簡單但慢）
- [ ] **選項 B** - 標記 dirty（推薦短期方案）
- [x] **選項 C** - 增量更新（推薦長期方案）✅ **已選擇**
- [ ] **選項 D** - 背景任務（過度設計）

---

## 🔴 決策 2：是否接受 16% 失敗率？

### 背景
當前系統由於 Ollama 限制，約 16% 文檔無法 embed（619/3840）。  
這些失敗會在每次重建時**持續失敗**（相同內容、相同限制）。

### 問題
如果保存包含 84% 成功文檔的向量存儲：
- ✅ 優點：下次啟動快（不需重新 embed 成功的）
- ❌ 缺點：失敗的 16% **永遠缺失**（除非強制重建）

### 選項比較

| 選項 | 策略 | 優點 | 缺點 |
|------|------|------|------|
| **A. 接受現狀** | 保存 84%，記錄失敗統計 | • 簡單<br>• 大部分內容可查 | • 部分查詢可能漏信息<br>• 失敗文檔持續缺失 |
| **B. 記錄並重試** | 保存成功的 + 記錄失敗清單 | • 可手動/自動重試<br>• 追蹤問題文檔 | • 需要額外實現<br>• 重試可能仍失敗 |
| **C. 改進清理邏輯** | 優化內容處理減少失敗 | • 提高成功率<br>• 根本解決問題 | • 需要深入調試<br>• 可能仍有失敗 |
| **D. 換 embedding 服務** | 使用更穩定的服務 | • 更高成功率<br>• 更好性能 | • 依賴外部服務<br>• 可能有成本 |

### 📊 影響分析

#### 84% 成功率意味著什麼？
假設 34 個 PDF：
- **成功**：約 29 個 PDF 的大部分內容可查詢
- **失敗**：約 5 個 PDF 的部分章節（目錄頁、封面等）缺失

#### 典型失敗內容：
1. 封面頁（大量空白和標題）
2. 目錄頁（很多點點和頁碼）
3. 特殊格式的表格
4. 超長段落（> 1800 bytes）

💡 **觀察**：失敗的通常是**非核心內容**（封面、目錄），對查詢影響較小。

### 💡 建議策略

**推薦：選項 A + B 組合**

```python
# 1. 保存成功的向量存儲（階段 1）
metadata = {
    'pdf_hash': '...',
    'processing_stats': {
        'total_chunks': 3840,
        'successful': 3221,
        'failed': 619,
        'success_rate': 0.84
    },
    'failed_documents': [  # 記錄失敗詳情
        {
            'source_file': 'paper1.pdf',
            'page': 1,
            'content_preview': '目錄...........',
            'error': 'EOF error',
            'byte_length': 2543
        },
        # ... 最多記錄前 50 個
    ]
}

# 2. 提供重試功能（階段 2）
def retry_failed_documents(self):
    """重試之前失敗的文檔"""
    if not hasattr(self, 'failed_documents'):
        return
    
    logger.info(f"Retrying {len(self.failed_documents)} failed documents...")
    # 嘗試用更激進的清理邏輯
    # ...
```

### ❓ 您的決策
- [x] **A. 接受 84% 成功率**（記錄統計即可）✅ **已選擇**
- [ ] **B. 實施失敗追蹤和重試**（建議加入）
- [ ] **C. 優先改進清理邏輯**（可作為長期目標）
- [ ] **D. 必須 100% 成功率**（考慮換服務）

---

## 🟡 決策 3：是否需要手動重建 API？

### 背景
有時管理員可能想強制重建向量存儲，但不想重啟整個服務。

### 使用場景
1. 懷疑向量存儲損壞
2. 批量上傳多個 PDF 後
3. 查詢質量下降，想重新 embed
4. 測試或除錯

### 選項

| 選項 | 描述 | 適用場景 |
|------|------|---------|
| **不需要** | 只能通過重啟服務重建 | 開發環境，偶爾重啟可接受 |
| **需要** | 提供 `/rag/rebuild` endpoint | 生產環境，高可用性需求 |

### 實施範例

```python
@app.route('/rag/rebuild', methods=['POST'])
def rebuild_rag():
    """手動觸發 RAG 系統重建"""
    if not rag_system:
        return jsonify({'error': 'RAG system not initialized'}), 500
    
    try:
        logger.info("Manual rebuild triggered via API")
        
        # 可選：背景任務執行
        import threading
        def rebuild_in_background():
            rag_system.reload(force_rebuild=True)
            logger.info("Background rebuild completed")
        
        thread = threading.Thread(target=rebuild_in_background)
        thread.start()
        
        return jsonify({
            'status': 'rebuilding',
            'message': '重建已開始，預計需要 6-10 分鐘',
            'estimated_time_seconds': 360
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
```

### ❓ 您的決策
- [x] **不需要**（通過重啟就好）✅ **已選擇**
- [ ] **需要基本版**（同步重建，阻塞 API）
- [ ] **需要進階版**（背景任務，不阻塞）

---

## 🟢 決策 4：向量存儲存放位置

### 選項

| 位置 | 路徑範例 | 優點 | 缺點 |
|------|---------|------|------|
| **專案根目錄** | `./vectorstore/` | 簡單，與代碼在一起 | 可能被 git 追蹤 |
| **data 目錄下** | `./data/vectorstore/` | 與 PDF 在一起 | 目錄結構混雜 |
| **系統臨時目錄** | `/tmp/vectorstore/` | 不占用專案空間 | 重啟可能被清除 |
| **用戶配置** | 可配置路徑 | 靈活性最高 | 需要額外配置 |

### 💡 建議

**推薦**：`./vectorstore/`（專案根目錄）✅ **已選擇**

記得加入 `.gitignore`：
```gitignore
# Vector store cache
vectorstore/
```

---

## 總結與建議實施順序

### ✅ 已選擇方案（基於您的決策）

**決策組合**：
1. **Upload 處理**：選項 C（增量更新）✅
2. **失敗率**：選項 A（接受 84%，記錄統計）✅
3. **手動重建**：不需要（通過重啟服務）✅
4. **存放位置**：`./vectorstore/`（專案根目錄）✅

**總實施時間**：~5-7 小時

**用戶體驗**：
- ✅ 第二次啟動快 50-100 倍（< 10 秒）
- ✅ Upload 後 10-30 秒即可查詢新 PDF
- ✅ 新 PDF 立即可搜尋
- ⚠️ 16% 文檔缺失但不影響主要內容（失敗的通常是封面、目錄等）

### � 實施階段

**Phase 1: 核心持久化**（2-3 小時）
- Hash 計算與驗證
- FAISS 保存/載入
- 統計追蹤

**Phase 2: 增量更新**（3-4 小時）
- `add_document()` 方法
- Upload endpoint 整合
- 測試新 PDF 上傳流程

**Phase 3: 優化穩定性**（可選，2-3 小時）
- 原子性保存
- 磁盤空間檢查
- 詳細日誌和監控

---

## 💭 您的決策摘要

請填寫您的選擇：

```yaml
decisions:
  upload_strategy: "C"  # 增量更新 - 只 embed 新 PDF，合併到現有存儲
  failure_handling: "A"  # 接受現狀 - 保存 84%，記錄失敗統計
  manual_rebuild_api: "no"  # 不需要 - 只能通過重啟服務重建
  storage_location: "./vectorstore/"  # 專案根目錄
  
implementation_plan: "Enhanced"  # 實施增量更新方案
  
notes: |
  選擇增量更新以獲得最佳用戶體驗：
  - Upload 後 10-30 秒即可查詢新 PDF
  - 接受 84% 成功率，記錄失敗統計
  - 不實施手動重建 API，保持系統簡單
  - 向量存儲保存在專案根目錄下
```

---

## 下一步

確定決策後：
1. 更新 `design.md` 反映決策
2. 創建 `tasks.md` 詳細任務清單
3. 開始實施 Phase 1
