# Design Review: Vector Store Persistence

## 系統適配性分析

### ✅ 完全適配的部分

1. **現有架構兼容**
   - `AcademicRAGSystem` 類已經封裝良好
   - `agent2.py` 只在啟動時初始化一次
   - 不需要修改 `agent2.py` 的任何代碼

2. **FAISS 支援**
   - LangChain FAISS 有原生的 `save_local()` 和 `load_local()` 方法
   - 已經在專案中使用 `langchain_community.vectorstores.FAISS`
   - 不需要額外依賴

3. **文檔處理流程**
   - `_load_documents()` 方法已經很完整
   - 文檔清理邏輯已經存在
   - 只需在最後加上保存邏輯

### ⚠️ 需要特別注意的部分

#### 1. **複雜的文檔處理流程**
**現況**：
- 系統使用**逐個文檔處理**（individual document processing）
- 每個文檔都有內容清理（1800 bytes 限制、700 字符截斷）
- 失敗率約 16%（619/3840 文檔）
- 處理時間長（6-10 分鐘）

**影響**：
- ✅ 好消息：即使有失敗，成功的文檔（84%）仍會被保存
- ⚠️ 注意：失敗的文檔不會包含在向量存儲中
- ⚠️ 問題：如果 PDF 沒變但處理邏輯改進，舊的失敗文檔不會自動重試

**建議**：
```python
# 在 metadata 中記錄處理統計
metadata = {
    'pdf_hash': self._get_pdf_hash(),
    'chunk_size': self.chunk_size,
    'chunk_overlap': self.chunk_overlap,
    'embedding_model': self.embedding_model,
    'processing_stats': {  # 新增
        'total_chunks': len(documents),
        'successful': success_count,
        'failed': fail_count,
        'success_rate': success_count / len(documents)
    },
    'content_cleaning_params': {  # 新增：記錄清理參數
        'max_bytes': 1800,
        'max_chars': 700
    }
}
```

#### 2. **動態 PDF 新增（Upload 功能）**
**現況**：
- `agent2.py` 有 `/upload_paper` endpoint
- 新 PDF 會被存到 `./data` 目錄
- **但不會自動更新 RAG 系統**

**問題場景**：
```
1. 啟動 agent2.py（加載舊的 vectorstore）
2. 用戶上傳新 PDF 到 ./data
3. RAG 查詢 → 找不到新 PDF 的內容！
```

**解決方案選項**：

**選項 A：上傳後立即重建（簡單但慢）**
```python
# 在 /upload_paper endpoint 最後加上
if rag_system:
    logger.info("Rebuilding RAG system with new PDF...")
    rag_system.reload(force_rebuild=True)
```
- ✅ 簡單實現
- ❌ 每次上傳都要重建全部（6-10 分鐘）
- ❌ 用戶體驗差

**選項 B：增量更新（複雜但快）** ⭐ **推薦**
```python
def add_document(self, pdf_path: str):
    """Add a single PDF to existing vectorstore"""
    # 1. 加載並處理新 PDF
    # 2. 嵌入新文檔
    # 3. 合併到現有 vectorstore
    # 4. 更新保存的索引
    pass
```
- ✅ 快速（只處理新文檔）
- ✅ 用戶體驗好
- ⚠️ 需要額外實現

**選項 C：延遲重建（平衡方案）**
```python
# 標記需要重建，下次啟動時重建
def mark_dirty(self):
    """Mark vectorstore as needing rebuild"""
    metadata_path = os.path.join(self.vectorstore_path, "metadata.pkl")
    if os.path.exists(metadata_path):
        os.remove(metadata_path)  # 刪除 metadata 強制下次重建
```
- ✅ 簡單實現
- ✅ 不阻塞上傳流程
- ⚠️ 新 PDF 要等下次重啟才能查詢

#### 3. **PDF 刪除/修改場景**
**問題**：
- 如果用戶從 `./data` 刪除 PDF，hash 會改變，會重建
- 但如果用戶**修改** PDF 內容但保持檔名，只靠 `mtime` 可能不夠準確

**建議**：
```python
def _get_pdf_hash(self) -> str:
    # 除了 mtime，還加上 file size 作為額外驗證
    for pdf_file in pdf_files:
        pdf_path = os.path.join(self.pdf_directory, pdf_file)
        mtime = os.path.getmtime(pdf_path)
        size = os.path.getsize(pdf_path)  # ✅ 已經有了
        hash_input += f"{pdf_file}:{mtime}:{size};"
```
- ✅ 提案中已經包含 `size`，這點沒問題

#### 4. **Ollama Embedding 的不穩定性**
**現況**：
- 16% 失敗率
- 500 Internal Server Error 常發生
- 逐個處理 + 0.1s 延遲

**擔憂**：
- 如果保存了"成功的 84%"，未來 Ollama 改進後，那 16% 還是缺失的
- 沒有機制重新處理失敗的文檔

**建議**：
```python
# 選項 1：記錄失敗的文檔索引
metadata = {
    ...,
    'failed_documents': [  # 記錄失敗文檔的來源
        {'file': 'paper1.pdf', 'page': 3, 'reason': 'EOF error'},
        ...
    ]
}

# 選項 2：提供手動重試功能
def retry_failed_documents(self):
    """Retry embedding previously failed documents"""
    pass
```

## 需要澄清的問題

### 🔴 高優先級

#### Q1: Upload 後如何處理向量存儲？
**選項**：
- A) 上傳後立即完整重建（慢但簡單）
- B) 實現增量更新功能（快但複雜）
- C) 標記為 dirty，下次重啟重建（簡單但延遲）

**✅ 決策**：**選項 C - 增量更新**（實施增量文檔添加功能）

#### Q2: 是否需要 API endpoint 手動觸發重建？
```python
@app.route('/rag/rebuild', methods=['POST'])
def rebuild_rag():
    """Manually trigger RAG system rebuild"""
    if rag_system:
        rag_system.reload(force_rebuild=True)
        return jsonify({'status': 'rebuilding'})
```

**場景**：
- 管理員發現查詢結果不準確
- 懷疑向量存儲損壞
- 想強制重建而不重啟服務

**✅ 決策**：**不需要**（保持系統簡單，通過重啟服務即可重建）

#### Q3: 16% 失敗率是否可接受？
**現況**：
- 3221 成功 / 3840 總計 = 84% 成功率
- 失敗原因：Ollama 限制、特殊內容格式

**問題**：
- 如果保存這個狀態，那 16% 永遠缺失
- 除非強制重建，否則不會重試

**選項**：
- A) 接受現狀，記錄失敗統計
- B) 實現失敗重試機制
- C) 改進內容清理邏輯減少失敗率

**✅ 決策**：**選項 A - 接受現狀**（保存 84% 成功文檔，記錄失敗統計）

### 🟡 中優先級

#### Q4: 多個向量存儲版本管理？
**場景**：
- 測試不同的 chunk_size
- 比較不同 embedding 模型
- A/B 測試查詢效果

**建議方案**：
```python
# 根據配置生成不同的存儲路徑
def _get_vectorstore_path(self):
    config_hash = hashlib.md5(
        f"{self.chunk_size}:{self.chunk_overlap}:{self.embedding_model}".encode()
    ).hexdigest()[:8]
    return f"./vectorstore_{config_hash}"
```

#### Q5: 磁盤空間限制？
**預估**：
- 34 PDF → ~3840 chunks
- FAISS index: ~30-50 MB
- Metadata: < 1 MB

**如果有 100 個 PDF**：
- 估計 ~11,000 chunks
- FAISS index: ~80-120 MB

**建議**：
- 加入磁盤空間檢查
- 提供清理舊版本的工具

#### Q6: 向量存儲損壞怎麼辦？
**可能原因**：
- 程序崩潰在保存過程中
- 磁盤滿了
- FAISS 版本不兼容

**建議**：
```python
def _save_vectorstore(self):
    try:
        # 1. 保存到臨時目錄
        temp_path = self.vectorstore_path + ".tmp"
        # 2. 驗證保存成功
        # 3. 原子性重命名
        os.rename(temp_path, self.vectorstore_path)
    except:
        # 清理臨時文件
        pass
```

### 🟢 低優先級

#### Q7: 是否需要向量存儲統計信息？
```python
{
    'vectorstore_size_mb': 45.2,
    'last_built': '2025-11-11 10:30:00',
    'build_duration_seconds': 487,
    'pdf_count': 34,
    'chunk_count': 3221,  # 成功的
    'failed_count': 619
}
```

#### Q8: 是否支援遠端存儲（S3, GCS）？
**用途**：
- 多台服務器共享同一個向量存儲
- 避免每台服務器都重建

**建議**：暫時不需要，未來可擴展

## 提案修正建議

### 需要補充的內容

#### 1. Upload 工作流程
```markdown
### PDF Upload Flow

When a new PDF is uploaded via `/upload_paper`:

**✅ Chosen: Option 3 - Incremental Update**
- Store PDF to ./data
- Call `rag_system.add_document(pdf_path)`
- Embed only new PDF (~10-30s)
- Merge into existing vectorstore
- ✅ Best UX
- ✅ Upload completes in 10-30 seconds
- ✅ New PDF immediately searchable

Implementation details:
```python
def add_document(self, pdf_path: str) -> Dict:
    """Add a single PDF to existing vectorstore"""
    # 1. Load and process new PDF
    # 2. Embed new documents
    # 3. Merge into existing FAISS index
    # 4. Update and save metadata
    return {'status': 'success', 'chunks_added': count}
```
```

#### 2. 失敗處理策略
```markdown
### Handling Embedding Failures

Current system has ~16% failure rate due to Ollama limitations.

**Proposed Strategy**:
1. **Accept Partial Success**: Save 84% successful embeddings
2. **Record Failures**: Store failed document metadata
3. **Provide Retry**: Add method to retry failed documents
4. **Log Statistics**: Include in saved metadata

```python
class AcademicRAGSystem:
    def retry_failed_documents(self, failed_list: List[Dict]):
        """Retry embedding for previously failed documents"""
        # Implementation in future PR
        pass
```
```

#### 3. 強制重建 API
```markdown
### Manual Rebuild API

Add endpoint for manual rebuild:

```python
@app.route('/rag/rebuild', methods=['POST'])
def rebuild_rag():
    """Force rebuild of RAG vectorstore"""
    if not rag_system:
        return jsonify({'error': 'RAG system not initialized'}), 500
    
    try:
        logger.info("Manual rebuild triggered")
        rag_system.reload(force_rebuild=True)
        return jsonify({
            'status': 'success',
            'message': 'RAG system rebuilt successfully'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
```

Use cases:
- Admin notices query quality issues
- After bulk PDF updates
- Suspecting vectorstore corruption
```

## 修正後的成功標準

```markdown
## Success Criteria (Updated)

### Must Have (MVP)
- ✅ First startup: Same time as current (~6-10 minutes)
- ✅ Subsequent startups: < 10 seconds (50-100x faster)
- ✅ Automatic detection of PDF file changes (name, mtime, size)
- ✅ Automatic detection of config changes (chunk_size, overlap, model)
- ✅ Zero code changes in `agent2.py`
- ✅ Graceful fallback on load failure
- ✅ Record embedding statistics (success/failure counts)

### Should Have (Phase 2)
- 🔄 Manual rebuild API endpoint
- 🔄 Handle PDF upload scenario (mark dirty on upload)
- 🔄 Atomic save operation (prevent corruption)
- 🔄 Failed document tracking in metadata

### Nice to Have (Future)
- ⭐ Incremental document addition
- ⭐ Multiple vectorstore versions
- ⭐ Retry failed documents
- ⭐ Disk space monitoring
```

## 風險評估更新

### 新增風險

#### Risk 4: Upload 後不一致性
**描述**：用戶上傳新 PDF，但 RAG 查詢找不到  
**影響**：用戶困惑，信任度降低  
**概率**：高（如果不處理）  
**緩解**：
- 短期：Upload 後標記 dirty，提示用戶重啟
- 長期：實現增量更新

#### Risk 5: 部分文檔缺失
**描述**：保存的向量存儲只包含 84% 文檔  
**影響**：某些查詢找不到相關內容  
**概率**：中  
**緩解**：
- 記錄失敗文檔詳情
- 提供重試機制
- 改進內容清理邏輯

## 實施階段修正

### Phase 1: 核心持久化（基礎）
**時間**：2-3 小時

✅ 包含：
- Hash 計算（PDF files + config）
- Save/load FAISS index
- 集成到 `_initialize()`
- 基本錯誤處理
- 記錄統計信息（成功/失敗計數）

### Phase 2: 增量更新功能（核心）
**時間**：3-4 小時

✅ 包含：
- 實現 `add_document()` 方法
- FAISS 索引合併邏輯
- Upload endpoint 整合
- 更新 metadata 追蹤
- 測試新 PDF 上傳流程

❌ 不包含：
- 手動重建 API（已決定不需要）
- 失敗文檔重試（接受現狀）

### Phase 3: 優化與穩定性（可選）
**時間**：2-3 小時

✅ 包含：
- 原子性保存（防止損壞）
- 更詳細的統計和日誌
- 磁盤空間檢查
- 性能優化

## 總結

### ✅ 提案基本可行
- FAISS 持久化技術成熟
- 與現有架構兼容良好
- 性能提升顯著（50-100x）

### ✅ 已確認決策
1. **Upload 後同步問題** → **增量更新**（10-30秒內可查詢新 PDF）
2. **16% 失敗率影響** → **接受現狀**（記錄統計，不實施重試）
3. **手動控制需求** → **不需要**（通過重啟服務重建）
4. **存放位置** → **./vectorstore/**（專案根目錄）

### 📋 實施計劃
1. ✅ Phase 1: 核心持久化（基礎保存/載入）
2. ✅ Phase 2: 增量更新功能（Upload 整合）
3. ⭐ Phase 3: 優化與穩定性（可選）

**預計總時間**：5-7 小時
**核心目標**：Upload 後 10-30 秒即可查詢新 PDF
