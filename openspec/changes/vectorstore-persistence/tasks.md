# 任務清單：向量存儲持久化實施

## 概述

根據 [DECISIONS.md](./DECISIONS.md) 的決策，本任務清單詳細列出實施步驟。

**核心目標**：
- ✅ 第二次啟動時間 < 10 秒（快 50-100 倍）
- ✅ Upload 後 10-30 秒即可查詢新 PDF
- ✅ 接受 84% 成功率，記錄失敗統計
- ✅ 向量存儲保存在 `./vectorstore/`

---

## Phase 1: 核心持久化（基礎）

**預計時間**：2-3 小時  
**目標**：實現基本的保存/載入功能

### Task 1.1: 添加配置和路徑管理
**文件**：`system_api/rag_system.py`

```python
class AcademicRAGSystem:
    def __init__(self, ...):
        # ... 現有代碼
        
        # 新增：向量存儲路徑
        self.vectorstore_path = "./vectorstore"
        self.metadata_path = os.path.join(self.vectorstore_path, "metadata.pkl")
```

- [ ] 添加 `vectorstore_path` 屬性
- [ ] 添加 `metadata_path` 屬性
- [ ] 確保路徑在初始化時創建

### Task 1.2: 實現 PDF Hash 計算
**文件**：`system_api/rag_system.py`

```python
def _get_pdf_hash(self) -> str:
    """計算 PDF 文件和配置的哈希值"""
    import hashlib
    
    pdf_files = [f for f in os.listdir(self.pdf_directory) 
                 if f.endswith('.pdf')]
    pdf_files.sort()
    
    hash_input = ""
    for pdf_file in pdf_files:
        pdf_path = os.path.join(self.pdf_directory, pdf_file)
        mtime = os.path.getmtime(pdf_path)
        size = os.path.getsize(pdf_path)
        hash_input += f"{pdf_file}:{mtime}:{size};"
    
    # 添加配置參數
    hash_input += f"chunk_size:{self.chunk_size};"
    hash_input += f"chunk_overlap:{self.chunk_overlap};"
    hash_input += f"embedding_model:{self.embedding_model};"
    
    return hashlib.sha256(hash_input.encode()).hexdigest()
```

- [ ] 實現 `_get_pdf_hash()` 方法
- [ ] 包含文件名、mtime、size
- [ ] 包含 chunk_size、chunk_overlap、embedding_model
- [ ] 測試 hash 計算正確性

### Task 1.3: 實現保存功能
**文件**：`system_api/rag_system.py`

```python
def _save_vectorstore(self) -> bool:
    """保存向量存儲到磁盤"""
    try:
        import time
        start_time = time.time()
        
        logger.info("Saving vectorstore to disk...")
        
        # 確保目錄存在
        os.makedirs(self.vectorstore_path, exist_ok=True)
        
        # 保存 FAISS 索引
        self.vectorstore.save_local(self.vectorstore_path)
        
        # 保存 metadata
        metadata = {
            'pdf_hash': self._get_pdf_hash(),
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'embedding_model': self.embedding_model,
            'created_at': time.time(),
            'processing_stats': {
                'total_chunks': len(self.vectorstore.docstore._dict),
                'successful': self._successful_count,  # 需要追蹤
                'failed': self._failed_count,  # 需要追蹤
                'success_rate': self._successful_count / (self._successful_count + self._failed_count)
            }
        }
        
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        duration = time.time() - start_time
        logger.info(f"Vectorstore saved successfully in {duration:.2f}s")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save vectorstore: {e}")
        return False
```

- [ ] 實現 `_save_vectorstore()` 方法
- [ ] 調用 `vectorstore.save_local()`
- [ ] 創建並保存 metadata（包含 hash、配置、統計）
- [ ] 添加錯誤處理
- [ ] 記錄保存時間

### Task 1.4: 實現載入功能
**文件**：`system_api/rag_system.py`

```python
def _load_vectorstore(self) -> bool:
    """從磁盤載入向量存儲"""
    try:
        import time
        start_time = time.time()
        
        # 檢查文件是否存在
        if not os.path.exists(self.metadata_path):
            logger.info("No saved vectorstore found")
            return False
        
        # 載入 metadata
        with open(self.metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # 驗證 hash
        current_hash = self._get_pdf_hash()
        if metadata['pdf_hash'] != current_hash:
            logger.info("PDF files or config changed, rebuild required")
            logger.info(f"  Saved hash: {metadata['pdf_hash'][:16]}...")
            logger.info(f"  Current hash: {current_hash[:16]}...")
            return False
        
        # 載入 FAISS 索引
        logger.info("Loading vectorstore from disk...")
        self.vectorstore = FAISS.load_local(
            self.vectorstore_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        # 重建 retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        duration = time.time() - start_time
        stats = metadata.get('processing_stats', {})
        logger.info(f"Vectorstore loaded successfully in {duration:.2f}s")
        logger.info(f"  Chunks: {stats.get('successful', 'unknown')}")
        logger.info(f"  Success rate: {stats.get('success_rate', 0):.1%}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to load vectorstore: {e}")
        return False
```

- [ ] 實現 `_load_vectorstore()` 方法
- [ ] 檢查 metadata 文件存在
- [ ] 載入並驗證 hash
- [ ] 調用 `FAISS.load_local()`
- [ ] 重建 retriever
- [ ] 添加詳細日誌
- [ ] 錯誤處理

### Task 1.5: 追蹤成功/失敗計數
**文件**：`system_api/rag_system.py`

在 `_load_documents()` 方法中：

```python
def _load_documents(self) -> List[Document]:
    # ... 現有代碼
    
    # 初始化計數器
    self._successful_count = 0
    self._failed_count = 0
    
    for doc in documents:
        try:
            # ... embed 邏輯
            self._successful_count += 1
        except Exception as e:
            self._failed_count += 1
            # ... 錯誤處理
```

- [ ] 添加 `_successful_count` 和 `_failed_count` 屬性
- [ ] 在文檔處理循環中更新計數
- [ ] 確保統計準確

### Task 1.6: 整合到 `_initialize()`
**文件**：`system_api/rag_system.py`

```python
def _initialize(self):
    """Initialize the RAG system"""
    if self.initialized:
        logger.info("RAG system already initialized")
        return
    
    try:
        logger.info("Initializing RAG system...")
        
        # 嘗試載入已保存的向量存儲
        if self._load_vectorstore():
            self.initialized = True
            logger.info("RAG system initialized from cache")
            return
        
        # 載入失敗，從頭開始構建
        logger.info("Building vectorstore from scratch...")
        documents = self._load_documents()
        
        # ... 現有的構建邏輯
        
        # 構建完成後保存
        self._save_vectorstore()
        
        self.initialized = True
        logger.info("RAG system initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG: {e}")
        raise
```

- [ ] 修改 `_initialize()` 方法
- [ ] 先嘗試 `_load_vectorstore()`
- [ ] 載入成功則直接返回
- [ ] 載入失敗則從頭構建
- [ ] 構建完成後調用 `_save_vectorstore()`
- [ ] 測試兩種路徑都正常工作

### Task 1.7: 添加到 `.gitignore`
**文件**：`.gitignore`（專案根目錄）

```gitignore
# Vector store cache
vectorstore/
```

- [ ] 確認 `.gitignore` 存在
- [ ] 添加 `vectorstore/` 條目
- [ ] 測試 git 不會追蹤向量存儲

---

## Phase 2: 增量更新功能（核心）

**預計時間**：3-4 小時  
**目標**：實現新 PDF 快速添加功能

### Task 2.1: 實現 `add_document()` 方法
**文件**：`system_api/rag_system.py`

```python
def add_document(self, pdf_path: str) -> Dict[str, Any]:
    """
    添加單個 PDF 到現有向量存儲
    
    Args:
        pdf_path: PDF 文件的絕對路徑
        
    Returns:
        包含狀態和統計的字典
    """
    import time
    start_time = time.time()
    
    try:
        if not self.initialized:
            raise RuntimeError("RAG system not initialized")
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        logger.info(f"Adding document: {pdf_path}")
        
        # 1. 載入新 PDF
        loader = PyMuPDFLoader(pdf_path)
        pages = loader.load()
        
        # 2. 分割文檔
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        new_documents = text_splitter.split_documents(pages)
        
        # 3. 清理內容（使用現有邏輯）
        cleaned_docs = []
        failed = 0
        
        for doc in new_documents:
            try:
                # 應用相同的清理邏輯
                content = doc.page_content
                if len(content.encode('utf-8')) > 1800:
                    content = content[:700]
                doc.page_content = content
                cleaned_docs.append(doc)
            except:
                failed += 1
        
        # 4. Embed 新文檔
        logger.info(f"Embedding {len(cleaned_docs)} chunks...")
        new_texts = [doc.page_content for doc in cleaned_docs]
        new_metadatas = [doc.metadata for doc in cleaned_docs]
        
        # 5. 添加到現有 FAISS 索引
        self.vectorstore.add_texts(
            texts=new_texts,
            metadatas=new_metadatas
        )
        
        # 6. 更新統計
        self._successful_count += len(cleaned_docs)
        self._failed_count += failed
        
        # 7. 保存更新後的向量存儲
        self._save_vectorstore()
        
        duration = time.time() - start_time
        
        result = {
            'status': 'success',
            'pdf_path': pdf_path,
            'chunks_added': len(cleaned_docs),
            'chunks_failed': failed,
            'duration_seconds': duration
        }
        
        logger.info(f"Document added successfully: {len(cleaned_docs)} chunks in {duration:.1f}s")
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to add document: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }
```

- [ ] 實現 `add_document()` 方法
- [ ] 載入並分割新 PDF
- [ ] 應用相同的清理邏輯
- [ ] Embed 新文檔
- [ ] 使用 `vectorstore.add_texts()` 合併
- [ ] 更新統計
- [ ] 保存向量存儲
- [ ] 返回詳細結果
- [ ] 添加錯誤處理

### Task 2.2: 整合到 Upload Endpoint
**文件**：`agent2.py`

找到 `/upload_paper` endpoint：

```python
@app.route('/upload_paper', methods=['POST'])
def upload_paper():
    try:
        # ... 現有的上傳邏輯
        
        # 保存 PDF
        storage = PDFStorage()
        stored_filename = storage.store(temp_file_path, paper_title)
        stored_pdf_path = os.path.join('./data', stored_filename)
        
        # 新增：添加到 RAG 系統
        if rag_system and rag_system.initialized:
            logger.info("Adding new PDF to RAG system...")
            result = rag_system.add_document(stored_pdf_path)
            
            if result['status'] == 'success':
                logger.info(f"PDF indexed: {result['chunks_added']} chunks")
                message = f'論文上傳成功並已加入搜尋索引（{result["chunks_added"]} 個片段）'
            else:
                logger.warning(f"PDF upload succeeded but indexing failed: {result.get('error')}")
                message = '論文上傳成功，但索引失敗。請重啟服務以更新搜尋。'
        else:
            message = '論文上傳成功'
        
        return jsonify({
            'success': True,
            'message': message,
            'filename': stored_filename
        })
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
```

- [ ] 找到 `upload_paper()` 函數
- [ ] 在 PDF 保存後調用 `rag_system.add_document()`
- [ ] 處理返回結果
- [ ] 更新返回消息
- [ ] 測試上傳流程

### Task 2.3: 測試增量更新
**測試步驟**：

1. **首次啟動測試**
   - [ ] 啟動 agent2.py
   - [ ] 確認從頭構建向量存儲
   - [ ] 確認保存成功
   - [ ] 檢查 `./vectorstore/` 目錄存在

2. **重啟測試**
   - [ ] 停止並重新啟動 agent2.py
   - [ ] 確認從緩存載入（< 10 秒）
   - [ ] 確認查詢功能正常

3. **Upload 測試**
   - [ ] 上傳新 PDF
   - [ ] 確認處理時間 10-30 秒
   - [ ] 立即查詢新 PDF 內容
   - [ ] 確認能找到新內容

4. **Hash 變化測試**
   - [ ] 刪除一個 PDF
   - [ ] 重啟服務
   - [ ] 確認重新構建（hash 不匹配）

---

## Phase 3: 優化與穩定性（可選）

**預計時間**：2-3 小時  
**目標**：提高系統穩定性和用戶體驗

### Task 3.1: 原子性保存
**文件**：`system_api/rag_system.py`

```python
def _save_vectorstore(self) -> bool:
    """原子性保存向量存儲"""
    try:
        import shutil
        
        temp_path = self.vectorstore_path + ".tmp"
        
        # 1. 保存到臨時目錄
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
        os.makedirs(temp_path, exist_ok=True)
        
        self.vectorstore.save_local(temp_path)
        
        # 2. 保存 metadata
        temp_metadata = os.path.join(temp_path, "metadata.pkl")
        with open(temp_metadata, 'wb') as f:
            pickle.dump(metadata, f)
        
        # 3. 原子性替換
        if os.path.exists(self.vectorstore_path):
            backup_path = self.vectorstore_path + ".old"
            if os.path.exists(backup_path):
                shutil.rmtree(backup_path)
            shutil.move(self.vectorstore_path, backup_path)
        
        shutil.move(temp_path, self.vectorstore_path)
        
        # 4. 清理舊備份
        if os.path.exists(backup_path):
            shutil.rmtree(backup_path)
        
        return True
        
    except Exception as e:
        logger.error(f"Atomic save failed: {e}")
        # 清理臨時文件
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
        return False
```

- [ ] 實現臨時目錄保存
- [ ] 驗證保存成功
- [ ] 原子性替換
- [ ] 清理舊備份
- [ ] 測試中斷場景

### Task 3.2: 磁盤空間檢查
**文件**：`system_api/rag_system.py`

```python
def _check_disk_space(self, required_mb: int = 100) -> bool:
    """檢查磁盤空間是否足夠"""
    import shutil
    
    try:
        stat = shutil.disk_usage(self.vectorstore_path)
        available_mb = stat.free / (1024 * 1024)
        
        if available_mb < required_mb:
            logger.warning(f"Low disk space: {available_mb:.1f} MB available")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Failed to check disk space: {e}")
        return True  # 不要因為檢查失敗而阻止操作
```

- [ ] 實現磁盤空間檢查
- [ ] 在保存前調用
- [ ] 記錄警告但不阻止
- [ ] 測試低空間場景

### Task 3.3: 詳細統計和日誌
**文件**：`system_api/rag_system.py`

增強 metadata：

```python
metadata = {
    'pdf_hash': self._get_pdf_hash(),
    'chunk_size': self.chunk_size,
    'chunk_overlap': self.chunk_overlap,
    'embedding_model': self.embedding_model,
    'created_at': time.time(),
    'updated_at': time.time(),
    'build_duration_seconds': duration,
    'processing_stats': {
        'total_chunks': self._successful_count + self._failed_count,
        'successful': self._successful_count,
        'failed': self._failed_count,
        'success_rate': self._successful_count / (self._successful_count + self._failed_count),
        'pdf_count': len([f for f in os.listdir(self.pdf_directory) if f.endswith('.pdf')])
    },
    'vectorstore_size_mb': self._get_vectorstore_size()
}
```

- [ ] 添加 `updated_at` 時間戳
- [ ] 記錄構建時長
- [ ] 記錄 PDF 數量
- [ ] 計算向量存儲大小
- [ ] 在日誌中顯示詳細統計

### Task 3.4: 性能監控
**文件**：`system_api/rag_system.py`

```python
def get_stats(self) -> Dict[str, Any]:
    """獲取 RAG 系統統計信息"""
    if not self.initialized:
        return {'initialized': False}
    
    metadata_path = self.metadata_path
    if os.path.exists(metadata_path):
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
    else:
        metadata = {}
    
    return {
        'initialized': True,
        'vectorstore_path': self.vectorstore_path,
        'chunk_count': len(self.vectorstore.docstore._dict),
        'metadata': metadata
    }
```

- [ ] 實現 `get_stats()` 方法
- [ ] 可選：添加 API endpoint `/rag/stats`
- [ ] 用於監控和除錯

---

## 測試清單

### 功能測試

- [ ] **首次啟動**：從頭構建並保存
- [ ] **第二次啟動**：快速載入（< 10 秒）
- [ ] **查詢功能**：載入後查詢正常工作
- [ ] **Upload 新 PDF**：10-30 秒內可查詢
- [ ] **PDF 變化檢測**：添加/刪除 PDF 觸發重建
- [ ] **配置變化檢測**：修改 chunk_size 觸發重建

### 錯誤處理測試

- [ ] **載入失敗**：損壞的 index 文件
- [ ] **保存失敗**：磁盤空間不足
- [ ] **Hash 不匹配**：正確重建
- [ ] **Upload 失敗**：不影響現有索引

### 性能測試

- [ ] **首次構建時間**：與當前系統相當（6-10 分鐘）
- [ ] **載入時間**：< 10 秒
- [ ] **Upload 處理時間**：10-30 秒
- [ ] **查詢延遲**：無明顯變化

---

## 驗收標準

### Must Have (Phase 1 + 2)
- ✅ 第二次啟動 < 10 秒
- ✅ Upload 後 10-30 秒可查詢新 PDF
- ✅ 自動檢測 PDF 變化（名稱、時間、大小）
- ✅ 自動檢測配置變化（chunk_size 等）
- ✅ 記錄成功/失敗統計
- ✅ `agent2.py` 無需修改（除了 upload endpoint）
- ✅ 向量存儲保存在 `./vectorstore/`

### Should Have (Phase 3)
- 🔄 原子性保存操作
- 🔄 磁盤空間檢查
- 🔄 詳細統計和日誌

### Won't Have (已決定不需要)
- ❌ 手動重建 API endpoint
- ❌ 失敗文檔重試機制
- ❌ 多版本向量存儲管理

---

## 實施順序建議

1. **Day 1**（2-3 小時）
   - Task 1.1 - 1.7（Phase 1 全部）
   - 測試首次啟動和重啟

2. **Day 2**（3-4 小時）
   - Task 2.1 - 2.3（Phase 2 全部）
   - 測試 Upload 流程

3. **Day 3**（可選，2-3 小時）
   - Task 3.1 - 3.4（Phase 3）
   - 完整測試和優化

---

## 完成檢查

Phase 1 完成標準：
- [ ] 所有 Task 1.1 - 1.7 完成
- [ ] 功能測試通過
- [ ] 重啟時間 < 10 秒

Phase 2 完成標準：
- [ ] 所有 Task 2.1 - 2.3 完成
- [ ] Upload 測試通過
- [ ] Upload 處理時間 10-30 秒

Phase 3 完成標準：
- [ ] 所有 Task 3.1 - 3.4 完成
- [ ] 錯誤處理測試通過
- [ ] 性能測試通過

---

## 問題追蹤

遇到問題時記錄在這裡：

### 問題 1：[標題]
**描述**：  
**解決方案**：  
**狀態**：

---

**最後更新**：2025-11-11  
**負責人**：  
**審核人**：