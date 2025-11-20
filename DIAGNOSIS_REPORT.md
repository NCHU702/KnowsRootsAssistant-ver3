# 索引一致性診斷報告

## 執行日期
2025-11-20

## 問題摘要

你報告的問題：「我明明添加新的論文，但資料庫並沒有更新。./test_data 中只有5篇論文，結果給出7篇論文。」

## 根本原因分析

### 1. **重複索引問題**（主要問題）

**發現：** Layer1 索引中存在重複的論文條目，同一篇 PDF 被索引了兩次，使用了不同的 `paper_id`：

#### 重複案例 1：`基於深度學習之鼻咽癌腫塊辨識_20251106_144224.pdf`
- **條目 A**: paper_id = `基於深度學習之鼻咽癌腫塊辨識_20251106_144224`（直接使用 filename）
- **條目 B**: paper_id = `6586db341cade520`（使用 MD5 hash）

#### 重複案例 2：`標準6_利用-RBF-UNet-以提昇刀具磨耗預測之準確率.pdf`
- **條目 A**: paper_id = `標準6_利用-RBF-UNet-以提昇刀具磨耗預測之準確率`（直接使用 filename）
- **條目 B**: paper_id = `774882dc48974abd`（使用 MD5 hash）

### 2. **paper_id 生成策略不一致**

系統中存在兩種不同的 paper_id 生成方式：

#### 方法 A：使用 filename 作為 paper_id
可能來源：
- 早期版本的 `Layer1VectorStore.add_abstract()`
- 某些手動添加的程式碼
- `AbstractExtractor` 的早期實作

#### 方法 B：使用 MD5(pdf_path) 作為 paper_id
來源：
- `HierarchicalRAGSystem.add_document()` (line 475):
  ```python
  paper_id = hashlib.md5(pdf_path.encode()).hexdigest()[:16]
  ```
- `IndexManager.add_document()` 同樣使用相同邏輯

### 3. **實際數量對照表**

| 數據源 | 數量 | 備註 |
|--------|------|------|
| 文件系統 (./test_data) | **5** | ✓ 正確 |
| Layer1 索引 | **7** | ✗ 包含 2 個重複項 |
| Layer2 索引 | **7** | ✗ 對應 Layer1 的 7 個條目 |
| Neo4j (Graph DB) | **未知** | ⚠️  無法連接（driver 問題）|

### 4. **Layer1 完整內容（7 個條目）**

```
[1] paper_id: 基於深度學習之鼻咽癌腫塊辨識_20251106_144224 (filename 方式)
    pdf_path: (空)

[2] paper_id: 標準6_利用-RBF-UNet-以提昇刀具磨耗預測之準確率 (filename 方式)
    pdf_path: (空)

[3] paper_id: 6586db341cade520 (hash 方式)
    pdf_path: ./test_data/基於深度學習之鼻咽癌腫塊辨識_20251106_144224.pdf
    ⚠️  與 [1] 重複

[4] paper_id: 5bfb26eca253fdbd (hash 方式)
    pdf_path: ./test_data/標準14_基於集成式生成對抗網路進行人流異常預測.pdf

[5] paper_id: c73994db08b45b08 (hash 方式)
    pdf_path: ./test_data/標準19_應用自動編碼器架構於影片生成文字敘述-以美國職業籃球聯賽比賽為例.pdf

[6] paper_id: 774882dc48974abd (hash 方式)
    pdf_path: ./test_data/標準6_利用-RBF-UNet-以提昇刀具磨耗預測之準確率.pdf
    ⚠️  與 [2] 重複

[7] paper_id: f83f120a885cf1e5 (hash 方式)
    pdf_path: ./test_data/標準8_以 CFRBF-UNet 進行全市人流預測的知識轉移學習.pdf
```

## 為什麼會出現重複？

### 推測的時間線：

1. **第一次索引**（可能是早期版本或測試）：
   - 某些論文使用 **filename** 作為 paper_id 被加入
   - 這些條目的 `pdf_path` 為空（可能是手動添加或早期程式碼的 bug）

2. **第二次索引**（使用當前的 HierarchicalRAGSystem）：
   - 系統檢測到 5 個 PDF 檔案
   - 使用 **MD5 hash** 作為 paper_id
   - 因為 paper_id 不同，系統認為這些是「新」論文
   - 結果：重複添加了已存在的論文

## 為什麼新添加的論文沒寫入 Neo4j？

### 可能的原因：

1. **IndexManager 的 Graph 元件未正確連接**
   - 在我剛才的修改中，已經添加了 `set_graph_components()` 方法
   - 但是在你的系統啟動時，可能還沒有執行這個連接
   - 需要重新啟動 agent2.py 才會生效

2. **Neo4j driver 安裝問題**
   - 當前 venv 中 neo4j driver 有安裝問題
   - 這可能導致 GraphManager 初始化失敗
   - 需要修復安裝

## 解決方案

### 方案 A：清理重複並重建索引（推薦）

#### 步驟 1：備份現有索引
```bash
cp -r ./vectorstore ./vectorstore_backup_$(date +%Y%m%d_%H%M%S)
```

#### 步驟 2：清空索引
```bash
rm -rf ./vectorstore/layer1/*
rm -rf ./vectorstore/layer2/*
```

#### 步驟 3：重新啟動 agent2.py
系統會自動檢測到索引為空，並重新索引所有 PDF（使用統一的 hash 方式）。

#### 步驟 4：執行 batch_index_to_graph.py
```bash
.venv/bin/python batch_index_to_graph.py --data-dir ./test_data
```

### 方案 B：手動清理重複項（較複雜）

創建一個清理腳本，識別並移除 Layer1 中使用 filename 作為 paper_id 的舊條目（[1] 和 [2]）。

### 方案 C：強制統一 paper_id 策略

修改所有相關代碼，確保**所有地方**都使用統一的 paper_id 生成方式（建議使用 MD5 hash）。

## 已完成的修改（task 1: domain_zh 支援）

✅ **GraphDataExtractor** 已經支援 `domain_zh`（第 155 行）
✅ **GraphManager.add_paper_metadata()** 已經接受並儲存 `domain_zh`
✅ **batch_index_to_graph.py** 已經傳遞 `domain_zh`（第 143 行）
✅ **agent2.py upload_paper()** 已經傳遞 `domain_zh`（第 1005 行）
✅ **IndexManager.add_document()** 已經支援在索引成功後呼叫 Graph ingestion（第 172-208 行）
✅ **agent2.py** 已經在初始化後將 graph 元件連接到 IndexManager（第 193-200 行）

## 建議的下一步操作

### 立即執行（推薦順序）：

1. **修復 neo4j driver 安裝**：
   ```bash
   .venv/bin/pip uninstall neo4j -y
   .venv/bin/pip install 'neo4j>=5.17.0,<6.0.0'
   ```

2. **備份並清空現有索引**：
   ```bash
   cp -r ./vectorstore ./vectorstore_backup_$(date +%Y%m%d)
   rm -rf ./vectorstore/layer1/abstract.*
   rm -rf ./vectorstore/layer1/metadata.pkl
   rm -rf ./vectorstore/layer2/chunks.jsonl
   ```

3. **重新啟動 agent2.py**：
   ```bash
   .venv/bin/python agent2.py
   ```
   系統會自動重建索引（5 篇論文，無重複）。

4. **執行 batch_index_to_graph.py**：
   ```bash
   .venv/bin/python batch_index_to_graph.py --data-dir ./test_data
   ```
   這會將所有論文的 metadata（包括 domain_zh）寫入 Neo4j。

5. **驗證一致性**：
   ```bash
   .venv/bin/python reconcile_indices.py
   ```
   應該顯示：
   - Filesystem: 5
   - Layer1: 5
   - Layer2: 5
   - Neo4j: 5

### 長期改進：

1. **統一 paper_id 生成策略**：
   - 建議在所有地方使用 `hashlib.md5(pdf_path.encode()).hexdigest()[:16]`
   - 或創建一個 `generate_paper_id(pdf_path)` 工具函數供全系統使用

2. **添加重複檢測**：
   - 在 `IndexManager.add_document()` 中加入檢查，防止同一個 PDF（根據路徑）被重複索引

3. **修復 requirements.txt**：
   - 添加 `neo4j>=5.17.0,<6.0.0` 到 requirements.txt

## 總結

**問題根源**：paper_id 生成策略不一致，導致同一篇論文被識別為兩篇不同的論文。

**影響範圍**：Layer1 和 Layer2 都有重複項，導致系統顯示 7 篇論文（實際只有 5 篇）。

**解決方案**：清空並重建索引，確保使用統一的 paper_id 策略，並重新執行 Graph indexing。

**Graph RAG (domain_zh) 支援**：✅ 已完成並集成到所有路徑（upload_paper 和 incremental add_document）。
