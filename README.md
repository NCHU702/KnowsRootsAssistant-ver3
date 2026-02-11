# KnowsRoots Assistant - 階層式 RAG 系統

學術論文智能檢索與分析系統，整合 Agentic AI 與 GraphRAG 技術。

## 系統簡介

學術論文檢索與問答系統，採用**階層式 RAG（Hierarchical RAG）**架構，提供高精確度的論文檢索和智能問答功能。

### 核心特性

- **兩層索引架構**：摘要層 + 區塊層
- **智能信心評估**：LLM 評估答案品質
- **早期終止優化**：50-70% 查詢更快
- **動態上下文擴展**：自動擴展相關內容
- **查詢監控**：完整的性能統計

---

## 系統需求


| 項目       | 需求                        |
| ---------- | --------------------------- |
| **Python** | 3.10 或以上版本             |
| **Ollama** | 本地 LLM 服務（必須運行中） |
| **Docker** | 用於運行 Neo4j 圖資料庫     |
| **記憶體** | 建議 8GB 以上               |
| **硬碟**   | 至少 10GB 可用空間          |

---

## 安裝步驟

### 步驟 1：安裝 Ollama

前往 [Ollama 官網](https://ollama.ai) 下載並安裝。

啟動 Ollama 服務：

```bash
ollama serve
```

### 步驟 2：下載所需的 Ollama 模型

**必要模型（3 個）：**

```bash
# 1. 主要 LLM 模型（用於推理、對話、分析）
ollama pull jcai/llama-3-taiwan-8b-instruct:q4_k_m

# 2. 嵌入模型（用於向量檢索）
ollama pull quentinz/bge-large-zh-v1.5:latest

# 3. 摘要模型（用於 Summarization 分塊模式）
ollama pull llama3.2:latest
```

**可選模型（提升效能）：**

```bash
# Reranker 模型（提升 Layer 2 檢索精度，強烈建議安裝）
ollama pull qllama/bce-reranker-base_v1:latest
```

> **模型說明：**
>
> - `jcai/llama-3-taiwan-8b-instruct:q4_k_m`：針對繁體中文優化的 LLM
> - `quentinz/bge-large-zh-v1.5:latest`：中文語義嵌入模型
> - `llama3.2:latest`：用於段落摘要生成
> - `qllama/bce-reranker-base_v1:latest`：跨編碼器重排序模型

### 步驟 3：安裝並啟動 Neo4j（GraphRAG 圖資料庫）

使用 Docker 快速部署 Neo4j：

```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 \
  -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest
```

**驗證 Neo4j 是否運行：**

開啟瀏覽器訪問 http://localhost:7474，使用以下帳號登入：

- 使用者名稱：`neo4j`
- 密碼：`password`

> **提示：** 若需修改密碼，請在 `system_api/graph_manager.py` 中調整連線參數。

### 步驟 4：安裝 Python 依賴

```bash
# 建議使用虛擬環境
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# 或 .venv\Scripts\activate  # Windows

# 安裝依賴套件
pip install -r requirements.txt
```

### 步驟 5：準備論文資料

將 PDF 論文放入 `test_data/` 資料夾：

```bash
mkdir -p test_data
cp your_papers/*.pdf test_data/
```

> **注意：** 系統啟動時會自動掃描 `test_data/` 資料夾中的 PDF 並建立索引。

### 步驟 6：啟動系統

```bash
python agent2.py
```

**系統會自動執行：**

1. ✓ 檢查 PDF 檔案
2. ✓ 建立 Layer 1 向量索引（論文摘要）
3. ✓ 建立 Layer 2 向量索引（論文區塊）
4. ✓ 提取論文結構化資訊至 Neo4j 圖資料庫
5. ✓ 啟動 Flask Web 服務

**成功啟動後，開啟瀏覽器訪問：**

- **Web 介面**: http://localhost:4000
- **Neo4j 圖資料庫**: http://localhost:7474

---

## 使用方式

### Web 介面功能


| 功能模組                 | 說明         | 使用情境                                      |
| ------------------------ | ------------ | --------------------------------------------- |
| **General Assistant**    | 一般助理     | 查詢論文內容、跨論文比較分析、方法/資料集查詢 |
| **Inheritance Analysis** | 研究傳承分析 | 查看研究脈絡、作者關係分析                    |
| **Data Categories**      | 論文分類搜尋 | 依年份、主題、方法篩選論文，快速瀏覽元資料    |
| **Add New Paper**        | 新增論文     | 上傳 PDF、自動提取元資料、即時更新索引        |

### 查詢範例

**跨論文查詢（使用 GraphRAG）：**

```
哪些論文使用 YOLO 進行物件偵測？
比較使用 LSTM 和 Transformer 的論文
列出所有關於智慧交通的研究
```

**單篇論文查詢（使用 RAG）：**

```
芒果分類那篇論文用什麼資料集？
交通流量預測論文的研究方法是什麼？
鼻咽癌辨識論文的準確率是多少？
```

---

## API 端點

### 查詢

- `POST /query` - 標準查詢
- `POST /query_stream` - 流式查詢

### 監控

- `GET /rag/health` - 健康檢查
- `GET /rag/stats` - 系統統計
- `GET /rag/mode` - 模式資訊

### API 使用範例

```bash
# 健康檢查
curl http://localhost:4000/rag/health

# 查詢統計
curl http://localhost:4000/rag/stats

# 流式查詢（推薦）
curl -X POST http://localhost:4000/query_stream \
  -H "Content-Type: application/json" \
  -d '{"input": "哪些論文使用 YOLO？"}'
```

---

## 系統架構

### 智能路由機制

```
使用者查詢
    ↓
┌─────────────────┐
│   智能路由器      │
│ (Query Router)  │
└────────┬─────────┘
         │
    ┌────┴────┐
    ↓         ↓
┌──────────────┐  ┌──────────────┐
│  GraphRAG    │  │  RAG         │
│  (關係查詢)   │  │  (內容查詢)   │
│              │  │              │
│ • 跨論文比較  │  │ • 單篇論文    │
│ • 方法統計   │  │ • 詳細內容    │
│ • 資料集查詢  │  │ • 深入分析    │
└──────┬───────┘  └──────┬───────┘
       │                 │
       └────────┬────────┘
                ↓
       ┌──────────────┐
       │  LLM 生成答案 │
       └──────────────┘
```

### Hierarchical RAG 檢索流程

```
查詢輸入
    ↓
┌─────────────────────────────────────┐
│ Layer 1: 論文級檢索（基於摘要）        │
│ • 快速篩選相關論文                    │
│ • 向量相似度搜尋                      │
│ • 混合檢索（語義 + 關鍵詞）            │
└────────────┬────────────────────────┘
             ↓
┌─────────────────────────────────────┐
│ 信心度評估 (Confidence Evaluator)   │
│ • LLM 評估答案品質                   │
│ • 決策：是否需進入 Layer 2            │
└────────────┬────────────────────────┘
       ┌─────┴─────┐
       ↓           ↓
   信心度高      信心度低
       ↓           ↓
   返回答案   ┌─────────────────────────┐
             │ Layer 2: 區塊級檢索       │
             │ • 詳細內容檢索            │
             │ • Cross-Encoder Reranking│
             │ • 上下文擴展              │
             └────────┬────────────────┘
                      ↓
              ┌──────────────┐
              │  生成最終答案 │
              └──────────────┘
```

---

## 專案結構

```
KnowsRootsAssistant-ver3/
├── agent2.py                       # 主程式（Flask 應用）
├── requirements.txt                # Python 依賴
├── test_data/                      # PDF 論文存放處
├── data/                           # 已處理的論文
├── vectorstore/                    # 向量索引儲存
│   ├── layer1/                     # Layer 1: 論文摘要索引
│   └── layer2/                     # Layer 2: 區塊級索引
├── system_api/                     # 核心模組
│   ├── hierarchical_rag_system.py  # 階層式 RAG 主系統
│   ├── layer1_vectorstore.py       # 論文級檢索
│   ├── layer2_vectorstore.py       # 區塊級檢索
│   ├── graph_manager.py            # Neo4j 圖譜管理
│   ├── graph_extractor.py          # 論文結構化提取
│   ├── abstract_extractor.py       # 摘要提取
│   ├── confidence_evaluator.py     # 信心評估
│   └── context_expander.py         # 上下文擴展
├── scripts/                        # 工具腳本
├── tests/                          # 測試
├── templates/                      # Web 前端
└── json_files/                     # 論文元資料
```

---

## 測試

```bash
# 運行組件測試
python tests/test_hierarchical_components.py

# 運行整合測試
python tests/test_hierarchical_integration.py

# 運行 agent2.py 整合測試
python tests/test_agent2_integration.py

# 或手動測試
curl http://localhost:4000/rag/health
```

---

## 配置

### 環境變數（可選）


| 變數名稱              | 預設值          | 說明                            |
| --------------------- | --------------- | ------------------------------- |
| `LAYER1_K_DOCUMENTS`  | 10              | Layer 1 檢索的論文數量          |
| `LAYER1_THRESHOLD`    | 0.6             | Layer 1 信心度閾值              |
| `LAYER2_THRESHOLD`    | 0.6             | Layer 2 信心度閾值              |
| `ENABLE_EXPANSION`    | true            | 啟用上下文擴展                  |
| `CHUNKING_MODE`       | naive           | 分塊模式（naive/summarization） |
| `SUMMARIZATION_MODEL` | llama3.2:latest | 摘要模型                        |
| `CACHE_SIZE`          | 10              | 快取大小                        |

**使用範例：**

```bash
export LAYER1_K_DOCUMENTS=15
export LAYER1_THRESHOLD=0.7
export LAYER2_THRESHOLD=0.8
export ENABLE_EXPANSION=true
export CHUNKING_MODE=summarization
python agent2.py
```

---

## 效能調校建議

### GPU 加速（推薦）

```bash
# 安裝 GPU 版本的 FAISS
pip uninstall faiss-cpu
pip install faiss-gpu
```

### 記憶體優化

```bash
# 減少 Layer 1 檢索數量
export LAYER1_K_DOCUMENTS=5

# 減少 Reranking 候選數量（修改 agent2.py）
'max_candidates': 100  # 預設 300
```

---

## 常見問題

### Q1: 啟動時顯示 "Failed to connect to Neo4j"

**解決方法：**

1. 確認 Docker 已啟動
2. 檢查 Neo4j 容器是否運行：
   ```bash
   docker ps | grep neo4j
   ```
3. 若未運行，重新啟動容器：
   ```bash
   docker start neo4j
   ```

### Q2: 啟動後顯示 "RAG system not initialized"

**解決方法：**

- 確認 `test_data/` 資料夾中至少有一個 PDF 檔案
- 檢查 Ollama 服務是否運行：
  ```bash
  curl http://localhost:11434/api/tags
  ```

### Q3: 查詢速度很慢或超時

**可能原因與解決方法：**

1. **Ollama 模型未載入**：首次查詢會載入模型，需等待 10-30 秒
2. **CPU 運算較慢**：建議使用 GPU 或減少 PDF 數量
3. **Layer 2 Reranking 過慢**：可在 `agent2.py` 中調整 `max_candidates` 參數

### Q4: 如何新增論文？

**方法 1（推薦）：使用 Web 介面**

1. 進入 "Add New Paper" 頁面
2. 上傳 PDF 檔案
3. 系統自動提取元資料並更新索引

**方法 2：手動放置**

1. 將 PDF 放入 `test_data/` 資料夾
2. 重啟系統或手動重建索引：
   ```bash
   curl -X POST http://localhost:4000/rag/rebuild?force=true
   ```

### Q5: 如何切換分塊模式？

**兩種分塊模式：**

- `naive`：傳統固定大小分塊（快速）
- `summarization`：基於段落摘要分塊（精確）

**切換方法：**

```bash
export CHUNKING_MODE=summarization
python agent2.py
```

### Q6: Neo4j 預設密碼需要修改嗎？

**修改方法：**

1. 修改 Docker 啟動指令中的 `NEO4J_AUTH`
2. 更新 `system_api/graph_manager.py` 中的連線參數：
   ```python
   def __init__(
       self,
       llm: OllamaLLM,
       url: str = "bolt://localhost:7687",
       username: str = "neo4j",
       password: str = "your_new_password"  # 修改此處
   ):
   ```

---

## 技術支援

### 檢查系統狀態

```bash
# 檢查系統健康狀態
curl http://localhost:4000/rag/health

# 查看系統統計資訊
curl http://localhost:4000/rag/stats

# 查看當前 RAG 模式
curl http://localhost:4000/rag/mode
```

### 日誌位置

系統運行日誌會輸出到終端機，若需保存日誌：

```bash
python agent2.py > system.log 2>&1
```

---

## 詳細文檔

- **快速開始指南**: [QUICKSTART.md](QUICKSTART.md)
- **完整使用手冊**: [HIERARCHICAL_RAG_USAGE.md](HIERARCHICAL_RAG_USAGE.md)

---

## 授權與使用限制

本專案僅供學術研究使用，不得用於商業用途。

**引用方式：**

```
KnowsRoots Assistant - Hierarchical RAG System for Academic Paper Analysis
Repository: NCHU702/KnowsRootsAssistant-ver3
```
