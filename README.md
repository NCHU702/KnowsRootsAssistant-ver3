# KnowsRoots Assistant# KnowsRoots Assistant# KnowsRoots Assistant - Hierarchical RAG System



學術論文智能檢索與分析系統，整合 Agentic AI 與 GraphRAG 技術。



---學術論文智能檢索與分析系統，整合 Agentic AI 與 GraphRAG 技術。## 🎯 系統簡介



## 系統需求



| 項目 | 需求 |---學術論文檢索與問答系統，採用 **階層式 RAG（Hierarchical RAG）** 架構，提供高精確度的論文檢索和智能問答功能。

|------|------|

| **Python** | 3.10 或以上版本 |

| **Ollama** | 本地 LLM 服務（必須運行中） |

| **Docker** | 用於運行 Neo4j 圖資料庫 |## 系統需求### 核心特性

| **記憶體** | 建議 8GB 以上 |

| **硬碟** | 至少 10GB 可用空間 |



---- **Python**: 3.10 或以上版本- ✨ **兩層索引架構**：摘要層 + 區塊層



## 安裝步驟- **Ollama**: 本地 LLM 服務（必須運行中）- 🧠 **智能信心評估**：LLM 評估答案品質



### 步驟 1：安裝 Ollama- **記憶體**: 建議 8GB 以上- ⚡ **早期終止優化**：50-70% 查詢更快



前往 [Ollama 官網](https://ollama.ai) 下載並安裝。- **硬碟**: 至少 5GB 可用空間（用於模型和向量索引）- 📈 **動態上下文擴展**：自動擴展相關內容



啟動 Ollama 服務：- 📊 **查詢監控**：完整的性能統計



```bash---

ollama serve

```## 🚀 快速開始



### 步驟 2：下載所需的 Ollama 模型## 安裝步驟



**必要模型（3 個）：**### 1. 啟動 Ollama 服務



```bash### 1. 安裝 Ollama

# 1. 主要 LLM 模型（用於推理、對話、分析）

ollama pull jcai/llama-3-taiwan-8b-instruct:q4_k_m確保 Ollama 已安裝並運行：



# 2. 嵌入模型（用於向量檢索）前往 [Ollama 官網](https://ollama.ai) 下載並安裝。

ollama pull quentinz/bge-large-zh-v1.5:latest

```bash

# 3. 摘要模型（用於 Summarization 分塊模式）

ollama pull llama3.2:latest啟動 Ollama 服務：ollama serve

```

```

**可選模型（提升效能）：**

```bash

```bash

# Reranker 模型（提升 Layer 2 檢索精度，強烈建議安裝）ollama serve### 2. 安裝依賴

ollama pull qllama/bce-reranker-base_v1:latest

``````



> **模型說明：**```bash

> - `jcai/llama-3-taiwan-8b-instruct:q4_k_m`：針對繁體中文優化的 LLM

> - `quentinz/bge-large-zh-v1.5:latest`：中文語義嵌入模型### 2. 下載所需模型pip install -r requirements.txt

> - `llama3.2:latest`：用於段落摘要生成

> - `qllama/bce-reranker-base_v1:latest`：跨編碼器重排序模型```



### 步驟 3：安裝並啟動 Neo4j（GraphRAG 圖資料庫）```bash



使用 Docker 快速部署 Neo4j：# LLM 模型### 3. 準備 PDF 文件



```bashollama pull jcai/llama-3-taiwan-8b-instruct:q4_k_m

docker run -d \

  --name neo4j \```bash

  -p 7474:7474 \

  -p 7687:7687 \# 嵌入模型# 將 PDF 論文放入 data 目錄

  -e NEO4J_AUTH=neo4j/password \

  neo4j:latestollama pull quentinz/bge-large-zh-v1.5:latestmkdir -p data

```

cp your_papers/*.pdf data/

**驗證 Neo4j 是否運行：**

# Reranker 模型（可選，提升檢索精度）```

開啟瀏覽器訪問 http://localhost:7474，使用以下帳號登入：

- 使用者名稱：`neo4j`ollama pull qllama/bce-reranker-base_v1:latest

- 密碼：`password`

```### 4. 建立索引

> **提示：** 若需修改密碼，請在 `system_api/graph_manager.py` 中調整連線參數。



### 步驟 4：安裝 Python 依賴

### 3. 安裝 Python 依賴```bash

```bash

# 建議使用虛擬環境python scripts/migrate_to_hierarchical_rag.py --execute

python -m venv .venv

source .venv/bin/activate  # macOS/Linux```bash```

# 或 .venv\Scripts\activate  # Windows

# 建議使用虛擬環境

# 安裝依賴套件

pip install -r requirements.txtpython -m venv .venv### 5. 啟動服務

```

source .venv/bin/activate  # macOS/Linux

### 步驟 5：準備論文資料

# 或 .venv\Scripts\activate  # Windows```bash

將 PDF 論文放入 `test_data/` 資料夾：

python agent2.py

```bash

mkdir -p test_data# 安裝依賴```

cp your_papers/*.pdf test_data/

```pip install -r requirements.txt



> **注意：** 系統啟動時會自動掃描 `test_data/` 資料夾中的 PDF 並建立索引。```服務啟動後可在 http://localhost:4000 訪問。



### 步驟 6：啟動系統



```bash### 4. 準備資料### 6. 測試系統

python agent2.py

```



**系統會自動執行：**將 PDF 論文放入 `test_data/` 資料夾：```bash

1. ✓ 檢查 PDF 檔案

2. ✓ 建立 Layer 1 向量索引（論文摘要）# 運行整合測試

3. ✓ 建立 Layer 2 向量索引（論文區塊）

4. ✓ 提取論文結構化資訊至 Neo4j 圖資料庫```bashpython tests/test_agent2_integration.py

5. ✓ 啟動 Flask Web 服務

cp your_papers/*.pdf test_data/

**成功啟動後，開啟瀏覽器訪問：**

- **Web 介面**: http://localhost:4000```# 或手動測試

- **Neo4j 圖資料庫**: http://localhost:7474

curl http://localhost:4000/rag/health

---

### 5. 啟動系統```

## 使用方式



### 📌 Web 介面功能

```bash## 📖 詳細文檔

| 功能模組 | 說明 | 使用情境 |

|---------|------|---------|python agent2.py

| **General Assistant** | 一般助理 | 查詢論文內容、跨論文比較分析、方法/資料集查詢 |

| **Inheritance Analysis** | 研究傳承分析 | 查看研究脈絡、作者關係分析 |```- **快速開始指南**: [QUICKSTART.md](QUICKSTART.md)

| **Data Categories** | 論文分類搜尋 | 依年份、主題、方法篩選論文，快速瀏覽元資料 |

| **Add New Paper** | 新增論文 | 上傳 PDF、自動提取元資料、即時更新索引 |- **完整使用手冊**: [HIERARCHICAL_RAG_USAGE.md](HIERARCHICAL_RAG_USAGE.md)



### 📌 查詢範例系統會自動：



**跨論文查詢（使用 GraphRAG）：**- 檢查 PDF 檔案## 🔗 API 端點

```

哪些論文使用 YOLO 進行物件偵測？- 建立向量索引

比較使用 LSTM 和 Transformer 的論文

列出所有關於智慧交通的研究- 啟動 Web 服務### 查詢

```

- `POST /query` - 標準查詢

**單篇論文查詢（使用 RAG）：**

```成功啟動後，開啟瀏覽器訪問 **http://localhost:4000**- `POST /query_stream` - 流式查詢

芒果分類那篇論文用什麼資料集？

交通流量預測論文的研究方法是什麼？

鼻咽癌辨識論文的準確率是多少？

```---### 監控



### 📌 API 端點- `GET /rag/health` - 健康檢查



```bash## 使用方式- `GET /rag/stats` - 系統統計

# 健康檢查

curl http://localhost:4000/rag/health- `GET /rag/mode` - 模式資訊



# 查詢統計### Web 介面功能

curl http://localhost:4000/rag/stats

## ⚙️ 配置

# 流式查詢（推薦）

curl -X POST http://localhost:4000/query_stream \1. **General Assistant**（一般助理）

  -H "Content-Type: application/json" \

  -d '{"input": "哪些論文使用 YOLO？"}'   - 查詢論文內容通過環境變數調整系統：

```

   - 跨論文比較分析

---

   - 方法、資料集、研究目標查詢```bash

## 專案結構

export LAYER1_THRESHOLD=0.7    # Layer 1 閾值

```

KnowsRootsAssistant-ver3/2. **Inheritance Analysis**（研究傳承分析）export LAYER2_THRESHOLD=0.8    # Layer 2 閾值

├── agent2.py                    # 主程式（Flask 應用）

├── requirements.txt             # Python 依賴   - 查看研究脈絡export ENABLE_EXPANSION=true   # 啟用上下文擴展

├── test_data/                   # PDF 論文存放處

├── data/                        # 已處理的論文   - 作者關係分析export CACHE_SIZE=10           # 快取大小

├── vectorstore/                 # 向量索引儲存

│   ├── layer1/                  # Layer 1: 論文摘要索引```

│   └── layer2/                  # Layer 2: 區塊級索引

├── system_api/                  # 核心模組3. **Data Categories**（論文分類搜尋）

│   ├── hierarchical_rag_system.py  # 階層式 RAG 主系統

│   ├── layer1_vectorstore.py       # 論文級檢索   - 依年份、主題、方法篩選論文## 📊 架構

│   ├── layer2_vectorstore.py       # 區塊級檢索

│   ├── graph_manager.py            # Neo4j 圖譜管理   - 快速瀏覽論文元資料

│   ├── graph_extractor.py          # 論文結構化提取

│   └── ...                         # 其他輔助模組```

├── templates/                   # Web 前端

└── json_files/                  # 論文元資料4. **Add New Paper**（新增論文）┌─────────────────────────────────────┐

```

   - 上傳 PDF│         User Query                  │

---

   - 自動提取元資料└─────────────┬───────────────────────┘

## 系統架構

   - 即時更新索引              │

### 🔍 智能路由機制

              ▼

```

                    使用者查詢### API 端點┌─────────────────────────────────────┐

                        ↓

              ┌─────────────────┐│  Layer 1: Abstract Search           │

              │   智能路由器      │

              │ (Query Router)  │```bash│  - 快速過濾相關論文                  │

              └────────┬─────────┘

                       │# 健康檢查│  - 基於摘要的語義搜索                │

        ┌──────────────┴──────────────┐

        ↓                             ↓curl http://localhost:4000/rag/health└─────────────┬───────────────────────┘

┌──────────────┐              ┌──────────────┐

│  GraphRAG    │              │  RAG         │              │

│  (關係查詢)   │              │  (內容查詢)   │

│              │              │              │# 查詢統計              ▼

│ • 跨論文比較  │              │ • 單篇論文    │

│ • 方法統計   │              │ • 詳細內容    │curl http://localhost:4000/rag/stats┌─────────────────────────────────────┐

│ • 資料集查詢  │              │ • 深入分析    │

└──────┬───────┘              └──────┬───────┘│  Confidence Evaluation              │

       │                             │

       └──────────────┬──────────────┘# 流式查詢（推薦）│  - LLM 評估答案信心度                │

                      ↓

              ┌──────────────┐curl -X POST http://localhost:4000/query_stream \│  - 決定是否需要深入檢索              │

              │  LLM 生成答案 │

              └──────────────┘  -H "Content-Type: application/json" \└─────────────┬───────────────────────┘

```

  -d '{"input": "哪些論文使用 YOLO？"}'              │

### 📊 Hierarchical RAG 檢索流程

```         ┌────┴────┐

```

查詢輸入         │         │

    ↓

┌─────────────────────────────────────┐---    信心度高     信心度低

│ Layer 1: 論文級檢索（基於摘要）        │

│ • 快速篩選相關論文                    │         │         │

│ • 向量相似度搜尋                      │

│ • 混合檢索（語義 + 關鍵詞）            │## 專案結構         ▼         ▼

└────────────┬────────────────────────┘

             ↓    ┌────────┐  ┌─────────────────────┐

┌─────────────────────────────────────┐

│ 信心度評估 (Confidence Evaluator)   │```    │ 返回答案│  │ Layer 2: Chunk Search│

│ • LLM 評估答案品質                   │

│ • 決策：是否需進入 Layer 2            │KnowsRootsAssistant-ver3/    └────────┘  │ - 詳細區塊檢索        │

└────────────┬────────────────────────┘

             │├── agent2.py                    # 主程式（Flask 應用）                │ - 上下文擴展          │

      ┌──────┴──────┐

      ↓             ↓├── requirements.txt             # Python 依賴                └──────────┬────────────┘

   信心度高      信心度低

      ↓             ↓├── test_data/                   # PDF 論文存放處                           │

   返回答案   ┌─────────────────────────┐

             │ Layer 2: 區塊級檢索       │├── data/                        # 已處理的論文                           ▼

             │ • 詳細內容檢索            │

             │ • Cross-Encoder Reranking│├── vectorstore/                 # 向量索引儲存                    ┌─────────────┐

             │ • 上下文擴展              │

             └────────┬────────────────┘│   ├── layer1/                  # Layer 1: 論文摘要索引                    │  生成最終答案 │

                      ↓

              ┌──────────────┐│   └── layer2/                  # Layer 2: 區塊級索引                    └─────────────┘

              │  生成最終答案 │

              └──────────────┘├── system_api/                  # 核心模組```

```

│   ├── hierarchical_rag_system.py  # 階層式 RAG 主系統

---

│   ├── layer1_vectorstore.py       # 論文級檢索## 🧪 測試

## 常見問題

│   ├── layer2_vectorstore.py       # 區塊級檢索

<details>

<summary><b>Q1: 啟動時顯示 "Failed to connect to Neo4j"</b></summary>│   ├── graph_manager.py            # Neo4j 圖譜管理```bash



**解決方法：**│   ├── graph_extractor.py          # 論文結構化提取# 運行組件測試

1. 確認 Docker 已啟動

2. 檢查 Neo4j 容器是否運行：│   └── ...                         # 其他輔助模組python tests/test_hierarchical_components.py

   ```bash

   docker ps | grep neo4j├── templates/                   # Web 前端

   ```

3. 若未運行，重新啟動容器：└── json_files/                  # 論文元資料# 運行整合測試

   ```bash

   docker start neo4j```python tests/test_hierarchical_integration.py

   ```

</details>



<details>---# 運行 agent2.py 整合測試

<summary><b>Q2: 啟動後顯示 "RAG system not initialized"</b></summary>

python tests/test_agent2_integration.py

**解決方法：**

- 確認 `test_data/` 資料夾中至少有一個 PDF 檔案## 系統架構```

- 檢查 Ollama 服務是否運行：

  ```bash

  curl http://localhost:11434/api/tags

  ``````## 📦 專案結構

</details>

使用者查詢

<details>

<summary><b>Q3: 查詢速度很慢或超時</b></summary>    ↓```



**可能原因與解決方法：**智能路由（Graph vs RAG）KnowsRootsAssistant-ver3/

1. **Ollama 模型未載入**：首次查詢會載入模型，需等待 10-30 秒

2. **CPU 運算較慢**：建議使用 GPU 或減少 PDF 數量    ↓├── agent2.py                       # 主應用

3. **Layer 2 Reranking 過慢**：可在 `agent2.py` 中調整 `max_candidates` 參數

</details>┌─────────────┬────────────┐├── system_api/                     # 核心模組



<details>│  GraphRAG   │  RAG       ││   ├── hierarchical_rag_system.py # 階層式 RAG

<summary><b>Q4: 如何新增論文？</b></summary>

│  (關係查詢)  │  (內容查詢) ││   ├── abstract_extractor.py      # 摘要提取

**方法 1（推薦）：使用 Web 介面**

1. 進入 "Add New Paper" 頁面└─────────────┴────────────┘│   ├── layer1_vectorstore.py      # Layer 1 索引

2. 上傳 PDF 檔案

3. 系統自動提取元資料並更新索引         ↓│   ├── layer2_vectorstore.py      # Layer 2 索引



**方法 2：手動放置**    LLM 生成答案│   ├── confidence_evaluator.py    # 信心評估

1. 將 PDF 放入 `test_data/` 資料夾

2. 重啟系統或手動重建索引：```│   └── context_expander.py        # 上下文擴展

   ```bash

   curl -X POST http://localhost:4000/rag/rebuild?force=true├── scripts/                        # 工具腳本

   ```

</details>### RAG 檢索流程├── tests/                          # 測試



<details>└── data/                          # PDF 文件

<summary><b>Q5: 如何切換分塊模式？</b></summary>

1. **Layer 1**：快速檢索相關論文（基於摘要）```

**兩種分塊模式：**

- `naive`：傳統固定大小分塊（快速）2. **信心評估**：判斷是否需進入 Layer 2

- `summarization`：基於段落摘要分塊（精確）

3. **Layer 2**：深入檢索論文區塊（含 Reranking）## 🤝 貢獻

**切換方法：**

```bash4. **上下文擴展**：依信心度動態調整上下文範圍

export CHUNKING_MODE=summarization

python agent2.py5. **LLM 生成**：合成最終答案歡迎提交 Issue 和 Pull Request！

```

</details>



<details>---## 📄 授權

<summary><b>Q6: Neo4j 預設密碼需要修改嗎？</b></summary>



**修改方法：**

1. 修改 Docker 啟動指令中的 `NEO4J_AUTH`## 常見問題[授權資訊]

2. 更新 `system_api/graph_manager.py` 中的連線參數：

   ```python

   def __init__(

       self,### Q: 啟動後顯示 "RAG system not initialized"

       llm: OllamaLLM,**A**: 確認 `test_data/` 資料夾中至少有一個 PDF 檔案。

       url: str = "bolt://localhost:7687",

       username: str = "neo4j",### Q: 查詢速度很慢

       password: str = "your_new_password"  # 修改此處**A**: 

   ):- 檢查 Ollama 是否正常運行

   ```- 若使用 CPU，建議減少 PDF 數量或升級至 GPU

</details>

### Q: 如何新增論文？

---**A**: 

1. 使用 Web 介面 "Add New Paper" 功能上傳

## 進階配置2. 或直接將 PDF 放入 `test_data/`，重啟系統自動索引



### 環境變數（可選）### Q: 如何重建索引？

**A**: 

| 變數名稱 | 預設值 | 說明 |```bash

|---------|--------|------|curl -X POST http://localhost:4000/rag/rebuild?force=true

| `LAYER1_K_DOCUMENTS` | 10 | Layer 1 檢索的論文數量 |```

| `LAYER1_THRESHOLD` | 0.6 | Layer 1 信心度閾值 |

| `LAYER2_THRESHOLD` | 0.6 | Layer 2 信心度閾值 |---

| `ENABLE_EXPANSION` | true | 啟用上下文擴展 |

| `CHUNKING_MODE` | naive | 分塊模式（naive/summarization） |## 進階配置

| `SUMMARIZATION_MODEL` | llama3.2:latest | 摘要模型 |

### 環境變數（可選）

**使用範例：**

```bash| 變數名稱 | 預設值 | 說明 |

export LAYER1_K_DOCUMENTS=15|---------|--------|------|

export CHUNKING_MODE=summarization| `LAYER1_K_DOCUMENTS` | 10 | Layer 1 檢索的論文數量 |

python agent2.py| `LAYER1_THRESHOLD` | 0.6 | Layer 1 信心度閾值 |

```| `LAYER2_THRESHOLD` | 0.6 | Layer 2 信心度閾值 |

| `ENABLE_EXPANSION` | true | 啟用上下文擴展 |

### 效能調校建議| `CHUNKING_MODE` | naive | 分塊模式（naive/summarization） |

| `SUMMARIZATION_MODEL` | llama3.2:latest | 摘要模型 |

**GPU 加速（推薦）：**

```bash**使用範例：**

# 安裝 GPU 版本的 FAISS```bash

pip uninstall faiss-cpuexport LAYER1_K_DOCUMENTS=15

pip install faiss-gpuexport CHUNKING_MODE=summarization

```python agent2.py

```

**記憶體優化：**

```bash### 效能調校建議

# 減少 Layer 1 檢索數量

export LAYER1_K_DOCUMENTS=5**GPU 加速（推薦）：**

```bash

# 減少 Reranking 候選數量（修改 agent2.py）# 安裝 GPU 版本的 FAISS

'max_candidates': 100  # 預設 300pip uninstall faiss-cpu

```pip install faiss-gpu

```

---

**記憶體優化：**

## 技術支援```bash

# 減少 Layer 1 檢索數量

### 檢查系統狀態export LAYER1_K_DOCUMENTS=5



```bash# 減少 Reranking 候選數量（修改 agent2.py）

# 檢查系統健康狀態'max_candidates': 100  # 預設 300

curl http://localhost:4000/rag/health```



# 查看系統統計資訊---

curl http://localhost:4000/rag/stats

## 技術支援

# 查看當前 RAG 模式

curl http://localhost:4000/rag/mode### 檢查系統狀態

```

```bash

### 日誌位置# 檢查系統健康狀態

curl http://localhost:4000/rag/health

系統運行日誌會輸出到終端機，若需保存日誌：

# 查看系統統計資訊

```bashcurl http://localhost:4000/rag/stats

python agent2.py > system.log 2>&1

```# 查看當前 RAG 模式

curl http://localhost:4000/rag/mode

---```



## 授權與使用限制### 日誌位置



本專案僅供學術研究使用，不得用於商業用途。系統運行日誌會輸出到終端機，若需保存日誌：



**引用方式：**```bash

```python agent2.py > system.log 2>&1

KnowsRoots Assistant - Hierarchical RAG System for Academic Paper Analysis```

Repository: NCHU702/KnowsRootsAssistant-ver3

```---


## 授權與使用限制

本專案僅供學術研究使用，不得用於商業用途。

**引用方式：**
```
KnowsRoots Assistant - Hierarchical RAG System for Academic Paper Analysis
Repository: NCHU702/KnowsRootsAssistant-ver3
```
