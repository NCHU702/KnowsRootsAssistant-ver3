# KnowsRoots Assistant# KnowsRoots Assistant - Hierarchical RAG System



學術論文智能檢索與分析系統，整合 Agentic AI 與 GraphRAG 技術。## 🎯 系統簡介



---學術論文檢索與問答系統，採用 **階層式 RAG（Hierarchical RAG）** 架構，提供高精確度的論文檢索和智能問答功能。



## 系統需求### 核心特性



- **Python**: 3.10 或以上版本- ✨ **兩層索引架構**：摘要層 + 區塊層

- **Ollama**: 本地 LLM 服務（必須運行中）- 🧠 **智能信心評估**：LLM 評估答案品質

- **記憶體**: 建議 8GB 以上- ⚡ **早期終止優化**：50-70% 查詢更快

- **硬碟**: 至少 5GB 可用空間（用於模型和向量索引）- 📈 **動態上下文擴展**：自動擴展相關內容

- 📊 **查詢監控**：完整的性能統計

---

## 🚀 快速開始

## 安裝步驟

### 1. 啟動 Ollama 服務

### 1. 安裝 Ollama

確保 Ollama 已安裝並運行：

前往 [Ollama 官網](https://ollama.ai) 下載並安裝。

```bash

啟動 Ollama 服務：ollama serve

```

```bash

ollama serve### 2. 安裝依賴

```

```bash

### 2. 下載所需模型pip install -r requirements.txt

```

```bash

# LLM 模型### 3. 準備 PDF 文件

ollama pull jcai/llama-3-taiwan-8b-instruct:q4_k_m

```bash

# 嵌入模型# 將 PDF 論文放入 data 目錄

ollama pull quentinz/bge-large-zh-v1.5:latestmkdir -p data

cp your_papers/*.pdf data/

# Reranker 模型（可選，提升檢索精度）```

ollama pull qllama/bce-reranker-base_v1:latest

```### 4. 建立索引



### 3. 安裝 Python 依賴```bash

python scripts/migrate_to_hierarchical_rag.py --execute

```bash```

# 建議使用虛擬環境

python -m venv .venv### 5. 啟動服務

source .venv/bin/activate  # macOS/Linux

# 或 .venv\Scripts\activate  # Windows```bash

python agent2.py

# 安裝依賴```

pip install -r requirements.txt

```服務啟動後可在 http://localhost:4000 訪問。



### 4. 準備資料### 6. 測試系統



將 PDF 論文放入 `test_data/` 資料夾：```bash

# 運行整合測試

```bashpython tests/test_agent2_integration.py

cp your_papers/*.pdf test_data/

```# 或手動測試

curl http://localhost:4000/rag/health

### 5. 啟動系統```



```bash## 📖 詳細文檔

python agent2.py

```- **快速開始指南**: [QUICKSTART.md](QUICKSTART.md)

- **完整使用手冊**: [HIERARCHICAL_RAG_USAGE.md](HIERARCHICAL_RAG_USAGE.md)

系統會自動：

- 檢查 PDF 檔案## 🔗 API 端點

- 建立向量索引

- 啟動 Web 服務### 查詢

- `POST /query` - 標準查詢

成功啟動後，開啟瀏覽器訪問 **http://localhost:4000**- `POST /query_stream` - 流式查詢



---### 監控

- `GET /rag/health` - 健康檢查

## 使用方式- `GET /rag/stats` - 系統統計

- `GET /rag/mode` - 模式資訊

### Web 介面功能

## ⚙️ 配置

1. **General Assistant**（一般助理）

   - 查詢論文內容通過環境變數調整系統：

   - 跨論文比較分析

   - 方法、資料集、研究目標查詢```bash

export LAYER1_THRESHOLD=0.7    # Layer 1 閾值

2. **Inheritance Analysis**（研究傳承分析）export LAYER2_THRESHOLD=0.8    # Layer 2 閾值

   - 查看研究脈絡export ENABLE_EXPANSION=true   # 啟用上下文擴展

   - 作者關係分析export CACHE_SIZE=10           # 快取大小

```

3. **Data Categories**（論文分類搜尋）

   - 依年份、主題、方法篩選論文## 📊 架構

   - 快速瀏覽論文元資料

```

4. **Add New Paper**（新增論文）┌─────────────────────────────────────┐

   - 上傳 PDF│         User Query                  │

   - 自動提取元資料└─────────────┬───────────────────────┘

   - 即時更新索引              │

              ▼

### API 端點┌─────────────────────────────────────┐

│  Layer 1: Abstract Search           │

```bash│  - 快速過濾相關論文                  │

# 健康檢查│  - 基於摘要的語義搜索                │

curl http://localhost:4000/rag/health└─────────────┬───────────────────────┘

              │

# 查詢統計              ▼

curl http://localhost:4000/rag/stats┌─────────────────────────────────────┐

│  Confidence Evaluation              │

# 流式查詢（推薦）│  - LLM 評估答案信心度                │

curl -X POST http://localhost:4000/query_stream \│  - 決定是否需要深入檢索              │

  -H "Content-Type: application/json" \└─────────────┬───────────────────────┘

  -d '{"input": "哪些論文使用 YOLO？"}'              │

```         ┌────┴────┐

         │         │

---    信心度高     信心度低

         │         │

## 專案結構         ▼         ▼

    ┌────────┐  ┌─────────────────────┐

```    │ 返回答案│  │ Layer 2: Chunk Search│

KnowsRootsAssistant-ver3/    └────────┘  │ - 詳細區塊檢索        │

├── agent2.py                    # 主程式（Flask 應用）                │ - 上下文擴展          │

├── requirements.txt             # Python 依賴                └──────────┬────────────┘

├── test_data/                   # PDF 論文存放處                           │

├── data/                        # 已處理的論文                           ▼

├── vectorstore/                 # 向量索引儲存                    ┌─────────────┐

│   ├── layer1/                  # Layer 1: 論文摘要索引                    │  生成最終答案 │

│   └── layer2/                  # Layer 2: 區塊級索引                    └─────────────┘

├── system_api/                  # 核心模組```

│   ├── hierarchical_rag_system.py  # 階層式 RAG 主系統

│   ├── layer1_vectorstore.py       # 論文級檢索## 🧪 測試

│   ├── layer2_vectorstore.py       # 區塊級檢索

│   ├── graph_manager.py            # Neo4j 圖譜管理```bash

│   ├── graph_extractor.py          # 論文結構化提取# 運行組件測試

│   └── ...                         # 其他輔助模組python tests/test_hierarchical_components.py

├── templates/                   # Web 前端

└── json_files/                  # 論文元資料# 運行整合測試

```python tests/test_hierarchical_integration.py



---# 運行 agent2.py 整合測試

python tests/test_agent2_integration.py

## 系統架構```



```## 📦 專案結構

使用者查詢

    ↓```

智能路由（Graph vs RAG）KnowsRootsAssistant-ver3/

    ↓├── agent2.py                       # 主應用

┌─────────────┬────────────┐├── system_api/                     # 核心模組

│  GraphRAG   │  RAG       ││   ├── hierarchical_rag_system.py # 階層式 RAG

│  (關係查詢)  │  (內容查詢) ││   ├── abstract_extractor.py      # 摘要提取

└─────────────┴────────────┘│   ├── layer1_vectorstore.py      # Layer 1 索引

         ↓│   ├── layer2_vectorstore.py      # Layer 2 索引

    LLM 生成答案│   ├── confidence_evaluator.py    # 信心評估

```│   └── context_expander.py        # 上下文擴展

├── scripts/                        # 工具腳本

### RAG 檢索流程├── tests/                          # 測試

└── data/                          # PDF 文件

1. **Layer 1**：快速檢索相關論文（基於摘要）```

2. **信心評估**：判斷是否需進入 Layer 2

3. **Layer 2**：深入檢索論文區塊（含 Reranking）## 🤝 貢獻

4. **上下文擴展**：依信心度動態調整上下文範圍

5. **LLM 生成**：合成最終答案歡迎提交 Issue 和 Pull Request！



---## 📄 授權



## 常見問題[授權資訊]



### Q: 啟動後顯示 "RAG system not initialized"
**A**: 確認 `test_data/` 資料夾中至少有一個 PDF 檔案。

### Q: 查詢速度很慢
**A**: 
- 檢查 Ollama 是否正常運行
- 若使用 CPU，建議減少 PDF 數量或升級至 GPU

### Q: 如何新增論文？
**A**: 
1. 使用 Web 介面 "Add New Paper" 功能上傳
2. 或直接將 PDF 放入 `test_data/`，重啟系統自動索引

### Q: 如何重建索引？
**A**: 
```bash
curl -X POST http://localhost:4000/rag/rebuild?force=true
```

---

## 環境變數配置（可選）

```bash
# Layer 1 檢索數量
export LAYER1_K_DOCUMENTS=10

# 信心度閾值
export LAYER1_THRESHOLD=0.6
export LAYER2_THRESHOLD=0.6

# 啟用/停用功能
export ENABLE_EXPANSION=true
export CHUNKING_MODE=summarization  # naive 或 summarization
```

---

## 授權

本專案僅供學術研究使用。
