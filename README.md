# KnowsRoots Assistant - Hierarchical RAG System

## 🎯 系統簡介

學術論文檢索與問答系統，採用 **階層式 RAG（Hierarchical RAG）** 架構，提供高精確度的論文檢索和智能問答功能。

### 核心特性

- ✨ **兩層索引架構**：摘要層 + 區塊層
- 🧠 **智能信心評估**：LLM 評估答案品質
- ⚡ **早期終止優化**：50-70% 查詢更快
- 📈 **動態上下文擴展**：自動擴展相關內容
- 📊 **查詢監控**：完整的性能統計

## 🚀 快速開始

### 1. 啟動 Ollama 服務

確保 Ollama 已安裝並運行：

```bash
ollama serve
```

### 2. 安裝依賴

```bash
pip install -r requirements.txt
```

### 3. 準備 PDF 文件

```bash
# 將 PDF 論文放入 data 目錄
mkdir -p data
cp your_papers/*.pdf data/
```

### 4. 建立索引

```bash
python scripts/migrate_to_hierarchical_rag.py --execute
```

### 5. 啟動服務

```bash
python agent2.py
```

服務啟動後可在 http://localhost:4000 訪問。

### 6. 測試系統

```bash
# 運行整合測試
python tests/test_agent2_integration.py

# 或手動測試
curl http://localhost:4000/rag/health
```

## 📖 詳細文檔

- **快速開始指南**: [QUICKSTART.md](QUICKSTART.md)
- **完整使用手冊**: [HIERARCHICAL_RAG_USAGE.md](HIERARCHICAL_RAG_USAGE.md)

## 🔗 API 端點

### 查詢
- `POST /query` - 標準查詢
- `POST /query_stream` - 流式查詢

### 監控
- `GET /rag/health` - 健康檢查
- `GET /rag/stats` - 系統統計
- `GET /rag/mode` - 模式資訊

## ⚙️ 配置

通過環境變數調整系統：

```bash
export LAYER1_THRESHOLD=0.7    # Layer 1 閾值
export LAYER2_THRESHOLD=0.8    # Layer 2 閾值
export ENABLE_EXPANSION=true   # 啟用上下文擴展
export CACHE_SIZE=10           # 快取大小
```

## 📊 架構

```
┌─────────────────────────────────────┐
│         User Query                  │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Layer 1: Abstract Search           │
│  - 快速過濾相關論文                  │
│  - 基於摘要的語義搜索                │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Confidence Evaluation              │
│  - LLM 評估答案信心度                │
│  - 決定是否需要深入檢索              │
└─────────────┬───────────────────────┘
              │
         ┌────┴────┐
         │         │
    信心度高     信心度低
         │         │
         ▼         ▼
    ┌────────┐  ┌─────────────────────┐
    │ 返回答案│  │ Layer 2: Chunk Search│
    └────────┘  │ - 詳細區塊檢索        │
                │ - 上下文擴展          │
                └──────────┬────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  生成最終答案 │
                    └─────────────┘
```

## 🧪 測試

```bash
# 運行組件測試
python tests/test_hierarchical_components.py

# 運行整合測試
python tests/test_hierarchical_integration.py

# 運行 agent2.py 整合測試
python tests/test_agent2_integration.py
```

## 📦 專案結構

```
KnowsRootsAssistant-ver3/
├── agent2.py                       # 主應用
├── system_api/                     # 核心模組
│   ├── hierarchical_rag_system.py # 階層式 RAG
│   ├── abstract_extractor.py      # 摘要提取
│   ├── layer1_vectorstore.py      # Layer 1 索引
│   ├── layer2_vectorstore.py      # Layer 2 索引
│   ├── confidence_evaluator.py    # 信心評估
│   └── context_expander.py        # 上下文擴展
├── scripts/                        # 工具腳本
├── tests/                          # 測試
└── data/                          # PDF 文件
```

## 🤝 貢獻

歡迎提交 Issue 和 Pull Request！

## 📄 授權

[授權資訊]

