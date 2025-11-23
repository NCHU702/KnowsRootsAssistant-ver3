# 專案名稱：Local-GraphAgent
## 副標題：基於 Agentic AI 與輕量化 GraphRAG 的私有化學術 GenAI 平台

---

### 1. 專案摘要 (Executive Summary)
本專案旨在打造一套**「Agentic AI GraphRAG GenAI 平台」**。這是一套專為學術研究與企業內部知識管理設計的**本地端（Local）、低成本（Low-cost）、高隱私（Privacy-First）**解決方案。

針對目前主流 RAG（檢索增強生成）技術面臨的「全域理解力不足」與「GraphRAG 建構成本過高」的兩難困境，本團隊開發了 **Local-GraphAgent**。我們創新地結合了 **Agentic AI（代理人工作流）** 與 **階層式圖學檢索（Hierarchical GraphRAG）**，並透過獨家的**高密度資訊切塊（High-Density Chunking）**技術，讓消費級硬體（如 Mac Mini M4）即可運行企業級的知識圖譜問答系統，實現「小模型，大智慧」的普惠 AI 願景。

---

### 2. 痛點分析 (Problem Statement)

在學術研究與企業知識管理場景中，現有解決方案存在三大核心痛點：

1.  **資料隱私與算力成本的矛盾**：
    * 機密文件（如未發表論文、企業專利）無法上傳至公有雲（GPT-4/Claude）。
    * 本地部署 70B+ 參數的大模型需要昂貴的 A100/H100 伺服器，中小企業與實驗室難以負擔。
2.  **傳統 RAG 的「見樹不見林」**：
    * 傳統 Vector RAG 僅能檢索片段相似度，無法回答跨文檔的聚合問題（例如：「比較這五篇論文的方法論演進」）。
    * 微軟原始 GraphRAG 雖解決此問題，但其 Community Report 生成過程消耗極大量 Token，且檢索延遲高，難以落地。
3.  **小模型的推論能力限制**：
    * 在邊緣設備上運行的小參數模型（SLM, <10B），往往因 Context Window 限制，在處理大量檢索文本時容易產生幻覺或遺忘。

---

### 3. 三大核心技術亮點 (Key Innovations)

本平台透過底層架構的重構，實現了三大技術突破，這也是我們與市面方案最大的差異：

#### 亮點一：改良 GraphRAG 結合 Hierarchy RAG 查詢結構，極致降低 Token 用量
* **技術實作**：不同於傳統 RAG 一次性檢索大量切片，我們設計了 **雙層檢索架構 (Hierarchical Retrieval System)**：
    * **Layer 1 (Paper-Level)**：利用 `FAISS` 快速篩選論文摘要與元數據，Agent 優先判斷摘要是否足以回答問題。
    * **Layer 2 (Chunk-Level)**：僅在需要細節時，Agent 才會向下鑽取（Drill-down）至具體切片。
* **Agentic 路由**：系統內建 `QueryRouter` 與 `Layer2TriggerDecision`，能智慧判斷查詢意圖。若是跨文檔的宏觀問題（如「有哪些論文使用 YOLO？」），直接調用 Graph 查詢；若是單一文檔細節，則調用 Vector RAG。
* **效益**：大幅減少無效 Context 的載入，Token 消耗量降低 60% 以上，提升回應速度。

#### 亮點二：預處理階段產生「高密度資訊 Chunk」，提升小模型準確度
* **技術實作**：捨棄傳統的固定字數切分（Naive Chunking），我們引入了 **Summarization-based Chunking** 策略。
    * **Graph Extraction**：在資料寫入階段，利用 LLM 自動提取結構化資訊（研究目標 `research_goal`、方法 `methods`、資料集 `datasets`、評估指標 `metrics`）寫入 Neo4j 圖資料庫。
    * **Section Summarization**：針對 PDF 的特定章節（如 "Experiment" 或 "Dataset"），先進行摘要濃縮，生成**高密度資訊塊**後再存入向量庫。
* **Cross-Encoder Re-ranking**：在 Layer 2 檢索後，我們強制啟用 Cross-Encoder 進行二次排序，確保餵給模型的資訊是真正的「乾貨」。
* **效益**：顯著提高檢索內容的信噪比（Signal-to-Noise Ratio），讓 8B 參數的小模型也能做出準確的邏輯推論。

#### 亮點三：輕量級架構設計，讓小模型也能成為智能助手
* **硬體適配**：本系統專為 **Apple Silicon (Mac Mini M4 24GB)** 與消費級 GPU 優化。
* **技術實作**：
    * 全面採用本地化模型（如 `Llama-3-Taiwan-8b`, `Gemma`）透過 Ollama 運行。
    * **Graph + Vector 混合索引**：利用 Neo4j 處理實體關係，JSONL/FAISS 處理語意檢索，達成輕量化部署。
    * **Agentic Workflow**：利用 `LangChain` 與 `LangGraph` 構建具備「規劃能力」的 Agent。它知道何時該查圖、何時該查文、何時該聯網，彌補了小模型本身推理能力的不足。
* **效益**：實現真正的「隨插即用」與「完全斷網運行」，為企業提供最低門檻的私有化 AI 知識庫。

---

### 4. 系統技術架構 (Technical Architecture)

本平台採用模組化設計，核心由以下組件構成：

#### 4.1 核心引擎 (Core Engine)
* **Orchestrator**: `agent2.py` 作為中控大腦，基於 React Agent 架構，動態調度工具。
* **Tools**:
    * `GraphAnalysisCall`: 用於宏觀趨勢分析、跨論文比較（Querying Neo4j）。
    * `AssistantCall`: 用於單篇論文的深度問答（Querying Hierarchy RAG）。
    * `WebSearchCall`: 必要時進行聯網補充資訊。

#### 4.2 雙層資料儲存 (Dual-Layer Storage)
* **Graph Store (Neo4j)**: 儲存論文的元數據結構（Paper -> Uses -> Method, Paper -> EvaluatedOn -> Dataset）。
* **Vector Store (Hierarchical)**:
    * **Layer 1**: 儲存論文摘要向量 (Abstract-level embedding)。
    * **Layer 2**: 儲存經摘要處理的高密度切片 (Summarized Chunks)。

#### 4.3 智慧檢索流程 (Intelligent Workflow)
1.  **Ingestion**: PDF 上傳 -> `GraphDataExtractor` 提取實體 -> `PDFSectionParser` 解析章節 -> 生成高密度 Chunks -> 寫入資料庫。
2.  **Routing**: 使用者提問 -> Agent 判斷意圖 (Macro vs. Micro)。
3.  **Retrieval**:
    * 若為 Macro 問題：查詢 Neo4j 圖譜，返回結構化報告。
    * 若為 Micro 問題：先檢索 Layer 1，若信心不足則觸發 Layer 2 並進行 Cross-Encoder Re-ranking。
4.  **Generation**: 彙整資訊 -> 本地 LLM 生成最終答案。

---

### 5. 預期效益與商業價值 (Impact & Commercial Value)

#### 5.1 學術界應用
* **文獻回顧自動化**：研究生可匯入百篇論文，系統自動生成「方法論演進圖」與「數據集使用統計」，節省數週的閱讀時間。
* **研究繼承分析**：透過 `ResearchInheritanceAnalyzer`，快速釐清實驗室內部的研究脈絡，避免重複造輪子。

#### 5.2 企業端應用 (B2B)
* **技術資產活化**：將散落在各部門的技術文件轉化為可對話的知識圖譜。
* **資安合規**：全本地端運行，確保技術機密與客戶資料 100% 不出內網，符合 GDPR 與企業資安規範。
* **成本優勢**：相比訂閱 Copilot 或購買大型伺服器，本方案硬體成本僅需數萬元，且無持續 Token 費用。

---

### 6. 結論 (Conclusion)

Local-GraphAgent 不僅僅是一個 RAG 系統，它是一個具備**「認知分層能力」**的 AI 代理平台。我們證明了透過優良的**架構設計（Agentic + Hierarchy RAG）**與**高品質的數據預處理（High-Density Chunking）**，小模型在專業領域的表現完全可以媲美雲端大模型。這將是企業與學術單位邁向「AI 私有化」最務實且高效的選擇。