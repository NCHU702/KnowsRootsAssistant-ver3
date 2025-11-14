# KnowsRoots Assistant - 完整系統架構與邏輯

## 📋 目錄

1. [系統概述](#系統概述)
2. [核心架構](#核心架構)
3. [階層式 RAG 系統](#階層式-rag-系統)
4. [索引管理邏輯](#索引管理邏輯)
5. [查詢處理流程](#查詢處理流程)
6. [組件詳解](#組件詳解)
7. [API 端點](#api-端點)
8. [數據流圖](#數據流圖)
9. [錯誤處理](#錯誤處理)

---

## 系統概述

### 🎯 核心目標
學術論文檢索與問答系統，解決傳統 RAG 的「語義相似但主題不相關」問題。

### 🏗️ 技術棧
- **後端框架**: Flask (Python)
- **LLM**: Ollama (gemma3:12b)
- **Embedding**: embeddinggemma:latest
- **向量存儲**: FAISS
- **PDF 處理**: PyPDF2
- **前端**: HTML/JavaScript (templates/index.html)

### 📦 主要功能模組
1. **Hierarchical RAG System** - 階層式檢索系統
2. **Agent Router** - 智能路由（RAG vs Web Search）
3. **Paper Management** - 論文上傳與分類
4. **Inheritance Analysis** - 研究繼承分析
5. **Index Management** - 智能索引管理

---

## 核心架構

### 系統層次結構

```
┌─────────────────────────────────────────────────────────────┐
│                      Flask Web Server                        │
│                     (agent2.py - Port 4000)                  │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────────┐    ┌──────────────┐
│   Frontend   │    │  Agent Router    │    │   API Layer  │
│  (HTML/JS)   │    │   (LLM-based)    │    │  (REST API)  │
└──────────────┘    └──────────────────┘    └──────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────────┐    ┌──────────────┐
│ Hierarchical │    │  Web Researcher  │    │  Inheritance │
│  RAG System  │    │   (Web Search)   │    │   Analyzer   │
└──────────────┘    └──────────────────┘    └──────────────┘
        │
        ▼
┌────────────────────────────────────────────────────────────┐
│              Core RAG Components                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │  Layer1  │  │  Layer2  │  │Confidence│  │ Context  │  │
│  │Abstracts │  │  Chunks  │  │Evaluator │  │ Expander │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
└────────────────────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────────────────┐
│                Data Storage Layer                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ vectorstore/ │  │    data/     │  │ json_files/  │    │
│  │  (FAISS)     │  │   (PDFs)     │  │  (Metadata)  │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
└────────────────────────────────────────────────────────────┘
```

---

## 階層式 RAG 系統

### 核心理念

傳統 RAG 問題：
```
Query: "深度學習在醫療的應用"
傳統 RAG 結果: 
  ✓ "深度學習模型" (相關)
  ✗ "深度學習股價預測" (不相關，但語義相似)
  ✗ "類神經網路鼻咽癌" (部分相關)
```

Hierarchical RAG 解決方案：
```
Layer 1: 先在摘要層過濾 → 只保留醫療相關論文
Layer 2: 再在區塊層精確檢索 → 找到具體內容
```

### 架構設計

```
┌─────────────────────────────────────────────────────────────┐
│                    User Query                                │
│              "深度學習在醫療的應用是什麼？"                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Layer 1: Abstract Search                    │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ • 搜索所有論文的摘要（Abstract）                         │ │
│  │ • 使用 Embedding 相似度搜索                              │ │
│  │ • 找出最相關的 k 篇論文（預設 k=15）                     │ │
│  │ • 只保留這些論文，其他完全排除                           │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  結果: [論文A: 鼻咽癌辨識, 論文B: 醫療AI應用, ...]           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Confidence Evaluation (Early Termination)       │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ • LLM 評估 Layer 1 結果的信心度                          │ │
│  │ • Prompt: "根據這些摘要，能否回答問題？(0-1分)"          │ │
│  │ • 閾值: 0.7 (可配置)                                     │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│         ┌─────────────────┴─────────────────┐               │
│         │                                   │               │
│    信心度 ≥ 0.7                        信心度 < 0.7         │
│         │                                   │               │
│         ▼                                   ▼               │
│  ┌────────────┐                    ┌──────────────────┐    │
│  │ 早期終止    │                    │ 繼續到 Layer 2    │    │
│  │ (50-70%)   │                    │                  │    │
│  └────────────┘                    └──────────────────┘    │
└─────────────────────────────────────────────────────────────┘
        │                                   │
        │                                   ▼
        │              ┌─────────────────────────────────────┐
        │              │     Layer 2: Chunk Search            │
        │              │ ┌─────────────────────────────────┐ │
        │              │ │ • 只在 Layer 1 選中的論文中搜索  │ │
        │              │ │ • 將論文分割為小區塊（800字）    │ │
        │              │ │ • 檢索最相關的 k 個區塊          │ │
        │              │ │ • 使用 Context Expander 擴展     │ │
        │              │ └─────────────────────────────────┘ │
        │              └─────────────────────────────────────┘
        │                              │
        │                              ▼
        │              ┌─────────────────────────────────────┐
        │              │   Confidence Evaluation (再次評估)   │
        │              │  • 評估 Layer 2 結果的信心度         │
        │              │  • 閾值: 0.8 (更高標準)             │
        │              └─────────────────────────────────────┘
        │                              │
        └──────────────────────────────┘
                       │
                       ▼
        ┌────────────────────────────────────────────┐
        │         Context Expansion (動態擴展)        │
        │  • 根據信心度決定擴展範圍：                 │
        │    - 高信心 (≥0.85): ±1 chunks            │
        │    - 中信心 (0.75-0.85): ±2 chunks        │
        │    - 低信心 (<0.75): ±3 chunks            │
        │  • 保持上下文連貫性                         │
        └────────────────────────────────────────────┘
                       │
                       ▼
        ┌────────────────────────────────────────────┐
        │          LLM Answer Generation              │
        │  • 使用檢索到的上下文                        │
        │  • 生成完整答案                              │
        │  • 支援流式輸出 (SSE)                        │
        └────────────────────────────────────────────┘
                       │
                       ▼
               最終答案返回給用戶
```

### 層次設計細節

#### Layer 1: Abstract VectorStore
**檔案**: `system_api/layer1_vectorstore.py`

**數據結構**:
```python
Document {
    page_content: "論文摘要全文...",
    metadata: {
        'paper_id': '論文ID',
        'title': '論文標題',
        'pdf_path': '實際路徑',
        'source': 'extracted/llm_generated/metadata',
        'confidence': 0.0-1.0,
        'extraction_method': '提取方法'
    }
}
```

**索引位置**: `vectorstore/layer1/`
- `index.faiss` - FAISS 向量索引
- `index.pkl` - 文檔內容與元數據

**功能**:
- `build_index(abstracts)` - 建立索引
- `search(query, k=15)` - 搜索相關論文
- `load()` - 載入現有索引
- `get_stats()` - 獲取統計資訊

#### Layer 2: Chunk VectorStore
**檔案**: `system_api/layer2_vectorstore.py`

**數據結構**:
```python
Document {
    page_content: "論文區塊內容...(約800字)",
    metadata: {
        'paper_id': '論文ID',
        'chunk_id': '論文ID_chunk_0',
        'chunk_index': 0,
        'pdf_path': '實際路徑',
        'source': 'hierarchical_build'
    }
}
```

**索引位置**: `vectorstore/layer2/`
- 按論文組織的區塊索引
- 支援快取機制 (LRU cache)

**功能**:
- `build_index(chunks)` - 建立索引
- `search_by_papers(query, paper_ids, k=10)` - 在指定論文中搜索
- `get_chunk_neighbors(chunk, range=2)` - 獲取相鄰區塊
- `load()` - 載入現有索引

#### Confidence Evaluator
**檔案**: `system_api/confidence_evaluator.py`

**評估邏輯**:
```python
def evaluate(query, contexts, layer):
    prompt = f"""
    問題: {query}
    
    上下文:
    {contexts}
    
    請評估這些上下文是否足以回答問題。
    給出 0-1 的信心分數。
    
    僅回答數字。
    """
    
    response = llm.invoke(prompt)
    confidence = parse_score(response)  # 0.0-1.0
    
    return confidence
```

**決策邏輯**:
```python
if layer == 1:
    if confidence >= 0.7:
        return "terminate"  # 早期終止
    else:
        return "continue"   # 進入 Layer 2
        
elif layer == 2:
    if confidence >= 0.8:
        return "sufficient"
    else:
        return "expand"     # 擴展上下文
```

#### Context Expander
**檔案**: `system_api/context_expander.py`

**擴展策略**:
```python
def expand(chunks, confidence):
    if confidence >= 0.85:
        range = 1  # ±1 chunks
    elif confidence >= 0.75:
        range = 2  # ±2 chunks
    else:
        range = 3  # ±3 chunks
    
    expanded_chunks = []
    for chunk in chunks:
        neighbors = layer2.get_chunk_neighbors(chunk, range)
        expanded_chunks.extend(neighbors)
    
    return deduplicate_and_sort(expanded_chunks)
```

---

## 索引管理邏輯

### 智能檢查與更新系統

**檔案**: `system_api/hierarchical_rag_system.py` - `check_and_update_indices()`

#### 啟動時自動檢查流程

```
agent2.py 啟動
    ↓
HierarchicalRAGSystem.__init__()
    ↓
_load_indices()  # 嘗試載入現有索引
    ↓
check_and_update_indices()
    ↓
    ├─ 檢查 1: 索引是否就緒？
    │   ├─ Layer 1 initialized? 
    │   └─ Layer 2 initialized?
    │
    ├─ 檢查 2: PDF 數量比對
    │   ├─ 當前 data/ 目錄: X 個 PDF
    │   ├─ 索引中記錄: Y 個 PDF
    │   └─ X vs Y 比較
    │       ├─ X > Y  → 有新增 → 重建索引
    │       ├─ X < Y  → 有刪除 → 重建索引
    │       └─ X == Y → 繼續檢查
    │
    ├─ 檢查 3: 索引完整性驗證
    │   ├─ Layer 1 stats 可讀取？
    │   ├─ Layer 2 stats 可讀取？
    │   └─ 元數據完整？
    │
    └─ 決策與執行
        ├─ 全部通過 → 跳過，使用現有索引
        ├─ 發現問題 → 執行 build_indices()
        └─ 強制重建 → 執行 build_indices()
```

#### 索引建立流程

```python
def build_indices(pdf_paths=None):
    """
    完整的索引建立流程
    """
    # 1. 掃描 PDF 文件
    if pdf_paths is None:
        pdf_paths = _get_all_pdfs()  # 從 ./data 目錄
    
    abstracts = []
    all_chunks = []
    
    # 2. 處理每個 PDF
    for pdf_path in pdf_paths:
        # 2.1 讀取 PDF
        reader = PdfReader(pdf_path)
        pdf_text = extract_all_text(reader)
        
        # 2.2 提取摘要（給 Layer 1）
        abstract = abstract_extractor.extract(
            pdf_path, 
            pdf_text, 
            paper_id
        )
        abstracts.append(abstract)
        
        # 2.3 分割為區塊（給 Layer 2）
        chunks = text_splitter.create_documents(
            texts=[pdf_text],
            metadatas=[{
                'paper_id': paper_id,
                'pdf_path': pdf_path
            }]
        )
        
        # 2.4 添加區塊元數據
        for idx, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = f"{paper_id}_chunk_{idx}"
            chunk.metadata['chunk_index'] = idx
        
        all_chunks.extend(chunks)
    
    # 3. 建立 Layer 1 索引
    layer1.build_index(abstracts)
    
    # 4. 建立 Layer 2 索引
    layer2.build_index(all_chunks)
    
    # 5. 返回統計資訊
    return {
        'status': 'success',
        'papers_processed': len(pdf_paths),
        'abstracts_extracted': len(abstracts),
        'chunks_created': len(all_chunks)
    }
```

### Abstract 提取器

**檔案**: `system_api/abstract_extractor.py`

**多重策略提取**:

```python
def extract(pdf_path, pdf_text, paper_id):
    """
    按順序嘗試多種提取方法
    """
    # 策略 1: Regex 提取（中英文）
    abstract = _try_regex_extraction(pdf_text)
    if abstract:
        return create_document(abstract, 'regex', confidence=0.9)
    
    # 策略 2: 從 Metadata 提取
    abstract = _try_metadata_extraction(pdf_path)
    if abstract:
        return create_document(abstract, 'metadata', confidence=0.8)
    
    # 策略 3: LLM 生成（Fallback）
    abstract = _generate_with_llm(pdf_text[:3000])
    if abstract:
        return create_document(abstract, 'llm_generated', confidence=0.7)
    
    # 策略 4: 使用前 N 個字作為 Fallback
    abstract = pdf_text[:500]
    return create_document(abstract, 'fallback', confidence=0.3)
```

**Regex 模式** (支援中英文):
```python
ABSTRACT_PATTERNS = [
    # 英文
    r'Abstract\s*[:：]?\s*\n(.+?)(?=\n\s*(?:Keywords|Introduction|1\.|$))',
    
    # 中文
    r'摘\s*要\s*[:：]?\s*\n(.+?)(?=\n\s*(?:關鍵字|前言|一、|$))',
    r'【摘要】\s*(.+?)(?=\n\s*(?:【關鍵字】|$))',
    
    # 混合
    r'(?:Abstract|摘要)\s*[:：]?\s*\n(.+?)(?=\n\s*(?:Keywords|關鍵字|$))',
]
```

---

## 查詢處理流程

### 完整查詢管線

```
用戶發送查詢
    ↓
POST /query 或 POST /query_stream
    ↓
agent_executor (Agent 路由)
    ↓
決策: AssistantCall 或 WebSearchCall?
    ↓
AssistantCall (RAG 查詢)
    ↓
rag_system.query() 或 rag_system.query_stream()
    ↓
_hierarchical_retrieval(query)
    ↓
┌─────────────────────────────────────┐
│  Step 1: Layer 1 Search             │
│  • layer1.search(query, k=15)       │
│  • 獲得 15 篇相關論文的摘要           │
│  • 提取 paper_ids                    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Step 2: Confidence Evaluation      │
│  • evaluator.evaluate(query, layer1)│
│  • 獲得信心分數 (0-1)                │
└─────────────────────────────────────┘
    ↓
    信心度 >= 0.7?
    ├─ Yes → 早期終止，使用 Layer 1 結果
    └─ No  → 繼續
        ↓
┌─────────────────────────────────────┐
│  Step 3: Layer 2 Search             │
│  • layer2.search_by_papers(         │
│      query, paper_ids, k=10)        │
│  • 只在選定論文中搜索區塊             │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Step 4: Confidence Evaluation      │
│  • evaluator.evaluate(query, layer2)│
│  • 獲得信心分數 (0-1)                │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Step 5: Context Expansion          │
│  • expander.expand_contexts(        │
│      chunks, confidence)            │
│  • 根據信心度擴展 ±1-3 chunks        │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Step 6: Query Logger               │
│  • logger.log_query(query, result)  │
│  • 記錄統計資訊                       │
└─────────────────────────────────────┘
    ↓
返回檢索結果 (contexts + metadata)
    ↓
_generate_answer(query, result)
    ↓
使用 LLM 生成答案
    ↓
返回給用戶
```

### 查詢返回格式

#### 標準查詢 (POST /query)

```json
{
  "answer": "深度學習在醫療領域的應用包括...",
  "rag_mode": "hierarchical",
  "retrieval_stats": {
    "termination_layer": "layer2",
    "confidence": 0.85,
    "papers_used": [
      "鼻咽癌辨識.pdf",
      "醫療AI應用.pdf"
    ],
    "chunks_retrieved": 10,
    "expansion_applied": true,
    "retrieval_time": {
      "layer1": 2.3,
      "evaluation1": 8.5,
      "layer2": 12.1,
      "evaluation2": 9.2,
      "expansion": 1.5,
      "total": 33.6
    }
  }
}
```

#### 流式查詢 (POST /query_stream)

Server-Sent Events 格式:

```
data: {"type": "start", "message": "Starting query processing"}

data: {"type": "retrieval", "stats": {...}}

data: {"type": "chunk", "content": "深度學習"}

data: {"type": "chunk", "content": "在醫療"}

data: {"type": "chunk", "content": "領域"}

data: {"type": "done", "message": "Query completed"}
```

---

## 組件詳解

### 1. Agent Router (agent2.py)

**功能**: 智能路由用戶查詢到正確的工具

```python
tools = [
    Tool(
        name="AssistantCall",
        func=assistant_call,
        description="用於論文摘要、解釋或翻譯。最適合分析學術內容。"
    ),
    Tool(
        name="WebSearchCall",
        func=web_search_call,
        description="用於網路搜索、尋找最新資訊。"
    )
]

# ReAct Agent 決策範例
"""
Question: 總結這篇論文的主要發現
Thought: 這需要分析學術內容，應該使用 AssistantCall
Action: AssistantCall
Action Input: 總結這篇論文的主要發現
Observation: [RAG 系統返回結果]
Thought: 我現在知道最終答案了
Final Answer: 這篇論文的主要發現是...
"""
```

### 2. Index Manager (system_api/index_manager.py)

**功能**: 管理索引的儲存與載入

**目錄結構**:
```
vectorstore/
├── layer1_abstracts/
│   ├── index.faiss          # FAISS 向量索引
│   ├── index.pkl            # 序列化的文檔數據
│   └── metadata.json        # 索引元數據
├── layer2_chunks/
│   ├── index.faiss
│   ├── index.pkl
│   └── metadata.json
└── [legacy files...]        # 舊的單層索引（可刪除）
```

**操作**:
```python
# 儲存索引
index_manager.save_index(
    vectorstore=layer1_store,
    path="vectorstore/layer1",
    metadata={
        'paper_count': 32,
        'created_at': '2025-11-13',
        'version': '1.0'
    }
)

# 載入索引
vectorstore, metadata = index_manager.load_index(
    path="vectorstore/layer1"
)
```

### 3. Query Logger (system_api/query_logger.py)

**功能**: 記錄查詢歷史與性能統計

**日誌格式**:
```python
{
    'query_id': 'uuid-xxxx',
    'timestamp': '2025-11-13 10:30:45',
    'query': '用戶問題',
    'termination_layer': 'layer1' or 'layer2',
    'confidence': 0.85,
    'papers_used': ['paper1.pdf', 'paper2.pdf'],
    'chunks_retrieved': 10,
    'retrieval_time': {
        'layer1': 2.3,
        'layer2': 12.1,
        'total': 33.6
    },
    'answer_length': 500,
    'success': true
}
```

**統計分析**:
```python
# 獲取統計
stats = query_logger.get_statistics()

{
    'total_queries': 100,
    'termination_distribution': {
        'layer1': 65,  # 65% 早期終止
        'layer2': 35   # 35% 需要深入檢索
    },
    'average_confidence': {
        'layer1': 0.82,
        'layer2': 0.87
    },
    'average_retrieval_time': {
        'layer1_terminated': 15.2,  # 秒
        'layer2_completed': 45.6    # 秒
    }
}
```

### 4. Paper Management (agent2.py)

**功能**: 論文上傳、分類、更新

**API 端點**:

#### POST /upload_paper
```python
# 上傳流程
1. 接收 PDF 文件
2. 驗證 PDF (PDFValidator)
3. 提取資訊 (PaperExtractor)
   - 標題
   - 作者
   - 年份
   - 關鍵字
4. 自動分類 (LLM)
5. 儲存到 ./data/
6. 更新 data_categories.json
7. [可選] 觸發索引更新
```

#### POST /update_paper
```python
# 更新流程
1. 接收更新資料
2. 從 data_categories.json 移除舊記錄
3. 添加新記錄
4. 如標題改變，重命名 PDF 文件
5. 返回新分類
```

### 5. Inheritance Analyzer (system_api/node_support.py)

**功能**: 分析論文之間的研究繼承關係

**數據結構** (json_files/relationships.json):
```json
{
  "nodes": [
    {
      "id": "paper1",
      "label": "論文標題",
      "type": "基礎",
      "year": 2020
    }
  ],
  "edges": [
    {
      "from": "paper1",
      "to": "paper2",
      "relationship": "延伸",
      "strength": 0.8
    }
  ]
}
```

**分析流程**:
```python
def analyze_inheritance(paper_title, paper_abstract):
    # 1. 找出所有相關論文
    related = find_related_papers(paper_abstract)
    
    # 2. LLM 分析關係
    for candidate in related:
        prompt = f"""
        論文A: {paper_title}
        論文B: {candidate.title}
        
        分析兩者的研究繼承關係：
        - 是否有延伸關係？
        - 關係強度 (0-1)？
        - 關係類型（延伸/應用/改進）？
        """
        
        relationship = llm.analyze(prompt)
    
    # 3. 構建關係圖
    graph = build_graph(relationships)
    
    return graph
```

---

## API 端點

### 查詢端點

#### POST /query
**功能**: 標準查詢（阻塞式）

**請求**:
```json
{
  "input": "深度學習在醫療的應用是什麼？"
}
```

**響應**:
```json
{
  "answer": "深度學習在醫療領域...",
  "rag_mode": "hierarchical",
  "retrieval_stats": {...}
}
```

#### POST /query_stream
**功能**: 流式查詢（SSE）

**請求**: 同上

**響應**: Server-Sent Events
```
data: {"type": "start"}
data: {"type": "chunk", "content": "..."}
data: {"type": "done"}
```

### 監控端點

#### GET /rag/stats
**功能**: 系統統計資訊

**響應**:
```json
{
  "mode": "hierarchical",
  "system_ready": true,
  "system_stats": {
    "layer1": {
      "paper_count": 32,
      "is_initialized": true
    },
    "layer2": {
      "chunk_count": 517,
      "paper_count": 32,
      "is_initialized": true
    }
  },
  "query_stats": {
    "total_queries": 100,
    "termination_distribution": {
      "layer1": 65,
      "layer2": 35
    }
  },
  "performance_summary": {
    "early_termination_rate": 0.65,
    "average_time": {
      "layer1_terminated": 15.2,
      "layer2_completed": 45.6
    }
  }
}
```

#### GET /rag/health
**功能**: 健康檢查

**響應**:
```json
{
  "status": "healthy",
  "components": {
    "rag_system": "up",
    "indices": "ready",
    "layer1": "up",
    "layer2": "up"
  },
  "issues": []
}
```

#### GET /rag/mode
**功能**: 模式資訊

**響應**:
```json
{
  "mode": "hierarchical",
  "type": "HierarchicalRAGSystem",
  "features": [
    "two-layer",
    "confidence-evaluation",
    "context-expansion"
  ],
  "config": {
    "layer1_threshold": 0.7,
    "layer2_threshold": 0.8,
    "expansion_enabled": true,
    "cache_size": 10
  }
}
```

#### POST /rag/rebuild
**功能**: 手動觸發索引重建

**請求**:
```bash
# 智能更新
curl -X POST http://localhost:4000/rag/rebuild

# 強制重建
curl -X POST "http://localhost:4000/rag/rebuild?force=true"
```

**響應**:
```json
{
  "status": "success",
  "action_taken": "incremental_update",
  "details": {
    "papers_processed": 35,
    "abstracts_extracted": 35,
    "chunks_created": 565,
    "duration": 127.5
  }
}
```

### 論文管理端點

#### POST /upload_paper
上傳新論文

#### POST /update_paper
更新論文資訊

#### POST /analyze_inheritance
分析研究繼承關係

---

## 數據流圖

### 查詢數據流

```
User Browser
    │
    │ HTTP POST /query
    │ {"input": "問題"}
    ▼
Flask (agent2.py)
    │
    │ agent_executor.invoke()
    ▼
ReAct Agent
    │
    │ 決策: 使用 AssistantCall
    ▼
assistant_call(query)
    │
    │ rag_system.query()
    ▼
HierarchicalRAGSystem
    │
    │ _hierarchical_retrieval()
    ▼
Layer 1 Search
    │ layer1.search(query)
    │ → FAISS 向量搜索
    │ → 返回 15 篇論文摘要
    ▼
Confidence Evaluator
    │ evaluate(query, contexts)
    │ → LLM 評估信心度
    │ → 返回分數 (0-1)
    ▼
決策分支
    ├─ 信心度 ≥ 0.7 → 早期終止
    │   └─→ 直接生成答案
    │
    └─ 信心度 < 0.7 → 繼續檢索
        ▼
    Layer 2 Search
        │ layer2.search_by_papers()
        │ → 在選定論文中搜索
        │ → 返回 10 個區塊
        ▼
    Confidence Evaluator
        │ evaluate(query, chunks)
        ▼
    Context Expander
        │ expand_contexts(chunks, conf)
        │ → 擴展 ±1-3 區塊
        ▼
    Query Logger
        │ log_query(...)
        ▼
Answer Generation
    │ llm.invoke(prompt + contexts)
    │ → 生成最終答案
    ▼
Response
    │ JSON 格式
    ▼
User Browser
```

### 索引建立數據流

```
PDF Files (./data/*.pdf)
    │
    │ HierarchicalRAGSystem.build_indices()
    ▼
PDF Processing
    │
    ├─→ PyPDF2.PdfReader
    │   └─→ extract_text()
    │
    ├─→ AbstractExtractor
    │   ├─→ Regex 提取
    │   ├─→ Metadata 提取
    │   └─→ LLM 生成
    │   └─→ Fallback
    │
    └─→ Text Splitter
        └─→ 分割為 800 字區塊
    │
    ▼
Data Organization
    │
    ├─→ abstracts[]
    │   └─→ Document(摘要, metadata)
    │
    └─→ chunks[]
        └─→ Document(區塊, metadata)
    │
    ▼
Vector Indexing
    │
    ├─→ Layer 1 Build
    │   ├─→ OllamaEmbeddings.embed_documents()
    │   ├─→ FAISS.from_documents()
    │   └─→ save to vectorstore/layer1/
    │
    └─→ Layer 2 Build
        ├─→ OllamaEmbeddings.embed_documents()
        ├─→ FAISS.from_documents()
        └─→ save to vectorstore/layer2/
    │
    ▼
Index Files
    │
    ├─→ vectorstore/layer1/
    │   ├─→ index.faiss
    │   └─→ index.pkl
    │
    └─→ vectorstore/layer2/
        ├─→ index.faiss
        └─→ index.pkl
```

---

## 錯誤處理

### 系統層級錯誤處理

#### 1. 索引載入失敗
```python
try:
    layer1.load()
    layer2.load()
except Exception as e:
    logger.error(f"Failed to load indices: {e}")
    # 自動觸發重建
    build_indices()
```

#### 2. PDF 處理失敗
```python
for pdf_path in pdf_paths:
    try:
        process_pdf(pdf_path)
    except Exception as e:
        logger.error(f"Failed to process {pdf_path}: {e}")
        # 跳過該 PDF，繼續處理其他
        continue
```

#### 3. LLM 調用失敗
```python
def evaluate_with_retry(query, contexts, max_retries=3):
    for attempt in range(max_retries):
        try:
            return llm.invoke(prompt)
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error("LLM call failed after retries")
                return default_confidence  # 使用預設值
            time.sleep(2 ** attempt)  # 指數退避
```

#### 4. 查詢超時
```python
@app.route('/query', methods=['POST'])
def query():
    try:
        # 設置超時
        result = rag_system.query(query, timeout=60)
    except TimeoutError:
        return jsonify({
            'error': 'Query timeout',
            'message': 'Please try a simpler query'
        }), 504
```

### API 錯誤碼

| 狀態碼 | 說明 | 範例 |
|--------|------|------|
| 200 | 成功 | 查詢正常返回 |
| 400 | 錯誤請求 | 缺少必需參數 |
| 500 | 伺服器錯誤 | LLM 調用失敗 |
| 503 | 服務不可用 | RAG 系統未初始化 |
| 504 | 請求超時 | 查詢處理時間過長 |

---

## 配置與環境變數

### 環境變數

```bash
# Layer 閾值配置
export LAYER1_THRESHOLD=0.7      # Layer 1 信心閾值
export LAYER2_THRESHOLD=0.8      # Layer 2 信心閾值

# 功能開關
export ENABLE_EXPANSION=true     # 啟用上下文擴展

# 性能配置
export CACHE_SIZE=10             # Layer 2 快取大小
```

### 系統配置

**檔案**: `system_api/hierarchical_rag_system.py`

```python
HIERARCHICAL_RAG_CONFIG = {
    'layer1': {
        'k_documents': 15,           # 檢索論文數
        'confidence_threshold': 0.7,  # 早期終止閾值
    },
    'layer2': {
        'k_documents': 10,           # 檢索區塊數
        'confidence_threshold': 0.8,  # 擴展閾值
    },
    'expansion': {
        'enabled': True,
        'ranges': {
            'high': 1,    # 高信心: ±1 chunks
            'medium': 2,  # 中信心: ±2 chunks
            'low': 3      # 低信心: ±3 chunks
        }
    },
    'performance': {
        'use_cache': True,
        'cache_size': 10,
    }
}
```

---

## 性能特性

### 查詢性能

| 場景 | 平均時間 | 說明 |
|------|----------|------|
| Layer 1 早期終止 | 10-16s | ~50-70% 查詢 |
| Layer 2 完整檢索 | 60-75s | ~30-50% 查詢 |
| 早期終止率 | 50-70% | 根據實際測試 |

### 索引性能

| 操作 | 時間 | 備註 |
|------|------|------|
| 載入索引 | <1s | 已建立的索引 |
| 建立 32 篇論文索引 | 5-10min | 首次建立 |
| 增量更新（+3 篇） | 2-3min | 只處理新 PDF |

### 記憶體使用

| 組件 | 記憶體 | 備註 |
|------|--------|------|
| Layer 1 Index | ~50MB | 32 篇論文 |
| Layer 2 Index | ~200MB | 517 個區塊 |
| LLM Cache | ~1GB | Ollama 模型 |
| 總計 | ~1.5GB | 運行時 |

---

## 部署架構

### 開發環境

```
MacBook Local
├── Ollama Server (localhost:11434)
│   ├── gemma3:12b
│   └── embeddinggemma:latest
│
├── Flask App (localhost:4000)
│   └── agent2.py
│
└── File System
    ├── ./data/*.pdf
    ├── ./vectorstore/
    └── ./json_files/
```

### 生產環境建議

```
Load Balancer
    │
    ├─→ App Server 1
    │   ├── Flask + Gunicorn
    │   ├── Shared Storage (NFS)
    │   │   ├── ./data/
    │   │   └── ./vectorstore/
    │   └── → Ollama Server
    │
    └─→ App Server 2
        ├── Flask + Gunicorn
        ├── Shared Storage (NFS)
        └── → Ollama Server

Ollama Cluster
    ├── Node 1 (LLM)
    └── Node 2 (Embeddings)
```

---

## 監控與維護

### 健康檢查腳本

```bash
#!/bin/bash
# health_check.sh

# 檢查服務
curl -f http://localhost:4000/rag/health || exit 1

# 檢查統計
stats=$(curl -s http://localhost:4000/rag/stats)
paper_count=$(echo $stats | jq '.system_stats.layer1.paper_count')

if [ $paper_count -eq 0 ]; then
    echo "Error: No papers indexed"
    exit 1
fi

echo "Health check passed"
```

### 日誌監控

```bash
# 查看錯誤日誌
tail -f agent2.log | grep ERROR

# 查看查詢統計
curl http://localhost:4000/rag/stats | jq '.query_stats'

# 查看性能
curl http://localhost:4000/rag/stats | jq '.performance_summary'
```

### 備份策略

```bash
#!/bin/bash
# backup.sh

DATE=$(date +%Y%m%d_%H%M%S)

# 備份索引
tar -czf backup_vectorstore_$DATE.tar.gz vectorstore/

# 備份論文
tar -czf backup_data_$DATE.tar.gz data/

# 備份元數據
tar -czf backup_json_$DATE.tar.gz json_files/

echo "Backup completed: $DATE"
```

---

## 故障排除

### 常見問題

#### 1. 索引未建立
**症狀**: `No existing indices found`

**解決**:
```bash
# 自動建立（重啟）
python agent2.py

# 或手動觸發
curl -X POST http://localhost:4000/rag/rebuild
```

#### 2. 查詢很慢
**症狀**: 查詢超過 60 秒

**可能原因**:
- Ollama 服務未運行
- 第一次查詢（模型載入）
- Layer 2 深度檢索（正常）

**解決**:
```bash
# 檢查 Ollama
ollama list

# 重啟 Ollama
ollama serve

# 檢查查詢統計
curl http://localhost:4000/rag/stats | jq '.query_stats'
```

#### 3. PDF 處理失敗
**症狀**: 部分 PDF 無法索引

**檢查**:
```bash
# 查看日誌
tail agent2.log | grep "Failed to process"

# 手動測試 PDF
python -c "from PyPDF2 import PdfReader; PdfReader('problem.pdf')"
```

**解決**:
- 確認 PDF 未加密
- 確認 PDF 未損壞
- 移除問題 PDF 或修復後重建

#### 4. 記憶體不足
**症狀**: 系統崩潰或變慢

**解決**:
```bash
# 減少快取大小
export CACHE_SIZE=5

# 減少檢索數量
# 修改 config: k_documents 從 15 降到 10

# 重啟服務
python agent2.py
```

---

## 擴展性

### 水平擴展

**支援**:
- ✅ 多個 Flask 實例（無狀態）
- ✅ 共享檔案系統（NFS）
- ✅ 負載均衡器

**不支援**:
- ❌ 分散式索引（目前單機 FAISS）

### 垂直擴展

**建議配置**:

| 論文數量 | CPU | 記憶體 | 磁碟 |
|----------|-----|--------|------|
| <50 篇   | 4核 | 8GB    | 20GB |
| 50-200 篇 | 8核 | 16GB   | 50GB |
| 200+ 篇  | 16核 | 32GB   | 100GB |

---

## 總結

### 系統優勢

✅ **高精確度**: 階層式檢索減少不相關結果  
✅ **性能優化**: 50-70% 查詢早期終止  
✅ **智能管理**: 自動檢測與更新索引  
✅ **擴展性**: 動態上下文擴展  
✅ **監控完善**: 詳細的統計與日誌  

### 技術亮點

1. **兩層索引架構**: 摘要層 + 區塊層
2. **LLM 信心評估**: 動態決策檢索深度
3. **智能索引管理**: 自動檢測更新
4. **上下文擴展**: 根據信心度調整
5. **完整監控**: 健康、統計、性能追蹤

### 適用場景

✅ 學術論文檢索與問答  
✅ 大規模文檔知識庫  
✅ 需要高精確度的 RAG 應用  
✅ 研究繼承關係分析  

---

**文檔版本**: 1.0  
**最後更新**: 2025-11-14  
**維護者**: NCHU702 Team
