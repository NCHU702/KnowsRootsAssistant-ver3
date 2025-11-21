"""
Stage 4: Graph-first Flow 整合計劃

================================================================================
📋 架構分析
================================================================================

當前 HierarchicalRAGSystem 的檢索流程:
1. Query Enhancement (Query Expansion + Adaptive Weights)
2. Layer 1: Paper-level retrieval (abstracts)
3. Layer 1 Evaluation + Layer 2 Trigger Decision
4. Layer 2: Chunk-level retrieval (filtered by paper_ids)
5. Layer 2 Evaluation
6. Context Expansion (optional)
7. Final Answer Generation

================================================================================
🎯 整合目標
================================================================================

在 Layer 1 之前插入 Graph-first 階段:

新流程:
0. Query Enhancement (保持不變)
1. **GRAPH STAGE: Query Neo4j for papers** (新增)
   - GraphRetriever: 查詢 Neo4j 找相關 papers
   - GraphIntegrator: LLM 整合 Graph 結果
   - 評估 confidence 決定是否下降
   
2a. **如果 Graph 足夠**: 直接返回答案 (terminated_at='graph')
2b. **如果 Graph 不足**: 下降到 Layer2 (skip Layer1)
   - 使用 Graph 找到的 paper_ids 過濾
   - 使用 GraphIntegrator 決定的 chunk_types 過濾
   
3. Layer 2 Retrieval (with dual filtering)
4. Final Answer Generation (merge Graph + Layer2)

================================================================================
📁 需要修改的檔案
================================================================================

1. system_api/hierarchical_rag_system.py
   - __init__(): 初始化 GraphRetriever + GraphIntegrator + ChunkClassifier
   - _hierarchical_retrieval(): 插入 Graph-first 階段
   - 配置: 新增 graph_first 區段

2. system_api/index_manager.py
   - add_document(): 在 Step 4 後使用 ChunkClassifier 標註 chunks
   - 確保 chunk.metadata['chunk_type'] 被正確設置

3. 新增測試檔案
   - test_graph_first_stage4_integration.py: 測試整合邏輯

================================================================================
🔧 詳細實作步驟
================================================================================

Step 1: 更新 HIERARCHICAL_RAG_CONFIG
-------------------------------------
新增配置區段:
```python
'graph_first': {
    'enabled': True,  # 啟用 Graph-first 模式
    'confidence_threshold': 0.7,  # Graph 結果的信心閾值
    'min_graph_hits': 1,  # 最少需要找到的 paper 數量
    'skip_layer1': True,  # 是否跳過 Layer1（直接 Graph → Layer2）
},
'chunk_classification': {
    'enabled': True,  # 在索引時標註 chunk types
    'use_llm_fallback': False,  # Heuristic-only (faster)
}
```

Step 2: 修改 __init__() - 初始化新組件
----------------------------------------
在現有組件初始化後添加:
```python
# Graph-first components (if graph is available)
self.graph_retriever: Optional[GraphRetriever] = None
self.graph_integrator: Optional[GraphIntegrator] = None
self.chunk_classifier: Optional[ChunkClassifier] = None

if self.index_manager.graph_manager and self.config['graph_first']['enabled']:
    from system_api.graph_retriever import GraphRetriever
    from system_api.graph_integrator import GraphIntegrator
    
    self.graph_retriever = GraphRetriever(self.index_manager.graph_manager)
    self.graph_integrator = GraphIntegrator(self.llm)
    logger.info("✓ Graph-first components initialized")

if self.config['chunk_classification']['enabled']:
    from system_api.chunk_classifier import ChunkClassifier
    
    use_llm = self.config['chunk_classification'].get('use_llm_fallback', False)
    self.chunk_classifier = ChunkClassifier(
        llm=self.llm if use_llm else None,
        use_llm_fallback=use_llm
    )
    logger.info("✓ ChunkClassifier initialized")
```

Step 3: 修改 _hierarchical_retrieval() - 插入 Graph 階段
--------------------------------------------------------
在 STEP 0 (Query Enhancement) 之後，Layer 1 之前插入:

```python
# ============================================================
# GRAPH STAGE: Query Neo4j for papers (if enabled)
# ============================================================
if self.graph_retriever and self.graph_integrator:
    logger.info("="*60)
    logger.info("GRAPH STAGE: Querying Knowledge Graph")
    logger.info("="*60)
    
    graph_start = time.time()
    
    # Step 1: Query graph for papers
    graph_results = self.graph_retriever.query_graph_for_papers(
        query=query,
        top_k=15  # Same as Layer1 k
    )
    
    result['timings']['graph_query'] = time.time() - graph_start
    result['layers_used'].append('graph')
    result['graph_results'] = graph_results
    
    logger.info(f"Found {len(graph_results)} papers from Graph")
    
    if graph_results:
        # Step 2: Integrate graph results using LLM
        integration_start = time.time()
        
        integration = self.graph_integrator.integrate_graph_results(
            query=query,
            graph_results=graph_results,
            max_papers=5  # Limit context size
        )
        
        result['timings']['graph_integration'] = time.time() - integration_start
        result['graph_integration'] = integration
        
        logger.info(f"Graph Integration:")
        logger.info(f"  Confidence: {integration['confidence']:.2f}")
        logger.info(f"  Should descend: {integration['should_descend']}")
        logger.info(f"  Missing info: {integration['missing_info']}")
        
        # Step 3: Decision - should we descend to Layer2?
        graph_threshold = self.config['graph_first']['confidence_threshold']
        
        if not integration['should_descend']:
            logger.info(f"✓ Graph results sufficient (confidence: {integration['confidence']:.2f} >= {graph_threshold})")
            logger.info("  No detailed chunk retrieval needed")
            
            result['final_answer'] = integration['text']
            result['final_confidence'] = integration['confidence']
            result['terminated_at'] = 'graph'
            result['referenced_papers'] = integration['referenced_papers']
            result['timings']['total'] = time.time() - start_time
            
            return result
        
        logger.info("→ Graph results insufficient, descending to Layer2 for details")
        
        # Step 4: Determine chunk types to retrieve
        chunk_types = self.graph_integrator.determine_chunk_types_for_query(
            query=query,
            missing_info=integration['missing_info']
        )
        
        result['target_chunk_types'] = chunk_types
        
        # Step 5: Extract paper_ids for Layer2 filtering
        graph_paper_ids = [r['paper_id'] for r in graph_results]
        
        logger.info(f"Will retrieve chunk types {chunk_types} from {len(graph_paper_ids)} papers")
        
        # Skip to Layer 2 directly (bypass Layer1)
        if self.config['graph_first'].get('skip_layer1', True):
            logger.info("  Skipping Layer1 (Graph → Layer2 direct)")
            
            # Jump to Layer 2 with filtering
            # [Insert Layer2 retrieval code with dual filtering]
            # ...
```

Step 4: 修改 Layer2 retrieval - 應用 chunk type 過濾
----------------------------------------------------
在 Layer2 檢索時:
```python
# Use filter_chunk_types if determined by Graph
filter_chunk_types = result.get('target_chunk_types', None)

if filter_chunk_types:
    logger.info(f"Applying chunk type filter: {filter_chunk_types}")

for paper_id in paper_ids:
    paper_results = self.layer2.search_with_scores(
        query=query,
        k=CHUNKS_PER_PAPER,
        filter_paper_ids=[paper_id],
        filter_chunk_types=filter_chunk_types  # NEW: Apply chunk type filter
    )
    # ...
```

Step 5: 修改 index_manager.py - 標註 chunk types
------------------------------------------------
在 add_document() 的 Step 4 (Create chunks) 之後:

```python
# Step 4: Creating chunks...
chunks = self.layer2.text_splitter.create_documents(...)

# Add chunk metadata
for idx, chunk in enumerate(chunks):
    chunk.metadata['chunk_id'] = f"{paper_id}_chunk_{idx}"
    chunk.metadata['chunk_index'] = idx

# NEW: Classify chunks if classifier is available
if self.chunk_classifier:
    logger.info("Step 4.5: Classifying chunks...")
    classifications = self.chunk_classifier.classify_chunks_batch(chunks)
    
    for chunk, classification in zip(chunks, classifications):
        chunk.metadata['chunk_type'] = classification['chunk_type']
        chunk.metadata['classification_confidence'] = classification['confidence']
    
    logger.info(f"  ✓ Chunks classified")
```

Step 6: 傳遞 chunk_classifier 到 index_manager
----------------------------------------------
在 HierarchicalRAGSystem.__init__() 中:
```python
self.index_manager = IndexManager(
    layer1=self.layer1,
    layer2=self.layer2,
    abstract_extractor=self.abstract_extractor,
    backup_dir=os.path.join(vectorstore_path, "backups"),
    max_backups=5,
    chunk_classifier=self.chunk_classifier  # NEW
)
```

在 IndexManager.__init__() 中:
```python
def __init__(
    self,
    layer1: Layer1VectorStore,
    layer2: Layer2VectorStore,
    abstract_extractor: AbstractExtractor,
    backup_dir: str = "./vectorstore/backups",
    max_backups: int = 5,
    chunk_classifier=None  # NEW
):
    # ...
    self.chunk_classifier = chunk_classifier
    # ...
```

================================================================================
🧪 測試策略
================================================================================

測試重點:
1. Graph-first flow 正確執行
2. Confidence 高時提前終止
3. Confidence 低時正確下降到 Layer2
4. Chunk type 過濾正確應用
5. 索引時 chunk classification 正確執行

測試案例:
- Case 1: High-confidence query (應該在 Graph 階段終止)
- Case 2: Low-confidence query (應該下降到 Layer2)
- Case 3: 特定類型查詢 (dataset/method/metric) → 檢查過濾

================================================================================
⚠️  注意事項
================================================================================

1. **向後兼容**: 
   - graph_first.enabled = False 時系統應該回退到原流程
   - 沒有 graph 組件時不會報錯

2. **錯誤處理**:
   - Graph 查詢失敗時 fallback 到 Layer1
   - ChunkClassifier 失敗時使用預設類別 'other'

3. **性能考量**:
   - Graph 查詢應該快速 (<1s)
   - Chunk classification 使用 heuristic (no LLM)

4. **日誌清晰**:
   - 每個階段都有明確的分隔線和狀態日誌
   - 顯示決策理由和數據統計

================================================================================
✅ 準備就緒，可以開始實作
================================================================================
"""

if __name__ == '__main__':
    with open(__file__, 'r', encoding='utf-8') as f:
        content = f.read()
        docstring = content.split('"""')[1]
        print(docstring)
