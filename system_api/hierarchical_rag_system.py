"""
Hierarchical RAG System - Main Orchestrator

This is the core system that integrates all hierarchical RAG components:
- Layer 1: Paper-level retrieval (abstracts)
- Layer 2: Chunk-level retrieval (filtered by Layer 1 results)
- Confidence Evaluation: Early termination decision
- Context Expansion: Dynamic context around retrieved chunks
"""

import os
import logging
import time
from typing import List, Dict, Any, Optional, Tuple, Generator
from datetime import datetime

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.documents import Document

from system_api.abstract_extractor import AbstractExtractor
from system_api.layer1_vectorstore import Layer1VectorStore
from system_api.layer2_vectorstore import Layer2VectorStore
from system_api.confidence_evaluator import ConfidenceEvaluator
from system_api.context_expander import ContextExpander
from system_api.progress_embeddings import ProgressEmbeddings
from system_api.hybrid_retriever import HybridRetriever
from system_api.query_expander import QueryExpander
from system_api.adaptive_weights import AdaptiveWeightAdjuster

logger = logging.getLogger(__name__)


# Configuration
HIERARCHICAL_RAG_CONFIG = {
    'layer1': {
        'k_documents': 15,  # Maximum number of papers to retrieve
        'confidence_threshold': 0.7,  # Early termination threshold
        'similarity_threshold': None,  # Optional: Only retrieve papers with similarity >= threshold (0-1)
    },
    'layer2': {
        'k_documents': 10,  # Number of chunks to retrieve
        'confidence_threshold': 0.8,  # Early termination threshold
    },
    'expansion': {
        'enabled': True,
        'ranges': {
            'high': 1,    # ±1 chunks when confidence ≥ 0.85
            'medium': 2,  # ±2 chunks when 0.75 ≤ confidence < 0.85
            'low': 3      # ±3 chunks when confidence < 0.75
        }
    },
    'hybrid_search': {
        'enabled': True,  # 啟用混合檢索
        'semantic_weight': 0.5,  # 語義搜尋權重 (0-1)
        'keyword_weight': 0.5,   # 關鍵詞搜尋權重 (0-1)
        'use_jieba': True,       # 使用 jieba 中文分詞
        'enable_smart_weighting': True,  # 啟用智能詞彙權重
    },
    'query_expansion': {
        'enabled': True,  # 啟用查詢擴展
        'min_word_count': 5,  # 少於此詞數才擴展
    },
    'adaptive_weights': {
        'enabled': True,  # 啟用自適應權重調整
        'default_semantic_weight': 0.5,
        'default_keyword_weight': 0.5,
    },
    'performance': {
        'use_cache': True,
        'cache_size': 10,
    }
}


class HierarchicalRAGSystem:
    """
    Main Hierarchical RAG System
    
    Query flow:
    1. Layer 1: Retrieve relevant papers by searching abstracts
    2. Evaluate: Check if Layer 1 results are sufficient
    3. Layer 2: If needed, retrieve specific chunks from selected papers
    4. Evaluate: Check if Layer 2 results are sufficient
    5. Expand: Dynamically expand context based on confidence
    6. Generate: Use LLM to generate final answer
    """
    
    def __init__(
        self,
        pdf_directory: str = "./data",
        model_name: str = "gemma3:12b",
        embedding_model: str = "embeddinggemma:latest",
        vectorstore_path: str = "./vectorstore",
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Hierarchical RAG System
        
        Args:
            pdf_directory: Directory containing PDF files
            model_name: LLM model name
            embedding_model: Embedding model name
            vectorstore_path: Base path for vector stores
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            config: Optional configuration override
        """
        self.pdf_directory = pdf_directory
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.vectorstore_path = vectorstore_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Merge config
        self.config = HIERARCHICAL_RAG_CONFIG.copy()
        if config:
            self.config.update(config)
        
        # Initialize components
        logger.info("Initializing Hierarchical RAG System...")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Embeddings: {embedding_model}")
        logger.info(f"  PDF Directory: {pdf_directory}")
        logger.info(f"  Vector Store: {vectorstore_path}")
        
        # LLM and Embeddings
        # Note: OllamaLLM uses temperature parameter during initialization, not per-call
        self.llm = OllamaLLM(model=model_name)
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        
        # Separate LLM for confidence evaluation (use same model, deterministic mode preferred)
        # Note: If temperature is not supported, rely on model's default behavior
        self.evaluation_llm = OllamaLLM(model=model_name)
        
        # Core components
        self.abstract_extractor = AbstractExtractor(self.llm)
        
        self.layer1 = Layer1VectorStore(
            embeddings=self.embeddings,
            vectorstore_path=os.path.join(vectorstore_path, "layer1")
        )
        
        self.layer2 = Layer2VectorStore(
            embeddings=self.embeddings,
            vectorstore_path=os.path.join(vectorstore_path, "layer2"),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            cache_size=self.config['performance']['cache_size']
        )
        
        self.confidence_evaluator = ConfidenceEvaluator(self.evaluation_llm)
        self.context_expander = ContextExpander(self.layer2)
        
        # Query expansion and adaptive weights
        self.query_expander: Optional[QueryExpander] = None
        self.weight_adjuster: Optional[AdaptiveWeightAdjuster] = None
        
        # Initialize query expander if enabled
        if self.config.get('query_expansion', {}).get('enabled', False):
            self.query_expander = QueryExpander(
                llm=self.llm,
                min_word_count=self.config['query_expansion'].get('min_word_count', 5)
            )
            logger.info("✓ Query Expander initialized")
        
        # Initialize adaptive weight adjuster if enabled
        if self.config.get('adaptive_weights', {}).get('enabled', False):
            aw_config = self.config['adaptive_weights']
            self.weight_adjuster = AdaptiveWeightAdjuster(
                default_semantic_weight=aw_config.get('default_semantic_weight', 0.5),
                default_keyword_weight=aw_config.get('default_keyword_weight', 0.5)
            )
            logger.info("✓ Adaptive Weight Adjuster initialized")
        
        # Hybrid retriever (initialized after loading indices)
        self.hybrid_retriever: Optional[HybridRetriever] = None
        
        # Try to load existing indices
        self._load_indices()
        
        # Initialize hybrid retriever if enabled
        if self.config.get('hybrid_search', {}).get('enabled', False):
            self._initialize_hybrid_retriever()
        
        logger.info("Hierarchical RAG System initialized successfully")
    
    def _initialize_hybrid_retriever(self):
        """初始化混合檢索器"""
        try:
            hybrid_config = self.config.get('hybrid_search', {})
            use_jieba = hybrid_config.get('use_jieba', True)
            enable_smart_weighting = hybrid_config.get('enable_smart_weighting', True)
            
            use_llm_analyzer = hybrid_config.get('use_llm_analyzer', True)
            
            logger.info("初始化混合檢索器 (Hybrid Retriever)...")
            logger.info(f"  語義權重: {hybrid_config.get('semantic_weight', 0.5):.2f}")
            logger.info(f"  關鍵詞權重: {hybrid_config.get('keyword_weight', 0.5):.2f}")
            logger.info(f"  中文分詞: {'啟用' if use_jieba else '停用'}")
            logger.info(f"  智能權重（字典）: {'啟用' if enable_smart_weighting else '停用'}")
            logger.info(f"  LLM 詞彙分析: {'啟用' if use_llm_analyzer else '停用'}")
            
            self.hybrid_retriever = HybridRetriever(
                layer1_vectorstore=self.layer1,
                use_jieba=use_jieba,
                build_index_on_init=True,  # 自動建立 BM25 索引
                enable_smart_weighting=enable_smart_weighting,  # 字典式智能權重
                llm=self.llm,  # 傳遞 LLM 給分析器
                use_llm_analyzer=use_llm_analyzer  # 啟用 LLM 自動分析
            )
            
            logger.info("✓ 混合檢索器初始化成功")
            
        except Exception as e:
            logger.error(f"混合檢索器初始化失敗: {e}", exc_info=True)
            logger.warning("將回退到純語義搜尋")
            self.hybrid_retriever = None
    
    def _load_indices(self):
        """Load existing vector store indices"""
        logger.info("Loading existing indices...")
        
        layer1_loaded = self.layer1.load()
        layer2_loaded = self.layer2.load()
        
        if layer1_loaded and layer2_loaded:
            logger.info("✓ Both indices loaded successfully")
            stats1 = self.layer1.get_stats()
            stats2 = self.layer2.get_stats()
            logger.info(f"  Layer 1: {stats1['paper_count']} papers")
            logger.info(f"  Layer 2: {stats2['chunk_count']} chunks from {stats2['paper_count']} papers")
        elif layer1_loaded:
            logger.warning("⚠️  Only Layer 1 loaded, Layer 2 needs building")
        elif layer2_loaded:
            logger.warning("⚠️  Only Layer 2 loaded, Layer 1 needs building")
        else:
            logger.info("ℹ️  No existing indices found, need to build from scratch")
    
    def build_indices(self, pdf_paths: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Build hierarchical indices from PDF files
        
        Args:
            pdf_paths: Optional list of specific PDF paths. If None, uses all PDFs in pdf_directory
            
        Returns:
            Dictionary with build statistics
        """
        start_time = time.time()
        logger.info("="*80)
        logger.info("Building Hierarchical Indices")
        logger.info("="*80)
        
        # Get PDF paths
        if pdf_paths is None:
            pdf_paths = self._get_all_pdfs()
        
        if not pdf_paths:
            logger.error("No PDF files found")
            return {'status': 'error', 'message': 'No PDF files found'}
        
        logger.info(f"Processing {len(pdf_paths)} PDF files...")
        
        # Extract and process documents
        abstracts = []
        all_chunks = []
        
        from PyPDF2 import PdfReader
        
        for i, pdf_path in enumerate(pdf_paths, 1):
            try:
                logger.info(f"[{i}/{len(pdf_paths)}] Processing: {os.path.basename(pdf_path)}")
                
                # Read PDF
                reader = PdfReader(pdf_path)
                pdf_text = ""
                for page in reader.pages:
                    pdf_text += page.extract_text() + "\n"
                
                # Generate paper_id from filename
                paper_id = os.path.splitext(os.path.basename(pdf_path))[0]
                
                # Extract abstract (for Layer 1)
                abstract_result = self.abstract_extractor.extract(
                    pdf_path=pdf_path,
                    pdf_text=pdf_text,
                    paper_id=paper_id
                )
                
                # Add simple metadata (use filename as title)
                if abstract_result:
                    abstract_result['title'] = paper_id
                    abstract_result['authors'] = []
                    abstract_result['year'] = None
                    abstracts.append(abstract_result)
                    logger.info(f"  ✓ Abstract: {abstract_result['source']} (confidence: {abstract_result['confidence']:.2f})")
                
                # Split into chunks (for Layer 2)
                chunks = self.layer2.text_splitter.create_documents(
                    texts=[pdf_text],
                    metadatas=[{
                        'paper_id': paper_id,
                        'pdf_path': pdf_path,
                        'source': 'hierarchical_build'
                    }]
                )
                
                # Add chunk metadata
                for idx, chunk in enumerate(chunks):
                    chunk.metadata['chunk_id'] = f"{paper_id}_chunk_{idx}"
                    chunk.metadata['chunk_index'] = idx
                
                all_chunks.extend(chunks)
                logger.info(f"  ✓ Chunks: {len(chunks)}")
                
            except Exception as e:
                logger.error(f"  ✗ Failed to process {pdf_path}: {e}")
                continue
        
        # Build Layer 1 index
        logger.info("\n" + "="*80)
        logger.info("Building Layer 1 Index (Abstracts)")
        logger.info("="*80)
        
        if abstracts:
            success = self.layer1.build_index(abstracts)
            if success:
                logger.info("✓ Layer 1 index built successfully")
            else:
                logger.error("✗ Layer 1 index build failed")
        else:
            logger.warning("⚠️  No abstracts extracted, skipping Layer 1")
        
        # Build Layer 2 index
        logger.info("\n" + "="*80)
        logger.info("Building Layer 2 Index (Chunks)")
        logger.info("="*80)
        
        if all_chunks:
            success = self.layer2.build_index(all_chunks)
            if success:
                logger.info("✓ Layer 2 index built successfully")
            else:
                logger.error("✗ Layer 2 index build failed")
        else:
            logger.warning("⚠️  No chunks created, skipping Layer 2")
        
        # Summary
        duration = time.time() - start_time
        stats = self.get_stats()
        
        logger.info("\n" + "="*80)
        logger.info("Build Complete")
        logger.info("="*80)
        logger.info(f"Duration: {duration:.1f}s")
        logger.info(f"Papers: {stats['layer1']['paper_count']}")
        logger.info(f"Chunks: {stats['layer2']['chunk_count']}")
        logger.info("="*80)
        
        return {
            'status': 'success',
            'duration': duration,
            'papers_processed': len(pdf_paths),
            'abstracts_extracted': len(abstracts),
            'chunks_created': len(all_chunks),
            'stats': stats
        }
    
    def _get_all_pdfs(self) -> List[str]:
        """Get all PDF files in pdf_directory"""
        pdf_paths = []
        
        if not os.path.exists(self.pdf_directory):
            logger.warning(f"PDF directory does not exist: {self.pdf_directory}")
            return pdf_paths
        
        for filename in os.listdir(self.pdf_directory):
            if filename.lower().endswith('.pdf'):
                pdf_paths.append(os.path.join(self.pdf_directory, filename))
        
        return sorted(pdf_paths)
    
    def query(self, query: str) -> str:
        """
        Query the hierarchical RAG system (blocking version)
        
        Args:
            query: User query string
            
        Returns:
            Generated answer string
        """
        result = self._hierarchical_retrieval(query)
        
        if result['status'] == 'error':
            return f"Error: {result['message']}"
        
        if result['status'] == 'no_results':
            return "抱歉，沒有找到與您的問題相關的論文。請嘗試使用不同的關鍵字或降低相似度閾值。"
        
        # Generate answer using LLM
        return self._generate_answer(query, result)
    
    def query_stream(self, query: str) -> Generator[str, None, None]:
        """
        Query the hierarchical RAG system (streaming version)
        
        Args:
            query: User query string
            
        Yields:
            Answer chunks as they are generated
        """
        # Perform hierarchical retrieval
        result = self._hierarchical_retrieval(query)
        
        if result['status'] == 'error':
            yield f"Error: {result['message']}"
            return
        
        if result['status'] == 'no_results':
            yield "抱歉，沒有找到與您的問題相關的論文。請嘗試使用不同的關鍵字或降低相似度閾值。"
            return
        
        # Generate answer with streaming
        yield from self._generate_answer_stream(query, result)
    
    def _hierarchical_retrieval(self, query: str) -> Dict[str, Any]:
        """
        Core hierarchical retrieval logic
        
        Args:
            query: User query
            
        Returns:
            Dictionary with retrieval results and metadata
        """
        start_time = time.time()
        logger.info(f"Starting hierarchical retrieval for: '{query[:100]}'")
        
        result = {
            'status': 'success',
            'query': query,
            'original_query': query,  # Save original query
            'layers_used': [],
            'timings': {},
            'evaluations': {}
        }
        
        try:
            # ============================================================
            # STEP 0: Query Enhancement (Expansion + Weight Adjustment)
            # ============================================================
            expansion_used = False
            weight_adjustment_used = False
            weight_reason = ""
            
            # Query Expansion
            if self.query_expander and self.config.get('query_expansion', {}).get('enabled', False):
                logger.info("="*60)
                logger.info("查詢擴展 (Query Expansion)")
                logger.info("="*60)
                
                expansion_start = time.time()
                expanded_query = self.query_expander.expand(query)
                result['timings']['query_expansion'] = time.time() - expansion_start
                
                if expanded_query != query:
                    logger.info(f"✓ 查詢已擴展:")
                    logger.info(f"  原始: {query}")
                    logger.info(f"  擴展: {expanded_query}")
                    query = expanded_query
                    expansion_used = True
                    result['expanded_query'] = expanded_query
                else:
                    logger.info("ℹ️  查詢無需擴展（詞數充足）")
            
            # Adaptive Weight Adjustment
            semantic_weight = self.config['hybrid_search'].get('semantic_weight', 0.5)
            keyword_weight = self.config['hybrid_search'].get('keyword_weight', 0.5)
            
            if self.weight_adjuster and self.config.get('adaptive_weights', {}).get('enabled', False):
                logger.info("="*60)
                logger.info("自適應權重調整 (Adaptive Weights)")
                logger.info("="*60)
                
                weight_start = time.time()
                semantic_weight, keyword_weight, weight_reason = self.weight_adjuster.adjust_weights(query)
                result['timings']['weight_adjustment'] = time.time() - weight_start
                
                logger.info(f"✓ 權重已調整:")
                logger.info(f"  語義權重: {semantic_weight:.2f}")
                logger.info(f"  關鍵詞權重: {keyword_weight:.2f}")
                logger.info(f"  調整原因: {weight_reason}")
                
                weight_adjustment_used = True
                result['adjusted_weights'] = {
                    'semantic': semantic_weight,
                    'keyword': keyword_weight,
                    'reason': weight_reason
                }
            
            # Store enhancement metadata
            result['enhancements'] = {
                'expansion_used': expansion_used,
                'weight_adjustment_used': weight_adjustment_used
            }
            
            # ============================================================
            # LAYER 1: Paper-level retrieval
            # ============================================================
            logger.info("="*60)
            logger.info("LAYER 1: Paper-level Retrieval")
            logger.info("="*60)
            
            layer1_start = time.time()
            
            if not self.layer1.is_initialized:
                raise ValueError("Layer 1 index not initialized")
            
            k1 = self.config['layer1']['k_documents']
            similarity_threshold = self.config['layer1'].get('similarity_threshold')
            
            # 判斷使用混合檢索還是純語義搜尋
            # 先獲取所有結果（不使用 threshold 過濾），用於顯示
            if self.hybrid_retriever and self.config.get('hybrid_search', {}).get('enabled', False):
                # 使用混合檢索 (語義 + 關鍵詞) with adjusted weights
                logger.info("使用混合檢索 (Hybrid Search: 語義 + 關鍵詞)")
                logger.info(f"  當前權重: 語義={semantic_weight:.2f}, 關鍵詞={keyword_weight:.2f}")
                
                # 先不使用 threshold，獲取所有結果
                layer1_all_results = self.hybrid_retriever.hybrid_search(
                    query=query,
                    k=k1,
                    semantic_weight=semantic_weight,
                    keyword_weight=keyword_weight,
                    score_threshold=None,  # 不過濾，顯示所有結果
                    return_scores_breakdown=False
                )
                
                # 再應用 threshold 過濾（用於後續處理）
                if similarity_threshold is not None:
                    layer1_results_with_scores = [
                        (doc, score) for doc, score in layer1_all_results 
                        if score >= similarity_threshold
                    ]
                else:
                    layer1_results_with_scores = layer1_all_results
            else:
                # 使用純語義搜尋
                logger.info("使用純語義搜尋 (Semantic Search)")
                
                # 先不使用 threshold
                layer1_all_results = self.layer1.search_with_scores(
                    query=query, 
                    k=k1,
                    score_threshold=None
                )
                
                # 再應用 threshold 過濾
                if similarity_threshold is not None:
                    layer1_results_with_scores = [
                        (doc, score) for doc, score in layer1_all_results 
                        if score >= similarity_threshold
                    ]
                else:
                    layer1_results_with_scores = layer1_all_results
            
            # 提取文檔（用於後續處理）
            layer1_docs = [doc for doc, score in layer1_results_with_scores]
            
            result['timings']['layer1_retrieval'] = time.time() - layer1_start
            result['layers_used'].append('layer1')
            result['layer1_docs'] = layer1_docs  # Save Layer 1 docs for reference display
            result['layer1_scores'] = [score for doc, score in layer1_results_with_scores]  # Save scores for display
            
            # 總是顯示前 5 名相似度最高的論文（從所有結果中）
            logger.info("\n📊 Layer 1 前 5 名相似度最高的論文:")
            logger.info("─" * 80)
            if layer1_all_results:
                for i, (doc, score) in enumerate(layer1_all_results[:5], 1):
                    title = doc.metadata.get('title', 'Unknown')
                    # 標記是否通過 threshold
                    passed = "✓" if similarity_threshold is None or score >= similarity_threshold else "✗"
                    logger.info(f"  {i}. [{score:.4f}] {passed} {title}")
                
                if similarity_threshold is not None:
                    passed_count = len(layer1_results_with_scores)
                    total_count = len(layer1_all_results)
                    logger.info(f"\n  通過閾值 ({similarity_threshold:.2f}): {passed_count}/{total_count} 篇")
            else:
                logger.warning("  ⚠️  沒有找到任何論文")
            logger.info("─" * 80)
            
            logger.info(f"\nRetrieved {len(layer1_docs)} papers"
                       f"{f' (threshold: {similarity_threshold:.2f})' if similarity_threshold else ''}")
            
            if not layer1_docs:
                logger.warning("No relevant papers found in Layer 1")
                result['status'] = 'no_results'
                result['message'] = 'No relevant papers found matching your query.'
                result['timings']['total'] = time.time() - start_time
                return result
            
            # Evaluate Layer 1 results
            eval1_start = time.time()
            threshold1 = self.config['layer1']['confidence_threshold']
            
            evaluation1 = self.confidence_evaluator.evaluate(
                query=query,
                retrieved_docs=layer1_docs,
                layer=1,
                threshold=threshold1
            )
            
            result['timings']['layer1_evaluation'] = time.time() - eval1_start
            result['evaluations']['layer1'] = evaluation1
            
            logger.info(f"Layer 1 Evaluation:")
            logger.info(f"  Confidence: {evaluation1['confidence']:.2f} (threshold: {threshold1})")
            logger.info(f"  Should continue: {evaluation1['should_continue']}")
            logger.info(f"  Reasoning: {evaluation1['reasoning'][:100]}...")
            
            # Early termination check
            if not evaluation1['should_continue']:
                logger.info("✓ Early termination: Layer 1 results sufficient")
                result['final_docs'] = layer1_docs
                result['final_confidence'] = evaluation1['confidence']
                result['terminated_at'] = 'layer1'
                result['timings']['total'] = time.time() - start_time
                return result
            
            # ============================================================
            # LAYER 2: Chunk-level retrieval
            # ============================================================
            logger.info("\n" + "="*60)
            logger.info("LAYER 2: Chunk-level Retrieval")
            logger.info("="*60)
            
            layer2_start = time.time()
            
            if not self.layer2.is_initialized:
                raise ValueError("Layer 2 index not initialized")
            
            # Extract paper IDs from Layer 1 results
            paper_ids = [doc.metadata['paper_id'] for doc in layer1_docs]
            logger.info(f"Filtering by {len(paper_ids)} papers from Layer 1")
            
            k2 = self.config['layer2']['k_documents']
            layer2_results_with_scores = self.layer2.search_with_scores(
                query=query,
                k=k2,
                filter_paper_ids=paper_ids
            )
            
            # 提取文檔
            layer2_docs = [doc for doc, score in layer2_results_with_scores]
            
            result['timings']['layer2_retrieval'] = time.time() - layer2_start
            result['layers_used'].append('layer2')
            
            logger.info(f"Retrieved {len(layer2_docs)} chunks")
            
            # 顯示前 5 名相似度最高的文檔片段
            if layer2_results_with_scores:
                logger.info("\n📊 前 5 名相似度最高的文檔片段:")
                logger.info("─" * 80)
                for i, (doc, score) in enumerate(layer2_results_with_scores[:5], 1):
                    chunk_id = doc.metadata.get('chunk_id', 'Unknown')
                    paper_title = doc.metadata.get('title', 'Unknown')
                    content_preview = doc.page_content[:60].replace('\n', ' ')
                    logger.info(f"  {i}. [{score:.4f}] {paper_title}")
                    logger.info(f"      Chunk: {chunk_id} | Preview: {content_preview}...")
                logger.info("─" * 80)
            
            if not layer2_docs:
                logger.warning("No chunks found in Layer 2, using Layer 1 results")
                result['final_docs'] = layer1_docs
                result['final_confidence'] = evaluation1['confidence']
                result['terminated_at'] = 'layer1_fallback'
                result['timings']['total'] = time.time() - start_time
                return result
            
            # Evaluate Layer 2 results
            eval2_start = time.time()
            threshold2 = self.config['layer2']['confidence_threshold']
            
            evaluation2 = self.confidence_evaluator.evaluate(
                query=query,
                retrieved_docs=layer2_docs,
                layer=2,
                threshold=threshold2
            )
            
            result['timings']['layer2_evaluation'] = time.time() - eval2_start
            result['evaluations']['layer2'] = evaluation2
            
            logger.info(f"Layer 2 Evaluation:")
            logger.info(f"  Confidence: {evaluation2['confidence']:.2f} (threshold: {threshold2})")
            logger.info(f"  Reasoning: {evaluation2['reasoning'][:100]}...")
            
            # ============================================================
            # CONTEXT EXPANSION
            # ============================================================
            if self.config['expansion']['enabled']:
                logger.info("\n" + "="*60)
                logger.info("CONTEXT EXPANSION")
                logger.info("="*60)
                
                expand_start = time.time()
                
                expanded_contexts = self.context_expander.expand(
                    chunks=layer2_docs,
                    confidence=evaluation2['confidence']
                )
                
                result['timings']['expansion'] = time.time() - expand_start
                result['expanded_contexts'] = expanded_contexts
                
                stats = self.context_expander.get_expansion_stats(evaluation2['confidence'])
                logger.info(f"Expanded {len(expanded_contexts)} contexts")
                logger.info(f"  Strategy: {stats['strategy']}")
                logger.info(f"  Range: ±{stats['expansion_range']} chunks")
            else:
                result['expanded_contexts'] = [doc.page_content for doc in layer2_docs]
            
            # Final results
            result['final_docs'] = layer2_docs
            result['final_confidence'] = evaluation2['confidence']
            result['terminated_at'] = 'layer2'
            result['timings']['total'] = time.time() - start_time
            
            logger.info("\n" + "="*60)
            logger.info(f"Retrieval Complete: {result['timings']['total']:.2f}s")
            logger.info("="*60)
            
            return result
            
        except Exception as e:
            logger.error(f"Hierarchical retrieval failed: {e}", exc_info=True)
            return {
                'status': 'error',
                'message': str(e),
                'query': query
            }
    
    def _generate_answer(self, query: str, retrieval_result: Dict[str, Any]) -> str:
        """
        Generate answer using LLM (blocking)
        
        Args:
            query: User query
            retrieval_result: Results from hierarchical retrieval
            
        Returns:
            Generated answer string with paper references
        """
        # Detect if query is in Chinese
        def is_chinese(text: str) -> bool:
            """Check if text contains significant Chinese characters"""
            chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
            return chinese_chars > len(text) * 0.3
        
        is_chinese_query = is_chinese(query)
        
        # Extract paper information from Layer 1 results with similarity scores
        referenced_papers = []
        if 'layers_used' in retrieval_result and 'layer1' in retrieval_result.get('layers_used', []):
            # Get paper metadata and scores from retrieval result
            layer1_docs = retrieval_result.get('layer1_docs', [])
            layer1_scores = retrieval_result.get('layer1_scores', [])
            
            for i, doc in enumerate(layer1_docs[:10]):  # Limit to top 10 papers
                metadata = doc.metadata
                title = metadata.get('title', 'Unknown Title')
                
                # Add similarity score if available
                if i < len(layer1_scores):
                    score = layer1_scores[i]
                    referenced_papers.append(f"- {title} (相似度: {score:.3f})")
                else:
                    referenced_papers.append(f"- {title}")
        
        # Format context
        if 'expanded_contexts' in retrieval_result:
            contexts = retrieval_result['expanded_contexts']
        else:
            contexts = [doc.page_content for doc in retrieval_result['final_docs']]
        
        context_text = "\n\n".join([f"[Context {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)])
        
        # Create prompt based on language
        if is_chinese_query:
            prompt = f"""你是一個學術研究助手。請根據以下研究論文的內容回答問題。

【重要】請務必使用「繁體中文（Traditional Chinese）」回答，不要使用簡體中文。

問題：{query}

參考文獻內容：
{context_text}

請根據以上內容提供完整的答案。如果內容不足以回答問題，請明確說明。
記住：回答必須使用繁體中文，例如「應用」而非「应用」，「資料」而非「数据」。

回答："""
        else:
            prompt = f"""Based on the following research paper contexts, please answer the question.

Question: {query}

Contexts:
{context_text}

Please provide a comprehensive answer based on the contexts above. If the contexts don't contain enough information, acknowledge that.

Answer:"""
        
        # Generate
        try:
            response = self.llm.invoke(prompt)
            
            # Prepend referenced papers list
            if referenced_papers:
                if is_chinese_query:
                    paper_list = "📚 參考論文：\n" + "\n".join(referenced_papers) + "\n\n" + "="*60 + "\n\n"
                else:
                    paper_list = "📚 Referenced Papers:\n" + "\n".join(referenced_papers) + "\n" + "="*60 + "\n"
                response = paper_list + response
            
            return response
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return f"Error generating answer: {e}"
    
    def _generate_answer_stream(self, query: str, retrieval_result: Dict[str, Any]) -> Generator[str, None, None]:
        """
        Generate answer using LLM (streaming)
        
        Args:
            query: User query
            retrieval_result: Results from hierarchical retrieval
            
        Yields:
            Answer chunks with paper references
        """
        # Detect if query is in Chinese
        def is_chinese(text: str) -> bool:
            """Check if text contains significant Chinese characters"""
            chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
            return chinese_chars > len(text) * 0.3
        
        is_chinese_query = is_chinese(query)
        
        # Extract paper information from Layer 1 results with similarity scores
        referenced_papers = []
        if 'layers_used' in retrieval_result and 'layer1' in retrieval_result.get('layers_used', []):
            # Get paper metadata and scores from retrieval result
            layer1_docs = retrieval_result.get('layer1_docs', [])
            layer1_scores = retrieval_result.get('layer1_scores', [])
            
            for i, doc in enumerate(layer1_docs[:10]):  # Limit to top 10 papers
                metadata = doc.metadata
                title = metadata.get('title', 'Unknown Title')
                
                # Add similarity score if available
                if i < len(layer1_scores):
                    score = layer1_scores[i]
                    referenced_papers.append(f"- {title} (相似度: {score:.3f})")
                else:
                    referenced_papers.append(f"- {title}")
        
        # Yield referenced papers first
        if referenced_papers:
            if is_chinese_query:
                yield "📚 參考論文：\n"
            else:
                yield "📚 Referenced Papers:\n"
            
            for paper in referenced_papers:
                yield paper + "\n"
            
            yield "\n" + "="*60 + "\n\n"
        
        # Format context
        if 'expanded_contexts' in retrieval_result:
            contexts = retrieval_result['expanded_contexts']
        else:
            contexts = [doc.page_content for doc in retrieval_result['final_docs']]
        
        context_text = "\n\n".join([f"[Context {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)])
        
        # Create prompt based on language
        if is_chinese_query:
            prompt = f"""你是一個學術研究助手。請根據以下研究論文的內容回答問題。

【重要】請務必使用「繁體中文（Traditional Chinese）」回答，不要使用簡體中文。

問題：{query}

參考文獻內容：
{context_text}

請根據以上內容提供完整的答案。如果內容不足以回答問題，請明確說明。
記住：回答必須使用繁體中文，例如「應用」而非「应用」，「資料」而非「数据」。

回答："""
        else:
            prompt = f"""Based on the following research paper contexts, please answer the question.

Question: {query}

Contexts:
{context_text}

Please provide a comprehensive answer based on the contexts above. If the contexts don't contain enough information, acknowledge that.

Answer:"""
        
        # Generate with streaming
        try:
            for chunk in self.llm.stream(prompt):
                yield chunk
        except Exception as e:
            logger.error(f"Streaming answer generation failed: {e}")
            yield f"Error generating answer: {e}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            'layer1': self.layer1.get_stats(),
            'layer2': self.layer2.get_stats(),
            'config': self.config,
            'initialized': self.layer1.is_initialized and self.layer2.is_initialized
        }
    
    def is_ready(self) -> bool:
        """Check if system is ready for queries"""
        return self.layer1.is_initialized and self.layer2.is_initialized
    
    def check_and_update_indices(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """
        智能檢查索引狀態並更新
        
        檢查項目：
        1. 索引是否存在
        2. 索引是否損壞
        3. 是否有新增的 PDF
        4. 索引的 PDF 數量是否匹配
        
        Args:
            force_rebuild: 強制重建索引
            
        Returns:
            操作結果字典
        """
        logger.info("="*80)
        logger.info("Checking Index Status")
        logger.info("="*80)
        
        result = {
            'status': 'up_to_date',
            'action_taken': None,
            'details': {}
        }
        
        # 檢查 1: 索引是否就緒
        if not self.is_ready() or force_rebuild:
            if force_rebuild:
                logger.info("Force rebuild requested")
                result['action_taken'] = 'rebuild'
            else:
                logger.warning("Indices not ready")
                result['action_taken'] = 'build'
            
            build_result = self.build_indices()
            result['status'] = build_result.get('status', 'error')
            result['details'] = build_result
            return result
        
        # 檢查 2: 獲取當前狀態
        try:
            stats1 = self.layer1.get_stats()
            stats2 = self.layer2.get_stats()
            current_pdfs = set(self._get_all_pdfs())
            
            indexed_count = stats1.get('paper_count', 0)
            current_count = len(current_pdfs)
            
            logger.info(f"Current PDF files: {current_count}")
            logger.info(f"Indexed papers: {indexed_count}")
            
            # 檢查 3: 數量不匹配 = 可能有新增或刪除
            if current_count != indexed_count:
                logger.warning(f"PDF count mismatch! Current: {current_count}, Indexed: {indexed_count}")
                
                if current_count > indexed_count:
                    logger.info(f"Detected {current_count - indexed_count} new PDF(s)")
                    result['action_taken'] = 'incremental_update'
                else:
                    logger.warning(f"Some PDFs were removed, rebuilding recommended")
                    result['action_taken'] = 'rebuild'
                
                # 重建索引
                build_result = self.build_indices()
                result['status'] = build_result.get('status', 'error')
                result['details'] = build_result
                return result
            
            # 檢查 4: 驗證索引完整性（簡單檢查）
            if not stats1.get('is_initialized') or not stats2.get('is_initialized'):
                logger.error("Indices appear corrupted")
                result['status'] = 'corrupted'
                result['action_taken'] = 'rebuild'
                
                build_result = self.build_indices()
                result['status'] = build_result.get('status', 'error')
                result['details'] = build_result
                return result
            
            # 全部檢查通過
            logger.info("✓ Indices are up-to-date and healthy")
            result['details'] = {
                'layer1': stats1,
                'layer2': stats2,
                'pdf_count': current_count
            }
            
        except Exception as e:
            logger.error(f"Error checking indices: {e}")
            result['status'] = 'error'
            result['details']['error'] = str(e)
            
            # 嘗試重建
            logger.info("Attempting to rebuild indices...")
            result['action_taken'] = 'rebuild'
            build_result = self.build_indices()
            result['status'] = build_result.get('status', 'error')
            result['details']['rebuild'] = build_result
        
        return result
