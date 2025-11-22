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
from system_api.layer2_trigger import Layer2TriggerDecision
from system_api.text_preprocessor import TextPreprocessor
from system_api.index_manager import IndexManager
from system_api.query_router import QueryRouter

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
    },
    'layer2_reranking': {
        'enabled': False,  # 是否啟用 Cross-Encoder Re-ranking（實驗性功能）
        'model': 'qllama/bce-reranker-base_v1:latest',
        'ollama_base_url': 'http://localhost:11434',
        'max_length': 512,
        'timeout': 30,
        'max_candidates': 300,  # 最大候選區塊數（避免 re-ranking 太慢）
        'cache_limit_mb': 100,  # JSONL 快取大小限制（MB）
    },
    'chunking': {
        'mode': 'naive',  # 'naive' or 'summarization' (default: naive for backward compatibility)
        'summarization': {
            'enabled': False,  # 是否啟用摘要式分塊（實驗性功能）
            'model': 'llama3:8b',  # Ollama model for summarization
            'section_parser': 'pymupdf_regex',  # 'pymupdf_regex' or 'grobid'
            'min_sections': 3,  # Minimum sections to consider PDF structured
            'map_reduce_threshold': 1500,  # Char count threshold for Map-Reduce
            'target_summary_length': 300,  # Target summary length in chars
            'store_original': False,  # Whether to store original full text
            'ollama_base_url': 'http://localhost:11434',
        }
    },
    'query_routing': {
        'enabled': True,  # 啟用智能查詢路由（自動判斷使用 Graph 或 RAG）
        'default_route': 'rag',  # 預設路由（當無法判斷時）：'graph' 或 'rag'
        'confidence_threshold': 0.7,  # 路由決策的最低信心度
    },
    'graph_first': {
        'enabled': True,  # 啟用 Graph-first 檢索模式
        'confidence_threshold': 0.7,  # Graph 結果的信心閾值（>= 此值且無需詳細資訊則不下降）
        'min_graph_hits': 1,  # 最少需要找到的 paper 數量（少於此數則 fallback 到 Layer1）
        'skip_layer1': True,  # Graph 不足時是否跳過 Layer1 直接到 Layer2
        'max_papers_for_integration': 5,  # LLM 整合時最多使用幾篇論文（避免 context 過大）
        'graph_query_top_k': 15,  # Graph 查詢返回最多幾篇論文
    },
    'chunk_classification': {
        'enabled': True,  # 在索引時對 chunks 進行語義分類
        'use_llm_fallback': False,  # 低信心時是否使用 LLM（False = 純 heuristic，更快）
        'categories': ['background', 'method', 'dataset', 'metric', 'domain', 'results'],  # 分類類別
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
        # Merge config first to allow overrides
        self.config = HIERARCHICAL_RAG_CONFIG.copy()
        if config:
            self.config.update(config)
        
        # Apply config overrides for initialization parameters
        self.pdf_directory = config.get('pdf_directory', pdf_directory) if config else pdf_directory
        self.model_name = config.get('model_name', model_name) if config else model_name
        self.embedding_model = config.get('embeddings_model', config.get('embedding_model', embedding_model)) if config else embedding_model
        self.vectorstore_path = config.get('vectorstore_path', vectorstore_path) if config else vectorstore_path
        self.chunk_size = config.get('chunk_size', chunk_size) if config else chunk_size
        self.chunk_overlap = config.get('chunk_overlap', chunk_overlap) if config else chunk_overlap
        
        # Initialize components
        logger.info("Initializing Hierarchical RAG System...")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  Embeddings: {self.embedding_model}")
        logger.info(f"  PDF Directory: {self.pdf_directory}")
        logger.info(f"  Vector Store: {self.vectorstore_path}")
        
        # LLM and Embeddings
        # Note: OllamaLLM uses temperature parameter during initialization, not per-call
        # Set num_ctx to 4096 for faster generation (13K chars ≈ 3.5K tokens should fit)
        self.llm = OllamaLLM(model=self.model_name, num_ctx=4096)
        self.embeddings = OllamaEmbeddings(model=self.embedding_model)
        
        # Separate LLM for confidence evaluation
        # Use a faster, smaller model for quick evaluation tasks
        # Default to llama3.2:latest (3B model) for speed
        evaluation_model = self.config.get('evaluation_model', 'llama3.2:latest')
        logger.info(f"  Evaluation Model: {evaluation_model}")
        self.evaluation_llm = OllamaLLM(model=evaluation_model, num_ctx=4096)
        
        # Core components
        self.abstract_extractor = AbstractExtractor(self.llm)
        self.text_preprocessor = TextPreprocessor()
        
        self.layer1 = Layer1VectorStore(
            embeddings=self.embeddings,
            vectorstore_path=os.path.join(self.vectorstore_path, "layer1")
        )
        
        # Initialize Layer 2 with optional re-ranking and chunking config
        reranking_config = self.config.get('layer2_reranking', {})
        use_reranker = reranking_config.get('enabled', False)
        
        # Prepare chunking config
        chunking_settings = self.config.get('chunking', {})
        chunking_mode = chunking_settings.get('mode', 'naive')
        chunking_config = None
        if chunking_mode == 'summarization':
            chunking_config = {
                'mode': 'summarization',
                **chunking_settings.get('summarization', {})
            }
        else:
            chunking_config = {'mode': 'naive'}
        
        self.layer2 = Layer2VectorStore(
            embeddings=self.embeddings,
            vectorstore_path=os.path.join(vectorstore_path, "layer2"),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            cache_size=self.config['performance']['cache_size'],
            use_reranker=use_reranker,
            reranker_config=reranking_config if use_reranker else None,
            chunking_config=chunking_config
        )
        
        self.confidence_evaluator = ConfidenceEvaluator(self.evaluation_llm)
        self.context_expander = ContextExpander(self.layer2)
        
        # Layer 2 觸發決策器
        self.layer2_trigger = Layer2TriggerDecision(llm=self.llm)
        logger.info("✓ Layer 2 Trigger Decision initialized")
        
        # Chunk Classifier (for semantic chunk classification)
        self.chunk_classifier: Optional[Any] = None
        if self.config.get('chunk_classification', {}).get('enabled', False):
            from system_api.chunk_classifier import ChunkClassifier
            
            use_llm = self.config['chunk_classification'].get('use_llm_fallback', False)
            self.chunk_classifier = ChunkClassifier(
                llm=self.llm if use_llm else None,
                use_llm_fallback=use_llm
            )
            logger.info("✓ Chunk Classifier initialized (heuristic-based)")
        
        # Index Manager (for incremental document updates)
        self.index_manager = IndexManager(
            layer1=self.layer1,
            layer2=self.layer2,
            abstract_extractor=self.abstract_extractor,
            backup_dir=os.path.join(vectorstore_path, "backups"),
            max_backups=5,
            chunk_classifier=self.chunk_classifier
        )
        logger.info("✓ Index Manager initialized")
        
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
        
        # Query Router (智能路由器) - 判斷使用 Graph 還是 RAG
        self.query_router: Optional[QueryRouter] = None
        if self.config.get('query_routing', {}).get('enabled', False):
            self.query_router = QueryRouter(llm=self.llm)
            logger.info("✓ Query Router initialized")
            logger.info("  Strategy: Graph-first for cross-paper queries, RAG for single-paper details")
        
        # Hybrid retriever (initialized after loading indices)
        self.hybrid_retriever: Optional[HybridRetriever] = None
        
        # Try to load existing indices
        self._load_indices()
        
        # Graph-first components (initialized after loading indices, requires graph_manager)
        self.graph_retriever: Optional[Any] = None
        self.graph_integrator: Optional[Any] = None
        
        # Always initialize GraphIntegrator (needed for single paper query chunk type filtering)
        try:
            from system_api.graph_integrator import GraphIntegrator
            self.graph_integrator = GraphIntegrator(self.llm)
            logger.info("✓ GraphIntegrator initialized (for chunk type determination)")
        except Exception as e:
            logger.warning(f"⚠️  Failed to initialize GraphIntegrator: {e}")
            logger.warning("  Chunk type filtering will be disabled")
        
        if self.config.get('graph_first', {}).get('enabled', False):
            # Check if graph components are available
            if self.index_manager.graph_manager:
                from system_api.graph_retriever import GraphRetriever
                
                self.graph_retriever = GraphRetriever(self.index_manager.graph_manager)
                
                logger.info("✓ Graph-first components initialized")
                logger.info(f"  Graph query top-k: {self.config['graph_first']['graph_query_top_k']}")
                logger.info(f"  Confidence threshold: {self.config['graph_first']['confidence_threshold']}")
                logger.info(f"  Skip Layer1: {self.config['graph_first']['skip_layer1']}")
            else:
                logger.warning("⚠️  Graph-first enabled but graph_manager not available")
                logger.warning("  Graph-first mode will be skipped")
        
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
                
                # Preprocess text: Remove references and appendix
                cleaned_text, preprocess_stats = self.text_preprocessor.preprocess(pdf_text)
                logger.info(f"  ✓ Preprocessed: {preprocess_stats['original_length']} → {preprocess_stats['final_length']} chars "
                           f"(removed: {preprocess_stats.get('removed_percentage', 0):.1f}%)")
                
                # Extract abstract (for Layer 1) - use original text for better abstract extraction
                abstract_result = self.abstract_extractor.extract(
                    pdf_path=pdf_path,
                    pdf_text=pdf_text,  # Use original text for abstract extraction
                    paper_id=paper_id
                )
                
                # Add simple metadata (use filename as title)
                if abstract_result:
                    abstract_result['title'] = paper_id
                    abstract_result['authors'] = []
                    abstract_result['year'] = None
                    abstracts.append(abstract_result)
                    logger.info(f"  ✓ Abstract: {abstract_result['source']} (confidence: {abstract_result['confidence']:.2f})")
                
                # Split into chunks (for Layer 2) - use cleaned text
                chunks = self.layer2.text_splitter.create_documents(
                    texts=[cleaned_text],  # Use preprocessed text for chunking
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
        
        # Step: Optional Graph Indexing (after both layers built)
        if self.index_manager.graph_manager and self.index_manager.graph_extractor:
            logger.info("\n" + "="*80)
            logger.info("Graph RAG Indexing (Optional)")
            logger.info("="*80)
            
            graph_success = 0
            graph_errors = 0
            
            for i, pdf_path in enumerate(pdf_paths, 1):
                try:
                    filename = os.path.basename(pdf_path)
                    logger.info(f"[{i}/{len(pdf_paths)}] Indexing to Graph: {filename}")
                    
                    # Read PDF text again (needed for graph extraction)
                    from PyPDF2 import PdfReader
                    reader = PdfReader(pdf_path)
                    pdf_text = ""
                    for page in reader.pages[:10]:  # First 10 pages sufficient
                        pdf_text += page.extract_text() + "\n"
                    
                    # Extract graph metadata
                    graph_data = self.index_manager.graph_extractor.extract(pdf_text)
                    
                    # Get paper_id (same as used in build)
                    paper_id = os.path.splitext(filename)[0]
                    
                    # Write to Neo4j
                    success = self.index_manager.graph_manager.add_paper_metadata(
                        paper_id=paper_id,
                        title=paper_id,  # Use filename as title
                        year='Unknown',
                        research_goal=graph_data.get('research_goal', ''),
                        methods=graph_data.get('methods', []),
                        datasets=graph_data.get('datasets', []),
                        domain=graph_data.get('domain', 'Unknown'),
                        domain_zh=graph_data.get('domain_zh', '未知領域'),
                        metrics=graph_data.get('metrics', [])
                    )
                    
                    if success:
                        logger.info(f"  ✓ Graph metadata indexed")
                        graph_success += 1
                    else:
                        logger.warning(f"  ✗ Graph indexing failed")
                        graph_errors += 1
                        
                except Exception as e:
                    logger.error(f"  ✗ Graph indexing error: {e}")
                    graph_errors += 1
            
            logger.info(f"Graph Indexing Complete: {graph_success} success, {graph_errors} errors")
        
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
    
    def add_document(self, pdf_path: str) -> Dict[str, Any]:
        """
        Add a single PDF to existing vectorstore using IndexManager
        
        This method provides incremental document addition with:
        - Automatic backup before changes
        - Text preprocessing (removes appendix, references, etc.)
        - Abstract extraction and Layer 1 indexing
        - Chunk creation and Layer 2 indexing
        - Transactional safety with rollback on failure
        
        Args:
            pdf_path: Absolute path to PDF file
            
        Returns:
            Dictionary with status and statistics:
            {
                'status': 'success' or 'error',
                'duration_seconds': float,
                'paper_id': str,
                'chunks_added': int,
                'backup_id': str,
                'error': str (if failed)
            }
        """
        start_time = time.time()
        
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF not found: {pdf_path}")
            
            logger.info(f"Adding document to Hierarchical RAG: {pdf_path}")
            
            # Extract PDF text
            from langchain_community.document_loaders import PyMuPDFLoader
            loader = PyMuPDFLoader(pdf_path)
            pages = loader.load()
            pdf_text = "\n".join([page.page_content for page in pages])
            
            # Generate paper ID (use filename without extension for consistency with build_indices)
            paper_id = os.path.splitext(os.path.basename(pdf_path))[0]
            
            # Extract metadata
            paper_metadata = {
                'title': paper_id,
                'pdf_path': pdf_path,
                'source_file': os.path.basename(pdf_path)
            }
            
            # Use IndexManager to add document (includes text preprocessing)
            result = self.index_manager.add_document(
                pdf_path=pdf_path,
                pdf_text=pdf_text,
                paper_id=paper_id,
                paper_metadata=paper_metadata
            )
            
            # Update hybrid retriever if enabled
            if self.hybrid_retriever and result['status'] == 'success':
                logger.info("Rebuilding BM25 index for hybrid search...")
                self.hybrid_retriever.build_bm25_index()
            
            duration = time.time() - start_time
            result['duration_seconds'] = duration
            
            logger.info(f"Document added successfully in {duration:.1f}s")
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Failed to add document: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'duration_seconds': duration
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
    
    def query_single_paper(self, query: str, return_metadata: bool = False) -> Any:
        """
        Query for a SINGLE SPECIFIC paper's details (specialized for AssistantCall)
        
        Flow:
        1. Layer1: Identify the specific paper (return top-1 most relevant)
        2. Check if abstract is sufficient to answer the query
        3. If not sufficient, go to Layer2 for detailed chunks with semantic filtering
        
        Args:
            query: User query about a specific paper
            return_metadata: If True, return full result dict with metadata
            
        Returns:
            Generated answer string, or full result dict if return_metadata=True
        """
        start_time = time.time()
        logger.info(f"=== Single Paper Query Mode ===")
        logger.info(f"Query: '{query[:100]}'")
        
        result = {
            'status': 'success',
            'query': query,
            'mode': 'single_paper',
            'layers_used': [],
            'timings': {}
        }
        
        try:
            # ============================================================
            # STEP 1: Layer 1 - Identify the target paper (top-1)
            # ============================================================
            logger.info("\n" + "="*60)
            logger.info("LAYER 1: Identify Target Paper")
            logger.info("="*60)
            
            layer1_start = time.time()
            
            if not self.layer1.is_initialized:
                raise ValueError("Layer 1 index not initialized")
            
            # STRATEGY: Get top-5, then use embedding similarity on titles to find best match
            layer1_results = self.layer1.search_with_scores(query=query, k=5)
            
            if not layer1_results:
                result['status'] = 'no_results'
                result['message'] = '未找到相關論文'
                result['timings']['total'] = time.time() - start_time
                if return_metadata:
                    result['answer'] = result['message']
                    return result
                return result['message']
            
            # Get query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Calculate title similarity for each paper in top-5
            best_title_match = None
            best_title_score = -1
            best_title_match_idx = -1
            
            for idx, (doc, abstract_score) in enumerate(layer1_results):
                paper_id = doc.metadata['paper_id']
                title = doc.metadata.get('title', paper_id)
                
                # Get title embedding and calculate cosine similarity
                title_embedding = self.embeddings.embed_query(paper_id)
                
                # Cosine similarity
                import numpy as np
                similarity = np.dot(query_embedding, title_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(title_embedding)
                )
                
                logger.info(f"  Paper {idx+1}: title_sim={similarity:.4f}, abstract_sim={abstract_score:.4f} - {paper_id[:50]}")
                
                if similarity > best_title_score:
                    best_title_score = similarity
                    best_title_match = doc
                    best_title_match_idx = idx
            
            # Use paper with highest title similarity if it's significantly higher than abstract-based top-1
            # Threshold: title similarity > 0.5 OR (title similarity > 0.3 AND higher than top-1's abstract score)
            abstract_top1_score = layer1_results[0][1]
            
            if best_title_score > 0.5 or (best_title_score > 0.3 and best_title_score > abstract_top1_score):
                target_paper_doc = best_title_match
                paper_score = best_title_score
                logger.info(f"✓ Using title-matched paper (rank {best_title_match_idx+1}, title_sim={best_title_score:.4f})")
            else:
                target_paper_doc, paper_score = layer1_results[0]
                logger.info(f"No strong title match, using top-1 by abstract similarity (score={paper_score:.4f})")
            
            target_paper_id = target_paper_doc.metadata['paper_id']
            target_paper_title = target_paper_doc.metadata.get('title', 'Unknown')
            target_paper_abstract = target_paper_doc.page_content
            
            result['timings']['layer1_identification'] = time.time() - layer1_start
            result['layers_used'].append('layer1')
            result['target_paper'] = {
                'paper_id': target_paper_id,
                'title': target_paper_title,
                'score': paper_score,
                'abstract': target_paper_abstract
            }
            
            logger.info(f"✓ Target paper identified:")
            logger.info(f"  Title: {target_paper_title}")
            logger.info(f"  Score: {paper_score:.4f}")
            logger.info(f"  Abstract length: {len(target_paper_abstract)} chars")
            
            # ============================================================
            # STEP 2: Check if abstract is sufficient
            # ============================================================
            # OPTIMIZATION: If title similarity is very high (>0.5), skip abstract check
            # This indicates user explicitly mentioned the paper name, so they want details
            skip_abstract_check = best_title_score > 0.5
            
            if skip_abstract_check:
                logger.info("\n" + "="*60)
                logger.info("STEP 2: Abstract Check SKIPPED (high title match)")
                logger.info("="*60)
                logger.info(f"  Title similarity: {best_title_score:.4f} > 0.5")
                logger.info(f"  → User explicitly mentioned paper name, going directly to Layer2 for details")
                result['timings']['abstract_evaluation'] = 0.0
                abstract_evaluation = {
                    'confidence': 0.0,
                    'should_continue': True,
                    'reasoning': 'Skipped due to high title similarity - user wants detailed information',
                    'missing_aspects': []  # No specific aspects identified
                }
                result['abstract_evaluation'] = abstract_evaluation
            else:
                logger.info("\n" + "="*60)
                logger.info("STEP 2: Abstract Sufficiency Check")
                logger.info("="*60)
                
                eval_start = time.time()
                
                # Use confidence evaluator to check if abstract is sufficient
                abstract_evaluation = self.confidence_evaluator.evaluate(
                    query=query,
                    retrieved_docs=[target_paper_doc],
                    layer=1,
                    threshold=0.7  # High threshold for abstract sufficiency
                )
                
                result['timings']['abstract_evaluation'] = time.time() - eval_start
                result['abstract_evaluation'] = abstract_evaluation
                
                logger.info(f"Abstract evaluation:")
                logger.info(f"  Confidence: {abstract_evaluation['confidence']:.2f}")
                logger.info(f"  Sufficient: {not abstract_evaluation['should_continue']}")
                logger.info(f"  Reasoning: {abstract_evaluation['reasoning'][:150]}...")
                
                # If abstract is sufficient, return answer based on abstract
                if not abstract_evaluation['should_continue']:
                    logger.info("\n✓ Abstract is sufficient to answer the query")
                    result['final_docs'] = [target_paper_doc]
                    result['terminated_at'] = 'abstract'
                    result['timings']['total'] = time.time() - start_time
                    
                    # Generate answer from abstract
                    answer = self._generate_answer(query, result)
                    if return_metadata:
                        result['answer'] = answer
                        return result
                    return answer
            
            logger.info("\n→ Abstract insufficient, retrieving detailed chunks from Layer2")
            
            # ============================================================
            # STEP 3: Layer2 - Retrieve detailed chunks with semantic filtering
            # ============================================================
            logger.info("\n" + "="*60)
            logger.info("LAYER 2: Detailed Chunk Retrieval with Semantic Filtering")
            logger.info("="*60)
            
            layer2_start = time.time()
            
            if not self.layer2.is_initialized:
                raise ValueError("Layer 2 index not initialized")
            
            # Determine which semantic chunk types are relevant to the query
            relevant_chunk_types = None
            if self.chunk_classifier and self.graph_integrator:
                # Use GraphIntegrator's method to determine chunk types
                relevant_chunk_types = self.graph_integrator.determine_chunk_types_for_query(
                    query=query,
                    missing_info=abstract_evaluation.get('missing_aspects', [])
                )
                logger.info(f"📌 Relevant chunk types: {relevant_chunk_types}")
            else:
                logger.info("ℹ️  No chunk type filtering (retrieving all types)")
            
            # Retrieve chunks from the target paper only, with chunk type filtering
            CHUNKS_TO_RETRIEVE = 5
            layer2_results = self.layer2.search_with_scores(
                query=query,
                k=CHUNKS_TO_RETRIEVE,
                paper_ids=[target_paper_id],
                filter_chunk_types=relevant_chunk_types
            )
            
            layer2_docs = [doc for doc, score in layer2_results]
            
            result['timings']['layer2_retrieval'] = time.time() - layer2_start
            result['layers_used'].append('layer2')
            result['layer2_docs'] = layer2_docs
            result['layer2_chunk_types'] = relevant_chunk_types
            
            logger.info(f"✓ Retrieved {len(layer2_docs)} detailed chunks")
            if layer2_results:
                logger.info("Top chunks:")
                for i, (doc, score) in enumerate(layer2_results[:3], 1):
                    chunk_type = doc.metadata.get('chunk_type', 'unknown')
                    preview = doc.page_content[:80].replace('\n', ' ')
                    logger.info(f"  {i}. [{score:.3f}] [{chunk_type}] {preview}...")
            
            # ============================================================
            # STEP 4: Generate final answer
            # ============================================================
            result['final_docs'] = layer2_docs
            result['terminated_at'] = 'layer2'
            result['timings']['total'] = time.time() - start_time
            
            logger.info(f"\n=== Query completed in {result['timings']['total']:.2f}s ===")
            
            # Generate answer
            answer = self._generate_answer(query, result)
            
            if return_metadata:
                result['answer'] = answer
                return result
            return answer
            
        except Exception as e:
            logger.error(f"Single paper query failed: {e}", exc_info=True)
            result['status'] = 'error'
            result['message'] = str(e)
            result['timings']['total'] = time.time() - start_time
            
            if return_metadata:
                result['answer'] = f"查詢失敗: {str(e)}"
                return result
            return f"查詢失敗: {str(e)}"
    
    def query(self, query: str, return_metadata: bool = False) -> Any:
        """
        Query the hierarchical RAG system (blocking version)
        
        Args:
            query: User query string
            return_metadata: If True, return full result dict with metadata instead of just answer string
            
        Returns:
            Generated answer string, or full result dict if return_metadata=True
        """
        result = self._hierarchical_retrieval(query)
        
        if result['status'] == 'error':
            if return_metadata:
                result['answer'] = f"Error: {result['message']}"
                return result
            return f"Error: {result['message']}"
        
        if result['status'] == 'no_results':
            if return_metadata:
                result['answer'] = "抱歉，沒有找到與您的問題相關的論文。請嘗試使用不同的關鍵字或降低相似度閾值。"
                return result
            return "抱歉，沒有找到與您的問題相關的論文。請嘗試使用不同的關鍵字或降低相似度閾值。"
        
        # Generate answer using LLM
        answer = self._generate_answer(query, result)
        
        if return_metadata:
            result['answer'] = answer
            return result
        return answer
    
    def query_single_paper_stream(self, query: str) -> Generator[str, None, None]:
        """
        Query for a SINGLE SPECIFIC paper's details with streaming (for AssistantCall)
        
        Flow:
        1. Layer1: Identify the specific paper (return top-1 most relevant)
        2. Check if abstract is sufficient to answer the query
        3. If not sufficient, go to Layer2 for detailed chunks with semantic filtering
        4. Stream the answer generation
        
        Args:
            query: User query about a specific paper
            
        Yields:
            Answer chunks as they are generated
        """
        start_time = time.time()
        logger.info(f"=== Single Paper Query Mode (Streaming) ===")
        logger.info(f"Query: '{query[:100]}'")
        
        result = {
            'status': 'success',
            'query': query,
            'mode': 'single_paper',
            'layers_used': [],
            'timings': {}
        }
        
        try:
            # ============================================================
            # STEP 1: Layer 1 - Identify the target paper (top-1)
            # ============================================================
            logger.info("\n" + "="*60)
            logger.info("LAYER 1: Identify Target Paper")
            logger.info("="*60)
            
            layer1_start = time.time()
            
            if not self.layer1.is_initialized:
                yield "錯誤: Layer 1 未初始化"
                return
            
            # STRATEGY: Check top-5 results for exact title match, prioritize if found
            layer1_results = self.layer1.search_with_scores(query=query, k=5)
            
            if not layer1_results:
                yield "未找到相關論文"
                return
            
            # Get query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Calculate title similarity for each paper in top-5
            best_title_match = None
            best_title_score = -1
            best_title_match_idx = -1
            
            import numpy as np
            for idx, (doc, abstract_score) in enumerate(layer1_results):
                paper_id = doc.metadata['paper_id']
                title = doc.metadata.get('title', paper_id)
                
                # Get title embedding and calculate cosine similarity
                title_embedding = self.embeddings.embed_query(paper_id)
                
                # Cosine similarity
                similarity = np.dot(query_embedding, title_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(title_embedding)
                )
                
                logger.info(f"  Paper {idx+1}: title_sim={similarity:.4f}, abstract_sim={abstract_score:.4f} - {paper_id[:50]}")
                
                if similarity > best_title_score:
                    best_title_score = similarity
                    best_title_match = doc
                    best_title_match_idx = idx
            
            # Use paper with highest title similarity if it's significantly higher
            abstract_top1_score = layer1_results[0][1]
            
            if best_title_score > 0.5 or (best_title_score > 0.3 and best_title_score > abstract_top1_score):
                target_paper_doc = best_title_match
                paper_score = best_title_score
                logger.info(f"✓ Using title-matched paper (rank {best_title_match_idx+1}, title_sim={best_title_score:.4f})")
            else:
                target_paper_doc, paper_score = layer1_results[0]
                logger.info(f"No strong title match, using top-1 by abstract similarity (score={paper_score:.4f})")
            
            target_paper_id = target_paper_doc.metadata['paper_id']
            target_paper_title = target_paper_doc.metadata.get('title', 'Unknown')
            target_paper_abstract = target_paper_doc.page_content
            
            result['timings']['layer1_identification'] = time.time() - layer1_start
            result['layers_used'].append('layer1')
            result['target_paper'] = {
                'paper_id': target_paper_id,
                'title': target_paper_title,
                'score': paper_score,
                'abstract': target_paper_abstract
            }
            
            logger.info(f"✓ Target paper identified:")
            logger.info(f"  Title: {target_paper_title}")
            logger.info(f"  Score: {paper_score:.4f}")
            
            # ============================================================
            # STEP 2: Check if abstract is sufficient
            # ============================================================
            # OPTIMIZATION: If title similarity is very high (>0.5), skip abstract check
            skip_abstract_check = best_title_score > 0.5
            
            if skip_abstract_check:
                logger.info("\n" + "="*60)
                logger.info("STEP 2: Abstract Check SKIPPED (high title match)")
                logger.info("="*60)
                logger.info(f"  Title similarity: {best_title_score:.4f} > 0.5")
                logger.info(f"  → User explicitly mentioned paper name, going directly to Layer2 for details")
                result['timings']['abstract_evaluation'] = 0.0
                abstract_evaluation = {
                    'confidence': 0.0,
                    'should_continue': True,
                    'reasoning': 'Skipped due to high title similarity',
                    'missing_aspects': []
                }
                result['abstract_evaluation'] = abstract_evaluation
            else:
                logger.info("\n" + "="*60)
                logger.info("STEP 2: Abstract Sufficiency Check")
                logger.info("="*60)
                
                eval_start = time.time()
                
                # Use confidence evaluator to check if abstract is sufficient
                abstract_evaluation = self.confidence_evaluator.evaluate(
                    query=query,
                    retrieved_docs=[target_paper_doc],
                    layer=1,
                    threshold=0.7
                )
                
                result['timings']['abstract_evaluation'] = time.time() - eval_start
                result['abstract_evaluation'] = abstract_evaluation
                
                logger.info(f"Abstract evaluation:")
                logger.info(f"  Confidence: {abstract_evaluation['confidence']:.2f}")
                logger.info(f"  Sufficient: {not abstract_evaluation['should_continue']}")
                
                # If abstract is sufficient, stream answer based on abstract
                if not abstract_evaluation['should_continue']:
                    logger.info("\n✓ Abstract is sufficient to answer the query")
                    result['final_docs'] = [target_paper_doc]
                    result['terminated_at'] = 'abstract'
                    result['timings']['total'] = time.time() - start_time
                    
                    # Stream answer from abstract
                    yield from self._generate_answer_stream(query, result)
                    return
            
            logger.info("\n→ Abstract insufficient, retrieving detailed chunks from Layer2")
            
            # ============================================================
            # STEP 3: Layer2 - Retrieve detailed chunks with semantic filtering
            # ============================================================
            logger.info("\n" + "="*60)
            logger.info("LAYER 2: Detailed Chunk Retrieval with Semantic Filtering")
            logger.info("="*60)
            
            layer2_start = time.time()
            
            if not self.layer2.is_initialized:
                yield "錯誤: Layer 2 未初始化"
                return
            
            # Determine which semantic chunk types are relevant to the query
            relevant_chunk_types = None
            if self.chunk_classifier and self.graph_integrator:
                relevant_chunk_types = self.graph_integrator.determine_chunk_types_for_query(
                    query=query,
                    missing_info=abstract_evaluation.get('missing_aspects', [])
                )
                logger.info(f"📌 Relevant chunk types: {relevant_chunk_types}")
            else:
                logger.info("ℹ️  No chunk type filtering (retrieving all types)")
            
            # Retrieve chunks from the target paper only, with chunk type filtering
            CHUNKS_TO_RETRIEVE = 5
            layer2_results = self.layer2.search_with_scores(
                query=query,
                k=CHUNKS_TO_RETRIEVE,
                paper_ids=[target_paper_id],
                filter_chunk_types=relevant_chunk_types
            )
            
            layer2_docs = [doc for doc, score in layer2_results]
            
            result['timings']['layer2_retrieval'] = time.time() - layer2_start
            result['layers_used'].append('layer2')
            result['layer2_docs'] = layer2_docs
            result['layer2_chunk_types'] = relevant_chunk_types
            
            logger.info(f"✓ Retrieved {len(layer2_docs)} detailed chunks")
            
            # ============================================================
            # STEP 4: Stream final answer
            # ============================================================
            result['final_docs'] = layer2_docs
            result['terminated_at'] = 'layer2'
            result['timings']['total'] = time.time() - start_time
            
            logger.info(f"\n=== Query completed in {result['timings']['total']:.2f}s ===")
            
            # Stream answer
            yield from self._generate_answer_stream(query, result)
            
        except Exception as e:
            logger.error(f"Single paper query stream failed: {e}", exc_info=True)
            yield f"查詢失敗: {str(e)}"
    
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
            # STEP -1: Query Routing (決定使用 Graph 或 RAG)
            # ============================================================
            if self.query_router and self.config.get('query_routing', {}).get('enabled', False):
                logger.info("="*60)
                logger.info("智能查詢路由 (Query Routing)")
                logger.info("="*60)
                
                routing_start = time.time()
                routing_decision = self.query_router.route(query)
                result['timings']['query_routing'] = time.time() - routing_start
                result['routing'] = routing_decision
                
                logger.info(f"📍 路由決策: {routing_decision['route'].upper()}")
                logger.info(f"📊 信心度: {routing_decision['confidence']:.2f}")
                logger.info(f"💭 理由: {routing_decision['reasoning']}")
                
                # 根據路由決策，設置 Graph-first 或跳過 Graph
                routing_confidence_threshold = self.config['query_routing'].get('confidence_threshold', 0.7)
                
                if routing_decision['confidence'] >= routing_confidence_threshold:
                    if routing_decision['route'] == 'graph':
                        # 跨論文查詢 → 優先使用 Graph
                        logger.info("→ 使用 Graph-first 模式（跨論文結構化查詢）")
                        result['preferred_route'] = 'graph'
                        # 路由器覆蓋配置，確保 Graph 會執行
                        result['skip_graph_for_this_query'] = False
                        if self.graph_retriever and self.graph_integrator:
                            logger.info("  ✓ Graph 檢索已就緒，將執行 Knowledge Graph 查詢")
                        else:
                            logger.warning("  ⚠️ Graph 組件未初始化，將回退到傳統 RAG 流程")
                    else:  # rag
                        # 單一論文詳細查詢 → 跳過 Graph，直接使用 RAG
                        logger.info("→ 使用 RAG 模式（單一論文詳細內容檢索）")
                        result['preferred_route'] = 'rag'
                        # 暫時禁用 Graph-first（針對此查詢）
                        result['skip_graph_for_this_query'] = True
                else:
                    logger.info(f"→ 信心度不足 ({routing_decision['confidence']:.2f} < {routing_confidence_threshold})，使用預設路由")
                    default_route = self.config['query_routing'].get('default_route', 'rag')
                    result['preferred_route'] = default_route
                    result['skip_graph_for_this_query'] = (default_route == 'rag')
            
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
            # GRAPH STAGE: Query Neo4j for papers (if enabled and routed)
            # ============================================================
            # 檢查是否應該跳過 Graph（根據路由決策）
            skip_graph = result.get('skip_graph_for_this_query', False)
            
            if self.graph_retriever and self.graph_integrator and not skip_graph:
                logger.info("="*60)
                logger.info("GRAPH STAGE: Querying Knowledge Graph")
                logger.info("="*60)
                
                graph_start = time.time()
                
                try:
                    # Step 1: Query graph for papers
                    graph_config = self.config.get('graph_first', {})
                    graph_top_k = graph_config.get('graph_query_top_k', 15)
                    
                    graph_results = self.graph_retriever.query_graph_for_papers(
                        query=query,
                        top_k=graph_top_k
                    )
                    
                    result['timings']['graph_query'] = time.time() - graph_start
                    result['layers_used'].append('graph')
                    result['graph_results'] = graph_results
                    
                    logger.info(f"Found {len(graph_results)} papers from Graph")
                    
                    # Display top results
                    if graph_results:
                        logger.info("\n📊 Graph 查詢結果 Top 5:")
                        logger.info("─" * 80)
                        for i, paper in enumerate(graph_results[:5], 1):
                            paper_id = paper['paper_id']
                            score = paper['score']
                            entities = paper['matched_entities']
                            logger.info(f"  {i}. [{score:.2f}] {paper_id[:50]}...")
                            logger.info(f"      Matched entities: {', '.join(entities[:5])}")
                        logger.info("─" * 80)
                    
                    # Check minimum hits threshold
                    min_hits = graph_config.get('min_graph_hits', 1)
                    
                    if len(graph_results) >= min_hits:
                        # Step 2: Integrate graph results using LLM
                        integration_start = time.time()
                        max_papers = graph_config.get('max_papers_for_integration', 5)
                        
                        integration = self.graph_integrator.integrate_graph_results(
                            query=query,
                            graph_results=graph_results,
                            max_papers=max_papers
                        )
                        
                        result['timings']['graph_integration'] = time.time() - integration_start
                        result['graph_integration'] = integration
                        
                        logger.info(f"\nGraph Integration:")
                        logger.info(f"  Confidence: {integration['confidence']:.2f}")
                        logger.info(f"  Should descend: {integration['should_descend']}")
                        logger.info(f"  Missing info: {integration['missing_info']}")
                        logger.info(f"  Answer preview: {integration['text'][:150]}...")
                        
                        # Step 3: Decision - should we descend to Layer2?
                        graph_threshold = graph_config.get('confidence_threshold', 0.7)
                        
                        if not integration['should_descend']:
                            logger.info(f"\n✓ Graph results sufficient (confidence: {integration['confidence']:.2f} >= {graph_threshold})")
                            logger.info("  No detailed chunk retrieval needed")
                            logger.info("  Terminating at Graph stage")
                            
                            result['final_docs'] = []  # No chunks retrieved
                            result['final_answer'] = integration['text']
                            result['final_confidence'] = integration['confidence']
                            result['terminated_at'] = 'graph'
                            result['referenced_papers'] = integration.get('referenced_papers', [])
                            result['timings']['total'] = time.time() - start_time
                            
                            return result
                        
                        logger.info("\n→ Graph results insufficient, descending to Layer2 for details")
                        
                        # Step 4: Determine chunk types to retrieve
                        chunk_types = self.graph_integrator.determine_chunk_types_for_query(
                            query=query,
                            missing_info=integration['missing_info']
                        )
                        
                        result['target_chunk_types'] = chunk_types
                        logger.info(f"  Target chunk types: {chunk_types}")
                        
                        # Step 5: Extract paper_ids for Layer2 filtering
                        graph_paper_ids = [r['paper_id'] for r in graph_results]
                        result['graph_paper_ids'] = graph_paper_ids
                        
                        logger.info(f"  Will filter {len(graph_paper_ids)} papers from Graph")
                        
                        # Skip Layer 1 if configured
                        skip_layer1 = graph_config.get('skip_layer1', True)
                        
                        if skip_layer1:
                            logger.info("\n  ⚡ Skipping Layer1 (Graph → Layer2 direct path)")
                            
                            # Jump directly to Layer 2 with filtering
                            # We'll set layer1_docs to empty and paper_ids from graph
                            # The Layer 2 code below will use graph_paper_ids
                            result['layer1_skipped'] = True
                            
                            # Continue to Layer 2 (skip Layer 1 block)
                            # We need to jump to Layer 2 section
                            # Set up variables that Layer 2 expects
                            layer1_docs = []  # No Layer1 docs
                            paper_ids = graph_paper_ids  # Use Graph paper IDs
                            
                            # Skip the entire Layer 1 block below
                            # Jump directly to Layer 2
                            # (We'll handle this by setting a flag and checking it)
                            
                        else:
                            logger.info("\n  → Continuing to Layer1 as backup")
                            # Continue to Layer 1 normally
                            result['layer1_skipped'] = False
                    
                    else:
                        logger.warning(f"⚠️  Graph returned {len(graph_results)} papers (< min: {min_hits})")
                        logger.warning("  Falling back to Layer1")
                        result['graph_insufficient'] = True
                
                except Exception as e:
                    logger.error(f"❌ Graph query failed: {e}", exc_info=True)
                    logger.warning("  Falling back to Layer1")
                    result['graph_error'] = str(e)
            
            # Check if we should skip Layer 1
            skip_layer1 = result.get('layer1_skipped', False)
            
            if not skip_layer1:
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
            
                # 保存 Layer1 文檔和分數
                result['layer1_docs'] = layer1_docs
                result['layer1_scores'] = [score for doc, score in layer1_results_with_scores]
            
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
                
                # ========== 使用智能觸發決策器 ==========
                logger.info("\n" + "="*60)
                logger.info("Layer 2 觸發決策分析")
                logger.info("="*60)
                
                trigger_decision = self.layer2_trigger.should_trigger_layer2(
                    query=query,
                    layer1_evaluation=evaluation1,
                    layer1_docs=layer1_docs,
                    base_threshold=threshold1
                )
                
                logger.info(f"決策結果:")
                logger.info(f"  是否觸發 Layer 2: {trigger_decision['should_trigger']}")
                logger.info(f"  查詢類型: {trigger_decision['decision_factors']['query_type']}")
                logger.info(f"  查詢複雜度: {trigger_decision['decision_factors']['query_complexity']}")
                logger.info(f"  基礎閾值: {trigger_decision['decision_factors']['base_threshold']:.2f}")
                logger.info(f"  調整後閾值: {trigger_decision['adjusted_threshold']:.2f}")
                logger.info(f"  決策理由: {trigger_decision['reason']}")
                
                # 記錄決策資訊
                result['layer2_trigger_decision'] = trigger_decision
                
                # 根據智能決策判斷是否進入 Layer 2
                if not trigger_decision['should_trigger']:
                    logger.info("✓ 智能決策: Layer 1 結果已足夠，無需進入 Layer 2")
                    result['final_docs'] = layer1_docs
                    result['final_confidence'] = evaluation1['confidence'] if evaluation1 else 0.0
                    result['terminated_at'] = 'layer1'
                    result['termination_reason'] = trigger_decision['reason']
                    result['timings']['total'] = time.time() - start_time
                    return result
                
                logger.info("→ 智能決策: 需要進入 Layer 2 獲取更詳細資訊")
            
            else:
                # Layer1 was skipped (Graph-first direct path)
                logger.info("\n" + "="*60)
                logger.info("Layer1 skipped - using Graph results for paper filtering")
                logger.info("="*60)
                
                # No Layer1 evaluation when skipped, set defaults
                layer1_docs = []  # Empty for consistency
                evaluation1 = None  # Mark as not evaluated
                trigger_decision = {'should_trigger': True}  # Always proceed to Layer2
            
            # ============================================================
            # LAYER 2: Chunk-level retrieval
            # ============================================================
            logger.info("\n" + "="*60)
            logger.info("LAYER 2: Chunk-level Retrieval")
            logger.info("="*60)
            
            layer2_start = time.time()
            
            if not self.layer2.is_initialized:
                raise ValueError("Layer 2 index not initialized")
            
            # Determine paper IDs source (Graph or Layer1)
            if result.get('layer1_skipped', False):
                # Use paper IDs from Graph
                paper_ids = result.get('graph_paper_ids', [])
                logger.info(f"Filtering by {len(paper_ids)} papers from Graph")
                logger.info(f"  Source: Graph-first direct path")
            else:
                # Use paper IDs from Layer 1
                paper_ids = [doc.metadata['paper_id'] for doc in layer1_docs]
                logger.info(f"Filtering by {len(paper_ids)} papers from Layer 1")
                logger.info(f"  Source: Traditional Layer1 → Layer2 path")
            
            # Get chunk type filter (if determined by Graph)
            filter_chunk_types = result.get('target_chunk_types', None)
            
            if filter_chunk_types:
                logger.info(f"📌 Applying chunk type filter: {filter_chunk_types}")
                logger.info(f"  Will only retrieve chunks of these semantic types")
            else:
                logger.info(f"  No chunk type filtering (retrieving all types)")
            
            # Strategy: Retrieve top-2 chunks PER PAPER (not globally)
            CHUNKS_PER_PAPER = 2
            logger.info(f"📋 Strategy: Retrieve top-{CHUNKS_PER_PAPER} chunks per paper")
            
            # Retrieve chunks for each paper separately
            layer2_docs_by_paper = {}
            all_layer2_results = []
            
            for paper_id in paper_ids:
                paper_results = self.layer2.search_with_scores(
                    query=query,
                    k=CHUNKS_PER_PAPER,
                    paper_ids=[paper_id],
                    filter_chunk_types=filter_chunk_types  # NEW: Apply chunk type filter
                )
                
                if paper_results:
                    layer2_docs_by_paper[paper_id] = [doc for doc, score in paper_results]
                    all_layer2_results.extend(paper_results)
                    logger.info(f"  ✓ {paper_id[:50]}...: {len(paper_results)} chunks")
            
            # Combine all results for display
            layer2_results_with_scores = all_layer2_results
            layer2_docs = [doc for doc, score in all_layer2_results]
            
            # For overview/general queries, ensure abstract chunks are included
            # (Only applicable when trigger_decision has decision_factors)
            query_type = trigger_decision.get('decision_factors', {}).get('query_type', 'specific')
            if query_type in ['overview', 'general']:
                logger.info("🔍 Overview query detected - ensuring abstract chunks are included")
                logger.info(f"  Paper IDs to check: {paper_ids}")
                
                # Find abstract chunks (chunk_0, chunk_1, or chunks with 'abstract' in content)
                abstract_chunks = []
                for paper_id in paper_ids:
                    # Try to get chunk_0 or chunk_1 from this paper
                    for chunk_num in [0, 1, 2]:
                        chunk_id_to_find = f"{paper_id}_chunk_{chunk_num}"
                        
                        logger.debug(f"  Looking for: {chunk_id_to_find}")
                        
                        # Check if already in results
                        already_included = any(
                            doc.metadata.get('chunk_id') == chunk_id_to_find 
                            for doc in layer2_docs
                        )
                        
                        if already_included:
                            logger.info(f"  ✓ Abstract chunk already in results: {chunk_id_to_find}")
                            break
                        else:
                            # Try to retrieve this chunk from Layer 2 store
                            try:
                                chunk_docs = self.layer2.get_chunks_by_ids([chunk_id_to_find])
                                if chunk_docs:
                                    abstract_chunks.extend(chunk_docs)
                                    logger.info(f"  ✓ Added abstract chunk: {chunk_id_to_find}")
                                    break  # Found abstract for this paper
                                else:
                                    logger.debug(f"  ✗ Chunk not found: {chunk_id_to_find}")
                            except Exception as e:
                                logger.debug(f"  ✗ Error retrieving {chunk_id_to_find}: {e}")
                
                # Prepend abstract chunks to results (they should come first for this paper)
                if abstract_chunks:
                    logger.info(f"  → Total {len(abstract_chunks)} abstract chunks prepended")
                    # Add to this paper's results
                    if paper_id in layer2_docs_by_paper:
                        layer2_docs_by_paper[paper_id] = abstract_chunks + layer2_docs_by_paper[paper_id]
                        # Limit to CHUNKS_PER_PAPER per paper
                        layer2_docs_by_paper[paper_id] = layer2_docs_by_paper[paper_id][:CHUNKS_PER_PAPER]
                else:
                    logger.warning("  ⚠️ No abstract chunks found to prepend")
            
            # Flatten all paper chunks for evaluation
            layer2_docs = []
            for paper_id, chunks in layer2_docs_by_paper.items():
                layer2_docs.extend(chunks)
            
            result['timings']['layer2_retrieval'] = time.time() - layer2_start
            result['layers_used'].append('layer2')
            result['layer2_docs_by_paper'] = layer2_docs_by_paper  # Store per-paper structure
            
            logger.info(f"Retrieved {len(layer2_docs)} chunks total from {len(layer2_docs_by_paper)} papers")
            
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
                # Safe access to evaluation1 (may be None if Layer1 was skipped)
                result['final_confidence'] = evaluation1['confidence'] if evaluation1 else 0.0
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
            # DYNAMIC ENTITY EXTRACTION is disabled (rollback)
            # ============================================================
            # The dynamic, query-time entity extractor was removed per user request.
            # We keep a placeholder in the result for compatibility with callers.
            logger.info("\n" + "="*60)
            logger.info("DYNAMIC ENTITY EXTRACTION: disabled")
            logger.info("="*60)
            result['timings']['entity_extraction'] = 0.0
            result['entity_extraction'] = {'status': 'disabled', 'entities': [], 'relationships': []}
            
            # ============================================================
            # CONTEXT EXPANSION (Per Paper)
            # ============================================================
            logger.info("\n" + "="*60)
            logger.info("CONTEXT EXPANSION (Per Paper)")
            logger.info("="*60)
            
            expand_start = time.time()
            expanded_contexts_by_paper = {}
            
            for paper_id, paper_chunks in layer2_docs_by_paper.items():
                if self.config['expansion']['enabled']:
                    paper_expanded = self.context_expander.expand(
                        chunks=paper_chunks,
                        confidence=evaluation2['confidence']
                    )
                    expanded_contexts_by_paper[paper_id] = paper_expanded
                    logger.info(f"  ✓ {paper_id[:50]}...: {len(paper_expanded)} contexts")
                else:
                    expanded_contexts_by_paper[paper_id] = [doc.page_content for doc in paper_chunks]
            
            result['timings']['expansion'] = time.time() - expand_start
            result['expanded_contexts_by_paper'] = expanded_contexts_by_paper
            
            stats = self.context_expander.get_expansion_stats(evaluation2['confidence'])
            logger.info(f"Expansion complete:")
            logger.info(f"  Strategy: {stats['strategy']}")
            logger.info(f"  Range: ±{stats['expansion_range']} chunks")
            logger.info(f"  Total papers: {len(expanded_contexts_by_paper)}")
            
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

【回答指引】
1. 仔細理解問題的重點關鍵詞（例如：資料集、方法、結果、貢獻等）
2. 如果問題問的是「資料集」，請重點說明：
   - 使用了什麼資料集？
   - 資料集有多少樣本/數據？
   - 資料來源是什麼？
3. 如果問題問的是「方法」，請重點說明使用的技術和演算法
4. 如果問題問的是「結果」，請重點說明實驗結果和性能指標
5. 請針對問題的核心關鍵詞回答，不要泛泛而談
6. 如果參考文獻中找不到相關資訊，請明確說明「論文中未提及相關資訊」

請根據以上內容提供完整的答案。
記住：回答必須使用繁體中文，例如「應用」而非「应用」，「資料」而非「数据」。

回答："""
        else:
            prompt = f"""Based on the following research paper contexts, please answer the question.

Question: {query}

Contexts:
{context_text}

【Answer Guidelines】
1. Carefully identify the key focus of the question (e.g., dataset, method, results, contribution)
2. If the question asks about "dataset", focus on:
   - What dataset(s) were used?
   - How many samples/data points?
   - What is the data source?
3. If the question asks about "method", focus on the techniques and algorithms used
4. If the question asks about "results", focus on experimental outcomes and performance metrics
5. Answer directly to the core keyword of the question, don't provide general summaries
6. If the information is not found in the contexts, clearly state "The paper does not mention this information"

Please provide a comprehensive answer based on the contexts above.

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
        
        # ✨ 新增：檢索摘要（選項 C - 顯示所有候選論文）
        if retrieval_result.get('layer1_ce_enabled', False):
            # 有 Cross-Encoder 評分的情況
            total_candidates = retrieval_result.get('layer1_ce_total_candidates', 0)
            passed_count = retrieval_result.get('layer1_ce_passed', 0)
            ce_scores = retrieval_result.get('layer1_ce_scores', [])
            hybrid_scores = retrieval_result.get('layer1_hybrid_scores', [])
            
            if is_chinese_query:
                yield "📊 檢索結果摘要\n"
                yield "─" * 60 + "\n"
                yield f"  階段 1 - 混合搜尋: {total_candidates} 篇候選\n"
                yield f"  階段 2 - 精確評分: {passed_count} 篇通過閾值\n"
                yield f"  最終選定: {len(layer1_docs)} 篇論文\n\n"
                yield "🎯 候選論文列表:\n"
            else:
                yield "📊 Retrieval Summary\n"
                yield "─" * 60 + "\n"
                yield f"  Stage 1 - Hybrid Search: {total_candidates} candidates\n"
                yield f"  Stage 2 - Precision Scoring: {passed_count} passed threshold\n"
                yield f"  Final Selection: {len(layer1_docs)} papers\n\n"
                yield "🎯 Candidate Papers:\n"
            
            # 顯示所有候選論文及雙重分數
            for i, doc in enumerate(layer1_docs, 1):
                title = doc.metadata.get('title', 'Unknown Title')
                ce_score = ce_scores[i-1] if i-1 < len(ce_scores) else 0.0
                hybrid_score = hybrid_scores[i-1] if i-1 < len(hybrid_scores) else 0.0
                
                if is_chinese_query:
                    yield f"  {i}. {title}\n"
                    yield f"     精確分數: {ce_score:.3f} | 混合分數: {hybrid_score:.3f}\n"
                else:
                    yield f"  {i}. {title}\n"
                    yield f"     CE Score: {ce_score:.3f} | Hybrid Score: {hybrid_score:.3f}\n"
            
            yield "\n" + "="*60 + "\n\n"
        
        # Yield referenced papers first (原有格式)
        elif referenced_papers:
            if is_chinese_query:
                yield "📚 參考論文：\n"
            else:
                yield "📚 Referenced Papers:\n"
            
            for paper in referenced_papers:
                yield paper + "\n"
            
            yield "\n" + "="*60 + "\n\n"
        
        # Check if we have per-paper contexts (Layer 2 with multiple papers)
        if 'expanded_contexts_by_paper' in retrieval_result:
            # Multiple papers: Generate answer for each paper separately
            expanded_contexts_by_paper = retrieval_result['expanded_contexts_by_paper']
            
            logger.info(f"📝 Generating answers for {len(expanded_contexts_by_paper)} papers separately")
            
            for paper_idx, (paper_id, contexts) in enumerate(expanded_contexts_by_paper.items(), 1):
                # Paper header
                if is_chinese_query:
                    yield f"\n{'='*60}\n"
                    yield f"📄 論文 {paper_idx}/{len(expanded_contexts_by_paper)}: {paper_id}\n"
                    yield f"{'='*60}\n\n"
                else:
                    yield f"\n{'='*60}\n"
                    yield f"📄 Paper {paper_idx}/{len(expanded_contexts_by_paper)}: {paper_id}\n"
                    yield f"{'='*60}\n\n"
                
                # Format context for this paper
                context_text = "\n\n".join([f"[Context {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)])
                
                # Debug log
                logger.info(f"📝 Paper {paper_idx} context:")
                logger.info(f"  Number of contexts: {len(contexts)}")
                logger.info(f"  Total context length: {len(context_text)} chars")
                
                # Create prompt
                if is_chinese_query:
                    prompt = f"""你是一個學術研究助手。以下是從一篇學術論文中擷取的相關段落，請根據這些段落回答問題。

問題：{query}

論文相關段落：
{context_text}

【回答指引】
1. 仔細理解問題的重點關鍵詞（例如：資料集、方法、結果、貢獻等）
2. 如果問題問的是「資料集」，請重點說明：使用了什麼資料集、有多少樣本、資料來源
3. 如果問題問的是「方法」，請重點說明使用的技術和演算法
4. 如果問題問的是「結果」，請重點說明實驗結果和性能指標
5. 請針對問題的核心關鍵詞回答，不要泛泛而談
6. 如果段落中找不到相關資訊，請明確說明「論文中未提及」

請根據以上段落，用繁體中文簡潔回答問題。"""
                else:
                    prompt = f"""Based on the following research paper contexts, please answer the question.

Question: {query}

Contexts:
{context_text}

【Answer Guidelines】
1. Identify the key focus of the question (e.g., dataset, method, results)
2. If asking about "dataset": specify what dataset, sample size, data source
3. If asking about "method": specify techniques and algorithms
4. If asking about "results": specify experimental outcomes and metrics
5. Answer directly to the core keyword, don't provide general summaries
6. If not found in contexts, state "Not mentioned in the paper"

Please provide a concise answer based on the contexts above.

Answer:"""
                
                # Generate answer for this paper
                try:
                    for chunk in self.llm.stream(prompt):
                        yield chunk
                    yield "\n\n"  # Separate papers
                except Exception as e:
                    logger.error(f"Error generating answer for paper {paper_idx}: {e}")
                    yield f"Error generating answer: {e}\n\n"
        
        else:
            # Single paper or Layer 1 only: Original logic
            if 'expanded_contexts' in retrieval_result:
                contexts = retrieval_result['expanded_contexts']
            else:
                contexts = [doc.page_content for doc in retrieval_result['final_docs']]
            
            context_text = "\n\n".join([f"[Context {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)])
            
            logger.info(f"📝 Context for LLM generation:")
            logger.info(f"  Number of contexts: {len(contexts)}")
            logger.info(f"  Total context length: {len(context_text)} chars")
            
            if is_chinese_query:
                prompt = f"""你是一個學術研究助手。以下是從學術論文中擷取的相關段落，請根據這些段落回答問題。

問題：{query}

論文相關段落：
{context_text}

【回答指引】
1. 仔細理解問題的重點關鍵詞（例如：資料集、方法、結果、貢獻等）
2. 如果問題問的是「資料集」，請重點說明：使用了什麼資料集、有多少樣本、資料來源
3. 如果問題問的是「方法」，請重點說明使用的技術和演算法
4. 如果問題問的是「結果」，請重點說明實驗結果和性能指標
5. 請針對問題的核心關鍵詞回答，不要泛泛而談
6. 如果段落中找不到相關資訊，請明確說明「論文中未提及」

請根據以上段落，用繁體中文完整回答問題。"""
            else:
                prompt = f"""Based on the following research paper contexts, please answer the question.

Question: {query}

Contexts:
{context_text}

【Answer Guidelines】
1. Identify the key focus of the question (e.g., dataset, method, results)
2. If asking about "dataset": specify what dataset, sample size, data source
3. If asking about "method": specify techniques and algorithms
4. If asking about "results": specify experimental outcomes and metrics
5. Answer directly to the core keyword, don't provide general summaries
6. If not found in contexts, state "Not mentioned in the paper"

Please provide a comprehensive answer based on the contexts above.

Answer:"""
            
            logger.info(f"📤 Prompt length: {len(prompt)} chars")
            
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
    
    @property
    def vectorstore(self):
        """
        Compatibility property for agent2.py checks
        Returns Layer 1 vectorstore if initialized, None otherwise
        """
        return self.layer1.vectorstore if self.layer1.is_initialized else None
    
    def check_and_update_indices(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """
        智能檢查索引狀態並更新（三資料庫同步邏輯）
        
        以 pdf_directory 為真相來源，確保 Layer1、Layer2、GraphRAG 三個資料庫一致：
        1. 索引是否存在且健康
        2. 掃描 pdf_directory 中所有 PDF（真相）
        3. 掃描 Vector DB (Layer1) 中已索引的論文
        4. 掃描 Graph DB 中已索引的論文
        5. 刪除多餘的索引（Vector + Graph）
        6. 新增缺少的索引（Vector + Graph）
        
        Args:
            force_rebuild: 強制重建所有索引
            
        Returns:
            操作結果字典
        """
        logger.info("="*80)
        logger.info("Checking Index Status (3-Database Sync)")
        logger.info("="*80)
        
        result = {
            'status': 'up_to_date',
            'action_taken': None,
            'details': {}
        }
        
        # 檢查 1: 索引是否就緒或強制重建
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
        
        # 檢查 2: 獲取三個資料庫的狀態
        try:
            # Step 2.1: 獲取 pdf_directory 中所有 PDF（真相來源）
            current_pdf_paths = set(self._get_all_pdfs())
            current_paper_ids = set()
            pdf_path_to_id = {}  # 用於後續查找
            
            for pdf_path in current_pdf_paths:
                # paper_id = filename without extension
                paper_id = os.path.splitext(os.path.basename(pdf_path))[0]
                current_paper_ids.add(paper_id)
                pdf_path_to_id[paper_id] = pdf_path
            
            logger.info(f"📁 PDF Directory: {len(current_paper_ids)} papers")
            
            # Step 2.2: 獲取 Vector DB (Layer1) 中已索引的 paper_id
            indexed_in_vector = set()
            if self.layer1.vectorstore:
                try:
                    docstore = self.layer1.vectorstore.docstore
                    if hasattr(docstore, '_dict'):
                        all_docs = list(docstore._dict.values())
                    else:
                        all_docs = []
                    
                    for doc in all_docs:
                        paper_id = doc.metadata.get('paper_id', '')
                        if paper_id:
                            indexed_in_vector.add(paper_id)
                except Exception as e:
                    logger.error(f"Error reading Vector DB: {e}")
            
            logger.info(f"📊 Vector DB (Layer1): {len(indexed_in_vector)} papers")
            
            # Step 2.3: 獲取 Graph DB 中已索引的 paper_id
            indexed_in_graph = set()
            if self.index_manager.graph_manager:
                try:
                    indexed_in_graph = set(self.index_manager.graph_manager.get_all_paper_ids())
                except Exception as e:
                    logger.error(f"Error reading Graph DB: {e}")
            
            logger.info(f"🕸️  Graph DB: {len(indexed_in_graph)} papers")
            
            # Step 2.4: 計算差異
            # 多餘的論文（需要刪除）
            extra_in_vector = indexed_in_vector - current_paper_ids
            extra_in_graph = indexed_in_graph - current_paper_ids
            
            # 缺少的論文（需要新增）
            missing_in_vector = current_paper_ids - indexed_in_vector
            missing_in_graph = current_paper_ids - indexed_in_graph
            
            logger.info("")
            logger.info("📈 Synchronization Status:")
            logger.info(f"  ➕ New papers to add: {len(missing_in_vector)}")
            logger.info(f"  ➖ Extra papers to remove from Vector DB: {len(extra_in_vector)}")
            logger.info(f"  ➖ Extra papers to remove from Graph DB: {len(extra_in_graph)}")
            
            # 檢查 3: 處理刪除（如果有多餘的論文）
            has_deletions = len(extra_in_vector) > 0 or len(extra_in_graph) > 0
            has_additions = len(missing_in_vector) > 0
            
            if has_deletions:
                logger.warning("⚠️  Detected extra papers in databases! Need to rebuild Vector DB to remove them.")
                logger.warning(f"  Extra in Vector DB: {list(extra_in_vector)[:5]}")
                if len(extra_in_vector) > 5:
                    logger.warning(f"  ... and {len(extra_in_vector) - 5} more")
                
                # Vector DB: FAISS 不支持單獨刪除，需要重建
                logger.info("Rebuilding Vector DB (Layer1 + Layer2) to remove extra papers...")
                result['action_taken'] = 'rebuild_due_to_deletions'
                
                build_result = self.build_indices()
                result['status'] = build_result.get('status', 'error')
                result['details'] = build_result
                result['details']['removed_from_vector'] = len(extra_in_vector)
                
                # Graph DB: 可以單獨刪除
                if len(extra_in_graph) > 0 and self.index_manager.graph_manager:
                    logger.info(f"Removing {len(extra_in_graph)} extra papers from Graph DB...")
                    removed_count = 0
                    for paper_id in extra_in_graph:
                        success = self.index_manager.graph_manager.delete_paper(paper_id)
                        if success:
                            removed_count += 1
                    logger.info(f"✓ Removed {removed_count}/{len(extra_in_graph)} papers from Graph DB")
                    result['details']['removed_from_graph'] = removed_count
                
                return result
            
            # 檢查 4: 處理新增（只有新增的論文）
            if has_additions:
                logger.info(f"✓ Detected {len(missing_in_vector)} new PDF(s) - using incremental update")
                
                # 顯示要新增的論文
                for paper_id in list(missing_in_vector)[:5]:
                    logger.info(f"  + {paper_id}")
                if len(missing_in_vector) > 5:
                    logger.info(f"  ... and {len(missing_in_vector) - 5} more")
                
                result['action_taken'] = 'incremental_update'
                
                # 增量更新：處理新增的 PDF
                added_to_vector = 0
                added_to_graph = 0
                failed_count = 0
                start_time = time.time()
                
                for i, paper_id in enumerate(sorted(missing_in_vector), 1):
                    try:
                        pdf_path = pdf_path_to_id.get(paper_id)
                        if not pdf_path:
                            logger.warning(f"  Cannot find PDF path for {paper_id}")
                            failed_count += 1
                            continue
                        
                        logger.info(f"[{i}/{len(missing_in_vector)}] Adding: {os.path.basename(pdf_path)}")
                        
                        # 新增到 Vector DB (自動包含 Layer1 + Layer2)
                        add_result = self.add_document(pdf_path)
                        if add_result['status'] == 'success':
                            added_to_vector += 1
                            logger.info(f"  ✓ Added to Vector DB")
                        else:
                            failed_count += 1
                            logger.warning(f"  ✗ Failed to add to Vector DB: {add_result.get('error', 'Unknown')}")
                            continue
                        
                        # 新增到 Graph DB (如果配置了 graph_manager)
                        if self.index_manager.graph_manager and self.index_manager.graph_extractor:
                            try:
                                # 提取 PDF 文本
                                from langchain_community.document_loaders import PyMuPDFLoader
                                loader = PyMuPDFLoader(pdf_path)
                                pages = loader.load()
                                pdf_text = "\n".join([page.page_content for page in pages])
                                
                                # 提取結構化數據
                                graph_data = self.index_manager.graph_extractor.extract(pdf_text)
                                
                                # 寫入 Neo4j
                                success = self.index_manager.graph_manager.add_paper_metadata(
                                    paper_id=paper_id,
                                    title=os.path.basename(pdf_path),
                                    year=graph_data.get('year', 'Unknown'),
                                    research_goal=graph_data.get('research_goal', ''),
                                    methods=graph_data.get('methods', []),
                                    datasets=graph_data.get('datasets', []),
                                    domain=graph_data.get('domain', 'Unknown'),
                                    metrics=graph_data.get('metrics', []),
                                    domain_zh=graph_data.get('domain_zh', '未知領域')
                                )
                                
                                if success:
                                    added_to_graph += 1
                                    logger.info(f"  ✓ Added to Graph DB")
                                else:
                                    logger.warning(f"  ⚠️ Failed to add to Graph DB")
                            except Exception as e:
                                logger.error(f"  ✗ Error adding to Graph DB: {e}")
                        
                    except Exception as e:
                        failed_count += 1
                        logger.error(f"  ✗ Error processing {paper_id}: {e}")
                
                duration = time.time() - start_time
                
                # 更新統計
                stats1 = self.layer1.get_stats()
                stats2 = self.layer2.get_stats()
                
                result['status'] = 'success'
                result['details'] = {
                    'new_papers_detected': len(missing_in_vector),
                    'added_to_vector': added_to_vector,
                    'added_to_graph': added_to_graph,
                    'failed': failed_count,
                    'duration': duration,
                    'layer1': stats1,
                    'layer2': stats2
                }
                
                logger.info("="*80)
                logger.info("Incremental Update Complete")
                logger.info(f"  Duration: {duration:.1f}s")
                logger.info(f"  Added to Vector DB: {added_to_vector}/{len(missing_in_vector)}")
                logger.info(f"  Added to Graph DB: {added_to_graph}/{len(missing_in_vector)}")
                if failed_count > 0:
                    logger.warning(f"  Failed: {failed_count}")
                logger.info(f"  Total Papers (Vector): {stats1['paper_count']}")
                logger.info(f"  Total Chunks (Vector): {stats2['chunk_count']}")
                logger.info("="*80)
                
                return result
            
            # 檢查 5: 全部檢查通過，所有資料庫已同步
            logger.info("✓ All databases are in sync and up-to-date")
            stats1 = self.layer1.get_stats()
            stats2 = self.layer2.get_stats()
            
            result['details'] = {
                'layer1': stats1,
                'layer2': stats2,
                'pdf_count': len(current_paper_ids),
                'vector_count': len(indexed_in_vector),
                'graph_count': len(indexed_in_graph)
            }
            
        except Exception as e:
            logger.error(f"Error checking indices: {e}", exc_info=True)
            result['status'] = 'error'
            result['details']['error'] = str(e)
            
            # 嘗試重建
            logger.info("Attempting to rebuild indices due to error...")
            result['action_taken'] = 'rebuild_due_to_error'
            build_result = self.build_indices()
            result['status'] = build_result.get('status', 'error')
            result['details']['rebuild'] = build_result
        
        return result

