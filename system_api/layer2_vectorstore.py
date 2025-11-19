"""
Layer 2 VectorStore Module (Re-ranking Only)

使用 Cross-Encoder Re-ranking 進行 chunk-level 檢索。
完全捨棄 FAISS 向量搜索，只使用語義 re-ranking。
"""

import os
import logging
import time
from typing import List, Dict, Optional, Any, Tuple
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class Layer2VectorStore:
    """
    Layer 2 (chunk-level) 檢索系統 - 純 Re-ranking 模式
    
    特性：
    - Cross-Encoder 語義 re-ranking
    - JSONL 文檔存儲（人類可讀）
    - 按 paper_id 過濾
    - 上下文檢索（surrounding chunks）
    """
    
    def __init__(
        self,
        embeddings: OllamaEmbeddings,
        vectorstore_path: str = "./vectorstore/layer2",
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        cache_size: int = 10,
        use_reranker: bool = True,  # 保留參數以兼容舊代碼，但強制 True
        reranker_config: Optional[Dict[str, Any]] = None,
        chunking_config: Optional[Dict[str, Any]] = None  # NEW: Summarization support
    ):
        """
        初始化 Layer 2 VectorStore (Re-ranking only)
        
        Args:
            embeddings: Embedding model（保留以兼容，但不使用）
            vectorstore_path: 存儲目錄
            chunk_size: Chunk 大小
            chunk_overlap: Chunk 重疊
            cache_size: 快取大小（保留以兼容）
            use_reranker: 是否使用 re-ranker（強制 True）
            reranker_config: Re-ranker 配置
            chunking_config: Chunking configuration (NEW: supports 'naive' or 'summarization' mode)
        """
        self.embeddings = embeddings  # 保留以兼容
        self.vectorstore_path = vectorstore_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_reranker = True  # 強制啟用
        
        # Parse chunking config
        self.chunking_config = chunking_config or {}
        self.chunking_mode = self.chunking_config.get('mode', 'naive')  # Default: naive (backward compatible)
        
        # 文件路徑
        self.jsonl_file = os.path.join(vectorstore_path, "chunks.jsonl")
        self.metadata_file = os.path.join(vectorstore_path, "metadata.pkl")
        
        # 統計
        self._chunk_count = 0
        self._paper_count = 0
        
        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Re-ranking 組件
        self._reranker = None
        self._document_store = None
        self._reranker_config = reranker_config or {}
        
        # Summarization components (if enabled)
        self.section_parser = None
        self.summarizer = None
        if self.chunking_mode == 'summarization':
            self._init_summarization_components()
        
        # 初始化
        os.makedirs(vectorstore_path, exist_ok=True)
        self._init_reranking_components()
        
        logger.info(f"✓ Layer 2 VectorStore initialized (chunking_mode={self.chunking_mode})")
    
    def _init_reranking_components(self):
        """初始化 Cross-Encoder re-ranker 和 document store"""
        try:
            from system_api.cross_encoder_reranker import CrossEncoderReranker
            from system_api.layer2_document_store import JSONLDocumentStore
            
            # Cross-Encoder
            model_name = self._reranker_config.get('model', 'qllama/bce-reranker-base_v1:latest')
            ollama_base_url = self._reranker_config.get('ollama_base_url', 'http://localhost:11434')
            max_length = self._reranker_config.get('max_length', 512)
            timeout = self._reranker_config.get('timeout', 30)
            
            self._reranker = CrossEncoderReranker(
                model_name=model_name,
                ollama_base_url=ollama_base_url,
                max_length=max_length,
                timeout=timeout
            )
            
            # Document store
            cache_limit_mb = self._reranker_config.get('cache_limit_mb', 100)
            self._document_store = JSONLDocumentStore(
                file_path=self.jsonl_file,
                cache_size_limit_mb=cache_limit_mb
            )
            
            # 載入現有數據統計
            self._update_statistics()
            
            logger.info("✓ Re-ranking components initialized")
            logger.info(f"  Model: {model_name}")
            logger.info(f"  Document store: {self.jsonl_file}")
            logger.info(f"  Chunks: {self._chunk_count}, Papers: {self._paper_count}")
            
        except Exception as e:
            logger.error(f"Failed to initialize re-ranking components: {e}", exc_info=True)
            raise RuntimeError("Cannot initialize Layer 2 without re-ranking components")
    
    def _init_summarization_components(self):
        """Initialize PDF section parser and LLM summarizer"""
        try:
            from system_api.pdf_section_parser import PDFSectionParser
            from system_api.llm_summarizer import LLMSummarizer
            
            # Section parser
            parser_method = self.chunking_config.get('section_parser', 'pymupdf_regex')
            min_sections = self.chunking_config.get('min_sections', 3)
            self.section_parser = PDFSectionParser(method=parser_method, min_sections=min_sections)
            
            # LLM summarizer
            summarizer_model = self.chunking_config.get('model', 'jcai/llama-3-taiwan-8b-instruct:q4_k_m')
            ollama_base_url = self.chunking_config.get('ollama_base_url', 'http://localhost:11434')
            map_reduce_threshold = self.chunking_config.get('map_reduce_threshold', 1500)
            target_summary_length = self.chunking_config.get('target_summary_length', 300)
            
            self.summarizer = LLMSummarizer(
                model_name=summarizer_model,
                ollama_base_url=ollama_base_url,
                map_reduce_threshold=map_reduce_threshold,
                target_summary_length=target_summary_length
            )
            
            logger.info("✓ Summarization components initialized")
            logger.info(f"  Parser: {parser_method}, Summarizer: {summarizer_model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize summarization components: {e}", exc_info=True)
            logger.warning("Falling back to naive chunking mode")
            self.chunking_mode = 'naive'
    
    def _update_statistics(self):
        """更新 chunk 和 paper 統計"""
        try:
            if not os.path.exists(self.jsonl_file):
                self._chunk_count = 0
                self._paper_count = 0
                return
            
            # 使用 document_store 的專用方法（更高效）
            if self._document_store:
                self._chunk_count = self._document_store.get_chunk_count()
                self._paper_count = self._document_store.get_paper_count()
            else:
                # 後備方案：直接讀取檔案計算
                logger.warning("Document store not available, using fallback statistics calculation")
                paper_ids = set()
                chunk_count = 0
                with open(self.jsonl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            import json
                            chunk_data = json.loads(line)
                            chunk_count += 1
                            paper_id = chunk_data.get('paper_id') or chunk_data.get('metadata', {}).get('paper_id')
                            if paper_id:
                                paper_ids.add(paper_id)
                        except Exception:
                            continue
                
                self._chunk_count = chunk_count
                self._paper_count = len(paper_ids)
                
        except Exception as e:
            logger.warning(f"Failed to update statistics: {e}")
            self._chunk_count = 0
            self._paper_count = 0
    
    @property
    def is_initialized(self) -> bool:
        """檢查索引是否已初始化"""
        return os.path.exists(self.jsonl_file) and self._chunk_count > 0
    
    def build_index(self, documents: List[Document]) -> bool:
        """
        建構索引（存儲所有 chunks 到 JSONL）
        
        Dispatches to either naive or summarization mode based on config.
        
        Args:
            documents: 文檔列表（每個 document 是一個 chunk）
            
        Returns:
            True if successful
        """
        if self.chunking_mode == 'summarization':
            return self._build_index_with_summarization(documents)
        else:
            return self._build_index_naive(documents)
    
    def _build_index_naive(self, documents: List[Document]) -> bool:
        """
        Build index using naive chunking (original behavior)
        
        Args:
            documents: Document list (each is a chunk)
            
        Returns:
            True if successful
        """
        try:
            if not documents:
                logger.warning("No documents provided to build index")
                return False
            
            logger.info(f"Building Layer 2 index (naive mode) with {len(documents)} chunks...")
            start_time = time.time()
            
            # 驗證
            valid_docs = []
            for doc in documents:
                if not doc.page_content or len(doc.page_content) < 50:
                    logger.warning(f"Skipping chunk - content too short")
                    continue
                if 'paper_id' not in doc.metadata:
                    logger.warning(f"Skipping chunk - missing paper_id")
                    continue
                valid_docs.append(doc)
            
            if not valid_docs:
                logger.error("No valid documents after validation")
                return False
            
            # 存儲到 JSONL
            success = self._document_store.store_chunks(valid_docs)
            
            if success:
                self._update_statistics()
                elapsed = time.time() - start_time
                logger.info(f"✓ Index built in {elapsed:.2f}s")
                logger.info(f"  Chunks: {self._chunk_count}, Papers: {self._paper_count}")
                return True
            else:
                logger.error("Failed to store chunks")
                return False
                
        except Exception as e:
            logger.error(f"Failed to build index (naive): {e}", exc_info=True)
            return False
    
    def _build_index_with_summarization(self, documents: List[Document]) -> bool:
        """
        Build index using summarization strategy
        
        Strategy:
        1. Group documents by paper_id
        2. For each paper:
           - Get PDF path from metadata
           - Parse sections (PDFSectionParser)
           - Summarize sections (LLMSummarizer)
           - Create Document objects with enhanced metadata
        3. Store in JSONL
        
        Args:
            documents: Document list (can be raw or pre-chunked)
            
        Returns:
            True if successful
        """
        try:
            if not documents:
                logger.warning("No documents provided to build index")
                return False
            
            logger.info(f"Building Layer 2 index (summarization mode) with {len(documents)} docs...")
            start_time = time.time()
            
            # Group by paper_id
            papers = {}
            for doc in documents:
                paper_id = doc.metadata.get('paper_id')
                if not paper_id:
                    logger.warning("Skipping document - missing paper_id")
                    continue
                if paper_id not in papers:
                    papers[paper_id] = []
                papers[paper_id].append(doc)
            
            logger.info(f"Processing {len(papers)} papers for summarization")
            
            # Process each paper
            summarized_chunks = []
            for paper_idx, (paper_id, paper_docs) in enumerate(papers.items(), 1):
                logger.info(f"[{paper_idx}/{len(papers)}] Processing paper {paper_id}")
                
                # Get PDF path
                pdf_path = paper_docs[0].metadata.get('pdf_path')
                if not pdf_path or not os.path.exists(pdf_path):
                    logger.warning(f"PDF not found for {paper_id}, falling back to naive chunking")
                    # Fallback: Use naive chunks for this paper
                    for doc in paper_docs:
                        if len(doc.page_content) >= 50:
                            doc.metadata['chunk_type'] = 'naive_chunk'
                            summarized_chunks.append(doc)
                    continue
                
                try:
                    # Parse sections
                    sections = self.section_parser.parse(pdf_path)
                    logger.info(f"  Extracted {len(sections)} sections")
                    
                    # Get paper context
                    paper_title = paper_docs[0].metadata.get('title', '')
                    
                    # Define sections to skip (matching Layer 1's text_preprocessor.py logic)
                    # These are the same sections that Layer 1 removes via text preprocessing
                    # Based on analysis of 45 PDFs (475 skip section headers found)
                    skip_section_keywords = [
                        # References (参考文献) - 315 occurrences
                        'bibliography', 'reference', 'references',
                        '参考文献', '參考文獻', '引用文獻', '文獻',
                        '文獻回顧', '文獻探討',  # Common in Chinese theses (17x + 9x)
                        
                        # Appendix (附录) - 3 occurrences
                        'appendix', 'appendices', '附录', '附錄',
                        
                        # Acknowledgements (致谢) - 32 occurrences
                        'acknowledgement', 'acknowledgements', 'acknowledgment',
                        '致謝', '誌謝', '謝誌', '謝辭',
                        
                        # Table of Contents (目录) - 111 occurrences
                        'contents', 'table of contents', '目录', '目錄',
                        '圖片目錄', '表格目錄',  # 1x each
                        
                        # List of Tables/Figures (表/图目录) - 34 occurrences each
                        'list of tables', '表目录', '表目錄',
                        'list of figures', '图目录', '圖目錄'
                    ]
                    
                    # Summarize each section
                    for section_idx, section in enumerate(sections, 1):
                        # Skip low-value sections (same as Layer 1 preprocessing)
                        section_name_lower = section.name.lower()
                        if any(keyword.lower() in section_name_lower for keyword in skip_section_keywords):
                            logger.info(f"  [{section_idx}/{len(sections)}] Skipping {section.name} (filtered by Layer 1 logic)")
                            continue
                        
                        # ✨ 檢測 Dataset 章節 - 基於真實分析結果（2025-11-19）
                        # 分析 35 篇論文，發現 Dataset 章節的命名模式：
                        # - "第三章 資料集" / "第三章 資料集與初步整理" (中文章節)
                        # - "5.1 資料集介紹" / "6.1 資料集與實驗參數介紹" (中文小節)
                        # - "Chapter 3 Dataset" / "4.1 Dataset" (英文)
                        is_dataset_section = (
                            '資料集' in section_name_lower or              # 最常見：中文「資料集」
                            '数据集' in section_name_lower or              # 簡體
                            'dataset' in section_name_lower or             # 英文 dataset
                            'data collection' in section_name_lower or     # 英文 data collection
                            ('data' in section_name_lower and              # 避免誤判：只有當 data 獨立出現時
                             (section_name_lower.strip() == 'data' or      # 單獨 "Data"
                              section_name_lower.startswith('data ') or    # "Data Introduction"
                              section_name_lower.endswith(' data')))       # "Training Data"
                        )
                        
                        if is_dataset_section:
                            logger.info(f"  [{section_idx}/{len(sections)}] 🔍 Dataset section detected: {section.name}")
                        else:
                            logger.info(f"  [{section_idx}/{len(sections)}] Summarizing {section.name}...")
                        
                        try:
                            # 根據章節類型選擇不同的摘要策略
                            if is_dataset_section:
                                # Dataset 專用 prompt：強調提取數據集關鍵信息
                                summary = self.summarizer.summarize_dataset_section(
                                    section.text,
                                    section.name,
                                    paper_context=paper_title
                                )
                                chunk_type = 'dataset_chunk'
                                logger.info(f"  ✓ Dataset summary: {len(section.text)} → {len(summary)} chars")
                            else:
                                # 一般章節使用標準摘要
                                summary = self.summarizer.summarize_section(
                                    section.text,
                                    section.name,
                                    paper_context=paper_title
                                )
                                chunk_type = 'section_summary'
                                logger.info(f"  ✓ Summary: {len(section.text)} → {len(summary)} chars")
                            
                            # Create Document with enhanced metadata
                            chunk_doc = Document(
                                page_content=summary,
                                metadata={
                                    'paper_id': paper_id,
                                    'chunk_id': f"{paper_id}_{section.name.lower().replace(' ', '_')}",
                                    'chunk_type': chunk_type,  # ✨ 區分 dataset_chunk 和 section_summary
                                    'section_name': section.name,
                                    'original_length': section.char_count,
                                    'summary_length': len(summary),
                                    'start_page': section.start_page,
                                    'end_page': section.end_page,
                                    **{k: v for k, v in paper_docs[0].metadata.items() 
                                       if k not in ['chunk_id', 'chunk_type', 'section_name']}  # Inherit paper metadata
                                }
                            )
                            summarized_chunks.append(chunk_doc)
                            
                            
                        except Exception as e:
                            logger.error(f"  Failed to summarize {section.name}: {e}")
                            # Fallback: Use extractive summary
                            extractive = section.text[:300] + "..." if len(section.text) > 300 else section.text
                            chunk_doc = Document(
                                page_content=extractive,
                                metadata={
                                    'paper_id': paper_id,
                                    'chunk_id': f"{paper_id}_{section.name.lower().replace(' ', '_')}",
                                    'chunk_type': 'extractive_summary',
                                    'section_name': section.name,
                                    'original_length': section.char_count,
                                    **paper_docs[0].metadata
                                }
                            )
                            summarized_chunks.append(chunk_doc)
                
                except Exception as e:
                    logger.error(f"Failed to process paper {paper_id}: {e}")
                    # Fallback: Use naive chunks
                    for doc in paper_docs:
                        if len(doc.page_content) >= 50:
                            doc.metadata['chunk_type'] = 'naive_chunk'
                            summarized_chunks.append(doc)
            
            if not summarized_chunks:
                logger.error("No summarized chunks generated")
                return False
            
            # Store in JSONL
            logger.info(f"Storing {len(summarized_chunks)} summarized chunks...")
            success = self._document_store.store_chunks(summarized_chunks)
            
            if success:
                self._update_statistics()
                elapsed = time.time() - start_time
                logger.info(f"✓ Index built in {elapsed:.2f}s")
                logger.info(f"  Chunks: {self._chunk_count}, Papers: {self._paper_count}")
                logger.info(f"  Average: {elapsed/len(papers):.1f}s per paper")
                return True
            else:
                logger.error("Failed to store chunks")
                return False
                
        except Exception as e:
            logger.error(f"Failed to build index (summarization): {e}", exc_info=True)
            return False
    
    def add_chunks(self, chunks: List[Document]) -> bool:
        """
        添加新 chunks 到現有索引
        
        Args:
            chunks: 要添加的 chunks
            
        Returns:
            True if successful
        """
        try:
            if not chunks:
                logger.warning("No chunks to add")
                return True
            
            logger.info(f"Adding {len(chunks)} chunks to index...")
            
            # 驗證
            valid_chunks = []
            for chunk in chunks:
                if not chunk.page_content or len(chunk.page_content) < 50:
                    logger.warning("Skipping chunk - content too short")
                    continue
                if 'paper_id' not in chunk.metadata:
                    logger.warning("Skipping chunk - missing paper_id")
                    continue
                valid_chunks.append(chunk)
            
            if not valid_chunks:
                logger.warning("No valid chunks to add")
                return True
            
            # 讀取現有 chunks
            existing_chunks = self._document_store.get_chunks_by_paper_ids(None)
            
            # 轉換新 chunks 為 dict
            new_chunk_dicts = [
                {'text': doc.page_content, 'metadata': doc.metadata}
                for doc in valid_chunks
            ]
            
            # 合併
            all_chunks_data = existing_chunks + new_chunk_dicts
            
            # 轉回 Documents 並存儲
            all_docs = [
                Document(page_content=chunk['text'], metadata=chunk['metadata'])
                for chunk in all_chunks_data
            ]
            
            success = self._document_store.store_chunks(all_docs)
            
            if success:
                self._update_statistics()
                logger.info(f"✓ Added {len(valid_chunks)} chunks")
                logger.info(f"  Total chunks: {self._chunk_count}")
                return True
            else:
                logger.error("Failed to store chunks")
                return False
                
        except Exception as e:
            logger.error(f"Failed to add chunks: {e}", exc_info=True)
            return False
    
    def search(
        self,
        query: str,
        k: int = 10,
        filter_paper_ids: Optional[List[str]] = None,
        paper_ids: Optional[List[str]] = None  # 向後兼容
    ) -> List[Document]:
        """
        搜索相關 chunks（使用 re-ranking）
        
        Args:
            query: 查詢文本
            k: 返回結果數量
            filter_paper_ids: 過濾的 paper IDs（None = 搜索全部）
            paper_ids: 別名，向後兼容
            
        Returns:
            相關文檔列表
        """
        # 向後兼容：如果提供了 paper_ids，使用它
        filter_ids = filter_paper_ids if filter_paper_ids is not None else paper_ids
        results = self.search_with_scores(query, k, filter_ids)
        return [doc for doc, score in results]
    
    def search_with_scores(
        self,
        query: str,
        k: int = 10,
        filter_paper_ids: Optional[List[str]] = None,
        paper_ids: Optional[List[str]] = None  # 向後兼容
    ) -> List[Tuple[Document, float]]:
        """
        搜索相關 chunks 並返回分數
        
        Args:
            query: 查詢文本
            k: 返回結果數量
            filter_paper_ids: 過濾的 paper IDs（None = 搜索全部）
            paper_ids: 別名，向後兼容
            
        Returns:
            (Document, score) 列表，按分數降序排列
        """
        # 向後兼容：如果提供了 paper_ids，使用它
        filter_ids = filter_paper_ids if filter_paper_ids is not None else paper_ids
        try:
            start_time = time.time()
            
            # 載入候選 chunks
            candidate_chunks = self._document_store.get_chunks_by_paper_ids(filter_ids)
            
            if not candidate_chunks:
                logger.warning(f"No chunks found for papers: {filter_ids}")
                return []
            
            # 限制候選數量以避免太慢
            max_candidates = self._reranker_config.get('max_candidates', 300)
            if len(candidate_chunks) > max_candidates:
                logger.info(f"Limiting candidates from {len(candidate_chunks)} to {max_candidates}")
                candidate_chunks = candidate_chunks[:max_candidates]
            
            # Re-rank（rank_chunks 需要 dict 格式）
            reranked_results = self._reranker.rank_chunks(
                query=query,
                chunks=candidate_chunks,  # 直接使用 dict 格式
                top_k=k
            )
            
            # 轉換結果為 (Document, score) 格式
            results_with_docs = [
                (Document(page_content=chunk_dict['text'], metadata=chunk_dict['metadata']), score)
                for chunk_dict, score in reranked_results
            ]
            reranked_results = results_with_docs
            
            elapsed = time.time() - start_time
            logger.info(f"Re-ranking search completed in {elapsed:.3f}s")
            logger.info(f"  Candidates: {len(candidate_chunks)}, Returned: {len(reranked_results)}")
            
            return reranked_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            return []
    
    def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[Document]:
        """
        根據 chunk_id 列表檢索特定的 chunks
        
        Args:
            chunk_ids: Chunk ID 列表（例如 ["paper1_chunk_0", "paper2_chunk_1"]）
            
        Returns:
            找到的 Document 列表
        """
        if not self.is_initialized:
            logger.warning("Index not initialized")
            return []
        
        try:
            found_docs = []
            
            # Read all chunks from JSONL (pass None to get all)
            all_chunks = self._document_store.get_chunks_by_paper_ids(None)
            
            # Filter by chunk_ids
            for chunk in all_chunks:
                chunk_id = chunk.get('chunk_id')
                if chunk_id in chunk_ids:
                    doc = Document(
                        page_content=chunk['text'],
                        metadata=chunk
                    )
                    found_docs.append(doc)
            
            logger.debug(f"Found {len(found_docs)}/{len(chunk_ids)} chunks by IDs")
            return found_docs
            
        except Exception as e:
            logger.error(f"Failed to get chunks by IDs: {e}")
            return []
    
    def get_surrounding_context(
        self,
        target_chunk: Document,
        window_size: int = 1
    ) -> List[Document]:
        """
        獲取目標 chunk 周圍的上下文
        
        Args:
            target_chunk: 目標 chunk
            window_size: 前後各取多少個 chunks
            
        Returns:
            上下文 chunks（包含目標 chunk）
        """
        try:
            paper_id = target_chunk.metadata.get('paper_id')
            chunk_id = target_chunk.metadata.get('chunk_id')
            
            if not paper_id or chunk_id is None:
                logger.warning("Cannot get context - missing paper_id or chunk_id")
                return [target_chunk]
            
            # 獲取同一篇論文的所有 chunks
            paper_chunks = self._document_store.get_chunks_by_paper_ids([paper_id])
            
            if not paper_chunks:
                return [target_chunk]
            
            # 轉換並排序
            docs = [
                Document(page_content=c['text'], metadata=c['metadata'])
                for c in paper_chunks
            ]
            
            # 排序函數：從 chunk_id 字串中提取數字
            def extract_chunk_index(doc):
                chunk_id_str = doc.metadata.get('chunk_id', '')
                if isinstance(chunk_id_str, str) and '_chunk_' in chunk_id_str:
                    try:
                        return int(chunk_id_str.split('_chunk_')[-1])
                    except:
                        return 0
                elif isinstance(chunk_id_str, int):
                    return chunk_id_str
                return 0
            
            docs.sort(key=extract_chunk_index)
            
            # 找到目標位置
            target_idx = None
            for idx, doc in enumerate(docs):
                if doc.metadata.get('chunk_id') == chunk_id:
                    target_idx = idx
                    break
            
            if target_idx is None:
                logger.warning(f"Target chunk not found in paper {paper_id}")
                return [target_chunk]
            
            # 取窗口
            start_idx = max(0, target_idx - window_size)
            end_idx = min(len(docs), target_idx + window_size + 1)
            
            context_chunks = docs[start_idx:end_idx]
            logger.debug(f"Retrieved {len(context_chunks)} context chunks (window={window_size})")
            
            return context_chunks
            
        except Exception as e:
            logger.error(f"Failed to get surrounding context: {e}", exc_info=True)
            return [target_chunk]

    def get_surrounding_chunks(
        self,
        paper_id: str = None,
        chunk_id: int = None,
        before: int = 1,
        after: int = 1,
        target_chunk: Document = None,
        window_size: int = None
    ) -> str:
        """
        兼容舊介面：獲取周圍 chunks 並返回拼接的文本
        
        支援兩種調用方式：
        1. get_surrounding_chunks(paper_id='xxx', chunk_id=0, before=2, after=2)
        2. get_surrounding_chunks(target_chunk=doc, window_size=2)
        
        Args:
            paper_id: Paper ID (舊接口)
            chunk_id: Chunk ID (舊接口)
            before: 前面取多少個 chunks (舊接口)
            after: 後面取多少個 chunks (舊接口)
            target_chunk: 目標 Document (新接口)
            window_size: 窗口大小 (新接口)
            
        Returns:
            拼接後的文本字串
        """
        # 如果使用新接口
        if target_chunk is not None:
            if window_size is None:
                window_size = 1
            context_docs = self.get_surrounding_context(target_chunk, window_size)
            return "\n\n".join(doc.page_content for doc in context_docs)
        
        # 使用舊接口
        if not paper_id or chunk_id is None:
            logger.warning("Missing paper_id or chunk_id")
            return ""
        
        try:
            # 獲取該論文的所有 chunks
            paper_chunks = self._document_store.get_chunks_by_paper_ids([paper_id])
            
            if not paper_chunks:
                return ""
            
            # 轉換並排序
            docs = [
                Document(page_content=c['text'], metadata=c['metadata'])
                for c in paper_chunks
            ]
            
            # 排序函數：從 chunk_id 字串中提取數字
            def extract_chunk_index(doc):
                chunk_id_str = doc.metadata.get('chunk_id', '')
                if isinstance(chunk_id_str, str) and '_chunk_' in chunk_id_str:
                    try:
                        return int(chunk_id_str.split('_chunk_')[-1])
                    except:
                        return 0
                elif isinstance(chunk_id_str, int):
                    return chunk_id_str
                return 0
            
            docs.sort(key=extract_chunk_index)
            
            # 找到目標位置（chunk_id 可能是整數或字串）
            target_idx = None
            for idx, doc in enumerate(docs):
                doc_chunk_id = doc.metadata.get('chunk_id', '')
                # 支援兩種格式：整數或 "paper_id_chunk_N"
                if doc_chunk_id == chunk_id or extract_chunk_index(doc) == chunk_id:
                    target_idx = idx
                    break
            
            if target_idx is None:
                logger.warning(f"Chunk {chunk_id} not found in paper {paper_id}")
                return ""
            
            # 取窗口
            start_idx = max(0, target_idx - before)
            end_idx = min(len(docs), target_idx + after + 1)
            
            context_chunks = docs[start_idx:end_idx]
            
            # 拼接文本
            return "\n\n".join(doc.page_content for doc in context_chunks)
            
        except Exception as e:
            logger.error(f"Failed to get surrounding chunks: {e}", exc_info=True)
            return ""
    
    def load_index(self) -> bool:
        """
        載入現有索引（更新統計信息）
        
        Returns:
            True if index exists and loaded
        """
        try:
            if not os.path.exists(self.jsonl_file):
                logger.warning(f"Index file not found: {self.jsonl_file}")
                return False
            
            # 確保 document_store 已初始化
            if self._document_store is None:
                logger.warning("Document store not initialized, attempting to initialize...")
                self._init_reranking_components()
            
            # 更新統計
            self._update_statistics()
            
            # 驗證載入成功
            if self._chunk_count == 0:
                logger.warning("Index file exists but contains no chunks")
                # 嘗試直接計算檔案行數作為後備
                try:
                    with open(self.jsonl_file, 'r', encoding='utf-8') as f:
                        line_count = sum(1 for _ in f)
                    if line_count > 0:
                        logger.info(f"  File contains {line_count} lines, re-initializing statistics...")
                        self._update_statistics()
                except Exception as count_error:
                    logger.error(f"Failed to count lines: {count_error}")
            
            if self._chunk_count > 0:
                logger.info("✓ Index loaded successfully")
                logger.info(f"  Chunks: {self._chunk_count}, Papers: {self._paper_count}")
                return True
            else:
                logger.warning("Failed to load index: no chunks found")
                return False
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}", exc_info=True)
            return False
    
    def save_index(self) -> bool:
        """
        保存索引（JSONL 自動保存，這裡只是更新統計）
        
        Returns:
            True if successful
        """
        try:
            self._update_statistics()
            logger.info("✓ Index statistics updated")
            return True
        except Exception as e:
            logger.error(f"Failed to save index: {e}", exc_info=True)
            return False
    
    # 向後兼容的別名方法
    def load(self) -> bool:
        """別名方法：向後兼容 HierarchicalRAGSystem"""
        return self.load_index()
    
    def save(self) -> bool:
        """別名方法：向後兼容 HierarchicalRAGSystem"""
        return self.save_index()
    
    def get_stats(self) -> Dict[str, Any]:
        """獲取統計信息"""
        return {
            'chunk_count': self._chunk_count,
            'paper_count': self._paper_count,
            'is_initialized': self.is_initialized,  # 添加 is_initialized 欄位
            'mode': 're-ranking',
            'document_store': self.jsonl_file,
            'reranker_model': self._reranker_config.get('model', 'qllama/bce-reranker-base_v1:latest')
        }
