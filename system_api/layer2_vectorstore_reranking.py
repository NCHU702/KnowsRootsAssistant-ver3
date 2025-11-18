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
        reranker_config: Optional[Dict[str, Any]] = None
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
        """
        self.embeddings = embeddings  # 保留以兼容
        self.vectorstore_path = vectorstore_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_reranker = True  # 強制啟用
        
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
        
        # 初始化
        os.makedirs(vectorstore_path, exist_ok=True)
        self._init_reranking_components()
        
        logger.info("✓ Layer 2 VectorStore initialized (Re-ranking only mode)")
    
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
    
    def _update_statistics(self):
        """更新 chunk 和 paper 統計"""
        try:
            if os.path.exists(self.jsonl_file):
                all_chunks = self._document_store.get_chunks_by_paper_ids(None)
                self._chunk_count = len(all_chunks)
                paper_ids = set(chunk['metadata'].get('paper_id') for chunk in all_chunks 
                               if 'paper_id' in chunk.get('metadata', {}))
                self._paper_count = len(paper_ids)
        except Exception as e:
            logger.warning(f"Failed to update statistics: {e}")
            self._chunk_count = 0
            self._paper_count = 0
    
    def build_index(self, documents: List[Document]) -> bool:
        """
        建構索引（存儲所有 chunks 到 JSONL）
        
        Args:
            documents: 文檔列表（每個 document 是一個 chunk）
            
        Returns:
            True if successful
        """
        try:
            if not documents:
                logger.warning("No documents provided to build index")
                return False
            
            logger.info(f"Building Layer 2 index with {len(documents)} chunks...")
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
            logger.error(f"Failed to build index: {e}", exc_info=True)
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
        paper_ids: Optional[List[str]] = None
    ) -> List[Document]:
        """
        搜索相關 chunks（使用 re-ranking）
        
        Args:
            query: 查詢文本
            k: 返回結果數量
            paper_ids: 過濾的 paper IDs（None = 搜索全部）
            
        Returns:
            相關文檔列表
        """
        results = self.search_with_scores(query, k, paper_ids)
        return [doc for doc, score in results]
    
    def search_with_scores(
        self,
        query: str,
        k: int = 10,
        paper_ids: Optional[List[str]] = None
    ) -> List[Tuple[Document, float]]:
        """
        搜索相關 chunks 並返回分數
        
        Args:
            query: 查詢文本
            k: 返回結果數量
            paper_ids: 過濾的 paper IDs（None = 搜索全部）
            
        Returns:
            (Document, score) 列表，按分數降序排列
        """
        try:
            start_time = time.time()
            
            # 載入候選 chunks
            candidate_chunks = self._document_store.get_chunks_by_paper_ids(paper_ids)
            
            if not candidate_chunks:
                logger.warning(f"No chunks found for papers: {paper_ids}")
                return []
            
            # 限制候選數量以避免太慢
            max_candidates = self._reranker_config.get('max_candidates', 300)
            if len(candidate_chunks) > max_candidates:
                logger.info(f"Limiting candidates from {len(candidate_chunks)} to {max_candidates}")
                candidate_chunks = candidate_chunks[:max_candidates]
            
            # 轉換為 Documents
            candidate_docs = [
                Document(page_content=chunk['text'], metadata=chunk['metadata'])
                for chunk in candidate_chunks
            ]
            
            # Re-rank
            reranked_results = self._reranker.rerank(
                query=query,
                documents=candidate_docs,
                top_k=k
            )
            
            elapsed = time.time() - start_time
            logger.info(f"Re-ranking search completed in {elapsed:.3f}s")
            logger.info(f"  Candidates: {len(candidate_chunks)}, Returned: {len(reranked_results)}")
            
            return reranked_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
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
            docs.sort(key=lambda d: d.metadata.get('chunk_id', 0))
            
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
            
            self._update_statistics()
            
            logger.info("✓ Index loaded")
            logger.info(f"  Chunks: {self._chunk_count}, Papers: {self._paper_count}")
            return True
            
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
    
    def get_stats(self) -> Dict[str, Any]:
        """獲取統計信息"""
        return {
            'chunk_count': self._chunk_count,
            'paper_count': self._paper_count,
            'mode': 're-ranking',
            'document_store': self.jsonl_file,
            'reranker_model': self._reranker_config.get('model', 'qllama/bce-reranker-base_v1:latest')
        }
