import os
import logging
import time
import re
import hashlib
import pickle
from typing import List, Dict, Any, Optional
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)


class AcademicRAGSystem:
    """RAG system for querying academic papers from PDF collection"""
    
    def __init__(
        self,
        pdf_directory: str = "./data",
        model_name: str = "gemma3:270m",
        embedding_model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k_documents: int = 4,
        vectorstore_path: str = "./vectorstore"
    ):
        """
        Initialize the RAG system
        
        Args:
            pdf_directory: Directory containing PDF files
            model_name: Ollama model for generation
            embedding_model: Ollama model for embeddings
            base_url: Ollama API base URL
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
            k_documents: Number of documents to retrieve
            vectorstore_path: Directory to save/load vector store
        """
        self.pdf_directory = pdf_directory
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.base_url = base_url
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k_documents = k_documents
        self.vectorstore_path = vectorstore_path
        self.metadata_path = os.path.join(vectorstore_path, "metadata.pkl")
        
        # Initialize components
        self.llm = None
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None
        
        # Track processing statistics
        self._successful_count = 0
        self._failed_count = 0
        
        # Initialize the system
        self._initialize()
    
    def _get_pdf_hash(self) -> str:
        """Calculate hash of PDF files and configuration"""
        try:
            pdf_files = [
                f for f in os.listdir(self.pdf_directory)
                if f.lower().endswith('.pdf')
            ]
            pdf_files.sort()
            
            hash_input = ""
            for pdf_file in pdf_files:
                pdf_path = os.path.join(self.pdf_directory, pdf_file)
                if os.path.exists(pdf_path):
                    mtime = os.path.getmtime(pdf_path)
                    size = os.path.getsize(pdf_path)
                    hash_input += f"{pdf_file}:{mtime}:{size};"
            
            # Add configuration parameters
            hash_input += f"chunk_size:{self.chunk_size};"
            hash_input += f"chunk_overlap:{self.chunk_overlap};"
            hash_input += f"embedding_model:{self.embedding_model};"
            
            return hashlib.sha256(hash_input.encode()).hexdigest()
        except Exception as e:
            logger.error(f"Error calculating PDF hash: {e}")
            return ""
    
    def _save_vectorstore(self) -> bool:
        """Save vector store to disk"""
        try:
            import time
            start_time = time.time()
            
            logger.info("Saving vectorstore to disk...")
            
            # Ensure directory exists
            os.makedirs(self.vectorstore_path, exist_ok=True)
            
            # Save FAISS index
            self.vectorstore.save_local(self.vectorstore_path)
            
            # Calculate success rate
            total_chunks = self._successful_count + self._failed_count
            success_rate = self._successful_count / total_chunks if total_chunks > 0 else 0
            
            # Save metadata
            metadata = {
                'pdf_hash': self._get_pdf_hash(),
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'embedding_model': self.embedding_model,
                'created_at': time.time(),
                'processing_stats': {
                    'total_chunks': total_chunks,
                    'successful': self._successful_count,
                    'failed': self._failed_count,
                    'success_rate': success_rate
                }
            }
            
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            duration = time.time() - start_time
            logger.info(f"Vectorstore saved successfully in {duration:.2f}s")
            logger.info(f"  Chunks: {self._successful_count} successful, {self._failed_count} failed")
            logger.info(f"  Success rate: {success_rate:.1%}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save vectorstore: {e}")
            return False
    
    def _load_vectorstore(self) -> bool:
        """Load vector store from disk"""
        try:
            import time
            start_time = time.time()
            
            # Check if files exist
            if not os.path.exists(self.metadata_path):
                logger.info("No saved vectorstore found")
                return False
            
            # Load metadata
            with open(self.metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            # Verify hash
            current_hash = self._get_pdf_hash()
            if metadata['pdf_hash'] != current_hash:
                logger.info("PDF files or config changed, rebuild required")
                logger.info(f"  Saved hash: {metadata['pdf_hash'][:16]}...")
                logger.info(f"  Current hash: {current_hash[:16]}...")
                return False
            
            # Load FAISS index
            logger.info("Loading vectorstore from disk...")
            self.vectorstore = FAISS.load_local(
                self.vectorstore_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            # Rebuild retriever
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.k_documents}
            )
            
            # Restore statistics
            stats = metadata.get('processing_stats', {})
            self._successful_count = stats.get('successful', 0)
            self._failed_count = stats.get('failed', 0)
            
            duration = time.time() - start_time
            logger.info(f"Vectorstore loaded successfully in {duration:.2f}s")
            logger.info(f"  Chunks: {stats.get('successful', 'unknown')}")
            logger.info(f"  Success rate: {stats.get('success_rate', 0):.1%}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load vectorstore: {e}")
            return False
    
    def _initialize(self):
        """Initialize LLM, embeddings, and vector store"""
        try:
            # Initialize LLM
            logger.info(f"Initializing LLM: {self.model_name}")
            self.llm = OllamaLLM(
                model=self.model_name,
                temperature=0.3,
                base_url=self.base_url
            )
            
            # Initialize embeddings
            logger.info(f"Initializing embeddings: {self.embedding_model}")
            self.embeddings = OllamaEmbeddings(
                model=self.embedding_model,
                base_url=self.base_url
            )
            
            # Try to load saved vectorstore
            if self._load_vectorstore():
                logger.info("RAG system initialized from cache")
                return
            
            # Load failed or no cache, build from scratch
            logger.info("Building vectorstore from scratch...")
            
            # Reset counters
            self._successful_count = 0
            self._failed_count = 0
            
            # Load and process documents
            logger.info("Loading and processing documents...")
            documents = self._load_documents()
            
            if not documents:
                logger.warning("No documents loaded. RAG system will be limited.")
                return
            
            # Test embedding before creating vector store
            logger.info("Testing embedding model...")
            try:
                test_embedding = self.embeddings.embed_query("test")
                logger.info(f"Embedding test successful. Dimension: {len(test_embedding)}")
            except Exception as embed_error:
                logger.error(f"Embedding test failed: {embed_error}")
                # Try to get more details from the error
                if hasattr(embed_error, 'status_code'):
                    logger.error(f"Status code: {embed_error.status_code}")
                raise
            
            # Clean and validate documents
            logger.info(f"Cleaning and validating {len(documents)} documents...")
            cleaned_documents = []
            skipped_short = 0
            skipped_long = 0
            skipped_invalid = 0
            
            for i, doc in enumerate(documents):
                try:
                    # Get document text
                    text = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                    
                    # Skip empty or very short documents
                    if not text or len(text.strip()) < 20:
                        skipped_short += 1
                        continue
                    
                    # 限制長度 - 以字節計算更安全
                    byte_length = len(text.encode('utf-8'))
                    if byte_length > 1800:
                        # 截斷到安全長度
                        text = text[:700]
                        skipped_long += 1
                    
                    # 移除控制字符（保留換行、回車、製表符）
                    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
                    
                    # 再次檢查清理後的長度
                    if len(text.strip()) < 20:
                        skipped_short += 1
                        continue
                    
                    doc.page_content = text
                    cleaned_documents.append(doc)
                    
                except Exception as e:
                    skipped_invalid += 1
                    logger.debug(f"Error cleaning document {i}: {e}")
                    continue
            
            logger.info(f"清理完成: 保留 {len(cleaned_documents)}, 跳過: 太短={skipped_short}, 太長被截斷={skipped_long}, 無效={skipped_invalid}")
            
            logger.info(f"Cleaned {len(cleaned_documents)} valid documents")
            documents = cleaned_documents
            
            if not documents:
                logger.error("No valid documents after cleaning")
                return
            
            # Create vector store with batch processing for large document sets
            logger.info(f"Creating vector store with {len(documents)} chunks...")
            
            # Process in smaller batches to avoid overwhelming the embedding service
            batch_size = 10  # Reduced from 50 to 10 for more stability
            if len(documents) > batch_size:
                logger.info(f"Processing documents in batches of {batch_size}...")
                # Create initial vectorstore with first batch
                first_batch = documents[:batch_size]
                logger.info(f"Creating initial vector store with {len(first_batch)} documents...")
                
                # 使用逐個處理來避免批次 API 的問題
                logger.info("Creating initial vector store using individual embeddings...")
                
                # 嘗試從第一個可用文檔創建向量存儲
                self.vectorstore = None
                init_success = False
                skipped_initial = 0
                
                for init_idx in range(len(first_batch)):
                    try:
                        time.sleep(0.1)
                        # 清理第一個文檔
                        doc = first_batch[init_idx]
                        content = doc.page_content
                        byte_len = len(content.encode('utf-8'))
                        if byte_len > 1800:
                            content = content[:700]
                        content = ''.join(char for char in content if ord(char) >= 32 or char in '\n\r\t')
                        doc.page_content = content
                        
                        if len(content.strip()) < 20:
                            skipped_initial += 1
                            continue
                            
                        self.vectorstore = FAISS.from_documents(
                            documents=[doc],
                            embedding=self.embeddings
                        )
                        init_success = True
                        logger.info(f"Initial vectorstore created from document {init_idx + 1}")
                        break
                    except Exception as e:
                        skipped_initial += 1
                        logger.warning(f"Failed to create vectorstore from document {init_idx + 1}: {e}")
                        continue
                
                if not init_success:
                    raise RuntimeError("Failed to create initial vectorstore from any document in first batch")
                
                # 逐個添加剩餘文檔
                start_idx = init_idx + 1
                for i, doc in enumerate(first_batch[start_idx:], 1):
                    try:
                        time.sleep(0.1)  # 短暫延遲
                        
                        # 清理內容
                        content = doc.page_content
                        
                        # 長度限制
                        byte_len = len(content.encode('utf-8'))
                        if byte_len > 1800:
                            content = content[:700]
                        
                        # 移除控制字符
                        content = ''.join(char for char in content if ord(char) >= 32 or char in '\n\r\t')
                        
                        if len(content.strip()) < 20:
                            continue
                            
                        doc.page_content = content
                        
                        single_vs = FAISS.from_documents(
                            documents=[doc],
                            embedding=self.embeddings
                        )
                        self.vectorstore.merge_from(single_vs)
                        if (i + start_idx) % 5 == 0:
                            logger.info(f"  已處理 {i + start_idx}/{len(first_batch)} 個初始文檔")
                    except Exception as e:
                        logger.debug(f"Failed to add document {i + start_idx} to initial batch: {e}")
                        continue
                
                logger.info(f"Initial vector store created with {len(first_batch)} documents (skipped {skipped_initial})")
                
                # Add remaining documents one by one (避免批次 API 問題)
                processed = len(first_batch)
                total_remaining = len(documents) - processed
                logger.info(f"Processing remaining {total_remaining} documents individually...")
                
                success_count = 0
                fail_count = 0
                
                for i in range(processed, len(documents)):
                    # Add small delay to avoid overwhelming the service
                    time.sleep(0.1)
                    
                    doc = documents[i]
                    
                    try:
                        # 額外的內容檢查和清理
                        content = doc.page_content
                        
                        # 檢查並限制長度（以字節為準更安全）
                        byte_len = len(content.encode('utf-8'))
                        if byte_len > 1800:
                            content = content[:700]
                        
                        # 移除可能導致問題的特殊控制字符
                        content = ''.join(char for char in content if ord(char) >= 32 or char in '\n\r\t')
                        
                        doc.page_content = content
                        
                        # 如果內容太短，跳過
                        if len(content.strip()) < 20:
                            fail_count += 1
                            continue
                        
                        single_vectorstore = FAISS.from_documents(
                            documents=[doc],
                            embedding=self.embeddings
                        )
                        self.vectorstore.merge_from(single_vectorstore)
                        success_count += 1
                        
                        # 每 50 個文檔報告一次進度
                        if (i - processed + 1) % 50 == 0:
                            logger.info(f"  進度: {i - processed + 1}/{total_remaining} 文檔 (成功: {success_count}, 失敗: {fail_count})")
                    except Exception as single_error:
                        fail_count += 1
                        # 記錄失敗文檔的詳細信息以便調試
                        if fail_count <= 10:  # 記錄前10個失敗的詳細信息
                            error_msg = str(single_error)
                            content_preview = doc.page_content[:100].replace('\n', ' ')
                            byte_len = len(doc.page_content.encode('utf-8'))
                            char_len = len(doc.page_content)
                            
                            logger.warning(
                                f"  失敗文檔 #{fail_count} (索引{i}):\n"
                                f"    字符長度: {char_len}, 字節長度: {byte_len}\n"
                                f"    內容預覽: {content_preview}...\n"
                                f"    錯誤: {error_msg}"
                            )
                            
                            # 嘗試分析錯誤類型
                            if "EOF" in error_msg:
                                logger.warning(f"    → 可能原因: 文本過長或包含特殊字符導致 Ollama 崩潰")
                            elif "500" in error_msg:
                                logger.warning(f"    → 可能原因: Ollama 內部錯誤，可能是文本編碼或格式問題")
                        elif fail_count == 11:
                            logger.info(f"  (後續失敗不再詳細記錄)")
                        continue
                
                logger.info(f"處理完成: 成功 {success_count}/{total_remaining}, 失敗 {fail_count}")
                
                # Update total counts
                self._successful_count = success_count + len(first_batch)
                self._failed_count = fail_count
            else:
                # Small number of documents, process all at once
                logger.info("Processing all documents in single batch...")
                self.vectorstore = FAISS.from_documents(
                    documents=documents,
                    embedding=self.embeddings
                )
                self._successful_count = len(documents)
                self._failed_count = 0
            
            # Save vectorstore to disk
            self._save_vectorstore()
            
            logger.info("RAG system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}", exc_info=True)
            raise
    
    def _load_documents(self) -> List[Any]:
        """Load and split PDF documents"""
        documents = []
        
        if not os.path.exists(self.pdf_directory):
            logger.warning(f"PDF directory does not exist: {self.pdf_directory}")
            return documents
        
        # Get all PDF files
        pdf_files = [
            f for f in os.listdir(self.pdf_directory)
            if f.lower().endswith('.pdf')
        ]
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.pdf_directory}")
            return documents
        
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        # Text splitter - 減小塊大小以避免 Ollama embedding 限制
        # Ollama nomic-embed-text 在 2000+ 字符時會失敗
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=min(self.chunk_size, 800),  # 限制最大 800 字符
            chunk_overlap=min(self.chunk_overlap, 100),  # 減少重疊
            separators=["\n\n", "\n", "。", ".", " ", ""]
        )
        
        # Load each PDF
        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.pdf_directory, pdf_file)
            try:
                logger.info(f"Loading: {pdf_file}")
                loader = PyPDFLoader(pdf_path)
                pages = loader.load()
                
                # Add metadata
                for page in pages:
                    page.metadata['source_file'] = pdf_file
                
                # Split documents
                splits = text_splitter.split_documents(pages)
                documents.extend(splits)
                
                logger.info(f"  Loaded {len(pages)} pages, created {len(splits)} chunks")
                
            except Exception as e:
                logger.error(f"Error loading {pdf_file}: {e}")
                continue
        
        logger.info(f"Total documents loaded: {len(documents)} chunks")
        return documents
    
    def add_document(self, pdf_path: str) -> Dict[str, Any]:
        """
        Add a single PDF to existing vectorstore
        
        Args:
            pdf_path: Absolute path to PDF file
            
        Returns:
            Dictionary with status and statistics
        """
        import time
        start_time = time.time()
        
        try:
            if not self.vectorstore:
                raise RuntimeError("RAG system not initialized")
            
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF not found: {pdf_path}")
            
            logger.info(f"Adding document: {pdf_path}")
            
            # Load new PDF
            loader = PyMuPDFLoader(pdf_path)
            pages = loader.load()
            
            # Add metadata
            pdf_filename = os.path.basename(pdf_path)
            for page in pages:
                page.metadata['source_file'] = pdf_filename
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=min(self.chunk_size, 800),
                chunk_overlap=min(self.chunk_overlap, 100),
                separators=["\n\n", "\n", "。", ".", " ", ""]
            )
            new_documents = text_splitter.split_documents(pages)
            
            # Clean content (use same logic as existing)
            cleaned_docs = []
            failed = 0
            
            for doc in new_documents:
                try:
                    content = doc.page_content
                    
                    # Length check
                    byte_len = len(content.encode('utf-8'))
                    if byte_len > 1800:
                        content = content[:700]
                    
                    # Remove control characters
                    content = ''.join(char for char in content if ord(char) >= 32 or char in '\n\r\t')
                    
                    # Skip if too short
                    if len(content.strip()) < 20:
                        failed += 1
                        continue
                    
                    doc.page_content = content
                    cleaned_docs.append(doc)
                except:
                    failed += 1
            
            # Embed new documents
            logger.info(f"Embedding {len(cleaned_docs)} chunks...")
            success_count = 0
            
            for i, doc in enumerate(cleaned_docs):
                try:
                    time.sleep(0.1)  # Rate limiting
                    single_vs = FAISS.from_documents(
                        documents=[doc],
                        embedding=self.embeddings
                    )
                    self.vectorstore.merge_from(single_vs)
                    success_count += 1
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"  Progress: {i + 1}/{len(cleaned_docs)} chunks")
                except Exception as e:
                    failed += 1
                    logger.debug(f"Failed to embed chunk {i}: {e}")
            
            # Update statistics
            self._successful_count += success_count
            self._failed_count += failed
            
            # Save updated vectorstore
            self._save_vectorstore()
            
            duration = time.time() - start_time
            
            result = {
                'status': 'success',
                'pdf_path': pdf_path,
                'chunks_added': success_count,
                'chunks_failed': failed,
                'duration_seconds': duration
            }
            
            logger.info(f"Document added successfully: {success_count} chunks in {duration:.1f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def query(self, user_input: str) -> str:
        """
        Query the RAG system
        
        Args:
            user_input: User's question or query
            
        Returns:
            Answer from the RAG system with source information
        """
        if not self.vectorstore:
            return "RAG system not properly initialized. Please check the logs."
        
        try:
            logger.info(f"Processing RAG query: {user_input}")
            
            # Retrieve relevant documents
            source_docs = self.vectorstore.similarity_search(user_input, k=self.k_documents)
            
            if not source_docs:
                return "No relevant documents found in the database."
            
            # Build context from retrieved documents
            context = "\n\n".join([
                f"文件 {i+1} ({doc.metadata.get('source_file', 'Unknown')}):\n{doc.page_content}"
                for i, doc in enumerate(source_docs)
            ])
            
            # Create prompt
            prompt = f"""你是一個學術論文助理。請根據以下提供的論文內容回答問題。

相關論文內容：
{context}

問題：{user_input}

請提供詳細且準確的回答，並在適當時引用來源論文。如果資料中沒有相關資訊，請誠實告知。

回答："""
            
            # Get response from LLM
            answer = self.llm.invoke(prompt)
            
            # Format response with sources
            result = f"{answer}\n\n"
            
            # Add source information
            result += "📚 來源論文：\n"
            sources = set()
            for doc in source_docs:
                source_file = doc.metadata.get('source_file', 'Unknown')
                sources.add(source_file)
            
            for source in sorted(sources):
                result += f"  • {source}\n"
            
            logger.info(f"RAG query completed, used {len(source_docs)} source chunks")
            return result
            
        except Exception as e:
            logger.error(f"Error in RAG query: {e}", exc_info=True)
            return f"Error processing query: {str(e)}"
    
    def query_stream(self, user_input: str):
        """
        Query the RAG system with streaming output
        
        Args:
            user_input: User's question or query
            
        Yields:
            Chunks of the response as they are generated
        """
        if not self.vectorstore:
            yield "RAG system not properly initialized. Please check the logs."
            return
        
        try:
            logger.info(f"Processing RAG query (streaming): {user_input}")
            
            # Retrieve relevant documents
            source_docs = self.vectorstore.similarity_search(user_input, k=self.k_documents)
            
            if not source_docs:
                yield "No relevant documents found in the database."
                return
            
            # Build context from retrieved documents
            context = "\n\n".join([
                f"文件 {i+1} ({doc.metadata.get('source_file', 'Unknown')}):\n{doc.page_content}"
                for i, doc in enumerate(source_docs)
            ])
            
            # Create prompt
            prompt = f"""你是一個學術論文助理。請根據以下提供的論文內容回答問題。

相關論文內容：
{context}

問題：{user_input}

請提供詳細且準確的回答，並在適當時引用來源論文。如果資料中沒有相關資訊，請誠實告知。

回答："""
            
            # Stream response from LLM
            try:
                for chunk in self.llm.stream(prompt):
                    yield chunk
            except GeneratorExit:
                # Generator was closed externally (e.g., user clicked stop)
                logger.info("RAG streaming interrupted by external close")
                raise
            
            # Add source information at the end
            yield "\n\n📚 來源論文：\n"
            sources = set()
            for doc in source_docs:
                source_file = doc.metadata.get('source_file', 'Unknown')
                sources.add(source_file)
            
            for source in sorted(sources):
                yield f"  • {source}\n"
            
            logger.info(f"RAG query streaming completed, used {len(source_docs)} source chunks")
            
        except Exception as e:
            logger.error(f"Error in RAG query streaming: {e}", exc_info=True)
            yield f"\n\nError processing query: {str(e)}"
    
    def search_documents(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        """
        Search for relevant documents
        
        Args:
            query: Search query
            k: Number of documents to return (default: self.k_documents)
            
        Returns:
            List of relevant documents with metadata
        """
        if not self.vectorstore:
            return []
        
        try:
            k = k or self.k_documents
            docs = self.vectorstore.similarity_search(query, k=k)
            
            results = []
            for doc in docs:
                results.append({
                    'content': doc.page_content[:500] + '...',  # Truncate for display
                    'source': doc.metadata.get('source_file', 'Unknown'),
                    'page': doc.metadata.get('page', 'Unknown')
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG system"""
        stats = {
            'initialized': self.vectorstore is not None,
            'model': self.model_name,
            'embedding_model': self.embedding_model,
            'pdf_directory': self.pdf_directory
        }
        
        if self.vectorstore:
            try:
                stats['total_documents'] = len(self.vectorstore.docstore._dict)
            except:
                stats['total_documents'] = 'Unknown'
        
        return stats
    
    def reload(self, force_rebuild: bool = False):
        """Reload documents and rebuild vector store
        
        Args:
            force_rebuild: If True, force rebuild even if cache exists
        """
        logger.info("Reloading RAG system...")
        if force_rebuild:
            # Delete cache to force rebuild
            import shutil
            if os.path.exists(self.vectorstore_path):
                logger.info("Force rebuild: removing cached vectorstore")
                shutil.rmtree(self.vectorstore_path)
        self._initialize()