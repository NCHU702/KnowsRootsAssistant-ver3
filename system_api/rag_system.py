import os
import logging
import time
import re
from typing import List, Dict, Any, Optional
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
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
        k_documents: int = 4
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
        """
        self.pdf_directory = pdf_directory
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.base_url = base_url
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k_documents = k_documents
        
        # Initialize components
        self.llm = None
        self.embeddings = None
        self.vectorstore = None
        
        # Initialize the system
        self._initialize()
    
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
                
                retry_count = 0
                max_retries = 3
                while retry_count < max_retries:
                    try:
                        # 使用逐個處理來避免批次 API 的問題
                        logger.info("Creating initial vector store using individual embeddings...")
                        
                        # 創建第一個文檔的向量存儲
                        self.vectorstore = FAISS.from_documents(
                            documents=[first_batch[0]],
                            embedding=self.embeddings
                        )
                        
                        # 逐個添加剩餘文檔
                        for i, doc in enumerate(first_batch[1:], 1):
                            time.sleep(0.1)  # 短暫延遲
                            
                            # 清理內容
                            # 清理內容
                            content = doc.page_content
                            
                            # 長度限制
                            byte_len = len(content.encode('utf-8'))
                            if byte_len > 1800:
                                content = content[:700]
                            
                            # 移除控制字符
                            content = ''.join(char for char in content if ord(char) >= 32 or char in '\n\r\t')
                            
                            doc.page_content = content
                            
                            single_vs = FAISS.from_documents(
                                documents=[doc],
                                embedding=self.embeddings
                            )
                            self.vectorstore.merge_from(single_vs)
                            if i % 5 == 0:
                                logger.info(f"  已處理 {i}/{len(first_batch)} 個初始文檔")
                        
                        logger.info("Initial vector store created successfully")
                        break
                    except Exception as init_error:
                        retry_count += 1
                        logger.error(f"Error creating initial vector store (attempt {retry_count}/{max_retries}): {init_error}")
                        if retry_count >= max_retries:
                            logger.error("Failed to create initial vector store after retries")
                            raise
                        # Try with even smaller batch
                        first_batch = documents[:max(1, batch_size // 2)]
                        logger.info(f"Retrying with smaller batch size: {len(first_batch)}")
                
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
            else:
                # Small number of documents, process all at once
                logger.info("Processing all documents in single batch...")
                self.vectorstore = FAISS.from_documents(
                    documents=documents,
                    embedding=self.embeddings
                )
            
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
    
    def reload(self):
        """Reload documents and rebuild vector store"""
        logger.info("Reloading RAG system...")
        self._initialize()