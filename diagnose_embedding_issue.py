#!/usr/bin/env python3
"""
診斷 Ollama Embedding 500 錯誤的腳本
"""

import sys
import time
import logging
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_embedding_basic():
    """測試基本的 embedding 功能"""
    logger.info("=" * 60)
    logger.info("測試 1: 基本 Embedding")
    logger.info("=" * 60)
    
    try:
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
        )
        
        # 測試單個簡單文本
        logger.info("測試簡單英文文本...")
        result = embeddings.embed_query("test")
        logger.info(f"✓ 成功 - 維度: {len(result)}")
        
        # 測試中文文本
        logger.info("測試中文文本...")
        result = embeddings.embed_query("測試")
        logger.info(f"✓ 成功 - 維度: {len(result)}")
        
        return True
    except Exception as e:
        logger.error(f"✗ 失敗: {e}")
        return False

def test_embedding_batch():
    """測試批次 embedding"""
    logger.info("\n" + "=" * 60)
    logger.info("測試 2: 批次 Embedding")
    logger.info("=" * 60)
    
    try:
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
        )
        
        # 測試小批次
        logger.info("測試 5 個文本的批次...")
        texts = [f"test {i}" for i in range(5)]
        result = embeddings.embed_documents(texts)
        logger.info(f"✓ 成功 - 返回 {len(result)} 個 embeddings")
        
        # 測試中等批次
        logger.info("測試 10 個文本的批次...")
        texts = [f"test {i}" for i in range(10)]
        result = embeddings.embed_documents(texts)
        logger.info(f"✓ 成功 - 返回 {len(result)} 個 embeddings")
        
        return True
    except Exception as e:
        logger.error(f"✗ 失敗: {e}")
        return False

def test_embedding_with_real_pdf():
    """測試實際 PDF 文本的 embedding"""
    logger.info("\n" + "=" * 60)
    logger.info("測試 3: 實際 PDF 文本 Embedding")
    logger.info("=" * 60)
    
    try:
        import os
        
        # 查找第一個 PDF 文件
        pdf_dir = "./data"
        pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
        
        if not pdf_files:
            logger.warning("未找到 PDF 文件")
            return True
        
        pdf_path = os.path.join(pdf_dir, pdf_files[0])
        logger.info(f"使用 PDF: {pdf_files[0]}")
        
        # 加載並分割 PDF
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        logger.info(f"加載了 {len(pages)} 頁")
        
        # 只取第一頁進行測試
        if pages:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents([pages[0]])
            logger.info(f"分割成 {len(chunks)} 個塊")
            
            # 測試第一個塊
            if chunks:
                embeddings = OllamaEmbeddings(
                    model="nomic-embed-text",
                    base_url="http://localhost:11434"
                )
                
                first_chunk = chunks[0].page_content
                logger.info(f"第一個塊長度: {len(first_chunk)} 字符")
                logger.info(f"前 100 字符: {first_chunk[:100]}...")
                
                # 檢查是否有特殊字符
                special_chars = [c for c in first_chunk if ord(c) > 127 and c not in '，。！？：；、「」『』（）【】']
                if special_chars:
                    logger.info(f"發現特殊字符: {set(special_chars[:20])}")
                
                logger.info("嘗試 embedding 第一個塊...")
                result = embeddings.embed_query(first_chunk)
                logger.info(f"✓ 成功 - 維度: {len(result)}")
                
                # 測試批次處理前 3 個塊
                if len(chunks) >= 3:
                    logger.info("嘗試批次 embedding 前 3 個塊...")
                    texts = [c.page_content for c in chunks[:3]]
                    result = embeddings.embed_documents(texts)
                    logger.info(f"✓ 成功 - 返回 {len(result)} 個 embeddings")
                
                # 測試批次處理前 10 個塊（可能觸發錯誤）
                if len(chunks) >= 10:
                    logger.info("嘗試批次 embedding 前 10 個塊...")
                    texts = [c.page_content for c in chunks[:10]]
                    time.sleep(0.5)  # 添加延遲
                    result = embeddings.embed_documents(texts)
                    logger.info(f"✓ 成功 - 返回 {len(result)} 個 embeddings")
        
        return True
    except Exception as e:
        logger.error(f"✗ 失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_text_length_limits():
    """測試不同長度文本的 embedding"""
    logger.info("\n" + "=" * 60)
    logger.info("測試 4: 文本長度限制")
    logger.info("=" * 60)
    
    try:
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
        )
        
        # 測試不同長度的文本
        lengths = [100, 500, 1000, 2000, 5000]
        
        for length in lengths:
            text = "測試" * (length // 2)  # 中文字符
            logger.info(f"測試長度 {length} 字符 ({len(text)} 實際長度)...")
            try:
                result = embeddings.embed_query(text)
                logger.info(f"  ✓ 成功")
            except Exception as e:
                logger.error(f"  ✗ 失敗在長度 {length}: {e}")
                return False
            time.sleep(0.2)  # 短暫延遲
        
        return True
    except Exception as e:
        logger.error(f"✗ 失敗: {e}")
        return False

def test_concurrent_requests():
    """測試連續請求是否會導致問題"""
    logger.info("\n" + "=" * 60)
    logger.info("測試 5: 連續請求壓力測試")
    logger.info("=" * 60)
    
    try:
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
        )
        
        # 連續發送 20 個請求
        logger.info("發送 20 個連續請求...")
        success_count = 0
        fail_count = 0
        
        for i in range(20):
            try:
                result = embeddings.embed_query(f"test {i}")
                success_count += 1
                if (i + 1) % 5 == 0:
                    logger.info(f"  已完成 {i + 1}/20 個請求")
            except Exception as e:
                fail_count += 1
                logger.error(f"  請求 {i + 1} 失敗: {e}")
            
            time.sleep(0.1)  # 短暫延遲
        
        logger.info(f"結果: {success_count} 成功, {fail_count} 失敗")
        return fail_count == 0
        
    except Exception as e:
        logger.error(f"✗ 失敗: {e}")
        return False

def main():
    """運行所有診斷測試"""
    logger.info("\n" + "=" * 60)
    logger.info("Ollama Embedding 問題診斷")
    logger.info("=" * 60 + "\n")
    
    results = {
        "基本 Embedding": test_embedding_basic(),
        "批次 Embedding": test_embedding_batch(),
        "實際 PDF 文本": test_embedding_with_real_pdf(),
        "文本長度限制": test_text_length_limits(),
        "連續請求壓力": test_concurrent_requests()
    }
    
    logger.info("\n" + "=" * 60)
    logger.info("診斷結果總結")
    logger.info("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ 通過" if passed else "✗ 失敗"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        logger.info("\n✓ 所有測試通過！問題可能在批次處理邏輯或特定文檔內容。")
    else:
        logger.info("\n✗ 發現問題！請檢查失敗的測試。")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
