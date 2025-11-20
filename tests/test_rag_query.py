#!/usr/bin/env python3
"""測試 RAG 查詢功能"""

from system_api.rag_system import AcademicRAGSystem
import logging

logging.basicConfig(level=logging.INFO)

print('初始化 RAG 系統...')
rag = AcademicRAGSystem(
    pdf_directory='./data',
    model_name='gemma3:270m',
    embedding_model='nomic-embed-text',
    chunk_size=800,
    chunk_overlap=100
)

print('\n=== RAG 系統統計 ===')
stats = rag.get_stats()
print(f"已初始化: {stats['initialized']}")
print(f"總文檔數: {stats.get('total_documents', 'Unknown')}")
print(f"成功 chunks: {rag._successful_count}")
print(f"失敗 chunks: {rag._failed_count}")

print('\n=== 測試查詢 ===')
query_text = '什麼是深度學習？'
print(f"查詢: {query_text}")

# 先測試文檔檢索
docs = rag.search_documents(query_text, k=5)
print(f"\n找到 {len(docs)} 個相關文檔片段:")
for i, doc in enumerate(docs, 1):
    print(f"\n{i}. 來源: {doc['source']}")
    print(f"   內容預覽: {doc['content'][:100]}...")

# 測試完整查詢
print('\n=== 完整 RAG 查詢 ===')
result = rag.query(query_text)
print(f"\n查詢結果 ({len(result)} 字元):")
print(result)
