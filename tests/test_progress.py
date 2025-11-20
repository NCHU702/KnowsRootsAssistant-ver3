"""
測試進度顯示功能
"""

from system_api.progress_embeddings import ProgressEmbeddings
from langchain_ollama import OllamaEmbeddings
import logging

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)

# 創建基礎 embeddings
base_embeddings = OllamaEmbeddings(model="quentinz/bge-large-zh-v1.5:latest")

# 包裝進度追蹤
progress_embeddings = ProgressEmbeddings(base_embeddings, label="Test")

# 測試文本
texts = [f"這是測試文本 {i}" for i in range(50)]

print("開始測試進度顯示...")
print("="*60)

# 執行 embedding
embeddings = progress_embeddings.embed_documents(texts)

print("="*60)
print(f"完成！生成了 {len(embeddings)} 個 embeddings")
print(f"每個 embedding 的維度: {len(embeddings[0])}")
