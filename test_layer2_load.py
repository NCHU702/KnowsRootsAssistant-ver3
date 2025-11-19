#!/usr/bin/env python3
"""
Test Layer 2 Load Functionality

驗證 Layer 2 是否能正確載入現有索引並返回 is_initialized=True
"""

import logging
import sys
from langchain_ollama import OllamaEmbeddings
from system_api.layer2_vectorstore import Layer2VectorStore

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_layer2_load():
    """測試 Layer 2 載入功能"""
    
    print("="*80)
    print("Layer 2 Load Test")
    print("="*80)
    
    # 初始化 embeddings（雖然不使用，但需要傳入）
    embeddings = OllamaEmbeddings(model="embeddinggemma:latest")
    
    # 初始化 Layer 2
    print("\n1️⃣ Initializing Layer 2 VectorStore...")
    layer2 = Layer2VectorStore(
        embeddings=embeddings,
        vectorstore_path="./vectorstore/layer2",
        use_reranker=True,
        reranker_config={
            'enabled': True,
            'model': 'qllama/bce-reranker-base_v1:latest',
            'ollama_base_url': 'http://localhost:11434',
        }
    )
    
    # 嘗試載入索引
    print("\n2️⃣ Loading existing index...")
    load_success = layer2.load()
    
    if load_success:
        print("✅ Load successful!")
    else:
        print("❌ Load failed!")
        return False
    
    # 檢查 is_initialized
    print(f"\n3️⃣ Checking is_initialized property...")
    is_init = layer2.is_initialized
    print(f"   is_initialized = {is_init}")
    
    # 獲取統計
    print(f"\n4️⃣ Getting statistics...")
    stats = layer2.get_stats()
    print(f"   Stats: {stats}")
    
    # 驗證
    print(f"\n5️⃣ Validation:")
    checks = [
        ("is_initialized property", is_init),
        ("is_initialized in stats", stats.get('is_initialized', False)),
        ("chunk_count > 0", stats.get('chunk_count', 0) > 0),
        ("paper_count > 0", stats.get('paper_count', 0) > 0),
    ]
    
    all_passed = True
    for check_name, result in checks:
        status = "✅" if result else "❌"
        print(f"   {status} {check_name}: {result}")
        if not result:
            all_passed = False
    
    print("\n" + "="*80)
    if all_passed:
        print("✅ ALL CHECKS PASSED - Layer 2 can load correctly!")
        print("="*80)
        return True
    else:
        print("❌ SOME CHECKS FAILED - Layer 2 load has issues")
        print("="*80)
        return False


if __name__ == "__main__":
    try:
        success = test_layer2_load()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        sys.exit(1)
