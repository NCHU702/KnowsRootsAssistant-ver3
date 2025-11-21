"""
測試 Layer2 Chunk Type 過濾功能
"""

import logging
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from system_api.layer2_vectorstore import Layer2VectorStore

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_chunk_type_filtering():
    """測試 chunk type 過濾功能"""
    logger.info("=" * 70)
    logger.info("🧪 Testing Layer2 Chunk Type Filtering")
    logger.info("=" * 70)
    
    # 1. 建立測試用的 Layer2 VectorStore
    logger.info("\n📦 Creating test Layer2 VectorStore...")
    
    embeddings = OllamaEmbeddings(model="quentinz/bge-large-zh-v1.5:latest")
    
    layer2 = Layer2VectorStore(
        embeddings=embeddings,
        vectorstore_path="./test_vectorstore_layer2",
        chunk_size=800,
        chunk_overlap=100,
        use_reranker=True,
        reranker_config={
            'model': 'qllama/bce-reranker-base_v1:latest',
            'ollama_base_url': 'http://localhost:11434'
        }
    )
    
    # 2. 建立測試文檔（模擬不同類型的 chunks）
    logger.info("\n📝 Creating test documents with different chunk types...")
    
    test_docs = [
        Document(
            page_content="This paper presents a novel LSTM architecture for time series forecasting. The method combines recurrent layers with attention mechanisms.",
            metadata={
                'paper_id': 'test_paper_1',
                'chunk_id': 'test_paper_1_method',
                'chunk_type': 'method',
                'section_name': 'Methodology'
            }
        ),
        Document(
            page_content="We evaluate our approach on three benchmark datasets: MNIST, CIFAR-10, and ImageNet. The datasets contain 60000, 50000, and 1.2M images respectively.",
            metadata={
                'paper_id': 'test_paper_1',
                'chunk_id': 'test_paper_1_dataset',
                'chunk_type': 'experiment',
                'section_name': 'Dataset'
            }
        ),
        Document(
            page_content="Our model achieves 95% accuracy on the test set, outperforming baseline models. The F1-score is 0.94 and precision is 0.96.",
            metadata={
                'paper_id': 'test_paper_1',
                'chunk_id': 'test_paper_1_results',
                'chunk_type': 'results',
                'section_name': 'Results'
            }
        ),
        Document(
            page_content="Abstract: This paper introduces a new approach to image classification using deep learning.",
            metadata={
                'paper_id': 'test_paper_1',
                'chunk_id': 'test_paper_1_abstract',
                'chunk_type': 'summary',
                'section_name': 'Abstract'
            }
        )
    ]
    
    logger.info(f"  Created {len(test_docs)} test documents:")
    for doc in test_docs:
        logger.info(f"    - {doc.metadata['chunk_type']}: {doc.page_content[:50]}...")
    
    # 3. 建立索引
    logger.info("\n🔨 Building index...")
    success = layer2._document_store.store_chunks(test_docs)
    
    if success:
        logger.info("  ✓ Index built successfully")
    else:
        logger.error("  ✗ Failed to build index")
        return False
    
    # 4. 測試無過濾的搜尋
    logger.info("\n🔍 Test 1: Search without filtering")
    query = "What datasets were used?"
    results = layer2.search_with_scores(query, k=10, filter_paper_ids=['test_paper_1'])
    
    logger.info(f"  Query: {query}")
    logger.info(f"  Results: {len(results)}")
    for i, (doc, score) in enumerate(results, 1):
        chunk_type = doc.metadata.get('chunk_type', 'unknown')
        logger.info(f"    {i}. [{score:.3f}] {chunk_type}: {doc.page_content[:60]}...")
    
    # 5. 測試過濾 'experiment' 類型
    logger.info("\n🔍 Test 2: Filter by chunk_type='experiment'")
    results = layer2.search_with_scores(
        query, 
        k=10, 
        filter_paper_ids=['test_paper_1'],
        filter_chunk_types=['experiment']
    )
    
    logger.info(f"  Query: {query}")
    logger.info(f"  Filter: chunk_type=['experiment']")
    logger.info(f"  Results: {len(results)}")
    for i, (doc, score) in enumerate(results, 1):
        chunk_type = doc.metadata.get('chunk_type', 'unknown')
        logger.info(f"    {i}. [{score:.3f}] {chunk_type}: {doc.page_content[:60]}...")
    
    # 驗證只返回 experiment 類型
    assert all(doc.metadata.get('chunk_type') == 'experiment' for doc, _ in results), \
        "Some results are not of type 'experiment'"
    logger.info("  ✓ All results are of type 'experiment'")
    
    # 6. 測試過濾多個類型
    logger.info("\n🔍 Test 3: Filter by multiple chunk_types")
    results = layer2.search_with_scores(
        query, 
        k=10, 
        filter_paper_ids=['test_paper_1'],
        filter_chunk_types=['experiment', 'method']
    )
    
    logger.info(f"  Query: {query}")
    logger.info(f"  Filter: chunk_type=['experiment', 'method']")
    logger.info(f"  Results: {len(results)}")
    for i, (doc, score) in enumerate(results, 1):
        chunk_type = doc.metadata.get('chunk_type', 'unknown')
        logger.info(f"    {i}. [{score:.3f}] {chunk_type}: {doc.page_content[:60]}...")
    
    # 驗證只返回 experiment 或 method 類型
    valid_types = {'experiment', 'method'}
    assert all(doc.metadata.get('chunk_type') in valid_types for doc, _ in results), \
        "Some results are not of type 'experiment' or 'method'"
    logger.info("  ✓ All results are of type 'experiment' or 'method'")
    
    # 7. 測試 summary 類型過濾
    logger.info("\n🔍 Test 4: Filter by chunk_type='summary'")
    query2 = "What is this paper about?"
    results = layer2.search_with_scores(
        query2, 
        k=10, 
        filter_paper_ids=['test_paper_1'],
        filter_chunk_types=['summary']
    )
    
    logger.info(f"  Query: {query2}")
    logger.info(f"  Filter: chunk_type=['summary']")
    logger.info(f"  Results: {len(results)}")
    for i, (doc, score) in enumerate(results, 1):
        chunk_type = doc.metadata.get('chunk_type', 'unknown')
        logger.info(f"    {i}. [{score:.3f}] {chunk_type}: {doc.page_content[:60]}...")
    
    # 8. 清理測試檔案
    logger.info("\n🧹 Cleaning up test files...")
    import shutil
    try:
        shutil.rmtree("./test_vectorstore_layer2")
        logger.info("  ✓ Test files removed")
    except Exception as e:
        logger.warning(f"  Failed to remove test files: {e}")
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ All Layer2 Chunk Type Filtering Tests Passed!")
    logger.info("=" * 70)
    
    return True


if __name__ == "__main__":
    import sys
    try:
        success = test_chunk_type_filtering()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n⚠️  Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}", exc_info=True)
        sys.exit(1)
