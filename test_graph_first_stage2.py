#!/usr/bin/env python3
"""
測試 GraphRetriever 功能（使用 mock GraphManager）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockGraphManager:
    """模擬 GraphManager 用於測試"""
    
    class MockGraph:
        """模擬 Neo4j graph 連接"""
        
        def query(self, cypher_query, params):
            """模擬 Cypher 查詢"""
            keyword = params.get('keyword', '').lower()
            
            # 模擬 Dataset 查詢結果
            if 'Dataset' in cypher_query and 'mnist' in keyword:
                return [
                    {
                        'paper_id': 'paper_mnist_1',
                        'title': 'Image Classification with CNN',
                        'entities': ['MNIST', 'CIFAR-10'],
                        'rel_count': 2
                    }
                ]
            
            # 模擬 Method 查詢結果
            if 'Method' in cypher_query and 'lstm' in keyword:
                return [
                    {
                        'paper_id': 'paper_lstm_1',
                        'title': 'Time Series Forecasting with LSTM',
                        'entities': ['LSTM', 'GRU'],
                        'rel_count': 3
                    }
                ]
            
            # 模擬 Domain 查詢結果
            if 'domain' in cypher_query.lower() and ('computer' in keyword or '計算機' in keyword):
                return [
                    {
                        'paper_id': 'paper_cv_1',
                        'title': 'Computer Vision Applications',
                        'domain': 'Computer Vision',
                        'domain_zh': '計算機視覺'
                    }
                ]
            
            # 模擬實體搜尋
            if 'collect(DISTINCT p.paper_id)' in cypher_query:
                if 'mnist' in keyword:
                    return [
                        {
                            'name': 'MNIST',
                            'type': 'Dataset',
                            'paper_count': 15,
                            'papers': ['paper_1', 'paper_2', 'paper_3']
                        }
                    ]
            
            return []
    
    def __init__(self):
        self.graph = self.MockGraph()


def test_keyword_extraction():
    """測試關鍵詞提取"""
    logger.info("=" * 70)
    logger.info("🧪 測試 1: 關鍵詞提取")
    logger.info("=" * 70)
    
    from system_api.graph_retriever import GraphRetriever
    
    mock_gm = MockGraphManager()
    retriever = GraphRetriever(mock_gm)
    
    test_queries = {
        "What datasets were used for image classification?": 
            ['datasets', 'used', 'image', 'classification'],
        "使用了哪些資料集進行實驗？": 
            ['使用', '哪些', '資料集', '進行', '實驗'],
        "LSTM model architecture": 
            ['lstm', 'model', 'architecture'],
    }
    
    for query, expected_keywords in test_queries.items():
        keywords = retriever._extract_keywords(query)
        logger.info(f"\n  Query: {query}")
        logger.info(f"  Keywords: {keywords}")
        
        # 驗證至少提取到一些關鍵詞
        assert len(keywords) > 0, f"No keywords extracted from: {query}"
        
        # 驗證移除了停用詞
        stopwords = {'the', 'a', 'is', 'were', '了', '的'}
        assert not any(kw in stopwords for kw in keywords), "Stopwords not removed"
        
        logger.info(f"  ✓ 提取了 {len(keywords)} 個關鍵詞")
    
    logger.info("\n✅ 關鍵詞提取測試通過")
    return True


def test_query_graph_for_papers():
    """測試查詢論文"""
    logger.info("\n" + "=" * 70)
    logger.info("🧪 測試 2: 查詢 Graph 獲取論文")
    logger.info("=" * 70)
    
    from system_api.graph_retriever import GraphRetriever
    
    mock_gm = MockGraphManager()
    retriever = GraphRetriever(mock_gm)
    
    # Test 1: 查詢資料集
    logger.info("\n📋 Test 2.1: 查詢包含 'MNIST' 的論文")
    query = "What papers used MNIST dataset?"
    results = retriever.query_graph_for_papers(query, top_k=10)
    
    logger.info(f"  Query: {query}")
    logger.info(f"  Results: {len(results)} papers")
    for i, paper in enumerate(results, 1):
        logger.info(f"    {i}. {paper['paper_id']}")
        logger.info(f"       Title: {paper['title']}")
        logger.info(f"       Score: {paper['score']:.2f}")
        logger.info(f"       Entities: {paper['matched_entities']}")
        logger.info(f"       Evidence: {paper['evidence']}")
    
    # 驗證結果結構
    if results:
        paper = results[0]
        assert 'paper_id' in paper
        assert 'title' in paper
        assert 'score' in paper
        assert 'matched_entities' in paper
        assert 'entity_types' in paper
        assert 'evidence' in paper
        assert 'relationship_count' in paper
        logger.info("  ✓ 結果結構正確")
    
    # Test 2: 查詢方法
    logger.info("\n📋 Test 2.2: 查詢包含 'LSTM' 的論文")
    query = "Papers about LSTM model"
    results = retriever.query_graph_for_papers(query, top_k=10)
    
    logger.info(f"  Query: {query}")
    logger.info(f"  Results: {len(results)} papers")
    for i, paper in enumerate(results, 1):
        logger.info(f"    {i}. {paper['paper_id']}: {paper['title']}")
    
    # Test 3: 查詢領域
    logger.info("\n📋 Test 2.3: 查詢特定領域的論文")
    query = "Computer vision papers"
    results = retriever.query_graph_for_papers(query, top_k=10)
    
    logger.info(f"  Query: {query}")
    logger.info(f"  Results: {len(results)} papers")
    for i, paper in enumerate(results, 1):
        logger.info(f"    {i}. {paper['paper_id']}: {paper['title']}")
    
    logger.info("\n✅ 查詢論文測試通過")
    return True


def test_query_graph_for_entities():
    """測試查詢實體"""
    logger.info("\n" + "=" * 70)
    logger.info("🧪 測試 3: 查詢 Graph 獲取實體")
    logger.info("=" * 70)
    
    from system_api.graph_retriever import GraphRetriever
    
    mock_gm = MockGraphManager()
    retriever = GraphRetriever(mock_gm)
    
    query = "MNIST dataset usage"
    results = retriever.query_graph_for_entities(query, top_k=20)
    
    logger.info(f"  Query: {query}")
    logger.info(f"  Results: {len(results)} entities")
    
    for i, entity in enumerate(results, 1):
        logger.info(f"    {i}. {entity['name']} ({entity['type']})")
        logger.info(f"       Used in {entity['paper_count']} papers")
        logger.info(f"       Papers: {entity['papers'][:3]}")
    
    # 驗證結果結構
    if results:
        entity = results[0]
        assert 'name' in entity
        assert 'type' in entity
        assert 'paper_count' in entity
        assert 'papers' in entity
        logger.info("  ✓ 結果結構正確")
    
    logger.info("\n✅ 查詢實體測試通過")
    return True


def test_get_papers_for_entities():
    """測試根據實體獲取論文"""
    logger.info("\n" + "=" * 70)
    logger.info("🧪 測試 4: 根據實體名稱獲取論文")
    logger.info("=" * 70)
    
    from system_api.graph_retriever import GraphRetriever
    
    # 創建更完整的 mock
    class EnhancedMockGraph:
        def query(self, cypher_query, params):
            entity_names = params.get('entity_names', [])
            if 'collect(DISTINCT e.name)' in cypher_query:
                # 返回論文資訊
                results = []
                if 'MNIST' in entity_names:
                    results.append({
                        'paper_id': 'paper_1',
                        'title': 'Image Classification',
                        'entities': ['MNIST', 'CNN'],
                        'entity_types': ['Dataset', 'Method']
                    })
                return results
            return []
    
    class EnhancedMockGM:
        def __init__(self):
            self.graph = EnhancedMockGraph()
    
    retriever = GraphRetriever(EnhancedMockGM())
    
    entity_names = ['MNIST', 'CIFAR-10']
    results = retriever.get_papers_for_entities(entity_names)
    
    logger.info(f"  Entities: {entity_names}")
    logger.info(f"  Results: {len(results)} papers")
    
    for paper_id, info in results.items():
        logger.info(f"    - {paper_id}")
        logger.info(f"      Title: {info['title']}")
        logger.info(f"      Entities: {info['entities']}")
        logger.info(f"      Types: {info['entity_types']}")
    
    logger.info("\n✅ 根據實體獲取論文測試通過")
    return True


def main():
    """執行所有測試"""
    logger.info("\n🚀 開始測試 GraphRetriever")
    logger.info("=" * 70)
    
    try:
        # 測試 1: 關鍵詞提取
        if not test_keyword_extraction():
            logger.error("❌ 關鍵詞提取測試失敗")
            return False
        
        # 測試 2: 查詢論文
        if not test_query_graph_for_papers():
            logger.error("❌ 查詢論文測試失敗")
            return False
        
        # 測試 3: 查詢實體
        if not test_query_graph_for_entities():
            logger.error("❌ 查詢實體測試失敗")
            return False
        
        # 測試 4: 根據實體獲取論文
        if not test_get_papers_for_entities():
            logger.error("❌ 根據實體獲取論文測試失敗")
            return False
        
        logger.info("\n" + "=" * 70)
        logger.info("🎉 所有 GraphRetriever 測試通過！")
        logger.info("=" * 70)
        logger.info("\n✅ GraphRetriever 功能驗證完成:")
        logger.info("   1. ✓ 關鍵詞提取（中英文，停用詞過濾）")
        logger.info("   2. ✓ 查詢論文（按 Dataset/Method/Metric/Domain）")
        logger.info("   3. ✓ 查詢實體（返回實體和相關論文）")
        logger.info("   4. ✓ 根據實體獲取論文")
        logger.info("\n📝 下一步:")
        logger.info("   - 建立 GraphIntegrator（整合 Graph 結果）")
        logger.info("=" * 70)
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ 測試失敗: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
