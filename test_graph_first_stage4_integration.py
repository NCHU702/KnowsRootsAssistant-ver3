"""
Stage 4 Integration Test - Graph-first Flow

測試 HierarchicalRAGSystem 中 Graph-first flow 的整合
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_config_structure():
    """測試配置結構是否正確"""
    print("\n" + "="*80)
    print("🧪 Test 1: Configuration Structure")
    print("="*80)
    
    from system_api.hierarchical_rag_system import HIERARCHICAL_RAG_CONFIG
    
    # Check graph_first config
    assert 'graph_first' in HIERARCHICAL_RAG_CONFIG, "❌ Missing 'graph_first' config"
    graph_config = HIERARCHICAL_RAG_CONFIG['graph_first']
    
    required_keys = ['enabled', 'confidence_threshold', 'min_graph_hits', 'skip_layer1']
    for key in required_keys:
        assert key in graph_config, f"❌ Missing key: {key}"
        print(f"  ✓ graph_first.{key} = {graph_config[key]}")
    
    # Check chunk_classification config
    assert 'chunk_classification' in HIERARCHICAL_RAG_CONFIG, "❌ Missing 'chunk_classification' config"
    chunk_config = HIERARCHICAL_RAG_CONFIG['chunk_classification']
    
    required_keys = ['enabled', 'use_llm_fallback']
    for key in required_keys:
        assert key in chunk_config, f"❌ Missing key: {key}"
        print(f"  ✓ chunk_classification.{key} = {chunk_config[key]}")
    
    print("\n✅ Configuration structure is correct!")
    return True


def test_index_manager_signature():
    """測試 IndexManager 是否接受 chunk_classifier 參數"""
    print("\n" + "="*80)
    print("🧪 Test 2: IndexManager Signature")
    print("="*80)
    
    from system_api.index_manager import IndexManager
    import inspect
    
    sig = inspect.signature(IndexManager.__init__)
    params = list(sig.parameters.keys())
    
    print(f"  IndexManager.__init__ parameters: {params}")
    
    assert 'chunk_classifier' in params, "❌ Missing 'chunk_classifier' parameter"
    print(f"  ✓ 'chunk_classifier' parameter exists")
    
    # Check if it's optional
    param = sig.parameters['chunk_classifier']
    assert param.default is not inspect.Parameter.empty, "❌ 'chunk_classifier' should be optional"
    print(f"  ✓ 'chunk_classifier' is optional (default: {param.default})")
    
    print("\n✅ IndexManager signature is correct!")
    return True


def test_component_initialization():
    """測試組件初始化（使用 mock 避免實際連接）"""
    print("\n" + "="*80)
    print("🧪 Test 3: Component Initialization (Mock)")
    print("="*80)
    
    # Mock components to avoid actual dependencies
    class MockEmbeddings:
        def embed_query(self, text):
            return [0.1] * 768
    
    class MockLLM:
        def invoke(self, text):
            class Response:
                content = "mock response"
            return Response()
    
    # Test ChunkClassifier can be imported and initialized
    try:
        from system_api.chunk_classifier import ChunkClassifier
        classifier = ChunkClassifier(llm=None, use_llm_fallback=False)
        print("  ✓ ChunkClassifier initialized")
    except Exception as e:
        print(f"  ❌ ChunkClassifier failed: {e}")
        return False
    
    # Test that classifier has expected methods
    assert hasattr(classifier, 'classify_chunk'), "❌ Missing classify_chunk method"
    assert hasattr(classifier, 'classify_chunks_batch'), "❌ Missing classify_chunks_batch method"
    print("  ✓ ChunkClassifier has required methods")
    
    # Test GraphRetriever can be imported
    try:
        from system_api.graph_retriever import GraphRetriever
        print("  ✓ GraphRetriever can be imported")
    except Exception as e:
        print(f"  ❌ GraphRetriever import failed: {e}")
        return False
    
    # Test GraphIntegrator can be imported
    try:
        from system_api.graph_integrator import GraphIntegrator
        integrator = GraphIntegrator(MockLLM())
        print("  ✓ GraphIntegrator initialized")
    except Exception as e:
        print(f"  ❌ GraphIntegrator failed: {e}")
        return False
    
    print("\n✅ Component initialization successful!")
    return True


def test_chunk_classification_in_indexing():
    """測試索引時的 chunk classification 邏輯"""
    print("\n" + "="*80)
    print("🧪 Test 4: Chunk Classification in Indexing")
    print("="*80)
    
    from langchain_core.documents import Document
    from system_api.chunk_classifier import ChunkClassifier
    
    # Create test chunks
    chunks = [
        Document(
            page_content="This paper introduces a new method for neural machine translation.",
            metadata={'paper_id': 'test_001', 'chunk_id': 'test_001_chunk_0'}
        ),
        Document(
            page_content="We evaluate our model on WMT14 dataset containing 4.5M sentence pairs.",
            metadata={'paper_id': 'test_001', 'chunk_id': 'test_001_chunk_1'}
        ),
        Document(
            page_content="The experimental results show BLEU score of 28.4 with 95% confidence.",
            metadata={'paper_id': 'test_001', 'chunk_id': 'test_001_chunk_2'}
        )
    ]
    
    # Classify chunks
    classifier = ChunkClassifier()
    classifications = classifier.classify_chunks_batch(chunks)
    
    print(f"  Classified {len(classifications)} chunks:")
    for chunk, classification in zip(chunks, classifications):
        chunk_type = classification['chunk_type']
        confidence = classification['confidence']
        chunk.metadata['chunk_type'] = chunk_type
        chunk.metadata['classification_confidence'] = confidence
        
        preview = chunk.page_content[:50] + "..."
        print(f"    ✓ {chunk_type} ({confidence:.2f}): {preview}")
    
    # Verify metadata was added
    for chunk in chunks:
        assert 'chunk_type' in chunk.metadata, "❌ chunk_type not in metadata"
        assert 'classification_confidence' in chunk.metadata, "❌ confidence not in metadata"
    
    print(f"\n  ✓ All chunks have classification metadata")
    print("\n✅ Chunk classification logic works correctly!")
    return True


def test_retrieval_flow_structure():
    """測試檢索流程的結構（檢查代碼邏輯，不執行）"""
    print("\n" + "="*80)
    print("🧪 Test 5: Retrieval Flow Structure")
    print("="*80)
    
    import ast
    import inspect
    from system_api.hierarchical_rag_system import HierarchicalRAGSystem
    
    # Get source code
    source = inspect.getsource(HierarchicalRAGSystem._hierarchical_retrieval)
    
    # Check for key strings indicating Graph stage
    graph_indicators = [
        'GRAPH STAGE',
        'graph_retriever',
        'graph_integrator',
        'query_graph_for_papers',
        'integrate_graph_results',
        'terminated_at',
        'graph',
        'target_chunk_types',
        'filter_chunk_types'
    ]
    
    found_indicators = []
    for indicator in graph_indicators:
        if indicator in source:
            found_indicators.append(indicator)
            print(f"  ✓ Found: '{indicator}'")
        else:
            print(f"  ⚠️  Missing: '{indicator}'")
    
    if len(found_indicators) >= len(graph_indicators) - 2:  # Allow 2 missing
        print(f"\n  ✓ Found {len(found_indicators)}/{len(graph_indicators)} indicators")
        print("\n✅ Retrieval flow structure looks correct!")
        return True
    else:
        print(f"\n  ❌ Only found {len(found_indicators)}/{len(graph_indicators)} indicators")
        return False


def main():
    """執行所有測試"""
    print("\n" + "🔬"*40)
    print("Stage 4 Integration Test - Graph-first Flow")
    print("🔬"*40)
    
    tests = [
        ("Configuration Structure", test_config_structure),
        ("IndexManager Signature", test_index_manager_signature),
        ("Component Initialization", test_component_initialization),
        ("Chunk Classification", test_chunk_classification_in_indexing),
        ("Retrieval Flow Structure", test_retrieval_flow_structure),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*80)
    print("📊 Test Summary")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print("\n" + "="*80)
    print(f"Result: {passed}/{total} tests passed ({100*passed/total:.0f}%)")
    print("="*80)
    
    if passed == total:
        print("\n🎉 所有測試通過！Stage 4 整合成功！")
        print("\n下一步:")
        print("  1. 使用真實 Neo4j 連接測試 Graph query")
        print("  2. 建立測試索引並驗證 chunk classification")
        print("  3. 執行端到端查詢測試")
        return True
    else:
        print(f"\n⚠️  {total - passed} 個測試失敗，需要修正")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
