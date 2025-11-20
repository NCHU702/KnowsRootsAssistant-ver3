"""
測試 Hierarchical RAG 各個組件

這個腳本測試 Phase 1 的所有基礎組件:
1. AbstractExtractor - 摘要提取
2. Layer1VectorStore - 論文級索引
3. Layer2VectorStore - 區塊級索引
4. ConfidenceEvaluator - 信心評估
5. ContextExpander - 上下文擴展
"""

import os
import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import components
from system_api.abstract_extractor import AbstractExtractor
from system_api.layer1_vectorstore import Layer1VectorStore
from system_api.layer2_vectorstore import Layer2VectorStore
from system_api.confidence_evaluator import ConfidenceEvaluator
from system_api.context_expander import ContextExpander

from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document


def print_section(title):
    """Print section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def test_abstract_extractor():
    """測試 Abstract Extractor"""
    print_section("測試 1: Abstract Extractor")
    
    try:
        # Initialize LLM
        llm = Ollama(model="gemma3:12b")
        extractor = AbstractExtractor(llm)
        
        # Test with sample text containing abstract
        sample_text = """
        Title: Deep Learning for Natural Language Processing
        
        Abstract
        This paper presents a novel approach to natural language processing using deep learning.
        We propose a new architecture that achieves state-of-the-art results on multiple benchmarks.
        Our method combines transformer models with hierarchical attention mechanisms.
        Experimental results demonstrate significant improvements over baseline methods.
        
        1. Introduction
        Natural language processing has seen tremendous progress...
        """
        
        print("📄 提取摘要 (使用正則表達式)...")
        result = extractor.extract(
            pdf_path="test_paper.pdf",
            pdf_text=sample_text,
            paper_id="test_001"
        )
        
        print(f"✓ 提取成功!")
        print(f"  來源: {result['source']}")
        print(f"  信心度: {result['confidence']:.2f}")
        print(f"  方法: {result['method']}")
        print(f"  摘要長度: {len(result['abstract'])} 字符")
        print(f"  摘要預覽: {result['abstract'][:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Abstract Extractor 測試失敗: {e}")
        logger.error("Abstract Extractor test failed", exc_info=True)
        return False


def test_layer1_vectorstore():
    """測試 Layer 1 VectorStore"""
    print_section("測試 2: Layer 1 VectorStore (論文級索引)")
    
    try:
        # Initialize embeddings
        embeddings = OllamaEmbeddings(model="embeddinggemma:latest")
        
        # Create test directory
        test_path = "./vectorstore_test/layer1"
        os.makedirs(test_path, exist_ok=True)
        
        vectorstore = Layer1VectorStore(embeddings, test_path)
        
        # Create test abstracts
        test_abstracts = [
            {
                'abstract': 'This paper discusses deep learning methods for computer vision tasks including image classification and object detection.',
                'paper_id': 'paper_001',
                'title': 'Deep Learning for Computer Vision',
                'authors': 'John Doe, Jane Smith',
                'year': 2023,
                'pdf_path': 'data/paper_001.pdf',
                'abstract_source': 'extracted',
                'abstract_confidence': 0.95
            },
            {
                'abstract': 'We present a novel natural language processing approach using transformer models for text generation and understanding.',
                'paper_id': 'paper_002',
                'title': 'Transformers for NLP',
                'authors': 'Alice Johnson',
                'year': 2024,
                'pdf_path': 'data/paper_002.pdf',
                'abstract_source': 'extracted',
                'abstract_confidence': 0.90
            },
            {
                'abstract': 'This study explores reinforcement learning algorithms for game playing and robotics control applications.',
                'paper_id': 'paper_003',
                'title': 'Reinforcement Learning Applications',
                'authors': 'Bob Chen',
                'year': 2023,
                'pdf_path': 'data/paper_003.pdf',
                'abstract_source': 'generated',
                'abstract_confidence': 0.75
            }
        ]
        
        print("📚 建立索引 (3 篇論文摘要)...")
        vectorstore.build_index(test_abstracts)
        
        # Get stats
        stats = vectorstore.get_stats()
        print(f"✓ 索引建立成功!")
        print(f"  論文數量: {stats['paper_count']}")
        print(f"  索引類型: {stats['type']}")
        print(f"  索引層級: Layer {stats['layer']}")
        
        # Test search
        print("\n🔍 測試搜索: 'computer vision deep learning'")
        results = vectorstore.search("computer vision deep learning", k=2)
        
        print(f"✓ 找到 {len(results)} 篇相關論文:")
        for i, doc in enumerate(results, 1):
            print(f"  {i}. {doc.metadata['title']}")
            print(f"     Paper ID: {doc.metadata['paper_id']}")
            print(f"     Year: {doc.metadata['year']}")
        
        # Test save/load
        print("\n💾 測試保存和載入...")
        vectorstore.save()
        
        new_vectorstore = Layer1VectorStore(embeddings, test_path)
        new_vectorstore.load()
        new_stats = new_vectorstore.get_stats()
        
        print(f"✓ 索引載入成功! 論文數量: {new_stats['paper_count']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Layer 1 VectorStore 測試失敗: {e}")
        logger.error("Layer 1 VectorStore test failed", exc_info=True)
        return False


def test_layer2_vectorstore():
    """測試 Layer 2 VectorStore"""
    print_section("測試 3: Layer 2 VectorStore (區塊級索引)")
    
    try:
        embeddings = OllamaEmbeddings(model="embeddinggemma:latest")
        
        test_path = "./vectorstore_test/layer2"
        os.makedirs(test_path, exist_ok=True)
        
        vectorstore = Layer2VectorStore(embeddings, test_path)
        
        # Create test documents (chunks)
        test_docs = []
        for paper_id in ['paper_001', 'paper_002']:
            for chunk_idx in range(5):
                doc = Document(
                    page_content=f"This is chunk {chunk_idx} from {paper_id}. It contains information about various topics in machine learning and artificial intelligence.",
                    metadata={
                        'paper_id': paper_id,
                        'chunk_id': f"{paper_id}_chunk_{chunk_idx}",
                        'chunk_index': chunk_idx,
                        'pdf_path': f'data/{paper_id}.pdf',
                        'page_number': chunk_idx // 2 + 1
                    }
                )
                test_docs.append(doc)
        
        print(f"📚 建立索引 ({len(test_docs)} 個區塊，來自 2 篇論文)...")
        vectorstore.build_index(test_docs)
        
        stats = vectorstore.get_stats()
        print(f"✓ 索引建立成功!")
        print(f"  區塊數量: {stats['chunk_count']}")
        print(f"  論文數量: {stats['paper_count']}")
        print(f"  索引層級: Layer {stats['layer']}")
        
        # Test unfiltered search
        print("\n🔍 測試未過濾搜索: 'machine learning'")
        results = vectorstore.search("machine learning", k=3)
        print(f"✓ 找到 {len(results)} 個區塊:")
        for i, doc in enumerate(results, 1):
            print(f"  {i}. {doc.metadata['chunk_id'][:30]}...")
        
        # Test filtered search
        print("\n🔍 測試過濾搜索 (只搜索 paper_001):")
        filtered_results = vectorstore.search(
            "machine learning",
            k=3,
            filter_paper_ids=['paper_001']
        )
        print(f"✓ 找到 {len(filtered_results)} 個區塊 (全部來自 paper_001):")
        for i, doc in enumerate(filtered_results, 1):
            print(f"  {i}. {doc.metadata['chunk_id']}")
        
        # Test context expansion
        print("\n📖 測試上下文檢索 (chunk 2 前後 ±1 個區塊):")
        context = vectorstore.get_surrounding_chunks(
            paper_id='paper_001',
            chunk_id='paper_001_chunk_2',
            before=1,
            after=1
        )
        if context:
            print(f"✓ 成功檢索上下文 ({len(context)} 字符)")
            print(f"  預覽: {context[:150]}...")
        
        # Test cache
        print("\n💾 測試子索引緩存:")
        cache_stats = vectorstore.get_cache_stats()
        print(f"✓ 緩存統計: {cache_stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Layer 2 VectorStore 測試失敗: {e}")
        logger.error("Layer 2 VectorStore test failed", exc_info=True)
        return False


def test_confidence_evaluator():
    """測試 Confidence Evaluator"""
    print_section("測試 4: Confidence Evaluator (信心評估)")
    
    try:
        llm = Ollama(model="gemma3:12b")
        evaluator = ConfidenceEvaluator(llm)
        
        # Test with Layer 1 documents (abstracts)
        query = "What are the recent advances in deep learning?"
        
        layer1_docs = [
            Document(
                page_content="This paper presents recent advances in deep learning including new architectures and training methods.",
                metadata={'paper_id': 'p1', 'title': 'Deep Learning Advances'}
            ),
            Document(
                page_content="We discuss convolutional neural networks for image processing applications.",
                metadata={'paper_id': 'p2', 'title': 'CNNs for Images'}
            )
        ]
        
        print(f"📊 測試 Layer 1 快速評估...")
        print(f"  查詢: '{query}'")
        print(f"  文檔數: {len(layer1_docs)}")
        
        result = evaluator.evaluate(query, layer1_docs, layer=1)
        
        print(f"✓ 評估完成!")
        print(f"  信心度: {result['confidence']:.2f}")
        print(f"  相關性: {result['relevance']:.2f}")
        print(f"  完整性: {result['completeness']:.2f}")
        print(f"  是否繼續: {result['should_continue']}")
        print(f"  推理: {result['reasoning'][:100]}...")
        
        # Test with Layer 2 documents (chunks)
        layer2_docs = [
            Document(
                page_content="Deep learning has made significant progress in recent years. Transformer architectures have become the dominant paradigm.",
                metadata={'paper_id': 'p1', 'chunk_id': 'c1'}
            )
        ]
        
        print(f"\n📊 測試 Layer 2 詳細評估...")
        result2 = evaluator.evaluate(query, layer2_docs, layer=2)
        
        print(f"✓ 評估完成!")
        print(f"  信心度: {result2['confidence']:.2f}")
        print(f"  相關性: {result2['relevance']:.2f}")
        print(f"  完整性: {result2['completeness']:.2f}")
        print(f"  深度: {result2['depth']:.2f}")
        print(f"  是否繼續: {result2['should_continue']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Confidence Evaluator 測試失敗: {e}")
        logger.error("Confidence Evaluator test failed", exc_info=True)
        return False


def test_context_expander():
    """測試 Context Expander"""
    print_section("測試 5: Context Expander (上下文擴展)")
    
    try:
        # First setup Layer 2 vectorstore
        embeddings = OllamaEmbeddings(model="embeddinggemma:latest")
        test_path = "./vectorstore_test/layer2"
        
        vectorstore = Layer2VectorStore(embeddings, test_path)
        
        # Ensure index is loaded
        if not vectorstore.is_initialized:
            print("⚠️  需要先初始化 Layer 2 索引，使用測試 3 的資料...")
            test_docs = []
            for paper_id in ['paper_001']:
                for chunk_idx in range(5):
                    doc = Document(
                        page_content=f"Content of chunk {chunk_idx} from {paper_id}. " * 10,
                        metadata={
                            'paper_id': paper_id,
                            'chunk_id': f"{paper_id}_chunk_{chunk_idx}",
                            'chunk_index': chunk_idx,
                            'pdf_path': f'data/{paper_id}.pdf',
                        }
                    )
                    test_docs.append(doc)
            vectorstore.build_index(test_docs)
        
        # Initialize expander
        expander = ContextExpander(vectorstore)
        
        # Create test chunks
        test_chunks = [
            Document(
                page_content="This is the target chunk.",
                metadata={
                    'paper_id': 'paper_001',
                    'chunk_id': 'paper_001_chunk_2',
                    'chunk_index': 2
                }
            )
        ]
        
        # Test different confidence levels
        confidence_levels = [0.9, 0.8, 0.7]
        
        for confidence in confidence_levels:
            print(f"\n📖 測試信心度 {confidence} 的擴展:")
            
            stats = expander.get_expansion_stats(confidence)
            print(f"  策略: {stats['strategy']}")
            print(f"  擴展範圍: ±{stats['expansion_range']}")
            print(f"  總區塊數: {stats['total_chunks_per_context']}")
            
            expanded = expander.expand(test_chunks, confidence)
            if expanded:
                print(f"✓ 擴展成功! 上下文長度: {len(expanded[0])} 字符")
                print(f"  預覽: {expanded[0][:100]}...")
        
        # Test with metadata
        print(f"\n📊 測試帶元數據的擴展:")
        results = expander.expand_with_metadata(test_chunks, 0.75)
        
        print(f"✓ 擴展完成!")
        for i, result in enumerate(results, 1):
            print(f"  結果 {i}:")
            print(f"    Paper ID: {result['paper_id']}")
            print(f"    Chunk ID: {result['chunk_id']}")
            print(f"    擴展範圍: ±{result['expansion_range']}")
            print(f"    信心度: {result['confidence']}")
            print(f"    上下文長度: {len(result['context'])} 字符")
        
        return True
        
    except Exception as e:
        print(f"❌ Context Expander 測試失敗: {e}")
        logger.error("Context Expander test failed", exc_info=True)
        return False


def main():
    """執行所有測試"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "Hierarchical RAG 組件測試" + " "*33 + "║")
    print("╚" + "="*78 + "╝")
    
    results = {
        'Abstract Extractor': False,
        'Layer 1 VectorStore': False,
        'Layer 2 VectorStore': False,
        'Confidence Evaluator': False,
        'Context Expander': False
    }
    
    # Run tests
    results['Abstract Extractor'] = test_abstract_extractor()
    results['Layer 1 VectorStore'] = test_layer1_vectorstore()
    results['Layer 2 VectorStore'] = test_layer2_vectorstore()
    results['Confidence Evaluator'] = test_confidence_evaluator()
    results['Context Expander'] = test_context_expander()
    
    # Summary
    print_section("測試總結")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for component, success in results.items():
        status = "✓ 通過" if success else "❌ 失敗"
        print(f"  {status}: {component}")
    
    print(f"\n總計: {passed}/{total} 個組件測試通過")
    
    if passed == total:
        print("\n🎉 所有組件測試通過！可以開始整合工作。")
    else:
        print("\n⚠️  部分組件測試失敗，請檢查錯誤日誌。")
    
    return passed == total


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  測試被用戶中斷")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 測試過程發生錯誤: {e}")
        logger.error("Test suite failed", exc_info=True)
        sys.exit(1)
