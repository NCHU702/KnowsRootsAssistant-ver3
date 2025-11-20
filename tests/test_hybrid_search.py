#!/usr/bin/env python3
"""
測試混合檢索功能

比較純語義搜尋 vs 混合檢索的效果
"""

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_ollama import OllamaEmbeddings
from system_api.layer1_vectorstore import Layer1VectorStore
from system_api.hybrid_retriever import HybridRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def print_results(title, results, show_scores=True):
    """印出搜尋結果"""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    
    if not results:
        print("  ❌ 沒有找到任何結果")
        return
    
    print(f"  找到 {len(results)} 篇論文:\n")
    
    for idx, item in enumerate(results, 1):
        if isinstance(item, tuple):
            doc, score = item
            title_text = doc.metadata.get('title', 'Unknown')
            if show_scores:
                print(f"  {idx}. [{score:.4f}] {title_text}")
            else:
                print(f"  {idx}. {title_text}")
        else:
            doc = item
            title_text = doc.metadata.get('title', 'Unknown')
            print(f"  {idx}. {title_text}")


def test_query(query, embeddings):
    """測試單一查詢"""
    print(f"\n\n{'#'*80}")
    print(f"測試查詢: 「{query}」")
    print(f"{'#'*80}")
    
    # 初始化
    layer1 = Layer1VectorStore(embeddings)
    if not layer1.load():
        print("❌ 無法載入 Layer 1 索引")
        return
    
    # 1. 純語義搜尋
    print("\n[方法 1] 純語義搜尋 (Semantic Search Only)")
    semantic_results = layer1.search_with_scores(
        query=query,
        k=10,
        score_threshold=None
    )
    print_results("純語義搜尋結果", semantic_results)
    
    # 2. 混合檢索
    print("\n[方法 2] 混合檢索 (Hybrid Search)")
    hybrid = HybridRetriever(layer1, use_jieba=True, build_index_on_init=True)
    
    hybrid_results = hybrid.hybrid_search(
        query=query,
        k=10,
        semantic_weight=0.5,
        keyword_weight=0.5,
        score_threshold=None,
        return_scores_breakdown=True
    )
    
    if hybrid_results:
        print(f"\n  找到 {len(hybrid_results)} 篇論文:\n")
        for idx, (doc, scores) in enumerate(hybrid_results, 1):
            combined, semantic, keyword = scores
            title = doc.metadata.get('title', 'Unknown')
            print(f"  {idx}. [{combined:.4f}] {title}")
            print(f"      (語義:{semantic:.4f} + 關鍵詞:{keyword:.4f})")
    else:
        print("  ❌ 沒有找到任何結果")
    
    # 3. 比較差異
    print("\n\n[比較分析]")
    print("-" * 80)
    
    semantic_titles = set(doc.metadata.get('title') for doc, _ in semantic_results[:5])
    hybrid_titles = set(doc.metadata.get('title') for doc, _ in hybrid_results[:5])
    
    only_in_semantic = semantic_titles - hybrid_titles
    only_in_hybrid = hybrid_titles - semantic_titles
    in_both = semantic_titles & hybrid_titles
    
    print(f"前 5 名重疊論文: {len(in_both)} 篇")
    
    if only_in_semantic:
        print(f"\n只在純語義搜尋中出現 ({len(only_in_semantic)} 篇):")
        for title in only_in_semantic:
            print(f"  - {title[:60]}")
    
    if only_in_hybrid:
        print(f"\n只在混合檢索中出現 ({len(only_in_hybrid)} 篇):")
        for title in only_in_hybrid:
            print(f"  - {title[:60]}")


def main():
    """主程式"""
    print("="*80)
    print("混合檢索測試工具")
    print("="*80)
    
    # 初始化 embeddings
    embedding_model = "quentinz/bge-large-zh-v1.5:latest"
    print(f"\n使用 Embedding 模型: {embedding_model}")
    embeddings = OllamaEmbeddings(model=embedding_model)
    
    # 測試查詢
    test_queries = [
        "深度學習在人流的應用是什麼",
        "如何使用神經網路預測人群密度",
        "LSTM 模型在時間序列預測的優勢",
        "捷運站的人流分析方法",
    ]
    
    for query in test_queries:
        test_query(query, embeddings)
    
    print("\n\n" + "="*80)
    print("測試完成！")
    print("="*80)
    
    print("\n總結:")
    print("  1. 純語義搜尋: 可能只匹配主要概念（如「深度學習」）")
    print("  2. 混合檢索: 確保次要概念也被考慮（如「人流」）")
    print("  3. 混合檢索更適合多概念查詢\n")


if __name__ == "__main__":
    main()
