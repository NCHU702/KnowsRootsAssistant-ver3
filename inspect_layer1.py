#!/usr/bin/env python3
"""
Layer 1 資料結構檢視工具

使用方式:
    python inspect_layer1.py
    python inspect_layer1.py --query "人流預測"
    python inspect_layer1.py --show-all
"""

import os
import sys
import argparse
import logging
import math
from typing import List, Tuple

# 設定路徑
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from system_api.layer1_vectorstore import Layer1VectorStore
from langchain_ollama import OllamaEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def print_separator(char="=", length=80):
    """印出分隔線"""
    print(char * length)


def print_document_detail(doc, similarity=None, index=None):
    """印出 Document 詳細資訊"""
    metadata = doc.metadata
    
    if index is not None:
        print(f"\n【論文 {index}】")
    else:
        print(f"\n【論文資訊】")
    
    if similarity is not None:
        print(f"相似度: {similarity:.4f}")
    
    print(f"標題: {metadata.get('title', 'Unknown')}")
    print(f"Paper ID: {metadata.get('paper_id', 'Unknown')}")
    print(f"作者: {', '.join(metadata.get('authors', []))}")
    print(f"年份: {metadata.get('year', 'Unknown')}")
    print(f"PDF: {metadata.get('pdf_path', 'Unknown')}")
    print(f"摘要來源: {metadata.get('abstract_source', 'Unknown')}")
    print(f"信心度: {metadata.get('abstract_confidence', 0.0):.2f}")
    print(f"Layer: {metadata.get('layer', 'Unknown')}")
    print(f"\n摘要內容 (前 200 字):")
    print(f"  {doc.page_content[:200]}...")


def show_statistics(layer1: Layer1VectorStore):
    """顯示統計資訊"""
    print_separator()
    print("📊 Layer 1 統計資訊")
    print_separator()
    
    stats = layer1.get_stats()
    print(f"Layer: {stats['layer']}")
    print(f"類型: {stats['type']}")
    print(f"論文數量: {stats['paper_count']}")
    print(f"已初始化: {stats['is_initialized']}")
    print(f"索引路徑: {stats['index_path']}")
    
    if layer1.vectorstore:
        print(f"\nFAISS 詳細資訊:")
        print(f"  向量總數: {layer1.vectorstore.index.ntotal}")
        print(f"  向量維度: {layer1.vectorstore.index.d}")
        print(f"  Document 數量: {len(layer1.vectorstore.docstore._dict)}")


def show_all_papers(layer1: Layer1VectorStore):
    """顯示所有論文列表"""
    print_separator()
    print("📚 所有論文列表")
    print_separator()
    
    if not layer1.vectorstore:
        print("❌ VectorStore 未初始化")
        return
    
    all_docs = list(layer1.vectorstore.docstore._dict.values())
    
    for idx, doc in enumerate(all_docs, 1):
        metadata = doc.metadata
        print(f"\n{idx}. {metadata.get('title', 'Unknown')}")
        print(f"   ID: {metadata.get('paper_id')}")
        print(f"   作者: {', '.join(metadata.get('authors', [])[:3])}")
        print(f"   年份: {metadata.get('year')}")
        print(f"   摘要長度: {len(doc.page_content)} 字")


def test_search(layer1: Layer1VectorStore, query: str, threshold: float = None):
    """測試搜尋功能"""
    print_separator()
    print(f"🔍 搜尋測試: \"{query}\"")
    if threshold:
        print(f"   閾值: {threshold}")
    print_separator()
    
    # 搜尋
    results_with_scores = layer1.search_with_scores(
        query=query,
        k=10,
        score_threshold=threshold
    )
    
    if not results_with_scores:
        print("\n❌ 沒有找到任何論文")
        if threshold:
            print(f"\n💡 提示: 閾值 {threshold} 可能太高，試試:")
            print(f"   python inspect_layer1.py --query \"{query}\" --threshold 0.3")
        return
    
    print(f"\n✓ 找到 {len(results_with_scores)} 篇相關論文:\n")
    
    for idx, (doc, similarity) in enumerate(results_with_scores, 1):
        metadata = doc.metadata
        print(f"{idx}. [{similarity:.4f}] {metadata.get('title', 'Unknown')}")
        print(f"   ID: {metadata.get('paper_id')}")
        print(f"   作者: {', '.join(metadata.get('authors', [])[:2])}")
        print(f"   年份: {metadata.get('year')}")
        print(f"   摘要: {doc.page_content[:100]}...")
        print()


def analyze_threshold_impact(layer1: Layer1VectorStore, query: str):
    """分析不同閾值的影響"""
    print_separator()
    print(f"📈 閾值影響分析: \"{query}\"")
    print_separator()
    
    # 取得所有結果 (無閾值)
    all_results = layer1.search_with_scores(
        query=query,
        k=50,
        score_threshold=None
    )
    
    if not all_results:
        print("\n❌ 沒有找到任何論文")
        return
    
    print(f"\n找到 {len(all_results)} 篇論文，分數分布:\n")
    
    # 分數統計
    scores = [sim for _, sim in all_results]
    print(f"最高分數: {max(scores):.4f}")
    print(f"最低分數: {min(scores):.4f}")
    print(f"平均分數: {sum(scores)/len(scores):.4f}")
    
    # 測試不同閾值
    print(f"\n不同閾值的通過率:")
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    
    for threshold in thresholds:
        passed = sum(1 for s in scores if s >= threshold)
        rate = (passed / len(scores)) * 100
        bar = "█" * int(rate / 5)
        print(f"  {threshold:.1f}: {passed:2d}/{len(scores):2d} ({rate:5.1f}%) {bar}")
    
    # 顯示前 10 篇的詳細分數
    print(f"\n前 10 篇論文分數:")
    for idx, (doc, sim) in enumerate(all_results[:10], 1):
        title = doc.metadata.get('title', 'Unknown')
        # 計算距離 (反推)
        distance = ((1/sim) - 1) ** 2
        print(f"  {idx}. [{sim:.4f}] (距離={distance:.4f}) {title[:60]}")


def compare_with_raw_distance(layer1: Layer1VectorStore, query: str):
    """比較距離與相似度的關係"""
    print_separator()
    print(f"📐 距離 vs 相似度對照")
    print_separator()
    
    # 獲取結果
    results = layer1.search_with_scores(
        query=query,
        k=10,
        score_threshold=None
    )
    
    if not results:
        print("\n❌ 沒有找到任何論文")
        return
    
    print(f"\n查詢: \"{query}\"\n")
    print(f"{'排名':<6} {'距離':<10} {'相似度':<10} {'標題':<50}")
    print("-" * 80)
    
    for idx, (doc, similarity) in enumerate(results, 1):
        # 反推距離
        distance = ((1/similarity) - 1) ** 2
        title = doc.metadata.get('title', 'Unknown')[:45]
        print(f"{idx:<6} {distance:<10.4f} {similarity:<10.4f} {title}")
    
    print("\n公式: similarity = 1 / (1 + sqrt(distance))")
    print("說明: 距離越小 → 相似度越高")


def show_faiss_internals(layer1: Layer1VectorStore):
    """顯示 FAISS 內部結構"""
    print_separator()
    print("🔬 FAISS 內部結構")
    print_separator()
    
    if not layer1.vectorstore:
        print("\n❌ VectorStore 未初始化")
        return
    
    vs = layer1.vectorstore
    
    print("\n1. FAISS Index:")
    print(f"   類型: {type(vs.index).__name__}")
    print(f"   向量總數: {vs.index.ntotal}")
    print(f"   向量維度: {vs.index.d}")
    print(f"   度量方式: L2 (歐氏距離)")
    print(f"   是否訓練: {vs.index.is_trained}")
    
    print("\n2. Document Store:")
    print(f"   類型: {type(vs.docstore).__name__}")
    print(f"   Document 數量: {len(vs.docstore._dict)}")
    
    print("\n3. Index to DocStore ID 映射:")
    print(f"   映射數量: {len(vs.index_to_docstore_id)}")
    print(f"   範例映射 (前 5 個):")
    for i in list(vs.index_to_docstore_id.keys())[:5]:
        doc_id = vs.index_to_docstore_id[i]
        doc = vs.docstore._dict[doc_id]
        title = doc.metadata.get('title', 'Unknown')[:40]
        print(f"     FAISS_idx[{i}] → DocID[{doc_id[:8]}...] → {title}")
    
    print("\n4. 檔案結構:")
    path = layer1.vectorstore_path
    if os.path.exists(path):
        for f in os.listdir(path):
            fpath = os.path.join(path, f)
            size = os.path.getsize(fpath) / 1024  # KB
            print(f"   {f}: {size:.1f} KB")


def main():
    parser = argparse.ArgumentParser(description="Layer 1 資料結構檢視工具")
    parser.add_argument("--query", "-q", type=str, help="測試查詢")
    parser.add_argument("--threshold", "-t", type=float, help="相似度閾值")
    parser.add_argument("--show-all", action="store_true", help="顯示所有論文")
    parser.add_argument("--analyze", "-a", action="store_true", help="分析閾值影響")
    parser.add_argument("--compare", "-c", action="store_true", help="比較距離與相似度")
    parser.add_argument("--internals", "-i", action="store_true", help="顯示 FAISS 內部結構")
    parser.add_argument("--embedding-model", default="mxbai-embed-large", 
                       help="Embedding 模型 (預設: mxbai-embed-large)")
    
    args = parser.parse_args()
    
    # 初始化
    print("🚀 初始化 Layer 1 VectorStore...")
    embeddings = OllamaEmbeddings(model=args.embedding_model)
    layer1 = Layer1VectorStore(embeddings)
    
    # 載入索引
    if not layer1.load():
        print("❌ 無法載入 Layer 1 索引")
        print("請先執行 agent2.py 建立索引")
        return
    
    print("✓ Layer 1 載入成功\n")
    
    # 執行功能
    show_statistics(layer1)
    
    if args.show_all:
        show_all_papers(layer1)
    
    if args.internals:
        show_faiss_internals(layer1)
    
    if args.query:
        if args.analyze:
            analyze_threshold_impact(layer1, args.query)
        elif args.compare:
            compare_with_raw_distance(layer1, args.query)
        else:
            test_search(layer1, args.query, args.threshold)
    
    # 如果沒有指定任何操作，顯示使用說明
    if not any([args.show_all, args.query, args.internals]):
        print_separator()
        print("💡 使用說明")
        print_separator()
        print("\n基本操作:")
        print("  python inspect_layer1.py                          # 顯示統計資訊")
        print("  python inspect_layer1.py --show-all               # 列出所有論文")
        print("  python inspect_layer1.py --internals              # 檢視 FAISS 結構")
        print("\n搜尋測試:")
        print("  python inspect_layer1.py --query '人流預測'       # 測試搜尋")
        print("  python inspect_layer1.py -q '深度學習' -t 0.5    # 使用閾值搜尋")
        print("\n進階分析:")
        print("  python inspect_layer1.py -q '人流預測' --analyze  # 分析閾值影響")
        print("  python inspect_layer1.py -q '人流預測' --compare  # 距離 vs 相似度")
        print()


if __name__ == "__main__":
    main()
