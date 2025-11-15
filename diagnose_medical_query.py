#!/usr/bin/env python3
"""
診斷查詢「深度學習在醫療的應用是什麼？」為何返回不相關結果
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from system_api.hierarchical_rag_system import HierarchicalRAGSystem
from system_api.query_expander import QueryExpander
from system_api.adaptive_weights import AdaptiveWeightAdjuster
from langchain_ollama import OllamaLLM
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)

print("=" * 80)
print("診斷查詢: 深度學習在醫療的應用是什麼？")
print("=" * 80)

# 初始化系統
print("\n1. 初始化系統...")
rag = HierarchicalRAGSystem(
    pdf_directory='./data',
    model_name='jcai/llama-3-taiwan-8b-instruct:q4_k_m',
    embedding_model='quentinz/bge-large-zh-v1.5:latest',
    vectorstore_path='./vectorstore',
    config={
        'layer1': {
            'k_documents': 10,
            'similarity_threshold': 0.48
        },
        'hybrid_search': {
            'enabled': True,
            'semantic_weight': 0.5,
            'keyword_weight': 0.5,
            'use_jieba': True
        },
        'query_expansion': {'enabled': True},
        'adaptive_weights': {'enabled': True}
    }
)

query = "深度學習在醫療的應用是什麼？"

# 測試查詢擴展
print("\n2. 測試查詢擴展...")
if rag.query_expander:
    expanded = rag.query_expander.expand(query)
    print(f"   原始: {query}")
    print(f"   擴展: {expanded}")
else:
    expanded = query
    print("   查詢擴展未啟用")

# 測試權重調整
print("\n3. 測試權重調整...")
if rag.weight_adjuster:
    sem_w, key_w, reason = rag.weight_adjuster.adjust_weights(expanded)
    print(f"   語義權重: {sem_w:.2f}")
    print(f"   關鍵詞權重: {key_w:.2f}")
    print(f"   原因: {reason}")
else:
    sem_w, key_w = 0.5, 0.5
    print("   權重調整未啟用")

# 測試純語義搜尋
print("\n4. 測試純語義搜尋 (FAISS)...")
semantic_results = rag.layer1.search_with_scores(query=expanded, k=10, score_threshold=None)
print(f"   找到 {len(semantic_results)} 篇論文")
print("   Top 5:")
for i, (doc, score) in enumerate(semantic_results[:5], 1):
    title = doc.metadata.get('title', 'Unknown')
    print(f"   {i}. [{score:.3f}] {title}")

# 測試 BM25 搜尋
print("\n5. 測試 BM25 關鍵詞搜尋...")
if rag.hybrid_retriever and rag.hybrid_retriever.bm25:
    query_tokens = rag.hybrid_retriever._tokenize(expanded)
    print(f"   查詢分詞: {query_tokens}")
    
    bm25_scores = rag.hybrid_retriever.bm25.get_scores(query_tokens)
    max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 else 1.0
    normalized_bm25 = bm25_scores / max_bm25 if max_bm25 > 0 else bm25_scores
    
    # 找出 BM25 分數最高的論文
    top_bm25_indices = normalized_bm25.argsort()[-10:][::-1]
    print(f"   BM25 分數範圍: [{normalized_bm25.min():.3f}, {normalized_bm25.max():.3f}]")
    print("   Top 5:")
    for i, idx in enumerate(top_bm25_indices[:5], 1):
        doc = rag.hybrid_retriever.documents[idx]
        score = normalized_bm25[idx]
        title = doc.metadata.get('title', 'Unknown')
        print(f"   {i}. [{score:.3f}] {title}")
else:
    print("   BM25 索引未建立")

# 測試混合搜尋
print("\n6. 測試混合搜尋 (權重 {}/{})...".format(sem_w, key_w))
if rag.hybrid_retriever:
    hybrid_results = rag.hybrid_retriever.hybrid_search(
        query=expanded,
        k=10,
        semantic_weight=sem_w,
        keyword_weight=key_w,
        score_threshold=0.48,
        return_scores_breakdown=True
    )
    
    print(f"   找到 {len(hybrid_results)} 篇論文")
    print("   Top 5 (組合分數, 語義分數, 關鍵詞分數):")
    for i, (doc, scores) in enumerate(hybrid_results[:5], 1):
        combined, semantic, keyword = scores
        title = doc.metadata.get('title', 'Unknown')
        print(f"   {i}. 組合:{combined:.3f} = 語義:{semantic:.3f} + 關鍵詞:{keyword:.3f}")
        print(f"      {title}")

# 檢查醫療相關論文
print("\n7. 檢查醫療相關論文的分數...")
medical_papers = [
    "基於深度學習之鼻咽癌腫塊辨識",
    "應用混合式深度學習開發適應於小資料集之鼻咽癌分類模型"
]

for paper_title_part in medical_papers:
    print(f"\n   查找: {paper_title_part}")
    
    # 在語義結果中查找
    for doc, score in semantic_results:
        if paper_title_part in doc.metadata.get('title', ''):
            print(f"   ✓ 語義搜尋: {score:.3f}")
            break
    else:
        print(f"   ✗ 語義搜尋: 未找到")
    
    # 在混合結果中查找
    if rag.hybrid_retriever:
        for doc, scores in hybrid_results:
            if paper_title_part in doc.metadata.get('title', ''):
                combined, semantic, keyword = scores
                print(f"   ✓ 混合搜尋: 組合={combined:.3f}, 語義={semantic:.3f}, 關鍵詞={keyword:.3f}")
                break
        else:
            print(f"   ✗ 混合搜尋: 未找到")

print("\n" + "=" * 80)
print("診斷完成")
print("=" * 80)
