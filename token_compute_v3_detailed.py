"""
Rigorous Token Consumption Comparative Analysis System
======================================================
Comparing three RAG architectures across two main phases:
1. Naive RAG
2. Microsoft GraphRAG
3. KnowRoots (Our System)

Test Data: Single master thesis (~40,000 words, 48,000 tokens)
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from dataclasses import dataclass, field

# Configure fonts for academic style
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.unicode_minus'] = False

# ==========================================
# 系統參數設定
# ==========================================

# 論文規模
PAPER_WORD_COUNT = 40000  # 字數
PAPER_TOKEN_COUNT = 60000  # Token 數（中文約 1.5 倍）
CHUNK_SIZE = 500  # 每個 chunk 的 token 數
CHUNKS_PER_PAPER = PAPER_TOKEN_COUNT // CHUNK_SIZE  # 約 120 個 chunks

# GPT-4o-mini Token 消耗標準（用於計算，不計算金額）
# Input tokens / Output tokens / Embedding tokens 的相對權重
INPUT_WEIGHT = 1.0
OUTPUT_WEIGHT = 4.0  # Output 約為 Input 的 4 倍消耗
EMBED_WEIGHT = 0.13  # Embedding 約為 Input 的 0.13 倍

@dataclass
class TokenUsage:
    """Token 使用量記錄"""
    llm_input: int = 0
    llm_output: int = 0
    embedding: int = 0
    
    def total_weighted(self) -> float:
        """計算加權總 token 數"""
        return (self.llm_input * INPUT_WEIGHT + 
                self.llm_output * OUTPUT_WEIGHT + 
                self.embedding * EMBED_WEIGHT)
    
    def __add__(self, other):
        return TokenUsage(
            self.llm_input + other.llm_input,
            self.llm_output + other.llm_output,
            self.embedding + other.embedding
        )

@dataclass
class PhaseBreakdown:
    """階段詳細分解"""
    extraction: TokenUsage = field(default_factory=TokenUsage)
    summarization: TokenUsage = field(default_factory=TokenUsage)
    storage: TokenUsage = field(default_factory=TokenUsage)
    
    def total(self) -> TokenUsage:
        return self.extraction + self.summarization + self.storage

@dataclass
class QueryBreakdown:
    """查詢階段詳細分解"""
    retrieval_strategy: TokenUsage = field(default_factory=TokenUsage)
    data_retrieval: TokenUsage = field(default_factory=TokenUsage)
    answer_generation: TokenUsage = field(default_factory=TokenUsage)
    
    def total(self) -> TokenUsage:
        return self.retrieval_strategy + self.data_retrieval + self.answer_generation


# ==========================================
# Phase 1: 資料建置與索引階段
# ==========================================

class Phase1_DataIndexing:
    """資料建置與索引階段的 Token 計算"""
    
    @staticmethod
    def naive_rag() -> PhaseBreakdown:
        """
        Naive RAG 建置流程：
        1. Extraction: 無需 LLM，直接分割文本
        2. Summarization: 不做摘要
        3. Storage: Embedding 所有 chunks
        """
        breakdown = PhaseBreakdown()
        
        # 1. Extraction: 直接分割，無 token 消耗
        breakdown.extraction = TokenUsage(0, 0, 0)
        
        # 2. Summarization: 不執行
        breakdown.summarization = TokenUsage(0, 0, 0)
        
        # 3. Storage: 對所有 chunks 進行 embedding
        breakdown.storage = TokenUsage(
            llm_input=0,
            llm_output=0,
            embedding=PAPER_TOKEN_COUNT  # 全文 embedding
        )
        
        return breakdown
    
    @staticmethod
    def microsoft_graphrag() -> PhaseBreakdown:
        """
        Microsoft GraphRAG 建置流程：
        1. Extraction: 多次完整讀取原文，提取實體和關係
        2. Summarization: 社群檢測後，對每個社群生成多層次摘要
        3. Storage: Embedding 實體、關係和摘要
        
        根據 Microsoft 論文與實測，token 消耗約為原文的 10-15 倍
        """
        breakdown = PhaseBreakdown()
        
        # 1. Extraction: 實體與關係提取（需要多次讀取原文）
        # - 第一輪: 全文實體提取（需讀取全文 + 生成實體列表）
        # - 第二輪: 關係提取（再次讀取全文 + 生成關係）
        # - 第三輪: 實體去重與合併（讀取中間結果 + 生成合併後的實體）
        extraction_input = PAPER_TOKEN_COUNT * 3  # 三次讀取
        extraction_output = PAPER_TOKEN_COUNT * 0.5  # 生成約 50% 的實體關係描述
        
        breakdown.extraction = TokenUsage(
            llm_input=extraction_input,
            llm_output=extraction_output,
            embedding=0
        )
        
        # 2. Summarization: 社群摘要（Hierarchical Community Summarization）
        # - 假設生成 10 個社群
        # - 每個社群需要讀取其所有實體關係 (約 4000 tokens/社群)
        # - 生成多層次摘要 (L0, L1, L2...)，每層約 1000 tokens
        num_communities = 10
        summarization_input = num_communities * 4000  # 讀取社群內容
        summarization_output = num_communities * 2000  # 每個社群生成約 2000 tokens 摘要
        
        breakdown.summarization = TokenUsage(
            llm_input=summarization_input,
            llm_output=summarization_output,
            embedding=0
        )
        
        # 3. Storage: Embedding 所有生成的內容
        # - 實體描述: ~8000 tokens
        # - 關係描述: ~8000 tokens  
        # - 社群摘要: ~20000 tokens
        total_embedding = 36000
        
        breakdown.storage = TokenUsage(
            llm_input=0,
            llm_output=0,
            embedding=total_embedding
        )
        
        return breakdown
    
    @staticmethod
    def improved_graphrag() -> PhaseBreakdown:
        """
        KnowRoots 建置流程（本系統）：
        1. Extraction: 智能截斷（前 6000 字符）提取結構化 metadata
        2. Summarization: Layer1 摘要（論文級別）
        3. Storage: Layer1 摘要 embedding + Layer2 全文 chunks embedding
        """
        breakdown = PhaseBreakdown()
        
        # 1. Extraction: Graph Metadata 提取
        # - 智能截斷: 只處理前 6000 字符 + 關鍵片段
        # - Input: 前 6000 chars (~7200 tokens) + system prompt (800 tokens)
        # - Output: JSON metadata (Paper, Method, Dataset, etc. ~800 tokens)
        breakdown.extraction = TokenUsage(
            llm_input=8000,   # 7200 + 800 prompt
            llm_output=800,   # 結構化 JSON
            embedding=0
        )
        
        # 2. Summarization: Layer1 摘要生成
        # - 輸入：論文前 6000 字符 + system prompt
        # - 輸出：結構化摘要（研究目標、方法、結果）
        breakdown.summarization = TokenUsage(
            llm_input=8000,   # 同 extraction，可能使用相同截斷內容
            llm_output=1000,  # 摘要約 1000 tokens
            embedding=0
        )
        
        # 3. Storage: 雙層 Embedding
        # - Layer1: 論文摘要 embedding (~1000 tokens)
        # - Layer2: 全文 chunks embedding (~48000 tokens)
        breakdown.storage = TokenUsage(
            llm_input=0,
            llm_output=0,
            embedding=1000 + PAPER_TOKEN_COUNT  # Layer1 + Layer2
        )
        
        return breakdown


# ==========================================
# Phase 2: 詢問與回答階段
# ==========================================

class Phase2_QueryAnswering:
    """詢問與回答階段的 Token 計算"""
    
    # 測試查詢範例（跨論文比較問題）
    QUERY_EXAMPLE = "比較深度學習模型在醫學影像辨識中的表現差異，分析 CNN 和 Transformer 架構的優缺點"
    QUERY_TOKENS = 25
    
    @staticmethod
    def naive_rag() -> QueryBreakdown:
        """
        Naive RAG 查詢流程：
        1. Retrieval Strategy: 無策略，直接向量檢索
        2. Data Retrieval: Top-K chunks 檢索（假設 K=10）
        3. Answer Generation: 基於檢索的 chunks 生成答案
        """
        breakdown = QueryBreakdown()
        
        # 1. Retrieval Strategy: 無需 LLM
        breakdown.retrieval_strategy = TokenUsage(0, 0, 0)
        
        # 2. Data Retrieval: 向量檢索（無 token 消耗）
        # 但需要返回的 chunks 會在下一步使用
        breakdown.data_retrieval = TokenUsage(0, 0, 0)
        
        # 3. Answer Generation: 
        # - Input: System prompt (500) + Query (25) + Top-10 chunks (10 * 500)
        # - Output: 答案 (600 tokens)
        k = 10
        breakdown.answer_generation = TokenUsage(
            llm_input=500 + Phase2_QueryAnswering.QUERY_TOKENS + (k * CHUNK_SIZE),
            llm_output=600,
            embedding=Phase2_QueryAnswering.QUERY_TOKENS  # Query embedding
        )
        
        return breakdown
    
    @staticmethod
    def microsoft_graphrag() -> QueryBreakdown:
        """
        Microsoft GraphRAG 查詢流程（Global Search）：
        1. Retrieval Strategy: 識別相關社群
        2. Data Retrieval: 並行 Map-Reduce
        3. Answer Generation: Reduce 階段生成最終答案
        """
        breakdown = QueryBreakdown()
        
        # 1. Retrieval Strategy: 向量檢索找出相關社群（無 token 消耗）
        breakdown.retrieval_strategy = TokenUsage(
            llm_input=0,
            llm_output=0,
            embedding=Phase2_QueryAnswering.QUERY_TOKENS  # Query embedding
        )
        
        # 2. Data Retrieval: Map 階段
        # - 假設找到 10 個相關社群
        # - 每個社群並行處理: Input = system prompt (500) + query (50) + 社群摘要 (1000)
        # - 每個社群生成中間答案: Output = 600 tokens
        num_communities = 10
        map_input = num_communities * (500 + Phase2_QueryAnswering.QUERY_TOKENS + 1000)
        map_output = num_communities * 600
        
        breakdown.data_retrieval = TokenUsage(
            llm_input=map_input,
            llm_output=map_output,
            embedding=0
        )
        
        # 3. Answer Generation: Reduce 階段
        # - Input: System prompt (500) + Query (50) + 10個中間答案 (10 * 600)
        # - Output: 最終答案 (800 tokens)
        breakdown.answer_generation = TokenUsage(
            llm_input=500 + Phase2_QueryAnswering.QUERY_TOKENS + map_output,
            llm_output=800,
            embedding=0
        )
        
        return breakdown
    
    @staticmethod
    def improved_graphrag() -> QueryBreakdown:
        """
        KnowRoots 查詢流程：
        1. Retrieval Strategy: Query Router 判斷路徑 + Graph 查詢
        2. Data Retrieval: Layer1 檢查 → Layer2 精確檢索
        3. Answer Generation: 基於精確檢索結果生成答案
        """
        breakdown = QueryBreakdown()
        
        # 1. Retrieval Strategy: 
        # - Query Router: Input = system prompt (300) + query (50)
        #                Output = 路由決策 (100 tokens)
        # - Graph Query (Text2Cypher): Input = prompt (400) + query (50)
        #                              Output = Cypher query (150 tokens)
        breakdown.retrieval_strategy = TokenUsage(
            llm_input=300 + Phase2_QueryAnswering.QUERY_TOKENS + 400 + Phase2_QueryAnswering.QUERY_TOKENS,
            llm_output=100 + 150,
            embedding=Phase2_QueryAnswering.QUERY_TOKENS  # Query embedding
        )
        
        # 2. Data Retrieval: Layer1 → Layer2
        # - Layer1: 向量檢索論文摘要（無 token 消耗）
        # - Layer2: 針對鎖定的論文檢索 Top-5 chunks（無 token 消耗）
        # - Confidence Evaluation: Input = prompt (200) + chunks (5 * 500)
        #                         Output = 信心評估 (50 tokens)
        breakdown.data_retrieval = TokenUsage(
            llm_input=200 + (5 * CHUNK_SIZE),
            llm_output=50,
            embedding=0
        )
        
        # 3. Answer Generation:
        # - Input: System prompt (500) + Query (50) + Top-5 chunks (5 * 500)
        # - Output: 答案 (700 tokens)
        # - 若信心度低，可能觸發 Context Expansion (+2 chunks)
        # 這裡以平均情況計算（不需要擴展）
        breakdown.answer_generation = TokenUsage(
            llm_input=500 + Phase2_QueryAnswering.QUERY_TOKENS + (5 * CHUNK_SIZE),
            llm_output=700,
            embedding=0
        )
        
        return breakdown


# ==========================================
# 結果整合與視覺化
# ==========================================

class TokenComparisonAnalyzer:
    """Token 消耗比較分析器"""
    
    def __init__(self):
        # Phase 1: 資料建置
        self.phase1_naive = Phase1_DataIndexing.naive_rag()
        self.phase1_ms = Phase1_DataIndexing.microsoft_graphrag()
        self.phase1_improved = Phase1_DataIndexing.improved_graphrag()
        
        # Phase 2: 查詢回答
        self.phase2_naive = Phase2_QueryAnswering.naive_rag()
        self.phase2_ms = Phase2_QueryAnswering.microsoft_graphrag()
        self.phase2_improved = Phase2_QueryAnswering.improved_graphrag()
    
    def print_phase1_details(self):
        """列印 Phase 1 詳細分解"""
        print("\n" + "="*80)
        print("PHASE 1: 資料建置與索引階段 (Data Indexing Phase)")
        print("="*80)
        print(f"測試資料: 單篇碩士論文 ({PAPER_WORD_COUNT:,} 字, {PAPER_TOKEN_COUNT:,} tokens)\n")
        
        methods = {
            "Naive RAG": self.phase1_naive,
            "Microsoft GraphRAG": self.phase1_ms,
            "KnowRoots": self.phase1_improved
        }
        
        for method_name, breakdown in methods.items():
            print(f"\n📊 {method_name}")
            print("-" * 80)
            
            # 1. Extraction
            ext = breakdown.extraction
            print(f"  1️⃣ Extraction (資料提取)")
            print(f"     LLM Input:  {ext.llm_input:>8,} tokens")
            print(f"     LLM Output: {ext.llm_output:>8,} tokens")
            print(f"     Embedding:  {ext.embedding:>8,} tokens")
            print(f"     小計:       {ext.total_weighted():>8,.1f} tokens")
            
            # 2. Summarization
            summ = breakdown.summarization
            print(f"\n  2️⃣ Summarization (摘要生成)")
            print(f"     LLM Input:  {summ.llm_input:>8,} tokens")
            print(f"     LLM Output: {summ.llm_output:>8,} tokens")
            print(f"     Embedding:  {summ.embedding:>8,} tokens")
            print(f"     小計:       {summ.total_weighted():>8,.1f} tokens")
            
            # 3. Storage
            stor = breakdown.storage
            print(f"\n  3️⃣ Storage (索引儲存)")
            print(f"     LLM Input:  {stor.llm_input:>8,} tokens")
            print(f"     LLM Output: {stor.llm_output:>8,} tokens")
            print(f"     Embedding:  {stor.embedding:>8,} tokens")
            print(f"     小計:       {stor.total_weighted():>8,.1f} tokens")
            
            # Total
            total = breakdown.total()
            print(f"\n  ✅ Phase 1 總計:")
            print(f"     LLM Input:  {total.llm_input:>8,} tokens")
            print(f"     LLM Output: {total.llm_output:>8,} tokens")
            print(f"     Embedding:  {total.embedding:>8,} tokens")
            print(f"     加權總計:   {total.total_weighted():>8,.1f} tokens")
    
    def print_phase2_details(self):
        """列印 Phase 2 詳細分解"""
        print("\n" + "="*80)
        print("PHASE 2: 詢問與回答階段 (Query & Answer Phase)")
        print("="*80)
        print(f"測試查詢: {Phase2_QueryAnswering.QUERY_EXAMPLE}\n")
        
        methods = {
            "Naive RAG": self.phase2_naive,
            "Microsoft GraphRAG": self.phase2_ms,
            "KnowRoots": self.phase2_improved
        }
        
        for method_name, breakdown in methods.items():
            print(f"\n📊 {method_name}")
            print("-" * 80)
            
            # 1. Retrieval Strategy
            ret_strat = breakdown.retrieval_strategy
            print(f"  1️⃣ Retrieval Strategy (檢索策略)")
            print(f"     LLM Input:  {ret_strat.llm_input:>8,} tokens")
            print(f"     LLM Output: {ret_strat.llm_output:>8,} tokens")
            print(f"     Embedding:  {ret_strat.embedding:>8,} tokens")
            print(f"     小計:       {ret_strat.total_weighted():>8,.1f} tokens")
            
            # 2. Data Retrieval
            data_ret = breakdown.data_retrieval
            print(f"\n  2️⃣ Data Retrieval (資料獲取)")
            print(f"     LLM Input:  {data_ret.llm_input:>8,} tokens")
            print(f"     LLM Output: {data_ret.llm_output:>8,} tokens")
            print(f"     Embedding:  {data_ret.embedding:>8,} tokens")
            print(f"     小計:       {data_ret.total_weighted():>8,.1f} tokens")
            
            # 3. Answer Generation
            ans_gen = breakdown.answer_generation
            print(f"\n  3️⃣ Answer Generation (答案生成)")
            print(f"     LLM Input:  {ans_gen.llm_input:>8,} tokens")
            print(f"     LLM Output: {ans_gen.llm_output:>8,} tokens")
            print(f"     Embedding:  {ans_gen.embedding:>8,} tokens")
            print(f"     小計:       {ans_gen.total_weighted():>8,.1f} tokens")
            
            # Total
            total = breakdown.total()
            print(f"\n  ✅ Phase 2 單次查詢總計:")
            print(f"     LLM Input:  {total.llm_input:>8,} tokens")
            print(f"     LLM Output: {total.llm_output:>8,} tokens")
            print(f"     Embedding:  {total.embedding:>8,} tokens")
            print(f"     加權總計:   {total.total_weighted():>8,.1f} tokens")
    
    def print_total_comparison(self, num_queries=1):
        """列印總成本比較（假設 N 次查詢）"""
        print("\n" + "="*80)
        print(f"總成本比較 (Total Cost Comparison)")
        print(f"假設場景: 1 篇論文 + {num_queries} 次查詢")
        print("="*80)
        
        methods = {
            "Naive RAG": (self.phase1_naive, self.phase2_naive),
            "Microsoft GraphRAG": (self.phase1_ms, self.phase2_ms),
            "KnowRoots": (self.phase1_improved, self.phase2_improved)
        }
        
        results = []
        for method_name, (p1, p2) in methods.items():
            phase1_total = p1.total().total_weighted()
            phase2_total = p2.total().total_weighted()
            total_cost = phase1_total + (phase2_total * num_queries)
            
            results.append({
                "Method": method_name,
                "Phase1": phase1_total,
                "Phase2_Single": phase2_total,
                "Phase2_Total": phase2_total * num_queries,
                "Grand_Total": total_cost
            })
        
        df = pd.DataFrame(results)
        print("\n")
        print(df.to_string(index=False, float_format=lambda x: f"{x:,.1f}"))
        
        # 計算相對比例
        print("\n" + "-"*80)
        print("相對比例 (以 Naive RAG 為基準 = 100%)")
        print("-"*80)
        baseline = df.loc[df['Method'] == 'Naive RAG', 'Grand_Total'].values[0]
        for _, row in df.iterrows():
            ratio = (row['Grand_Total'] / baseline) * 100
            print(f"{row['Method']:<25}: {ratio:>6.1f}%")
        
        return df
    
    def plot_comparison(self, num_queries=1):
        """Generate academic-style comparative visualization"""
        fig = plt.figure(figsize=(16, 13))
        
        # Add experimental conditions at the top
        fig.text(0.5, 0.98, 'Token Consumption Comparative Analysis of RAG Architectures', 
                 ha='center', va='top', fontsize=16, fontweight='bold')
        
        # Create subplot grid with adjusted spacing
        gs = fig.add_gridspec(2, 2, top=0.90, bottom=0.05, left=0.08, right=0.98, 
                             hspace=0.35, wspace=0.25)
        
        methods = ["Naive RAG", "Microsoft\nGraphRAG", "KnowRoots"]
        methods_short = ["Naive RAG", "MS GraphRAG", "KnowRoots"]
        colors = ['#7f8c8d', '#e74c3c', '#27ae60']  # Gray, Red, Green
        
        # === Figure (a): Phase 1 Sub-stage Breakdown ===
        ax1 = fig.add_subplot(gs[0, 0])
        phase1_data = {
            'Naive RAG': self.phase1_naive,
            'MS GraphRAG': self.phase1_ms,
            'KnowRoots': self.phase1_improved
        }
        
        extraction = [p.extraction.total_weighted() for p in phase1_data.values()]
        summarization = [p.summarization.total_weighted() for p in phase1_data.values()]
        storage = [p.storage.total_weighted() for p in phase1_data.values()]
        
        x = np.arange(len(methods))
        width = 0.25
        
        ax1.bar(x - width, extraction, width, label='Extraction', color='#3498db', alpha=0.85, edgecolor='black', linewidth=0.5)
        ax1.bar(x, summarization, width, label='Summarization', color='#f39c12', alpha=0.85, edgecolor='black', linewidth=0.5)
        ax1.bar(x + width, storage, width, label='Storage (Embedding)', color='#9b59b6', alpha=0.85, edgecolor='black', linewidth=0.5)
        
        ax1.set_ylabel('Tokens', fontsize=11, fontweight='bold')
        ax1.set_title('(a) Phase 1: Data Indexing Sub-stages', fontsize=12, fontweight='bold', pad=10)
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods, fontsize=9)
        ax1.legend(fontsize=9, loc='upper left', framealpha=0.9)
        ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # === Figure (b): Phase 1 Total Cost ===
        ax2 = fig.add_subplot(gs[0, 1])
        phase1_totals = [p.total().total_weighted() for p in phase1_data.values()]
        bars2 = ax2.bar(methods, phase1_totals, color=colors, alpha=0.85, edgecolor='black', linewidth=0.8)
        ax2.set_ylabel('Tokens', fontsize=11, fontweight='bold')
        ax2.set_title('(b) Phase 1: Total Indexing Cost', fontsize=12, fontweight='bold', pad=10)
        ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.set_xticklabels(methods, fontsize=9)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, height,
                    f'{int(height):,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # === Figure (c): Phase 2 Sub-stage Breakdown ===
        ax3 = fig.add_subplot(gs[1, 0])
        phase2_data = {
            'Naive RAG': self.phase2_naive,
            'MS GraphRAG': self.phase2_ms,
            'KnowRoots': self.phase2_improved
        }
        
        retrieval_strat = [p.retrieval_strategy.total_weighted() for p in phase2_data.values()]
        data_retrieval = [p.data_retrieval.total_weighted() for p in phase2_data.values()]
        answer_gen = [p.answer_generation.total_weighted() for p in phase2_data.values()]
        
        ax3.bar(x - width, retrieval_strat, width, label='Retrieval Strategy', color='#e67e22', alpha=0.85, edgecolor='black', linewidth=0.5)
        ax3.bar(x, data_retrieval, width, label='Data Retrieval', color='#1abc9c', alpha=0.85, edgecolor='black', linewidth=0.5)
        ax3.bar(x + width, answer_gen, width, label='Answer Generation', color='#34495e', alpha=0.85, edgecolor='black', linewidth=0.5)
        
        ax3.set_ylabel('Tokens (per query)', fontsize=11, fontweight='bold')
        ax3.set_title('(c) Phase 2: Query Processing Sub-stages', fontsize=12, fontweight='bold', pad=10)
        ax3.set_xticks(x)
        ax3.set_xticklabels(methods, fontsize=9)
        ax3.legend(fontsize=9, loc='upper left', framealpha=0.9)
        ax3.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        
        # === Figure (d): Total Cost Comparison ===
        ax4 = fig.add_subplot(gs[1, 1])
        
        phase1_totals_dict = {m: p.total().total_weighted() for m, p in phase1_data.items()}
        phase2_totals_dict = {m: p.total().total_weighted() for m, p in phase2_data.items()}
        
        phase1_list = [phase1_totals_dict[m] for m in methods_short]
        phase2_list = [phase2_totals_dict[m] * num_queries for m in methods_short]
        
        query_label = f'{num_queries} Query' if num_queries == 1 else f'{num_queries} Queries'
        bars_p1 = ax4.bar(x, phase1_list, width*2, label=f'Phase 1 (Indexing)', 
                         color='#c0392b', alpha=0.8, edgecolor='black', linewidth=0.8)
        bars_p2 = ax4.bar(x, phase2_list, width*2, bottom=phase1_list, 
                         label=f'Phase 2 ({query_label})', 
                         color='#16a085', alpha=0.8, edgecolor='black', linewidth=0.8)
        
        ax4.set_ylabel('Tokens', fontsize=11, fontweight='bold')
        query_text = 'Query' if num_queries == 1 else 'Queries'
        ax4.set_title(f'(d) Total Cost Comparison (1 Paper + {num_queries} {query_text})', 
                     fontsize=12, fontweight='bold', pad=10)
        ax4.set_xticks(x)
        ax4.set_xticklabels(methods, fontsize=9)
        ax4.legend(fontsize=9, loc='upper left', framealpha=0.9)
        ax4.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        
        # Add total cost labels
        for i, (p1, p2) in enumerate(zip(phase1_list, phase2_list)):
            total = p1 + p2
            ax4.text(i, total + max(phase1_list + phase2_list) * 0.02, 
                    f'{int(total):,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.savefig('token_comparison_detailed.png', dpi=300, bbox_inches='tight', facecolor='white')
        print("\n📊 Figure saved: token_comparison_detailed.png")
        plt.show()


# ==========================================
# 主程式執行
# ==========================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("Token 消耗嚴謹比較分析")
    print("="*80)
    print(f"系統參數:")
    print(f"  - 論文規模: {PAPER_WORD_COUNT:,} 字 ({PAPER_TOKEN_COUNT:,} tokens)")
    print(f"  - Chunk 大小: {CHUNK_SIZE} tokens")
    print(f"  - Chunks 總數: {CHUNKS_PER_PAPER} chunks")
    print(f"  - Token 權重: Input={INPUT_WEIGHT}, Output={OUTPUT_WEIGHT}, Embedding={EMBED_WEIGHT}")
    
    analyzer = TokenComparisonAnalyzer()
    
    # 列印詳細分解
    analyzer.print_phase1_details()
    analyzer.print_phase2_details()
    
    # 列印總成本比較
    df_comparison = analyzer.print_total_comparison(num_queries=1)
    
    # 繪製視覺化圖表
    analyzer.plot_comparison(num_queries=1)
    
    print("\n✅ 分析完成！")
