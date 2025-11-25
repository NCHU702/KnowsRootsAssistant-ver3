import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==========================================
# 1. 實驗數據設定
# ==========================================
# 論文數據: 《基礎3_基於深度學習之鼻咽癌腫塊辨識》
# 估計 Token 數 (含繁體中文加權)
PAPER_TOKENS = 48000  # 約 40,000 字

# LLM 定價模型 (參考 OpenAI GPT-4o-mini 比例，單位: USD per 1M tokens)
PRICE_INPUT = 0.15
PRICE_OUTPUT = 0.60 
# Embedding 成本極低，設為 Input 的 1/10
PRICE_EMBED = 0.02 

class RAGComparisonSimulator:
    def __init__(self, paper_tokens):
        self.tokens = paper_tokens
        print(f"📄 測試樣本: 單篇碩士論文 ({self.tokens:,} Tokens)")

    # ----------------------------------------------------------
    # Phase 1: 建置索引成本 (Indexing Cost) - 一次性費用
    # ----------------------------------------------------------
    def calc_indexing(self):
        # 1. Naive RAG
        # 流程: Chunking -> Embedding
        # 成本: 100% Input (Embedding)
        naive_in = self.tokens
        naive_out = 0
        naive_cost = (naive_in * PRICE_EMBED) / 1e6

        # 2. MS GraphRAG
        # 流程: Full Read -> Entity Extract -> Community Detect -> Summarization (Recursive)
        # 依據微軟論文與實測，放大倍率約 10x-12x
        # Input: 需多次讀取原文與中間產物
        ms_in = self.tokens * 8.0 
        # Output: 生成大量的實體、關係與摘要 (最貴的部分)
        ms_out = self.tokens * 2.0
        ms_cost = (ms_in * PRICE_INPUT + ms_out * PRICE_OUTPUT) / 1e6

        # 3. Improved GraphRAG (Naive Mode)
        # 流程: Smart Truncate (6k chars) -> Metadata Extract -> Layer 2 Vector
        # Graph Input: 限制 6000 chars ≈ 7200 tokens + System Prompt (800)
        imp_graph_in = 8000
        # Graph Output: JSON Metadata (少量)
        imp_graph_out = 500
        
        # Layer 2 Input: 全文 Embedding (同 Naive)
        imp_l2_in = self.tokens
        
        imp_total_in = imp_graph_in  # LLM Input
        imp_total_out = imp_graph_out # LLM Output
        # Cost = LLM Cost + Embedding Cost
        imp_cost = ((imp_total_in * PRICE_INPUT + imp_total_out * PRICE_OUTPUT) / 1e6) + \
                   ((imp_l2_in * PRICE_EMBED) / 1e6)

        return {
            "Naive RAG": {"tokens": naive_in, "cost": naive_cost},
            "MS GraphRAG": {"tokens": ms_in + ms_out, "cost": ms_cost},
            "Improved": {"tokens": imp_total_in + imp_total_out + imp_l2_in, "cost": imp_cost}
        }

    # ----------------------------------------------------------
    # Phase 2: 查詢成本 (Query Cost) - 針對"複雜比較型問題"
    # 問題範例: "比較這篇論文中 CNN 與 YOLO 方法的訓練成效差異"
    # ----------------------------------------------------------
    def calc_querying(self):
        # 1. Naive RAG (Top-K Retrieval)
        # 機制: 檢索 Top-10 chunks (假設每個 chunk 500 tokens)
        k = 10
        chunk_size = 500
        # Input: System Prompt + 10 Chunks
        naive_q_in = 1000 + (k * chunk_size)
        # Output: 回答
        naive_q_out = 500
        naive_q_cost = (naive_q_in * PRICE_INPUT + naive_q_out * PRICE_OUTPUT) / 1e6

        # 2. MS GraphRAG (Global Search / Map-Reduce)
        # 機制: 找出相關的 10 個社群摘要，並行執行 Map，再執行 Reduce
        num_communities = 10
        summary_size = 1000
        # Map Phase: 10 * (Prompt + Summary) -> 10 * Intermediate Answers
        map_in = num_communities * (1000 + summary_size)
        map_out = num_communities * 500 # 每個社群生成一段分析
        # Reduce Phase: 匯總 10 段分析
        reduce_in = 1000 + map_out
        reduce_out = 800 # 最終答案
        
        ms_q_total_in = map_in + reduce_in
        ms_q_total_out = map_out + reduce_out
        ms_q_cost = (ms_q_total_in * PRICE_INPUT + ms_q_total_out * PRICE_OUTPUT) / 1e6

        # 3. Improved GraphRAG (Graph Routing + Drill Down)
        # 機制: Graph 查出 Paper ID -> 針對該 Paper 檢索 Top-5 chunks (Layer 2)
        # Step 1: Graph Query (Text2Cypher)
        router_in = 500
        router_out = 100 
        
        # Step 2: Layer 2 Retrieval (同 Naive，但範圍更精準，假設取 Top-5)
        # 因為已經鎖定 Paper，不需要 Top-10 盲搜，Top-5 足夠
        k_refined = 5
        l2_in = 1000 + (k_refined * chunk_size)
        l2_out = 800 # 最終答案
        
        imp_q_total_in = router_in + l2_in
        imp_q_total_out = router_out + l2_out
        imp_q_cost = (imp_q_total_in * PRICE_INPUT + imp_q_total_out * PRICE_OUTPUT) / 1e6

        return {
            "Naive RAG": {"tokens": naive_q_in + naive_q_out, "cost": naive_q_cost},
            "MS GraphRAG": {"tokens": ms_q_total_in + ms_q_total_out, "cost": ms_q_cost},
            "Improved": {"tokens": imp_q_total_in + imp_q_total_out, "cost": imp_q_cost}
        }

# ==========================================
# 執行模擬與繪圖
# ==========================================
sim = RAGComparisonSimulator(PAPER_TOKENS)
idx_data = sim.calc_indexing()
qry_data = sim.calc_querying()

# 轉換為 DataFrame
df_idx = pd.DataFrame(idx_data).T
df_qry = pd.DataFrame(qry_data).T

# 繪製雙圖表
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# 圖 1: 建置索引成本 (Indexing)
colors = ['#999999', '#d62728', '#2ca02c'] # 灰(Naive), 紅(MS), 綠(Imp)
bars1 = ax1.bar(df_idx.index, df_idx['tokens'], color=colors, alpha=0.8)
ax1.set_title('Phase 1: 建置索引 Token 消耗 (Indexing)\n(單篇論文)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Total Tokens', fontsize=12)
ax1.grid(axis='y', linestyle='--', alpha=0.3)

# 標註數值
for rect in bars1:
    height = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width()/2, height, f"{int(height):,}", 
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# 圖 2: 單次查詢成本 (Querying)
bars2 = ax2.bar(df_qry.index, df_qry['tokens'], color=colors, alpha=0.8)
ax2.set_title('Phase 2: 單次複雜查詢 Token 消耗 (Querying)\n(Global/Complex Query)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Total Tokens', fontsize=12)
ax2.grid(axis='y', linestyle='--', alpha=0.3)

# 標註數值
for rect in bars2:
    height = rect.get_height()
    ax2.text(rect.get_x() + rect.get_width()/2, height, f"{int(height):,}", 
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()

# 輸出詳細數據
print("\n📊 詳細成本數據表 (Tokens):")
print("-" * 60)
print(f"{'Method':<20} | {'Indexing':<15} | {'Querying (Per Call)':<15}")
print("-" * 60)
for method in idx_data.keys():
    print(f"{method:<20} | {int(idx_data[method]['tokens']):<15,} | {int(qry_data[method]['tokens']):<15,}")
print("-" * 60)