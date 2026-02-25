"""
run_benchmark.py - KnowsRoots Graph vs Traditional RAG 精度基準測試
=================================================================

比較兩條檢索路徑在 Precision / Recall / F1 上的表現:
  1. Graph Path (Oracle Cypher) — 使用 ground_truth.json 中預寫的 Cypher
  2. RAG Path — 使用 Layer1 HybridRetriever (語義 + BM25)

輸出:
  - Terminal 格式化表格
  - benchmark/results/ 下的 CSV 檔案
"""

import json
import os
import sys
import csv
import time
import unicodedata
from datetime import datetime
from typing import Dict, List, Set, Tuple, Any

# ---------------------------------------------------------------------------
# Path Setup
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

GT_PATH = os.path.join(BASE_DIR, "benchmark", "ground_truth.json")
RESULTS_DIR = os.path.join(BASE_DIR, "benchmark", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Utility: Unicode normalization for comparison
# ---------------------------------------------------------------------------
def normalize(s: str) -> str:
    """NFKC normalize a string for safe comparison."""
    return unicodedata.normalize("NFKC", s)


def normalize_set(paper_ids: List[str]) -> Set[str]:
    return {normalize(pid) for pid in paper_ids}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def calc_metrics(retrieved: Set[str], ground_truth: Set[str]) -> Dict[str, float]:
    """Calculate Precision / Recall / F1 between retrieved and ground truth sets."""
    if not retrieved and not ground_truth:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not retrieved:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not ground_truth:
        return {"precision": 0.0, "recall": 0.0 if ground_truth else 1.0, "f1": 0.0}

    tp = len(retrieved & ground_truth)
    precision = tp / len(retrieved) if retrieved else 0.0
    recall = tp / len(ground_truth) if ground_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# Graph Path: Execute pre-written Cypher against Neo4j
# ---------------------------------------------------------------------------
class GraphBenchmark:
    """Graph path benchmark using direct Cypher queries."""

    def __init__(self):
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "password")
        )
        # Quick connectivity check
        with self.driver.session() as session:
            session.run("RETURN 1")
        print("  ✓ Neo4j connected")

    def query(self, cypher: str) -> Set[str]:
        """Run a Cypher query and return the set of paper_ids."""
        with self.driver.session() as session:
            result = session.run(cypher)
            paper_ids = set()
            for record in result:
                # The Cypher queries return p.paper_id
                val = record[0] if record else None
                if val:
                    paper_ids.add(normalize(str(val)))
            return paper_ids

    def close(self):
        self.driver.close()


# ---------------------------------------------------------------------------
# RAG Path: Layer1 HybridRetriever (semantic + BM25)
# ---------------------------------------------------------------------------
class RAGBenchmark:
    """
    RAG path benchmark using Layer1 HybridRetriever.
    
    Initializes the same components the production system uses,
    but runs retrieval ONLY (no LLM generation).
    """

    def __init__(self):
        from langchain_ollama import OllamaEmbeddings
        from system_api.layer1_vectorstore import Layer1VectorStore
        from system_api.hybrid_retriever import HybridRetriever

        embedding_model = "quentinz/bge-large-zh-v1.5:latest"
        vectorstore_path = os.path.join(BASE_DIR, "vectorstore")

        print("  Initializing embeddings …")
        embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url="http://localhost:11434"
        )

        print("  Loading Layer1 VectorStore …")
        self.layer1 = Layer1VectorStore(
            embeddings=embeddings,
            vectorstore_path=os.path.join(vectorstore_path, "layer1")
        )
        self.layer1.load()
        print(f"  ✓ Layer1 loaded ({self.layer1.get_stats().get('total_papers', '?')} papers)")

        print("  Building HybridRetriever …")
        self.hybrid = HybridRetriever(
            layer1_vectorstore=self.layer1,
            use_jieba=True,
            use_llm_analyzer=False  # Skip LLM analyzer for benchmark (no LLM needed)
        )
        print("  ✓ HybridRetriever ready")

    def query(self, query_text: str, k: int = 10,
              semantic_weight: float = 0.9,
              keyword_weight: float = 0.1,
              threshold: float | None = None) -> List[Tuple[str, float]]:
        """
        Run a hybrid search and return [(paper_id, score), …] sorted by score desc.
        """
        results = self.hybrid.hybrid_search(
            query=query_text,
            k=k,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight,
            score_threshold=threshold
        )
        out = []
        for doc, score in results:
            pid = doc.metadata.get("paper_id", "")
            out.append((normalize(pid), score))
        return out


# ---------------------------------------------------------------------------
# Main Benchmark Runner
# ---------------------------------------------------------------------------
def run_benchmark():
    print("=" * 78)
    print("  KnowsRoots Benchmark: Graph (Oracle Cypher) vs RAG (Hybrid Search)")
    print("=" * 78)
    print()

    # 1. Load Ground Truth
    with open(GT_PATH, "r", encoding="utf-8") as f:
        gt = json.load(f)
    queries = gt["queries"]
    print(f"📋 Loaded {len(queries)} queries from ground_truth.json\n")

    # 2. Initialize backends
    print("🔧 Initializing backends …")
    graph_bench = GraphBenchmark()
    rag_bench = RAGBenchmark()
    print()

    # 3. Cutoff levels for RAG path (top-k)
    rag_k_levels = [3, 5, 10]

    # 4. Run queries
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows: List[Dict[str, Any]] = []          # per-query rows
    agg: Dict[str, List[float]] = {}         # aggregation accumulators

    for q in queries:
        qid = q["query_id"]
        qtext = q["query_text"]
        qtype = q["query_type"]
        cypher = q["cypher"]
        gt_ids = normalize_set(q["ground_truth_paper_ids"])

        print(f"─── {qid}: {qtext} ({qtype}, GT={len(gt_ids)}) ───")

        # --- Graph Oracle ---
        t0 = time.time()
        graph_ids = graph_bench.query(cypher)
        graph_time = time.time() - t0
        graph_metrics = calc_metrics(graph_ids, gt_ids)
        print(f"  Graph Oracle  : P={graph_metrics['precision']:.2f}  R={graph_metrics['recall']:.2f}  F1={graph_metrics['f1']:.2f}  |ret|={len(graph_ids)}  t={graph_time:.3f}s")

        row: Dict[str, Any] = {
            "query_id": qid,
            "query_text": qtext,
            "query_type": qtype,
            "gt_count": len(gt_ids),
            "graph_retrieved": len(graph_ids),
            "graph_precision": graph_metrics["precision"],
            "graph_recall": graph_metrics["recall"],
            "graph_f1": graph_metrics["f1"],
            "graph_time_s": round(graph_time, 4),
        }

        # Accumulate
        _acc(agg, "graph_precision", graph_metrics["precision"])
        _acc(agg, "graph_recall", graph_metrics["recall"])
        _acc(agg, "graph_f1", graph_metrics["f1"])

        # --- RAG at various k levels ---
        for k in rag_k_levels:
            t0 = time.time()
            rag_results = rag_bench.query(qtext, k=k)
            rag_time = time.time() - t0
            rag_ids = {pid for pid, _ in rag_results}
            rag_metrics = calc_metrics(rag_ids, gt_ids)
            print(f"  RAG  k={k:<3}     : P={rag_metrics['precision']:.2f}  R={rag_metrics['recall']:.2f}  F1={rag_metrics['f1']:.2f}  |ret|={len(rag_ids)}  t={rag_time:.3f}s")

            row[f"rag_k{k}_retrieved"] = len(rag_ids)
            row[f"rag_k{k}_precision"] = rag_metrics["precision"]
            row[f"rag_k{k}_recall"] = rag_metrics["recall"]
            row[f"rag_k{k}_f1"] = rag_metrics["f1"]
            row[f"rag_k{k}_time_s"] = round(rag_time, 4)

            _acc(agg, f"rag_k{k}_precision", rag_metrics["precision"])
            _acc(agg, f"rag_k{k}_recall", rag_metrics["recall"])
            _acc(agg, f"rag_k{k}_f1", rag_metrics["f1"])

            # Detail: which papers were retrieved
            row[f"rag_k{k}_papers"] = "; ".join(
                f"{pid}({score:.3f})" for pid, score in rag_results
            )

        # Also save TP/FP/FN breakdown for the "main" k=10 level
        rag_main = {pid for pid, _ in rag_bench.query(qtext, k=10)}
        row["rag_k10_tp"] = "; ".join(sorted(rag_main & gt_ids))
        row["rag_k10_fp"] = "; ".join(sorted(rag_main - gt_ids))
        row["rag_k10_fn"] = "; ".join(sorted(gt_ids - rag_main))

        rows.append(row)
        print()

    # 5. Aggregate results
    n = len(queries)
    print("=" * 78)
    print("  AGGREGATE RESULTS (macro-average over 10 queries)")
    print("=" * 78)
    print()
    print(f"  {'Method':<22} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'─'*22} {'─'*10} {'─'*10} {'─'*10}")

    def _avg(key: str) -> float:
        return sum(agg.get(key, [])) / len(agg.get(key, [1])) if agg.get(key) else 0.0

    # Graph Oracle
    gp, gr, gf = _avg("graph_precision"), _avg("graph_recall"), _avg("graph_f1")
    print(f"  {'Graph (Oracle)':<22} {gp:>10.4f} {gr:>10.4f} {gf:>10.4f}")

    for k in rag_k_levels:
        rp = _avg(f"rag_k{k}_precision")
        rr = _avg(f"rag_k{k}_recall")
        rf = _avg(f"rag_k{k}_f1")
        print(f"  {f'RAG (k={k})':<22} {rp:>10.4f} {rr:>10.4f} {rf:>10.4f}")

    print()

    # 6. Breakdown by query type
    print("=" * 78)
    print("  BREAKDOWN BY QUERY TYPE")
    print("=" * 78)
    for qtype in ["domain", "method", "composite"]:
        subset = [r for r in rows if r["query_type"] == qtype]
        if not subset:
            continue
        ns = len(subset)
        print(f"\n  ── {qtype.upper()} ({ns} queries) ──")
        print(f"  {'Method':<22} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print(f"  {'─'*22} {'─'*10} {'─'*10} {'─'*10}")

        gp = sum(r["graph_precision"] for r in subset) / ns
        gr = sum(r["graph_recall"] for r in subset) / ns
        gf = sum(r["graph_f1"] for r in subset) / ns
        print(f"  {'Graph (Oracle)':<22} {gp:>10.4f} {gr:>10.4f} {gf:>10.4f}")

        for k_val in rag_k_levels:
            rp = sum(r[f"rag_k{k_val}_precision"] for r in subset) / ns
            rr = sum(r[f"rag_k{k_val}_recall"] for r in subset) / ns
            rf = sum(r[f"rag_k{k_val}_f1"] for r in subset) / ns
            print(f"  {f'RAG (k={k_val})':<22} {rp:>10.4f} {rr:>10.4f} {rf:>10.4f}")

    print()

    # 7. Write CSV
    csv_path = os.path.join(RESULTS_DIR, f"benchmark_{timestamp}.csv")
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"💾 Detailed CSV saved: {csv_path}")

    # 8. Write summary JSON
    summary = {
        "timestamp": timestamp,
        "total_queries": n,
        "aggregate": {
            "graph_oracle": {"precision": gp, "recall": gr, "f1": gf},
        },
        "per_query": rows
    }
    for k_val in rag_k_levels:
        summary["aggregate"][f"rag_k{k_val}"] = {
            "precision": _avg(f"rag_k{k_val}_precision"),
            "recall": _avg(f"rag_k{k_val}_recall"),
            "f1": _avg(f"rag_k{k_val}_f1"),
        }

    json_path = os.path.join(RESULTS_DIR, f"benchmark_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"💾 Summary JSON saved: {json_path}")

    # Cleanup
    graph_bench.close()
    print("\n✅ Benchmark complete!")


def _acc(agg: dict, key: str, value: float):
    agg.setdefault(key, []).append(value)


if __name__ == "__main__":
    run_benchmark()
