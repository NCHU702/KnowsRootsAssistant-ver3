"""
run_benchmark_v2.py - KnowsRoots 分層抽象化基準測試
=================================================================

比較兩條檢索路徑在 15 個查詢上的 Precision / Recall / F1 表現。
查詢按抽象化程度分三組：
  - Low  (L01–L05): 直接匹配 Graph 邊/節點
  - Mid  (M01–M05): 語義同義詞、概念等價
  - High (H01–H05): 多跳推理、跨域聯想

Graph 路徑分兩種模式：
  1. Graph Oracle — 使用預寫 Cypher (有 cypher 欄位才執行)
  2. Graph Approx — 使用近似 Cypher (有 cypher_approx 欄位才執行)

RAG 路徑: Layer1 HybridRetriever (語義 + BM25) at k=3, 5, 10
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

GT_PATH = os.path.join(BASE_DIR, "benchmark", "ground_truth_v2.json")
RESULTS_DIR = os.path.join(BASE_DIR, "benchmark", "results_v2")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Utility: Unicode normalization for comparison
# ---------------------------------------------------------------------------
def normalize(s: str) -> str:
    return unicodedata.normalize("NFKC", s)


def normalize_set(paper_ids: List[str]) -> Set[str]:
    return {normalize(pid) for pid in paper_ids}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def calc_metrics(retrieved: Set[str], ground_truth: Set[str]) -> Dict[str, float]:
    if not retrieved and not ground_truth:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}
    if not retrieved:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": len(ground_truth)}
    if not ground_truth:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": len(retrieved), "fn": 0}

    tp = len(retrieved & ground_truth)
    fp = len(retrieved - ground_truth)
    fn = len(ground_truth - retrieved)
    precision = tp / len(retrieved) if retrieved else 0.0
    recall = tp / len(ground_truth) if ground_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


# ---------------------------------------------------------------------------
# Graph Path: Execute Cypher against Neo4j
# ---------------------------------------------------------------------------
class GraphBenchmark:
    def __init__(self):
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(
            "bolt://localhost:7687", auth=("neo4j", "password")
        )
        with self.driver.session() as session:
            session.run("RETURN 1")
        print("  ✓ Neo4j connected")

    def query(self, cypher: str) -> Set[str]:
        if not cypher:
            return set()
        with self.driver.session() as session:
            result = session.run(cypher)
            return {normalize(str(r[0])) for r in result if r[0]}

    def close(self):
        self.driver.close()


# ---------------------------------------------------------------------------
# RAG Path: Layer1 HybridRetriever (semantic + BM25)
# ---------------------------------------------------------------------------
class RAGBenchmark:
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
            use_llm_analyzer=False
        )
        print("  ✓ HybridRetriever ready")

    def query(self, query_text: str, k: int = 10,
              semantic_weight: float = 0.9,
              keyword_weight: float = 0.1,
              threshold: float | None = None) -> List[Tuple[str, float]]:
        results = self.hybrid.hybrid_search(
            query=query_text, k=k,
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
    print("=" * 80)
    print("  KnowsRoots Benchmark v2: 分層抽象化 (Low / Medium / High)")
    print("=" * 80)
    print()

    # 1. Load Ground Truth
    with open(GT_PATH, "r", encoding="utf-8") as f:
        gt = json.load(f)
    queries = gt["queries"]
    print(f"📋 Loaded {len(queries)} queries from ground_truth_v2.json")
    for level in ["low", "medium", "high"]:
        count = sum(1 for q in queries if q.get("abstraction_level") == level)
        print(f"   {level.upper()}: {count} queries")
    print()

    # 2. Initialize backends
    print("🔧 Initializing backends …")
    graph_bench = GraphBenchmark()
    rag_bench = RAGBenchmark()
    print()

    # 3. Cutoff levels for RAG path (top-k)
    rag_k_levels = [3, 5, 10]

    # 4. Run queries
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows: List[Dict[str, Any]] = []
    agg: Dict[str, List[float]] = {}

    for q in queries:
        qid = q["query_id"]
        qtext = q["query_text"]
        qtype = q.get("query_type", "unknown")
        level = q.get("abstraction_level", "unknown")
        cypher_exact = q.get("cypher")
        cypher_approx = q.get("cypher_approx")
        gt_ids = normalize_set(q["ground_truth_paper_ids"])

        print(f"─── {qid} [{level.upper()}]: {qtext} (GT={len(gt_ids)}) ───")

        row: Dict[str, Any] = {
            "query_id": qid,
            "query_text": qtext,
            "abstraction_level": level,
            "query_type": qtype,
            "gt_count": len(gt_ids),
        }

        # --- Graph Oracle (exact Cypher) ---
        if cypher_exact:
            t0 = time.time()
            graph_ids = graph_bench.query(cypher_exact)
            graph_time = time.time() - t0
            graph_m = calc_metrics(graph_ids, gt_ids)
            print(f"  Graph Oracle  : P={graph_m['precision']:.2f}  R={graph_m['recall']:.2f}  F1={graph_m['f1']:.2f}  |ret|={len(graph_ids)}  t={graph_time:.3f}s")
        else:
            graph_ids = set()
            graph_m = calc_metrics(set(), gt_ids)
            graph_time = 0.0
            print(f"  Graph Oracle  : N/A (no exact Cypher)")

        row["graph_oracle_retrieved"] = len(graph_ids)
        row["graph_oracle_precision"] = graph_m["precision"]
        row["graph_oracle_recall"] = graph_m["recall"]
        row["graph_oracle_f1"] = graph_m["f1"]
        row["graph_oracle_time_s"] = round(graph_time, 4)
        _acc(agg, f"graph_oracle_precision_{level}", graph_m["precision"])
        _acc(agg, f"graph_oracle_recall_{level}", graph_m["recall"])
        _acc(agg, f"graph_oracle_f1_{level}", graph_m["f1"])
        _acc(agg, "graph_oracle_precision_all", graph_m["precision"])
        _acc(agg, "graph_oracle_recall_all", graph_m["recall"])
        _acc(agg, "graph_oracle_f1_all", graph_m["f1"])

        # --- Graph Approx (best-effort Cypher) ---
        active_cypher_approx = cypher_approx if cypher_approx else cypher_exact
        if active_cypher_approx:
            t0 = time.time()
            graph_approx_ids = graph_bench.query(active_cypher_approx)
            graph_approx_time = time.time() - t0
            graph_approx_m = calc_metrics(graph_approx_ids, gt_ids)
            print(f"  Graph Approx  : P={graph_approx_m['precision']:.2f}  R={graph_approx_m['recall']:.2f}  F1={graph_approx_m['f1']:.2f}  |ret|={len(graph_approx_ids)}  t={graph_approx_time:.3f}s")
        else:
            graph_approx_ids = set()
            graph_approx_m = calc_metrics(set(), gt_ids)
            graph_approx_time = 0.0
            print(f"  Graph Approx  : N/A (no Cypher available)")

        row["graph_approx_retrieved"] = len(graph_approx_ids)
        row["graph_approx_precision"] = graph_approx_m["precision"]
        row["graph_approx_recall"] = graph_approx_m["recall"]
        row["graph_approx_f1"] = graph_approx_m["f1"]
        row["graph_approx_time_s"] = round(graph_approx_time, 4)
        _acc(agg, f"graph_approx_precision_{level}", graph_approx_m["precision"])
        _acc(agg, f"graph_approx_recall_{level}", graph_approx_m["recall"])
        _acc(agg, f"graph_approx_f1_{level}", graph_approx_m["f1"])
        _acc(agg, "graph_approx_precision_all", graph_approx_m["precision"])
        _acc(agg, "graph_approx_recall_all", graph_approx_m["recall"])
        _acc(agg, "graph_approx_f1_all", graph_approx_m["f1"])

        # --- RAG at various k levels ---
        for k in rag_k_levels:
            t0 = time.time()
            rag_results = rag_bench.query(qtext, k=k)
            rag_time = time.time() - t0
            rag_ids = {pid for pid, _ in rag_results}
            rag_m = calc_metrics(rag_ids, gt_ids)
            print(f"  RAG  k={k:<3}     : P={rag_m['precision']:.2f}  R={rag_m['recall']:.2f}  F1={rag_m['f1']:.2f}  |ret|={len(rag_ids)}  t={rag_time:.3f}s")

            row[f"rag_k{k}_retrieved"] = len(rag_ids)
            row[f"rag_k{k}_precision"] = rag_m["precision"]
            row[f"rag_k{k}_recall"] = rag_m["recall"]
            row[f"rag_k{k}_f1"] = rag_m["f1"]
            row[f"rag_k{k}_time_s"] = round(rag_time, 4)

            _acc(agg, f"rag_k{k}_precision_{level}", rag_m["precision"])
            _acc(agg, f"rag_k{k}_recall_{level}", rag_m["recall"])
            _acc(agg, f"rag_k{k}_f1_{level}", rag_m["f1"])
            _acc(agg, f"rag_k{k}_precision_all", rag_m["precision"])
            _acc(agg, f"rag_k{k}_recall_all", rag_m["recall"])
            _acc(agg, f"rag_k{k}_f1_all", rag_m["f1"])

            # Paper list
            row[f"rag_k{k}_papers"] = "; ".join(
                f"{pid}({score:.3f})" for pid, score in rag_results
            )

        # TP/FP/FN detail at k=10
        rag_main = {pid for pid, _ in rag_bench.query(qtext, k=10)}
        row["rag_k10_tp"] = "; ".join(sorted(rag_main & gt_ids))
        row["rag_k10_fp"] = "; ".join(sorted(rag_main - gt_ids))
        row["rag_k10_fn"] = "; ".join(sorted(gt_ids - rag_main))

        rows.append(row)
        print()

    # 5. Aggregate results
    n = len(queries)
    def _avg(key: str) -> float:
        vals = agg.get(key, [])
        return sum(vals) / len(vals) if vals else 0.0

    # ── Overall ──
    print("=" * 80)
    print("  AGGREGATE RESULTS (macro-average)")
    print("=" * 80)
    print()
    print(f"  {'Method':<22} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'─'*22} {'─'*10} {'─'*10} {'─'*10}")

    gop = _avg("graph_oracle_precision_all")
    gor = _avg("graph_oracle_recall_all")
    gof = _avg("graph_oracle_f1_all")
    print(f"  {'Graph Oracle':<22} {gop:>10.4f} {gor:>10.4f} {gof:>10.4f}")

    gap = _avg("graph_approx_precision_all")
    gar = _avg("graph_approx_recall_all")
    gaf = _avg("graph_approx_f1_all")
    print(f"  {'Graph Approx':<22} {gap:>10.4f} {gar:>10.4f} {gaf:>10.4f}")

    for k in rag_k_levels:
        rp = _avg(f"rag_k{k}_precision_all")
        rr = _avg(f"rag_k{k}_recall_all")
        rf = _avg(f"rag_k{k}_f1_all")
        print(f"  {f'RAG (k={k})':<22} {rp:>10.4f} {rr:>10.4f} {rf:>10.4f}")

    # ── Per abstraction level ──
    print()
    print("=" * 80)
    print("  BREAKDOWN BY ABSTRACTION LEVEL")
    print("=" * 80)

    for level in ["low", "medium", "high"]:
        level_count = sum(1 for q in queries if q.get("abstraction_level") == level)
        if level_count == 0:
            continue
        print(f"\n  ── {level.upper()} ({level_count} queries) ──")
        print(f"  {'Method':<22} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print(f"  {'─'*22} {'─'*10} {'─'*10} {'─'*10}")

        p, r, f = _avg(f"graph_oracle_precision_{level}"), _avg(f"graph_oracle_recall_{level}"), _avg(f"graph_oracle_f1_{level}")
        print(f"  {'Graph Oracle':<22} {p:>10.4f} {r:>10.4f} {f:>10.4f}")
        p, r, f = _avg(f"graph_approx_precision_{level}"), _avg(f"graph_approx_recall_{level}"), _avg(f"graph_approx_f1_{level}")
        print(f"  {'Graph Approx':<22} {p:>10.4f} {r:>10.4f} {f:>10.4f}")

        for k in rag_k_levels:
            p = _avg(f"rag_k{k}_precision_{level}")
            r = _avg(f"rag_k{k}_recall_{level}")
            f = _avg(f"rag_k{k}_f1_{level}")
            print(f"  {f'RAG (k={k})':<22} {p:>10.4f} {r:>10.4f} {f:>10.4f}")

    print()

    # 6. Write CSV
    csv_path = os.path.join(RESULTS_DIR, f"benchmark_v2_{timestamp}.csv")
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"💾 Detailed CSV saved: {csv_path}")

    # 7. Write summary JSON
    summary = {
        "timestamp": timestamp,
        "total_queries": n,
        "abstraction_levels": {"low": 5, "medium": 5, "high": 5},
        "aggregate": {
            "overall": {
                "graph_oracle": {"precision": gop, "recall": gor, "f1": gof},
                "graph_approx": {"precision": gap, "recall": gar, "f1": gaf},
            },
        },
        "per_level": {},
        "per_query": rows
    }
    for k in rag_k_levels:
        summary["aggregate"]["overall"][f"rag_k{k}"] = {
            "precision": _avg(f"rag_k{k}_precision_all"),
            "recall": _avg(f"rag_k{k}_recall_all"),
            "f1": _avg(f"rag_k{k}_f1_all"),
        }
    for level in ["low", "medium", "high"]:
        summary["per_level"][level] = {
            "graph_oracle": {
                "precision": _avg(f"graph_oracle_precision_{level}"),
                "recall": _avg(f"graph_oracle_recall_{level}"),
                "f1": _avg(f"graph_oracle_f1_{level}"),
            },
            "graph_approx": {
                "precision": _avg(f"graph_approx_precision_{level}"),
                "recall": _avg(f"graph_approx_recall_{level}"),
                "f1": _avg(f"graph_approx_f1_{level}"),
            },
        }
        for k in rag_k_levels:
            summary["per_level"][level][f"rag_k{k}"] = {
                "precision": _avg(f"rag_k{k}_precision_{level}"),
                "recall": _avg(f"rag_k{k}_recall_{level}"),
                "f1": _avg(f"rag_k{k}_f1_{level}"),
            }

    json_path = os.path.join(RESULTS_DIR, f"benchmark_v2_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"💾 Summary JSON saved: {json_path}")

    # Cleanup
    graph_bench.close()
    print("\n✅ Benchmark v2 complete!")


def _acc(agg: dict, key: str, value: float):
    agg.setdefault(key, []).append(value)


if __name__ == "__main__":
    run_benchmark()
