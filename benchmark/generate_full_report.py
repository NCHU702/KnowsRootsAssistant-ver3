"""
generate_full_report.py - 產出完整 benchmark 結果
包含每題的查詢、Ground Truth、Graph 回傳、RAG 回傳、TP/FP/FN 分析
"""
import json
import os
import unicodedata
from neo4j import GraphDatabase

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT_PATH = os.path.join(BASE_DIR, "benchmark", "ground_truth.json")
RESULT_PATH = os.path.join(BASE_DIR, "benchmark", "results", "benchmark_20260224_151616.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "benchmark", "FULL_RESULTS.md")

def norm(s):
    return unicodedata.normalize("NFKC", s)

def main():
    with open(GT_PATH, "r", encoding="utf-8") as f:
        gt = json.load(f)
    with open(RESULT_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)

    # Get Graph returned papers from Neo4j
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
    graph_returned = {}
    with driver.session() as session:
        for q in gt["queries"]:
            qid = q["query_id"]
            cypher = q["cypher"]
            result = session.run(cypher)
            papers = sorted({r[0] for r in result if r[0]})
            graph_returned[qid] = papers
    driver.close()

    # Build result lookup
    result_lookup = {r["query_id"]: r for r in results["per_query"]}

    lines = []
    lines.append("# KnowsRoots Benchmark — 完整逐題結果")
    lines.append("")
    lines.append(f"**執行時間:** {results['timestamp']}")
    lines.append(f"**查詢總數:** {results['total_queries']}")
    lines.append(f"**Graph DB:** Neo4j ({gt['_metadata']['total_papers_in_graph']} papers)")
    lines.append(f"**RAG Engine:** Layer1 HybridRetriever (FAISS+BM25, sem=0.9, kw=0.1)")
    lines.append(f"**Embedding:** bge-large-zh-v1.5")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Aggregate table
    lines.append("## 總覽 (Macro-Average)")
    lines.append("")
    lines.append("| Method | Precision | Recall | F1 |")
    lines.append("|---|:---:|:---:|:---:|")
    agg = results["aggregate"]
    lines.append(f"| **Graph (Oracle Cypher)** | **{agg['graph_oracle']['precision']:.4f}** | **{agg['graph_oracle']['recall']:.4f}** | **{agg['graph_oracle']['f1']:.4f}** |")
    for k in [3, 5, 10]:
        rk = agg[f"rag_k{k}"]
        lines.append(f"| RAG (k={k}) | {rk['precision']:.4f} | {rk['recall']:.4f} | {rk['f1']:.4f} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Per-query detail
    for q in gt["queries"]:
        qid = q["query_id"]
        r = result_lookup[qid]
        gt_ids = sorted(q["ground_truth_paper_ids"])
        graph_ids = graph_returned[qid]

        lines.append(f"## {qid}: {q['query_text']}")
        lines.append("")
        lines.append(f"- **類型:** {q['query_type']}")
        lines.append(f"- **Cypher:** `{q['cypher']}`")
        lines.append(f"- **難度:** {q['difficulty']}")
        lines.append(f"- **備註:** {q['notes']}")
        lines.append("")

        # Metrics table
        lines.append(f"### 指標")
        lines.append("")
        lines.append("| Method | Retrieved | Precision | Recall | F1 | Time |")
        lines.append("|---|:---:|:---:|:---:|:---:|:---:|")
        lines.append(f"| **Graph** | **{r['graph_retrieved']}** | **{r['graph_precision']:.2f}** | **{r['graph_recall']:.2f}** | **{r['graph_f1']:.2f}** | {r['graph_time_s']:.3f}s |")
        for k in [3, 5, 10]:
            lines.append(f"| RAG k={k} | {r[f'rag_k{k}_retrieved']} | {r[f'rag_k{k}_precision']:.2f} | {r[f'rag_k{k}_recall']:.2f} | {r[f'rag_k{k}_f1']:.2f} | {r[f'rag_k{k}_time_s']:.3f}s |")
        lines.append("")

        # Ground Truth
        lines.append(f"### Ground Truth ({len(gt_ids)} 篇)")
        lines.append("")
        for i, pid in enumerate(gt_ids, 1):
            lines.append(f"{i}. {pid}")
        lines.append("")

        # Graph returned
        lines.append(f"### Graph 回傳 ({len(graph_ids)} 篇)")
        lines.append("")
        gt_set = {norm(p) for p in gt_ids}
        for i, pid in enumerate(graph_ids, 1):
            tag = "✅" if norm(pid) in gt_set else "❌"
            lines.append(f"{i}. {tag} {pid}")
        lines.append("")

        # RAG returned for each k
        for k in [3, 5, 10]:
            papers_str = r.get(f"rag_k{k}_papers", "")
            if not papers_str:
                continue
            # Parse "paper_id(score); paper_id(score); ..."
            items = [x.strip() for x in papers_str.split(";") if x.strip()]
            lines.append(f"### RAG k={k} 回傳 ({len(items)} 篇)")
            lines.append("")
            for i, item in enumerate(items, 1):
                # Extract paper_id and score
                if "(" in item:
                    pid = item[:item.rfind("(")].strip()
                    score = item[item.rfind("(")+1:item.rfind(")")]
                else:
                    pid = item
                    score = "?"
                tag = "✅" if norm(pid) in gt_set else "❌"
                lines.append(f"{i}. {tag} `{score}` {pid}")
            lines.append("")

        # TP / FP / FN for k=10
        tp_str = r.get("rag_k10_tp", "")
        fp_str = r.get("rag_k10_fp", "")
        fn_str = r.get("rag_k10_fn", "")

        tp_list = sorted([x.strip() for x in tp_str.split(";") if x.strip()]) if tp_str else []
        fp_list = sorted([x.strip() for x in fp_str.split(";") if x.strip()]) if fp_str else []
        fn_list = sorted([x.strip() for x in fn_str.split(";") if x.strip()]) if fn_str else []

        lines.append(f"### RAG k=10 誤差分析")
        lines.append("")
        lines.append(f"| 類別 | 數量 | 論文 |")
        lines.append(f"|---|:---:|---|")
        lines.append(f"| ✅ TP (正確找到) | {len(tp_list)} | {', '.join(tp_list) if tp_list else '—'} |")
        lines.append(f"| ❌ FP (錯誤找到) | {len(fp_list)} | {', '.join(fp_list) if fp_list else '—'} |")
        lines.append(f"| ⚠️ FN (遺漏) | {len(fn_list)} | {', '.join(fn_list) if fn_list else '—'} |")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Write
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✅ Full results written to: {OUTPUT_PATH}")
    print(f"   共 {len(gt['queries'])} 題完整分析")


if __name__ == "__main__":
    main()
