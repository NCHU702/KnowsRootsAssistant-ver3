"""
verify_ground_truth.py - 驗證 ground_truth.json 與 Graph 匯出 CSV 一致
"""
import json
import csv
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPH_EXPORTS = os.path.join(BASE_DIR, "graph_exports")
GT_PATH = os.path.join(BASE_DIR, "benchmark", "ground_truth.json")


def load_csv_edges(filename):
    """載入邊 CSV，回傳 {(source_key, target_key)} 集合"""
    edges = []
    path = os.path.join(GRAPH_EXPORTS, filename)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            edges.append((row["source_key"], row["target_key"]))
    return edges


def main():
    # 載入 Ground Truth
    with open(GT_PATH, "r", encoding="utf-8") as f:
        gt = json.load(f)

    # 載入 Graph 邊
    applied_in = load_csv_edges("APPLIED_IN_edges_20251123_151150.csv")
    uses_method = load_csv_edges("USES_METHOD_edges_20251123_151150.csv")

    # 建立查找表
    domain_papers = {}  # domain_name -> set(paper_ids)
    for paper_id, domain in applied_in:
        domain_papers.setdefault(domain, set()).add(paper_id)

    method_papers = {}  # method_name -> set(paper_ids)
    for paper_id, method in uses_method:
        method_papers.setdefault(method, set()).add(paper_id)

    print("=" * 70)
    print("Ground Truth 驗證報告")
    print("=" * 70)

    all_pass = True
    for q in gt["queries"]:
        qid = q["query_id"]
        qtext = q["query_text"]
        qtype = q["query_type"]
        gt_ids = set(q["ground_truth_paper_ids"])
        expected = q["expected_count"]

        # 從 CSV 計算實際值
        if qtype == "domain":
            target = q["graph_filter"]["target_value"]
            actual_ids = domain_papers.get(target, set())
        elif qtype == "method":
            target = q["graph_filter"]["target_value"]
            actual_ids = method_papers.get(target, set())
        elif qtype == "composite":
            target1 = q["graph_filter"]["target_value_1"]
            target2 = q["graph_filter"]["target_value_2"]
            domain_set = domain_papers.get(target1, set())
            method_set = method_papers.get(target2, set())
            actual_ids = domain_set & method_set
        else:
            actual_ids = set()

        # 比較
        match = gt_ids == actual_ids
        status = "✅ PASS" if match else "❌ FAIL"

        if not match:
            all_pass = False

        print(f"\n{status}  {qid}: {qtext}")
        print(f"  類型: {qtype} | 預期: {expected} 篇 | CSV 實際: {len(actual_ids)} 篇 | GT 定義: {len(gt_ids)} 篇")

        if not match:
            missing = actual_ids - gt_ids
            extra = gt_ids - actual_ids
            if missing:
                print(f"  ⚠️  GT 中缺少（CSV 有但 GT 沒列）:")
                for p in sorted(missing):
                    print(f"      - {p}")
            if extra:
                print(f"  ⚠️  GT 中多餘（GT 有但 CSV 沒有）:")
                for p in sorted(extra):
                    print(f"      - {p}")

    print("\n" + "=" * 70)
    if all_pass:
        print("✅ 全部驗證通過！Ground Truth 與 Graph CSV 完全一致。")
    else:
        print("❌ 部分驗證失敗，請修正 ground_truth.json。")
    print("=" * 70)

    # 額外統計
    print("\n📊 Graph 資料庫統計:")
    print(f"  Domain 分布:")
    for domain, papers in sorted(domain_papers.items()):
        print(f"    {domain}: {len(papers)} 篇")
    print(f"\n  Method 分布 (前 10):")
    sorted_methods = sorted(method_papers.items(), key=lambda x: len(x[1]), reverse=True)
    for method, papers in sorted_methods[:10]:
        print(f"    {method}: {len(papers)} 篇")


if __name__ == "__main__":
    main()
