"""
build_ground_truth.py - 從 Graph CSV 匯出自動生成 ground_truth.json
確保 paper_id 與 CSV 中的 bytes 完全一致（避免 Unicode 正規化差異）
"""
import csv
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPH_EXPORTS = os.path.join(BASE_DIR, "graph_exports")
OUTPUT_PATH = os.path.join(BASE_DIR, "benchmark", "ground_truth.json")


def load_edges(filename):
    """載入邊 CSV"""
    edges = []
    path = os.path.join(GRAPH_EXPORTS, filename)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            edges.append((row["source_key"], row["target_key"]))
    return edges


def build_lookup(edges):
    """建立 target_value -> set(paper_ids) 對照表"""
    lookup = {}
    for paper_id, target in edges:
        lookup.setdefault(target, set()).add(paper_id)
    return lookup


def main():
    # 載入所有邊
    applied_in = load_edges("APPLIED_IN_edges_20251123_151150.csv")
    uses_method = load_edges("USES_METHOD_edges_20251123_151150.csv")

    domain_papers = build_lookup(applied_in)
    method_papers = build_lookup(uses_method)

    # 定義 10 組查詢
    queries = [
        {
            "query_id": "Q01",
            "query_text": "哪些論文和醫療相關？",
            "query_type": "domain",
            "graph_filter": {
                "relationship": "APPLIED_IN",
                "target_node": "Domain",
                "target_value": "Healthcare"
            },
            "cypher": "MATCH (p:Paper)-[:APPLIED_IN]->(d:Domain {name: 'Healthcare'}) RETURN p.paper_id",
            "ground_truth_paper_ids": sorted(domain_papers.get("Healthcare", set())),
            "difficulty": "medium",
            "notes": "RAG 挑戰：鼻咽癌論文摘要未必包含「醫療」二字，而是強調「深度學習」「影像辨識」"
        },
        {
            "query_id": "Q02",
            "query_text": "哪些論文和智慧交通相關？",
            "query_type": "domain",
            "graph_filter": {
                "relationship": "APPLIED_IN",
                "target_node": "Domain",
                "target_value": "Smart Transportation"
            },
            "cypher": "MATCH (p:Paper)-[:APPLIED_IN]->(d:Domain {name: 'Smart Transportation'}) RETURN p.paper_id",
            "ground_truth_paper_ids": sorted(domain_papers.get("Smart Transportation", set())),
            "difficulty": "hard",
            "notes": "RAG 挑戰：「天際線查詢」「YouBike」「澳門公車」的摘要可能不直接包含「交通」關鍵字"
        },
        {
            "query_id": "Q03",
            "query_text": "哪些論文和環境監測相關？",
            "query_type": "domain",
            "graph_filter": {
                "relationship": "APPLIED_IN",
                "target_node": "Domain",
                "target_value": "Environmental Monitoring"
            },
            "cypher": "MATCH (p:Paper)-[:APPLIED_IN]->(d:Domain {name: 'Environmental Monitoring'}) RETURN p.paper_id",
            "ground_truth_paper_ids": sorted(domain_papers.get("Environmental Monitoring", set())),
            "difficulty": "easy",
            "notes": "只有 1 篇在 Graph 中，RAG 可能額外找到標準3（也是 PM2.5，但不在 Graph 中）"
        },
        {
            "query_id": "Q04",
            "query_text": "哪些論文和觀光旅遊相關？",
            "query_type": "domain",
            "graph_filter": {
                "relationship": "APPLIED_IN",
                "target_node": "Domain",
                "target_value": "Tourism"
            },
            "cypher": "MATCH (p:Paper)-[:APPLIED_IN]->(d:Domain {name: 'Tourism'}) RETURN p.paper_id",
            "ground_truth_paper_ids": sorted(domain_papers.get("Tourism", set())),
            "difficulty": "medium",
            "notes": "RAG 挑戰：「旅程推薦」和「旅遊區辨識」語義接近旅遊，但摘要可能偏向技術描述"
        },
        {
            "query_id": "Q05",
            "query_text": "哪些論文使用CNN？",
            "query_type": "method",
            "graph_filter": {
                "relationship": "USES_METHOD",
                "target_node": "Method",
                "target_value": "CNN"
            },
            "cypher": "MATCH (p:Paper)-[:USES_METHOD]->(m:Method {name: 'CNN'}) RETURN p.paper_id",
            "ground_truth_paper_ids": sorted(method_papers.get("CNN", set())),
            "difficulty": "hard",
            "notes": "RAG 挑戰：部分論文用 CNN 做前處理而非主模型，摘要可能不強調 CNN。Graph 另有「卷積神經網路」「金字塔卷積類神經網路」「3D-CNN」為獨立節點。"
        },
        {
            "query_id": "Q06",
            "query_text": "哪些論文使用LSTM？",
            "query_type": "method",
            "graph_filter": {
                "relationship": "USES_METHOD",
                "target_node": "Method",
                "target_value": "LSTM"
            },
            "cypher": "MATCH (p:Paper)-[:USES_METHOD]->(m:Method {name: 'LSTM'}) RETURN p.paper_id",
            "ground_truth_paper_ids": sorted(method_papers.get("LSTM", set())),
            "difficulty": "medium",
            "notes": "Graph 另有「mRBF-LSTM」為獨立節點（標準5），不包含在 LSTM 節點中"
        },
        {
            "query_id": "Q07",
            "query_text": "哪些論文使用YOLO？",
            "query_type": "method",
            "graph_filter": {
                "relationship": "USES_METHOD",
                "target_node": "Method",
                "target_value": "YOLO"
            },
            "cypher": "MATCH (p:Paper)-[:USES_METHOD]->(m:Method {name: 'YOLO'}) RETURN p.paper_id",
            "ground_truth_paper_ids": sorted(method_papers.get("YOLO", set())),
            "difficulty": "medium",
            "notes": "Graph 另有「YOLO-v5」為獨立節點（標準19），不包含在 YOLO 節點中"
        },
        {
            "query_id": "Q08",
            "query_text": "哪些論文使用Ensemble Learning？",
            "query_type": "method",
            "graph_filter": {
                "relationship": "USES_METHOD",
                "target_node": "Method",
                "target_value": "Ensemble Learning"
            },
            "cypher": "MATCH (p:Paper)-[:USES_METHOD]->(m:Method {name: 'Ensemble Learning'}) RETURN p.paper_id",
            "ground_truth_paper_ids": sorted(method_papers.get("Ensemble Learning", set())),
            "difficulty": "medium",
            "notes": "Graph 另有「集成式學習」為獨立中文節點（基礎5），不包含在此英文節點中"
        },
        {
            "query_id": "Q09",
            "query_text": "哪些醫療相關的論文使用CNN？",
            "query_type": "composite",
            "graph_filter": {
                "relationship_1": "APPLIED_IN",
                "target_node_1": "Domain",
                "target_value_1": "Healthcare",
                "relationship_2": "USES_METHOD",
                "target_node_2": "Method",
                "target_value_2": "CNN"
            },
            "cypher": "MATCH (p:Paper)-[:APPLIED_IN]->(d:Domain {name: 'Healthcare'}) WHERE (p)-[:USES_METHOD]->(:Method {name: 'CNN'}) RETURN p.paper_id",
            "ground_truth_paper_ids": sorted(
                domain_papers.get("Healthcare", set()) & method_papers.get("CNN", set())
            ),
            "difficulty": "hard",
            "notes": "Healthcare 有 4 篇，其中 3 篇使用 CNN。標準4（RBF-DNN）不使用 CNN → 排除。RAG 無法做交集運算，這是最大弱點。"
        },
        {
            "query_id": "Q10",
            "query_text": "哪些智慧交通相關的論文使用LSTM？",
            "query_type": "composite",
            "graph_filter": {
                "relationship_1": "APPLIED_IN",
                "target_node_1": "Domain",
                "target_value_1": "Smart Transportation",
                "relationship_2": "USES_METHOD",
                "target_node_2": "Method",
                "target_value_2": "LSTM"
            },
            "cypher": "MATCH (p:Paper)-[:APPLIED_IN]->(d:Domain {name: 'Smart Transportation'}) WHERE (p)-[:USES_METHOD]->(:Method {name: 'LSTM'}) RETURN p.paper_id",
            "ground_truth_paper_ids": sorted(
                domain_papers.get("Smart Transportation", set()) & method_papers.get("LSTM", set())
            ),
            "difficulty": "hard",
            "notes": "Smart Transportation 有 9 篇，其中只有 3 篇使用 LSTM。基礎9 使用 LSTM 但屬於 Smart Building → 排除。RAG 無法做交集運算。"
        }
    ]

    # 加入 expected_count
    for q in queries:
        q["expected_count"] = len(q["ground_truth_paper_ids"])

    # 組裝輸出
    output = {
        "_metadata": {
            "description": "Benchmark Ground Truth: KnowsRoots Graph vs Traditional RAG",
            "version": "1.0",
            "created": "2026-02-24",
            "source": "Neo4j Graph exports (2025-11-23), auto-generated by build_ground_truth.py",
            "total_papers_in_graph": len(set(p for p, _ in applied_in) | set(p for p, _ in uses_method)),
            "total_queries": len(queries),
            "notes": "Ground Truth 根據 Neo4j Graph 結構化邊（APPLIED_IN, USES_METHOD）定義。paper_id 直接從 CSV 讀取，確保 Unicode 完全一致。"
        },
        "queries": queries
    }

    # 寫入
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"✅ Ground Truth 已生成: {OUTPUT_PATH}")
    print(f"   共 {len(queries)} 組查詢")
    for q in queries:
        print(f"   {q['query_id']}: {q['query_text']} → {q['expected_count']} 篇")


if __name__ == "__main__":
    main()
