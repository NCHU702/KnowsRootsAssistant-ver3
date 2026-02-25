"""
build_ground_truth.py - 從 Neo4j 直接生成 ground_truth.json (v2)
=================================================================

確保 paper_id 與 live Neo4j 完全一致（不再依賴可能過時的 CSV 匯出）。
"""
import json
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(BASE_DIR, "benchmark", "ground_truth.json")


def neo4j_session():
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
    return driver


def query_paper_ids(session, cypher: str):
    """Execute Cypher and return sorted list of paper_ids."""
    result = session.run(cypher)
    return sorted({r[0] for r in result if r[0]})


def main():
    driver = neo4j_session()

    with driver.session() as session:
        # Helper: get papers for a domain
        def domain_papers(domain_name):
            return query_paper_ids(session,
                f"MATCH (p:Paper)-[:APPLIED_IN]->(d:Domain {{name: '{domain_name}'}}) RETURN p.paper_id")

        # Helper: get papers for a method
        def method_papers(method_name):
            return query_paper_ids(session,
                f"MATCH (p:Paper)-[:USES_METHOD]->(m:Method {{name: '{method_name}'}}) RETURN p.paper_id")

        # Helper: intersection
        def intersect(list_a, list_b):
            return sorted(set(list_a) & set(list_b))

        # Count total papers
        total_papers = session.run("MATCH (n:Paper) RETURN count(n)").single()[0]

        # ──── 10 Queries ────
        queries = [
            {
                "query_id": "Q01",
                "query_text": "哪些論文和醫療相關？",
                "query_type": "domain",
                "graph_filter": {"relationship": "APPLIED_IN", "target_node": "Domain", "target_value": "Healthcare"},
                "cypher": "MATCH (p:Paper)-[:APPLIED_IN]->(d:Domain {name: 'Healthcare'}) RETURN p.paper_id",
                "ground_truth_paper_ids": domain_papers("Healthcare"),
                "difficulty": "medium",
                "notes": "RAG 挑戰：鼻咽癌論文摘要未必包含「醫療」二字"
            },
            {
                "query_id": "Q02",
                "query_text": "哪些論文和智慧交通相關？",
                "query_type": "domain",
                "graph_filter": {"relationship": "APPLIED_IN", "target_node": "Domain", "target_value": "Smart Transportation"},
                "cypher": "MATCH (p:Paper)-[:APPLIED_IN]->(d:Domain {name: 'Smart Transportation'}) RETURN p.paper_id",
                "ground_truth_paper_ids": domain_papers("Smart Transportation"),
                "difficulty": "hard",
                "notes": "RAG 挑戰：「天際線查詢」「YouBike」「澳門公車」的摘要可能不直接包含「交通」關鍵字"
            },
            {
                "query_id": "Q03",
                "query_text": "哪些論文和環境監測相關？",
                "query_type": "domain",
                "graph_filter": {"relationship": "APPLIED_IN", "target_node": "Domain", "target_value": "Environmental Monitoring"},
                "cypher": "MATCH (p:Paper)-[:APPLIED_IN]->(d:Domain {name: 'Environmental Monitoring'}) RETURN p.paper_id",
                "ground_truth_paper_ids": domain_papers("Environmental Monitoring"),
                "difficulty": "easy",
                "notes": "只有少量論文，RAG 可能額外找到標準3（PM2.5，但不在 Graph 中）"
            },
            {
                "query_id": "Q04",
                "query_text": "哪些論文和觀光旅遊相關？",
                "query_type": "domain",
                "graph_filter": {"relationship": "APPLIED_IN", "target_node": "Domain", "target_value": "Tourism"},
                "cypher": "MATCH (p:Paper)-[:APPLIED_IN]->(d:Domain {name: 'Tourism'}) RETURN p.paper_id",
                "ground_truth_paper_ids": domain_papers("Tourism"),
                "difficulty": "medium",
                "notes": "RAG 挑戰：「旅程推薦」和「旅遊區辨識」語義接近旅遊，但摘要可能偏向技術描述"
            },
            {
                "query_id": "Q05",
                "query_text": "哪些論文使用CNN？",
                "query_type": "method",
                "graph_filter": {"relationship": "USES_METHOD", "target_node": "Method", "target_value": "CNN"},
                "cypher": "MATCH (p:Paper)-[:USES_METHOD]->(m:Method {name: 'CNN'}) RETURN p.paper_id",
                "ground_truth_paper_ids": method_papers("CNN"),
                "difficulty": "hard",
                "notes": "部分論文用 CNN 做前處理而非主模型。Graph 另有「卷積神經網路」「3D-CNN」為獨立節點"
            },
            {
                "query_id": "Q06",
                "query_text": "哪些論文使用LSTM？",
                "query_type": "method",
                "graph_filter": {"relationship": "USES_METHOD", "target_node": "Method", "target_value": "LSTM"},
                "cypher": "MATCH (p:Paper)-[:USES_METHOD]->(m:Method {name: 'LSTM'}) RETURN p.paper_id",
                "ground_truth_paper_ids": method_papers("LSTM"),
                "difficulty": "medium",
                "notes": "Graph 另有「mRBF-LSTM」為獨立節點，不包含在 LSTM 節點中"
            },
            {
                "query_id": "Q07",
                "query_text": "哪些論文使用YOLO？",
                "query_type": "method",
                "graph_filter": {"relationship": "USES_METHOD", "target_node": "Method", "target_value": "YOLO"},
                "cypher": "MATCH (p:Paper)-[:USES_METHOD]->(m:Method {name: 'YOLO'}) RETURN p.paper_id",
                "ground_truth_paper_ids": method_papers("YOLO"),
                "difficulty": "medium",
                "notes": "Graph 另有「YOLO-v5」為獨立節點，不包含在 YOLO 節點中"
            },
            {
                "query_id": "Q08",
                "query_text": "哪些論文使用Ensemble Learning？",
                "query_type": "method",
                "graph_filter": {"relationship": "USES_METHOD", "target_node": "Method", "target_value": "Ensemble Learning"},
                "cypher": "MATCH (p:Paper)-[:USES_METHOD]->(m:Method {name: 'Ensemble Learning'}) RETURN p.paper_id",
                "ground_truth_paper_ids": method_papers("Ensemble Learning"),
                "difficulty": "medium",
                "notes": "Graph 另有「集成式學習」為獨立中文節點，不包含在此英文節點中"
            },
            {
                "query_id": "Q09",
                "query_text": "哪些醫療相關的論文使用CNN？",
                "query_type": "composite",
                "graph_filter": {
                    "relationship_1": "APPLIED_IN", "target_node_1": "Domain", "target_value_1": "Healthcare",
                    "relationship_2": "USES_METHOD", "target_node_2": "Method", "target_value_2": "CNN"
                },
                "cypher": "MATCH (p:Paper)-[:APPLIED_IN]->(d:Domain {name: 'Healthcare'}) WHERE (p)-[:USES_METHOD]->(:Method {name: 'CNN'}) RETURN p.paper_id",
                "ground_truth_paper_ids": intersect(domain_papers("Healthcare"), method_papers("CNN")),
                "difficulty": "hard",
                "notes": "Healthcare ∩ CNN。RAG 無法做交集運算，這是最大弱點。"
            },
            {
                "query_id": "Q10",
                "query_text": "哪些智慧交通相關的論文使用LSTM？",
                "query_type": "composite",
                "graph_filter": {
                    "relationship_1": "APPLIED_IN", "target_node_1": "Domain", "target_value_1": "Smart Transportation",
                    "relationship_2": "USES_METHOD", "target_node_2": "Method", "target_value_2": "LSTM"
                },
                "cypher": "MATCH (p:Paper)-[:APPLIED_IN]->(d:Domain {name: 'Smart Transportation'}) WHERE (p)-[:USES_METHOD]->(:Method {name: 'LSTM'}) RETURN p.paper_id",
                "ground_truth_paper_ids": intersect(domain_papers("Smart Transportation"), method_papers("LSTM")),
                "difficulty": "hard",
                "notes": "Smart Transportation ∩ LSTM。RAG 無法做交集運算。"
            }
        ]

        # Add expected_count
        for q in queries:
            q["expected_count"] = len(q["ground_truth_paper_ids"])

        # Build output
        output = {
            "_metadata": {
                "description": "Benchmark Ground Truth: KnowsRoots Graph vs Traditional RAG",
                "version": "2.0",
                "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source": "Live Neo4j (bolt://localhost:7687) — queried directly for exact consistency",
                "total_papers_in_graph": total_papers,
                "total_queries": len(queries),
                "notes": "paper_id 直接從 Neo4j 讀取，確保與 benchmark Cypher 查詢完全一致。"
            },
            "queries": queries
        }

    driver.close()

    # Write
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"✅ Ground Truth 已生成 (from live Neo4j): {OUTPUT_PATH}")
    print(f"   共 {len(queries)} 組查詢, {total_papers} 篇 Paper in Graph")
    for q in queries:
        print(f"   {q['query_id']}: {q['query_text']} → {q['expected_count']} 篇")


if __name__ == "__main__":
    main()
