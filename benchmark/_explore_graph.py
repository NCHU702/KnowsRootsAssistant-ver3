"""
explore_graph.py - 完整探索 Neo4j Graph 結構
輸出所有 Node 和 Edge，用於設計不同抽象層級的 benchmark 查詢
"""
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

with driver.session() as s:
    # ── All Papers ──
    print("=" * 80)
    print("ALL PAPERS")
    print("=" * 80)
    result = s.run("MATCH (p:Paper) RETURN p.paper_id, p.title, p.year ORDER BY p.paper_id")
    papers = []
    for r in result:
        papers.append(r)
        print(f"  {r['p.paper_id']}")
    print(f"  Total: {len(papers)}")

    # ── All Domains ──
    print("\n" + "=" * 80)
    print("ALL DOMAINS (with papers)")
    print("=" * 80)
    result = s.run("""
        MATCH (d:Domain)<-[:APPLIED_IN]-(p:Paper)
        RETURN d.name, d.name_zh, collect(p.paper_id) as papers
        ORDER BY d.name
    """)
    for r in result:
        print(f"\n  [{r['d.name']}] ({r['d.name_zh']}): {len(r['papers'])} papers")
        for pid in sorted(r['papers']):
            print(f"    - {pid}")

    # ── All Methods ──
    print("\n" + "=" * 80)
    print("ALL METHODS (with papers)")
    print("=" * 80)
    result = s.run("""
        MATCH (m:Method)<-[:USES_METHOD]-(p:Paper)
        RETURN m.name, collect(p.paper_id) as papers
        ORDER BY size(collect(p.paper_id)) DESC, m.name
    """)
    methods = []
    for r in result:
        methods.append(r)
        print(f"\n  [{r['m.name']}]: {len(r['papers'])} papers")
        for pid in sorted(r['papers']):
            print(f"    - {pid}")

    # ── All Datasets ──
    print("\n" + "=" * 80)
    print("ALL DATASETS (with papers)")
    print("=" * 80)
    result = s.run("""
        MATCH (d:Dataset)<-[:EVALUATED_ON]-(p:Paper)
        RETURN d.name, collect(p.paper_id) as papers
        ORDER BY size(collect(p.paper_id)) DESC, d.name
    """)
    for r in result:
        print(f"\n  [{r['d.name']}]: {len(r['papers'])} papers")
        for pid in sorted(r['papers']):
            print(f"    - {pid}")

    # ── All Metrics ──
    print("\n" + "=" * 80)
    print("ALL METRICS (with papers)")
    print("=" * 80)
    result = s.run("""
        MATCH (met:Metric)<-[:EVALUATED_WITH]-(p:Paper)
        RETURN met.name, collect(p.paper_id) as papers
        ORDER BY size(collect(p.paper_id)) DESC, met.name
    """)
    for r in result:
        print(f"\n  [{r['met.name']}]: {len(r['papers'])} papers")
        for pid in sorted(r['papers']):
            print(f"    - {pid}")

    # ── All ResearchGoals ──
    print("\n" + "=" * 80)
    print("ALL RESEARCH GOALS")
    print("=" * 80)
    result = s.run("""
        MATCH (g:ResearchGoal)<-[:AIMS_TO]-(p:Paper)
        RETURN g.summary, collect(p.paper_id) as papers
        ORDER BY p.paper_id
    """)
    for r in result:
        print(f"\n  Goal: {r['g.summary'][:120]}...")
        for pid in sorted(r['papers']):
            print(f"    - {pid}")

    # ── Cross-reference: Paper → all connections ──
    print("\n" + "=" * 80)
    print("PAPER FULL PROFILE (first 5)")
    print("=" * 80)
    result = s.run("""
        MATCH (p:Paper)
        OPTIONAL MATCH (p)-[:APPLIED_IN]->(d:Domain)
        OPTIONAL MATCH (p)-[:USES_METHOD]->(m:Method)
        OPTIONAL MATCH (p)-[:EVALUATED_ON]->(ds:Dataset)
        OPTIONAL MATCH (p)-[:EVALUATED_WITH]->(met:Metric)
        OPTIONAL MATCH (p)-[:AIMS_TO]->(g:ResearchGoal)
        RETURN p.paper_id,
               collect(DISTINCT d.name) as domains,
               collect(DISTINCT m.name) as methods,
               collect(DISTINCT ds.name) as datasets,
               collect(DISTINCT met.name) as metrics,
               collect(DISTINCT left(g.summary, 80)) as goals
        ORDER BY p.paper_id
    """)
    for r in result:
        print(f"\n  📄 {r['p.paper_id']}")
        print(f"     Domains:  {r['domains']}")
        print(f"     Methods:  {r['methods']}")
        print(f"     Datasets: {r['datasets']}")
        print(f"     Metrics:  {r['metrics']}")
        if r['goals']:
            print(f"     Goal:     {r['goals'][0]}")

driver.close()
