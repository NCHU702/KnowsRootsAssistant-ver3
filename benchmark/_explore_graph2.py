"""explore_graph_part2.py - Get ResearchGoals + Paper profiles"""
from neo4j import GraphDatabase
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
with driver.session() as s:
    print("ALL RESEARCH GOALS")
    print("=" * 80)
    result = s.run("""
        MATCH (p:Paper)-[:AIMS_TO]->(g:ResearchGoal)
        RETURN p.paper_id, g.summary
        ORDER BY p.paper_id
    """)
    for r in result:
        print(f"  {r['p.paper_id']}")
        print(f"    → {r['g.summary'][:150]}")
        print()

    print("=" * 80)
    print("PAPER FULL PROFILES")
    print("=" * 80)
    result = s.run("""
        MATCH (p:Paper)
        OPTIONAL MATCH (p)-[:APPLIED_IN]->(d:Domain)
        OPTIONAL MATCH (p)-[:USES_METHOD]->(m:Method)
        OPTIONAL MATCH (p)-[:EVALUATED_ON]->(ds:Dataset)
        OPTIONAL MATCH (p)-[:EVALUATED_WITH]->(met:Metric)
        RETURN p.paper_id,
               collect(DISTINCT d.name) as domains,
               collect(DISTINCT m.name) as methods,
               collect(DISTINCT ds.name) as datasets,
               collect(DISTINCT met.name) as metrics
        ORDER BY p.paper_id
    """)
    for r in result:
        print(f"\n  📄 {r['p.paper_id']}")
        print(f"     Domain:   {r['domains']}")
        print(f"     Methods:  {r['methods']}")
        print(f"     Datasets: {r['datasets']}")
        print(f"     Metrics:  {r['metrics']}")
driver.close()
