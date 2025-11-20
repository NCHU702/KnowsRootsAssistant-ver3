"""
Quick test to verify Neo4j data and query
"""
from langchain_ollama import OllamaLLM
from system_api.graph_manager import GraphManager

# Initialize
llm = OllamaLLM(model="jcai/llama-3-taiwan-8b-instruct:q4_k_m", temperature=0)
gm = GraphManager(llm=llm)

if not gm.graph:
    print("❌ Failed to connect to Neo4j")
    exit(1)

print("✓ Connected to Neo4j\n")

# Check what's in database
print("="*60)
print("Checking Database Contents")
print("="*60)

# 1. Count papers
count_query = "MATCH (p:Paper) RETURN count(p) as count"
result = gm.graph.query(count_query)
print(f"\n📄 Total Papers: {result[0]['count']}")

# 2. List all papers with domains
papers_query = """
MATCH (p:Paper)-[:APPLIED_IN]->(dom:Domain)
RETURN p.paper_id as id, p.title as title, dom.name as domain
"""
papers = gm.graph.query(papers_query)
print(f"\n📋 Papers with Domains:")
for p in papers:
    print(f"  - {p['title']}")
    print(f"    Domain: {p['domain']}")
    print(f"    ID: {p['id']}\n")

# 3. List all domains
domain_query = "MATCH (d:Domain) RETURN d.name as name"
domains = gm.graph.query(domain_query)
print(f"🏥 All Domains:")
for d in domains:
    print(f"  - {d['name']}")

# 4. Test specific query
print("\n" + "="*60)
print("Testing Query: 'Which papers are related to healthcare?'")
print("="*60)

test_query = """
MATCH (p:Paper)-[:APPLIED_IN]->(dom:Domain)
WHERE toLower(dom.name) CONTAINS 'healthcare'
RETURN p.title as title, dom.name as domain
"""
results = gm.graph.query(test_query)
print(f"\nDirect Cypher Result: {len(results)} papers found")
for r in results:
    print(f"  - {r['title']} (Domain: {r['domain']})")

# 5. Test with LLM-generated query
print("\n" + "="*60)
print("Testing with GraphManager.query_graph()")
print("="*60)
llm_result = gm.query_graph("Which papers are related to healthcare?")
print(f"\nLLM Result:\n{llm_result}")
