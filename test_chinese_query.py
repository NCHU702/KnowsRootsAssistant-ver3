"""
Test Chinese query translation in Graph RAG
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

# Test Chinese queries
test_queries = [
    "哪些研究和醫療有關?",
    "有哪些論文使用深度學習?",
    "製造業相關的論文有哪些?",
]

print("="*70)
print("Testing Chinese Query Translation")
print("="*70)

for query in test_queries:
    print(f"\n🔍 Query: {query}")
    print("-" * 70)
    result = gm.query_graph(query)
    print(f"Result:\n{result}\n")

print("="*70)
print("✅ Test Complete")
print("="*70)
