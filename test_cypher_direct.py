"""
Test the exact Cypher query that agent generated
"""
from system_api.graph_manager import GraphManager
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model='jcai/llama-3-taiwan-8b-instruct:q4_k_m', base_url='http://localhost:11434')
gm = GraphManager(llm)

print('\n' + '='*80)
print('Testing Agent-Generated Cypher')
print('='*80)

# The exact query from agent logs
agent_cypher = """
MATCH (p:Paper)-[:USES_METHOD]->(m:Method)
WHERE toLower(m.name) CONTAINS "ensemble" OR toLower(m.name) CONTAINS "集成"
RETURN DISTINCT p.title, p.year
"""

print('\n📝 Query:')
print(agent_cypher)

print('\n🔍 Executing...')
try:
    result = gm.graph.query(agent_cypher)
    print(f'\n✅ Success! Found {len(result)} results:')
    for r in result:
        print(f'   - {r["p.title"]} ({r["p.year"]})')
except Exception as e:
    print(f'\n❌ Error: {e}')

# Also test the simpler version
print('\n' + '-'*80)
print('Testing Simpler Version')
print('-'*80)

simple_cypher = """
MATCH (p:Paper)-[:USES_METHOD]->(m:Method)
WHERE toLower(m.name) CONTAINS "集成"
RETURN p.title, p.year
"""

print('\n📝 Query:')
print(simple_cypher)

print('\n🔍 Executing...')
try:
    result = gm.graph.query(simple_cypher)
    print(f'\n✅ Success! Found {len(result)} results:')
    for r in result:
        print(f'   - {r["p.title"]} ({r["p.year"]})')
except Exception as e:
    print(f'\n❌ Error: {e}')

# Check exact method name
print('\n' + '-'*80)
print('Checking Exact Method Names')
print('-'*80)

check_methods = """
MATCH (m:Method)
RETURN m.name
ORDER BY m.name
"""

result = gm.graph.query(check_methods)
print(f'\nAll Methods in DB ({len(result)}):')
for r in result:
    name = r['m.name']
    has_ensemble = 'ensemble' in name.lower()
    has_jicheng = '集成' in name
    print(f'   - "{name}"  [ensemble: {has_ensemble}, 集成: {has_jicheng}]')

print('\n' + '='*80)
