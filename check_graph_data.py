"""
Quick script to check what's actually in Neo4j
"""
from system_api.graph_manager import GraphManager
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model='jcai/llama-3-taiwan-8b-instruct:q4_k_m', base_url='http://localhost:11434')
gm = GraphManager(llm)

print('\n' + '='*80)
print('Neo4j Database Content Check')
print('='*80)

# 1. Check Papers
print('\n📄 Papers in Database:')
papers = gm.graph.query('MATCH (p:Paper) RETURN p.paper_id, p.title')
print(f'   Total: {len(papers)} papers')
for i, p in enumerate(papers[:5], 1):
    print(f'   {i}. {p["p.title"]}')
if len(papers) > 5:
    print(f'   ... and {len(papers) - 5} more')

# 2. Check Methods
print('\n🔧 Methods in Database:')
methods = gm.graph.query('MATCH (m:Method) RETURN DISTINCT m.name ORDER BY m.name')
print(f'   Total: {len(methods)} unique methods')
for i, m in enumerate(methods, 1):
    print(f'   {i}. {m["m.name"]}')

# 3. Check Paper-Method relationships
print('\n🔗 Paper-Method Relationships:')
rels = gm.graph.query('''
    MATCH (p:Paper)-[:USES_METHOD]->(m:Method) 
    RETURN p.title, collect(m.name) as methods
''')
print(f'   Total: {len(rels)} papers with methods')
for r in rels[:5]:
    methods_str = ', '.join(r['methods'][:3])
    if len(r['methods']) > 3:
        methods_str += f' (+{len(r["methods"]) - 3} more)'
    print(f'   - {r["p.title"][:60]}')
    print(f'     Methods: {methods_str}')

# 4. Test specific queries
print('\n🔍 Test Query: Papers with "ensemble" or "集成"')
test1 = gm.graph.query('''
    MATCH (p:Paper)-[:USES_METHOD]->(m:Method)
    WHERE toLower(m.name) CONTAINS "ensemble" 
       OR toLower(m.name) CONTAINS "集成"
    RETURN p.title, m.name
''')
if test1:
    print(f'   Found {len(test1)} results:')
    for r in test1:
        print(f'   - {r["p.title"]}: {r["m.name"]}')
else:
    print('   ❌ No results found')
    print('   Checking if methods contain these terms...')
    all_methods = gm.graph.query('MATCH (m:Method) RETURN m.name')
    matching = [m['m.name'] for m in all_methods if 'ensemble' in m['m.name'].lower() or '集成' in m['m.name'].lower()]
    if matching:
        print(f'   Found matching methods: {matching}')
    else:
        print('   No methods contain "ensemble" or "集成"')

print('\n🔍 Test Query: Papers with "Mask R-CNN"')
test2 = gm.graph.query('''
    MATCH (p:Paper)-[:USES_METHOD]->(m:Method)
    WHERE toLower(m.name) CONTAINS "mask"
    RETURN p.title, m.name
''')
if test2:
    print(f'   Found {len(test2)} results:')
    for r in test2:
        print(f'   - {r["p.title"]}: {r["m.name"]}')
else:
    print('   ❌ No results found')
    print('   Checking if methods contain "mask"...')
    all_methods = gm.graph.query('MATCH (m:Method) RETURN m.name')
    matching = [m['m.name'] for m in all_methods if 'mask' in m['m.name'].lower() or 'rcnn' in m['m.name'].lower() or 'r-cnn' in m['m.name'].lower()]
    if matching:
        print(f'   Found matching methods: {matching}')
    else:
        print('   No methods contain "mask", "rcnn", or "r-cnn"')

print('\n' + '='*80)
