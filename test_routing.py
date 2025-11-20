"""
Test agent routing with Graph RAG
"""
import sys
sys.path.insert(0, '.')

# Simulate the query routing logic
query_tests = [
    ("我想知道哪些論文和醫療有關", "GraphAnalysisCall", "Macro query - 'which papers'"),
    ("Which papers use LSTM?", "GraphAnalysisCall", "Macro query - 'which papers'"),
    ("比較不同領域的研究目標", "GraphAnalysisCall", "Macro query - 'compare'"),
    ("有多少篇論文使用深度學習?", "GraphAnalysisCall", "Macro query - 'how many'"),
    ("總結第一篇論文", "AssistantCall", "Micro query - specific paper summary"),
    ("解釋這篇論文的方法論", "AssistantCall", "Micro query - content explanation"),
    ("搜尋網路上最新的AI論文", "WebSearchCall", "Explicit web search"),
]

print("="*70)
print("🧪 Agent Routing Test Cases")
print("="*70)

for query, expected_tool, reason in query_tests:
    # Analyze query characteristics
    query_lower = query.lower()
    
    # Check for GraphAnalysisCall keywords
    graph_keywords = ['which papers', '哪些論文', 'list all', '列出所有', 'compare', '比較', 
                     'how many', '多少', 'count', '計算', 'all papers', '所有論文']
    has_graph_keyword = any(kw in query_lower for kw in graph_keywords)
    
    # Check for AssistantCall keywords  
    assistant_keywords = ['summarize', '總結', 'explain', '解釋', 'what is', '什麼是',
                          'methodology', '方法論', 'abstract', '摘要']
    has_assistant_keyword = any(kw in query_lower for kw in assistant_keywords)
    
    # Check for WebSearchCall keywords
    web_keywords = ['search internet', '搜尋網路', 'search online', 'web search', '網路搜尋']
    has_web_keyword = any(kw in query_lower for kw in web_keywords)
    
    # Predict routing
    if has_web_keyword:
        predicted = "WebSearchCall"
    elif has_graph_keyword:
        predicted = "GraphAnalysisCall"
    elif has_assistant_keyword:
        predicted = "AssistantCall"
    else:
        predicted = "AssistantCall"  # Default
    
    # Check result
    status = "✅" if predicted == expected_tool else "❌"
    
    print(f"\n{status} Query: {query}")
    print(f"   Expected: {expected_tool}")
    print(f"   Predicted: {predicted}")
    print(f"   Reason: {reason}")
    if predicted != expected_tool:
        print(f"   ⚠️  MISMATCH - Agent may route incorrectly!")

print("\n" + "="*70)
print("📝 Summary:")
print("="*70)
print("The agent prompt has been updated with 3-tier routing rules.")
print("GraphAnalysisCall should handle: which/list/compare/count queries")
print("AssistantCall should handle: specific paper content queries")
print("WebSearchCall should handle: explicit internet search requests")
print("\n✅ Now restart agent2.py and test in the web interface!")
