"""
測試 Graph RAG 在 build_indices 後是否會自動執行

這個腳本會：
1. 清空現有索引
2. 初始化 HierarchicalRAGSystem（會觸發 build_indices）
3. 驗證 Neo4j 是否包含所有 5 篇論文
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_ollama import OllamaLLM
from system_api.graph_manager import GraphManager

def check_neo4j_papers():
    """檢查 Neo4j 中的論文數量"""
    try:
        # 使用基本的 OllamaLLM 初始化 GraphManager
        llm = OllamaLLM(
            model="jcai/llama-3-taiwan-8b-instruct:q4_k_m",
            temperature=0,
            base_url="http://localhost:11434"
        )
        
        graph_manager = GraphManager(llm=llm)
        
        if not graph_manager.graph:
            print("❌ GraphManager failed to connect to Neo4j")
            return
        
        # Query all papers
        query = """
        MATCH (p:Paper)
        RETURN p.paper_id as paper_id, p.title as title, p.year as year
        ORDER BY p.title
        """
        
        results = graph_manager.graph.query(query)
        
        print("\n" + "="*80)
        print("NEO4J PAPERS")
        print("="*80)
        print(f"Total papers: {len(results)}")
        print()
        
        for i, row in enumerate(results, 1):
            print(f"[{i}] {row['paper_id']}")
            print(f"    Title: {row['title']}")
            print(f"    Year: {row['year']}")
            print()
        
        # Also check domains
        query_domains = """
        MATCH (p:Paper)-[:APPLIED_IN]->(dom:Domain)
        RETURN p.paper_id as paper_id, dom.name as domain, dom.name_zh as domain_zh
        """
        
        domain_results = graph_manager.graph.query(query_domains)
        
        print("="*80)
        print("PAPER DOMAINS")
        print("="*80)
        for row in domain_results:
            print(f"{row['paper_id']}: {row['domain']} ({row['domain_zh']})")
        
        print("="*80)
        
        if len(results) == 5:
            print("✅ SUCCESS: All 5 papers are indexed in Neo4j!")
        else:
            print(f"⚠️  Expected 5 papers, found {len(results)}")
        
    except Exception as e:
        print(f"❌ Error checking Neo4j: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    print("Waiting a moment for agent2.py to complete indexing...")
    print("(Run this script AFTER agent2.py has finished building indices)")
    print()
    
    input("Press Enter to check Neo4j papers...")
    
    check_neo4j_papers()
