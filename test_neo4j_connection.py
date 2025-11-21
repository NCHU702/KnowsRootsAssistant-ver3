#!/usr/bin/env python3
"""
Quick Neo4j Connection Test
快速測試 Neo4j 連接是否正常
"""

import logging
from langchain_community.graphs import Neo4jGraph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_neo4j_connection():
    """測試 Neo4j 連接"""
    
    logger.info("Testing Neo4j connection...")
    
    # 嘗試連接
    try:
        graph = Neo4jGraph(
            url="bolt://localhost:7687",
            username="neo4j",
            password="password"  # 請修改為你的密碼
        )
        
        logger.info("✓ Connected to Neo4j successfully!")
        
        # 測試基本查詢
        result = graph.query("MATCH (n) RETURN count(n) as node_count")
        node_count = result[0]['node_count'] if result else 0
        logger.info(f"✓ Database has {node_count} nodes")
        
        # 檢查 Paper 節點
        result = graph.query("MATCH (p:Paper) RETURN count(p) as paper_count")
        paper_count = result[0]['paper_count'] if result else 0
        logger.info(f"✓ Found {paper_count} Paper nodes")
        
        if paper_count > 0:
            # 顯示一些論文
            result = graph.query("MATCH (p:Paper) RETURN p.paper_id, p.title LIMIT 3")
            logger.info("\nSample papers:")
            for i, row in enumerate(result, 1):
                logger.info(f"  {i}. {row['p.paper_id']}: {row['p.title']}")
        
        logger.info("\n✅ Neo4j is ready for testing!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to connect to Neo4j: {e}")
        logger.error("\nPlease check:")
        logger.error("1. Neo4j is running (check with: ps aux | grep neo4j)")
        logger.error("2. Bolt port 7687 is accessible")
        logger.error("3. Username and password are correct")
        return False

if __name__ == "__main__":
    test_neo4j_connection()
