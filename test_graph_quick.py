#!/usr/bin/env python3
"""
快速測試：驗證 Neo4j 中已存在的 Chunk-Level Entity Extraction 數據
不需要重新索引，只查詢數據庫
"""

import logging
from langchain_ollama import OllamaLLM
from system_api.graph_manager import GraphManager

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """快速測試流程"""
    
    logger.info("=" * 70)
    logger.info("🔍 Quick Test: Chunk-Level Entity Extraction Verification")
    logger.info("=" * 70)
    
    # 1. 初始化 Graph Manager
    logger.info("\n🔧 Initializing Graph Manager...")
    try:
        llm = OllamaLLM(
            model="jcai/llama-3-taiwan-8b-instruct:q4_k_m",
            temperature=0,
            base_url="http://localhost:11434"
        )
        graph_manager = GraphManager(llm=llm)
        logger.info("✅ Graph Manager initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize: {e}")
        return False
    
    # 2. 檢查 Paper 總數
    logger.info("\n📊 Checking database statistics...")
    try:
        paper_count_query = "MATCH (p:Paper) RETURN count(p) as count"
        result = graph_manager.graph.query(paper_count_query)
        paper_count = result[0]['count'] if result else 0
        logger.info(f"✅ Total papers in Neo4j: {paper_count}")
        
        if paper_count == 0:
            logger.warning("⚠️  No papers found! Please run agent2.py first to index papers.")
            return False
            
    except Exception as e:
        logger.error(f"❌ Query failed: {e}")
        return False
    
    # 3. 檢查是否有 Chunk nodes（chunk-level extraction 的證據）
    logger.info("\n🔍 Checking for Chunk nodes...")
    try:
        chunk_query = """
        MATCH (c:Chunk)
        RETURN count(c) as chunk_count
        """
        result = graph_manager.graph.query(chunk_query)
        chunk_count = result[0]['chunk_count'] if result else 0
        
        if chunk_count > 0:
            logger.info(f"✅ Found {chunk_count} Chunk nodes")
        else:
            logger.warning("⚠️  No Chunk nodes found!")
            logger.warning("   This means chunk-level extraction hasn't run yet.")
            logger.warning("   Please upload a new PDF or re-index an existing one.")
            return False
            
    except Exception as e:
        logger.error(f"❌ Query failed: {e}")
        return False
    
    # 4. 檢查 MENTIONED_IN relationships
    logger.info("\n🔗 Checking MENTIONED_IN relationships...")
    try:
        mentioned_query = """
        MATCH (e:Entity)-[r:MENTIONED_IN]->(c:Chunk)
        RETURN count(r) as count
        """
        result = graph_manager.graph.query(mentioned_query)
        mentioned_count = result[0]['count'] if result else 0
        
        if mentioned_count > 0:
            logger.info(f"✅ Found {mentioned_count} MENTIONED_IN relationships")
        else:
            logger.warning("⚠️  No MENTIONED_IN relationships found!")
            return False
            
    except Exception as e:
        logger.error(f"❌ Query failed: {e}")
        return False
    
    # 5. 隨機選一個 Paper 進行詳細檢查
    logger.info("\n📄 Checking random paper details...")
    try:
        # 獲取一個有 chunks 的 paper
        paper_query = """
        MATCH (p:Paper)-[:CONTAINS]->(c:Chunk)
        RETURN p.paper_id as paper_id, p.title as title, count(c) as chunk_count
        ORDER BY chunk_count DESC
        LIMIT 1
        """
        result = graph_manager.graph.query(paper_query)
        
        if not result:
            logger.warning("⚠️  No papers with chunks found")
            return False
        
        paper = result[0]
        paper_id = paper['paper_id']
        title = paper['title']
        chunk_count = paper['chunk_count']
        
        logger.info(f"📝 Selected paper: {title}")
        logger.info(f"   Paper ID: {paper_id}")
        logger.info(f"   Chunks: {chunk_count}")
        
        # 檢查該 paper 的 entities
        entity_query = """
        MATCH (p:Paper {paper_id: $paper_id})-[:CONTAINS]->(c:Chunk)<-[:MENTIONED_IN]-(e:Entity)
        RETURN e.name as entity_name, e.type as entity_type, 
               e.source_chunks as source_chunks, count(c) as mention_count
        ORDER BY mention_count DESC
        LIMIT 10
        """
        entities = graph_manager.graph.query(entity_query, {"paper_id": paper_id})
        
        if entities:
            logger.info(f"\n✅ Top 10 entities extracted from chunks:")
            for i, entity in enumerate(entities, 1):
                name = entity['entity_name']
                etype = entity['entity_type']
                mentions = entity['mention_count']
                source_chunks = entity.get('source_chunks', [])
                
                logger.info(f"   {i}. {name} ({etype})")
                logger.info(f"      └─ Mentioned in {mentions} chunks")
                if source_chunks:
                    logger.info(f"      └─ Source chunks: {len(source_chunks)} total")
                    logger.info(f"         First 3: {source_chunks[:3]}")
        else:
            logger.warning("⚠️  No entities found for this paper")
            return False
        
        # 測試 Local Search
        logger.info("\n🔍 Testing Local Search (Entity → Chunks)...")
        test_entity = entities[0]['entity_name']
        logger.info(f"   Query entity: {test_entity}")
        
        local_search_query = """
        MATCH (e:Entity {name: $entity_name})-[:MENTIONED_IN]->(c:Chunk)
        RETURN c.chunk_id as chunk_id, c.paper_id as paper_id
        LIMIT 5
        """
        chunks = graph_manager.graph.query(local_search_query, {"entity_name": test_entity})
        
        if chunks:
            logger.info(f"✅ Found {len(chunks)} chunks mentioning '{test_entity}':")
            for chunk in chunks:
                logger.info(f"      - {chunk['chunk_id']}")
        else:
            logger.warning(f"⚠️  No chunks found for '{test_entity}'")
        
    except Exception as e:
        logger.error(f"❌ Detail check failed: {e}")
        return False
    
    # 6. 統計摘要
    logger.info("\n" + "=" * 70)
    logger.info("📊 Summary Statistics")
    logger.info("=" * 70)
    
    try:
        # Papers with chunks
        papers_with_chunks_query = """
        MATCH (p:Paper)-[:CONTAINS]->(c:Chunk)
        RETURN count(DISTINCT p) as count
        """
        result = graph_manager.graph.query(papers_with_chunks_query)
        papers_with_chunks = result[0]['count'] if result else 0
        
        # Total entities with source_chunks
        entities_query = """
        MATCH (e:Entity)
        WHERE e.source_chunks IS NOT NULL
        RETURN count(e) as count
        """
        result = graph_manager.graph.query(entities_query)
        entities_with_sources = result[0]['count'] if result else 0
        
        logger.info(f"✅ Papers with chunks: {papers_with_chunks}/{paper_count}")
        logger.info(f"✅ Total chunks: {chunk_count}")
        logger.info(f"✅ Entities with source_chunks: {entities_with_sources}")
        logger.info(f"✅ MENTIONED_IN relationships: {mentioned_count}")
        
        # Calculate coverage
        if paper_count > 0:
            coverage = (papers_with_chunks / paper_count) * 100
            logger.info(f"✅ Chunk extraction coverage: {coverage:.1f}%")
            
            if coverage < 100:
                logger.info(f"\nℹ️  Note: {paper_count - papers_with_chunks} papers haven't been re-indexed yet")
                logger.info("   To update them, restart agent2.py or upload a new PDF")
        
    except Exception as e:
        logger.error(f"❌ Statistics failed: {e}")
        return False
    
    # 7. 結論
    logger.info("\n" + "=" * 70)
    logger.info("🎉 Chunk-Level Entity Extraction is Working!")
    logger.info("=" * 70)
    logger.info("✅ Neo4j schema verified:")
    logger.info("   - Paper → CONTAINS → Chunk")
    logger.info("   - Entity → MENTIONED_IN → Chunk")
    logger.info("   - Entities have source_chunks tracking")
    logger.info("✅ Local Search capability ready")
    logger.info("=" * 70)
    
    return True


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
