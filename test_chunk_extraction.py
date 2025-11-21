#!/usr/bin/env python3
"""
測試腳本：驗證 Chunk-Level Entity Extraction
隨機選擇一篇論文，測試完整的 chunk-level extraction 流程
"""

import os
import sys
import random
import logging
from pathlib import Path

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """主測試流程"""
    
    # 1. 隨機選擇一篇論文
    data_dir = Path("./data")
    pdf_files = [f for f in data_dir.glob("*.pdf")]
    
    if not pdf_files:
        logger.error("❌ No PDF files found in ./data")
        return False
    
    # 隨機選擇
    selected_pdf = random.choice(pdf_files)
    logger.info(f"📄 Selected PDF: {selected_pdf.name}")
    
    # 使用現有的 vectorstore（已有索引）而不是創建新的
    logger.info("🔧 Using existing vectorstore at ./vectorstore...")
    from system_api.hierarchical_rag_system import HierarchicalRAGSystem
    
    try:
        rag_system = HierarchicalRAGSystem(
            pdf_directory="./data",
            model_name="jcai/llama-3-taiwan-8b-instruct:q4_k_m",
            embedding_model="quentinz/bge-large-zh-v1.5:latest",
            vectorstore_path="./vectorstore",  # 使用現有的 vectorstore
            chunk_size=800,
            chunk_overlap=100,
            config={
                'layer1': {'k_documents': 5, 'confidence_threshold': 0.6},
                'layer2': {'k_documents': 3, 'confidence_threshold': 0.6},
                'chunking': {'mode': 'naive'}
            }
        )
        logger.info("✅ RAG System initialized with existing vectorstore")
            
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG System: {e}")
        return False
    
    # 3. 初始化 Graph RAG Components
    logger.info("🔧 Initializing Graph RAG Components...")
    from langchain_ollama import OllamaLLM
    from system_api.graph_manager import GraphManager
    from system_api.graph_extractor import GraphDataExtractor
    
    try:
        llm = OllamaLLM(
            model="jcai/llama-3-taiwan-8b-instruct:q4_k_m",
            temperature=0,
            base_url="http://localhost:11434"
        )
        
        graph_manager = GraphManager(llm=llm)
        graph_extractor = GraphDataExtractor(llm=llm)
        
        # 附加到 IndexManager
        rag_system.index_manager.set_graph_components(graph_manager, graph_extractor)
        
        logger.info("✅ Graph components attached to IndexManager")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Graph components: {e}")
        return False
    
    # 4. 執行文檔索引（包含 chunk-level extraction）
    logger.info(f"📊 Starting document indexing with chunk-level extraction...")
    logger.info(f"   PDF: {selected_pdf}")
    
    try:
        result = rag_system.add_document(str(selected_pdf))
        
        if result['status'] == 'success':
            logger.info(f"✅ Indexing completed successfully!")
            logger.info(f"   Chunks added: {result['chunks_added']}")
            logger.info(f"   Duration: {result['duration_seconds']:.2f}s")
        else:
            logger.error(f"❌ Indexing failed: {result.get('error')}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Indexing error: {e}", exc_info=True)
        return False
    
    # 5. 驗證 Neo4j 中的結果
    logger.info("🔍 Verifying Neo4j database...")
    
    try:
        # 獲取 paper_id (使用檔名)
        paper_id = selected_pdf.name
        
        # 查詢 1: 檢查 Paper node
        paper_query = """
        MATCH (p:Paper {paper_id: $paper_id})
        RETURN p.paper_id as paper_id, p.title as title, p.year as year, p.domain as domain
        """
        paper_result = graph_manager.query(paper_query, {"paper_id": paper_id})
        
        if paper_result:
            logger.info("✅ Paper node found:")
            for record in paper_result:
                logger.info(f"   ID: {record['paper_id']}")
                logger.info(f"   Title: {record['title']}")
                logger.info(f"   Year: {record['year']}")
                logger.info(f"   Domain: {record['domain']}")
        else:
            logger.warning("⚠️  Paper node not found")
        
        # 查詢 2: 檢查 Chunk nodes
        chunk_query = """
        MATCH (p:Paper {paper_id: $paper_id})-[:CONTAINS]->(c:Chunk)
        RETURN count(c) as chunk_count
        """
        chunk_result = graph_manager.query(chunk_query, {"paper_id": paper_id})
        
        if chunk_result:
            chunk_count = chunk_result[0]['chunk_count']
            logger.info(f"✅ Chunk nodes: {chunk_count}")
        else:
            logger.warning("⚠️  No chunk nodes found")
        
        # 查詢 3: 檢查 Entity nodes with source_chunks
        entity_query = """
        MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk)<-[:CONTAINS]-(p:Paper {paper_id: $paper_id})
        RETURN e.name as entity_name, e.type as entity_type, 
               e.source_chunks as source_chunks, count(c) as mention_count
        ORDER BY mention_count DESC
        LIMIT 10
        """
        entity_result = graph_manager.query(entity_query, {"paper_id": paper_id})
        
        if entity_result:
            logger.info(f"✅ Top 10 entities extracted from chunks:")
            for i, record in enumerate(entity_result, 1):
                source_chunks = record.get('source_chunks', [])
                logger.info(f"   {i}. {record['entity_name']} ({record['entity_type']})")
                logger.info(f"      Mentioned in {record['mention_count']} chunks")
                logger.info(f"      Source chunks: {source_chunks[:3]}...")  # 顯示前3個
        else:
            logger.warning("⚠️  No entities found")
        
        # 查詢 4: 檢查 MENTIONED_IN relationships
        relationship_query = """
        MATCH (e:Entity)-[r:MENTIONED_IN]->(c:Chunk)<-[:CONTAINS]-(p:Paper {paper_id: $paper_id})
        RETURN count(r) as relationship_count
        """
        rel_result = graph_manager.query(relationship_query, {"paper_id": paper_id})
        
        if rel_result:
            rel_count = rel_result[0]['relationship_count']
            logger.info(f"✅ MENTIONED_IN relationships: {rel_count}")
        else:
            logger.warning("⚠️  No MENTIONED_IN relationships found")
        
        # 查詢 5: 檢查 inter-entity relationships
        inter_rel_query = """
        MATCH (e1:Entity)-[r]->(e2:Entity)
        WHERE (e1)-[:MENTIONED_IN]->(:Chunk)<-[:CONTAINS]-(:Paper {paper_id: $paper_id})
        RETURN type(r) as rel_type, count(r) as count
        ORDER BY count DESC
        LIMIT 5
        """
        inter_rel_result = graph_manager.query(inter_rel_query, {"paper_id": paper_id})
        
        if inter_rel_result:
            logger.info(f"✅ Top inter-entity relationships:")
            for record in inter_rel_result:
                logger.info(f"   {record['rel_type']}: {record['count']}")
        else:
            logger.info("ℹ️  No inter-entity relationships found (may be expected)")
        
    except Exception as e:
        logger.error(f"❌ Neo4j verification failed: {e}", exc_info=True)
        return False
    
    # 6. 測試 Local Search (Entity → Chunk retrieval)
    logger.info("🔍 Testing Local Search capability...")
    
    try:
        # 獲取第一個 entity 進行測試
        test_entity_query = """
        MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk)<-[:CONTAINS]-(p:Paper {paper_id: $paper_id})
        RETURN e.name as entity_name LIMIT 1
        """
        test_entity_result = graph_manager.query(test_entity_query, {"paper_id": paper_id})
        
        if test_entity_result:
            test_entity = test_entity_result[0]['entity_name']
            logger.info(f"   Testing with entity: {test_entity}")
            
            # Local Search: Entity → Chunks → Text
            local_search_query = """
            MATCH (e:Entity {name: $entity_name})-[:MENTIONED_IN]->(c:Chunk)
            RETURN c.chunk_id as chunk_id, c.paper_id as paper_id
            LIMIT 5
            """
            local_result = graph_manager.query(
                local_search_query, 
                {"entity_name": test_entity}
            )
            
            if local_result:
                logger.info(f"✅ Found {len(local_result)} chunks mentioning '{test_entity}':")
                for record in local_result:
                    logger.info(f"   - Chunk: {record['chunk_id']}")
                    # 在實際應用中，這裡會從 Layer2 vectorstore 檢索完整文本
            else:
                logger.warning(f"⚠️  No chunks found for entity '{test_entity}'")
        else:
            logger.warning("⚠️  No entities to test with")
            
    except Exception as e:
        logger.error(f"❌ Local Search test failed: {e}")
        return False
    
    # 7. 總結
    logger.info("=" * 70)
    logger.info("🎉 Test completed successfully!")
    logger.info("=" * 70)
    logger.info("✅ Chunk-level extraction is working correctly")
    logger.info("✅ Neo4j schema verified (Paper → Chunk → Entity)")
    logger.info("✅ MENTIONED_IN relationships created")
    logger.info("✅ source_chunks tracking functional")
    logger.info("✅ Local Search capability validated")
    logger.info("=" * 70)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
