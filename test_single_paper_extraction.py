#!/usr/bin/env python3
"""
測試腳本：對單一論文執行 Chunk-Level Entity Extraction
選擇一篇現有論文，只對它執行 chunk-level extraction
"""

import logging
import random
from pathlib import Path
from tqdm import tqdm
from langchain_ollama import OllamaLLM
from system_api.graph_manager import GraphManager
from system_api.graph_extractor import GraphDataExtractor
from system_api.hierarchical_rag_system import HierarchicalRAGSystem

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """主測試流程"""
    
    logger.info("=" * 70)
    logger.info("🧪 Single Paper Chunk-Level Extraction Test")
    logger.info("=" * 70)
    
    # 1. 選擇一篇論文
    data_dir = Path("./data")
    pdf_files = [f for f in data_dir.glob("*.pdf") if f.name != '.DS_Store']
    
    if not pdf_files:
        logger.error("❌ No PDF files found in ./data")
        return False
    
    # 選擇一篇小的論文加快測試
    selected_pdf = random.choice(pdf_files)
    logger.info(f"\n📄 Selected PDF: {selected_pdf.name}")
    logger.info(f"   Size: {selected_pdf.stat().st_size / 1024:.1f} KB")
    
    # 2. 初始化組件
    logger.info("\n🔧 Initializing components...")
    
    try:
        # LLM
        llm = OllamaLLM(
            model="jcai/llama-3-taiwan-8b-instruct:q4_k_m",
            temperature=0,
            base_url="http://localhost:11434"
        )
        
        # Graph components
        graph_manager = GraphManager(llm=llm)
        graph_extractor = GraphDataExtractor(llm=llm)
        
        # RAG System (使用現有 vectorstore)
        rag_system = HierarchicalRAGSystem(
            pdf_directory="./data",
            model_name="jcai/llama-3-taiwan-8b-instruct:q4_k_m",
            embedding_model="quentinz/bge-large-zh-v1.5:latest",
            vectorstore_path="./vectorstore",
            chunk_size=800,
            chunk_overlap=100,
            config={
                'layer1': {'k_documents': 5, 'confidence_threshold': 0.6},
                'layer2': {'k_documents': 3, 'confidence_threshold': 0.6},
                'chunking': {'mode': 'naive'}
            }
        )
        
        # 附加 graph components
        rag_system.index_manager.set_graph_components(graph_manager, graph_extractor)
        
        logger.info("✅ Components initialized")
        
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        return False
    
    # 3. 執行索引（包含 chunk-level extraction）
    logger.info(f"\n📊 Starting indexing with chunk-level extraction...")
    logger.info(f"   This will take a few minutes due to LLM extraction...")
    
    try:
        result = rag_system.add_document(str(selected_pdf))
        
        if result['status'] == 'success':
            logger.info(f"\n✅ Indexing completed!")
            logger.info(f"   Chunks added: {result['chunks_added']}")
            logger.info(f"   Duration: {result['duration_seconds']:.1f}s")
        else:
            logger.error(f"❌ Indexing failed: {result.get('error')}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Indexing error: {e}", exc_info=True)
        return False
    
    # 4. 驗證結果
    logger.info("\n🔍 Verifying extraction results...")
    
    paper_id = selected_pdf.name
    
    try:
        # 檢查 Chunk nodes
        chunk_query = """
        MATCH (p:Paper {paper_id: $paper_id})-[:CONTAINS]->(c:Chunk)
        RETURN count(c) as chunk_count
        """
        result = graph_manager.graph.query(chunk_query, {"paper_id": paper_id})
        chunk_count = result[0]['chunk_count'] if result else 0
        
        logger.info(f"✅ Chunk nodes created: {chunk_count}")
        
        if chunk_count == 0:
            logger.warning("⚠️  No chunks created!")
            return False
        
        # 檢查 Entity nodes
        entity_query = """
        MATCH (p:Paper {paper_id: $paper_id})-[:CONTAINS]->(c:Chunk)<-[:MENTIONED_IN]-(e:Entity)
        RETURN e.name as entity_name, e.type as entity_type, 
               e.source_chunks as source_chunks, count(c) as mention_count
        ORDER BY mention_count DESC
        LIMIT 10
        """
        entities = graph_manager.graph.query(entity_query, {"paper_id": paper_id})
        
        if entities:
            logger.info(f"\n✅ Extracted {len(entities)} entities (showing top 10):")
            for i, entity in enumerate(entities, 1):
                name = entity['entity_name']
                etype = entity['entity_type']
                mentions = entity['mention_count']
                source_chunks = entity.get('source_chunks', [])
                
                logger.info(f"   {i}. {name} ({etype})")
                logger.info(f"      └─ {mentions} mentions, {len(source_chunks)} source chunks")
        else:
            logger.warning("⚠️  No entities extracted!")
            return False
        
        # 檢查 MENTIONED_IN relationships
        rel_query = """
        MATCH (e:Entity)-[r:MENTIONED_IN]->(c:Chunk)<-[:CONTAINS]-(p:Paper {paper_id: $paper_id})
        RETURN count(r) as count
        """
        result = graph_manager.graph.query(rel_query, {"paper_id": paper_id})
        rel_count = result[0]['count'] if result else 0
        
        logger.info(f"\n✅ MENTIONED_IN relationships: {rel_count}")
        
    except Exception as e:
        logger.error(f"❌ Verification failed: {e}")
        return False
    
    # 5. 成功
    logger.info("\n" + "=" * 70)
    logger.info("🎉 Test Completed Successfully!")
    logger.info("=" * 70)
    logger.info(f"✅ Paper: {selected_pdf.name}")
    logger.info(f"✅ Chunks: {chunk_count}")
    logger.info(f"✅ Entities: {len(entities)}")
    logger.info(f"✅ Relationships: {rel_count}")
    logger.info("\n💡 Now run test_graph_quick.py to see overall statistics")
    logger.info("=" * 70)
    
    return True


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
