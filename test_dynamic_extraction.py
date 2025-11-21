#!/usr/bin/env python3
"""
測試動態實體提取（查詢時提取）
"""

import logging
from langchain_ollama import OllamaLLM
from system_api.hierarchical_rag_system import HierarchicalRAGSystem
from system_api.graph_manager import GraphManager
from system_api.graph_extractor import GraphDataExtractor

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 70)
    logger.info("🧪 Testing Dynamic Entity Extraction (Query-Time)")
    logger.info("=" * 70)
    
    # 1. 初始化 RAG System
    logger.info("\n🔧 Initializing RAG System...")
    
    rag_system = HierarchicalRAGSystem(
        pdf_directory="./data",
        model_name="jcai/llama-3-taiwan-8b-instruct:q4_k_m",
        embedding_model="quentinz/bge-large-zh-v1.5:latest",
        vectorstore_path="./vectorstore",
        chunk_size=800,
        chunk_overlap=100,
        config={
            'layer1': {'k_documents': 5, 'confidence_threshold': 0.6},
            'layer2': {'k_documents': 3, 'confidence_threshold': 0.6}
            # chunking mode is always 'summarization' now
        }
    )
    
    # 2. 初始化 Graph Components
    logger.info("🔧 Initializing Graph Components...")
    
    llm = OllamaLLM(
        model="jcai/llama-3-taiwan-8b-instruct:q4_k_m",
        temperature=0,
        base_url="http://localhost:11434"
    )
    
    graph_manager = GraphManager(llm=llm)
    graph_extractor = GraphDataExtractor(llm=llm)
    
    # 附加到 index_manager（會自動初始化 dynamic_extractor）
    rag_system.index_manager.set_graph_components(graph_manager, graph_extractor)
    
    logger.info("✅ System initialized")
    
    # 2.5 檢查索引狀態，如果沒有就隨機選一篇建立
    if not rag_system.is_ready():
        logger.info("📄 No existing indices, will build index with smallest paper for testing...")
        
        # 選擇一篇論文
        from pathlib import Path
        import random
        
        data_dir = Path("./data")
        pdf_files = [f for f in data_dir.glob("*.pdf") if f.name != '.DS_Store']
        
        if not pdf_files:
            logger.error("❌ No PDF files found in ./data")
            return
        
        # 選擇最小的論文（快速測試）
        pdf_files_sorted = sorted(pdf_files, key=lambda f: f.stat().st_size)
        selected_pdf = pdf_files_sorted[0]
        
        logger.info(f"   Selected: {selected_pdf.name} ({selected_pdf.stat().st_size / 1024:.1f} KB)")
        logger.info(f"   Building index (this will take 1-2 minutes)...")
        
        try:
            # Use build_indices instead of add_document for initial index creation
            result = rag_system.build_indices([str(selected_pdf)])
            if result.get('status') == 'success':
                logger.info(f"✅ Index built successfully:")
                logger.info(f"   Papers: {result.get('papers_processed', 0)}")
                logger.info(f"   Chunks: {result.get('chunks_created', 0)}")
                logger.info(f"   Duration: {result.get('duration', 0):.1f}s")
            else:
                logger.error(f"❌ Index build failed: {result.get('message', 'Unknown error')}")
                return
        except Exception as e:
            logger.error(f"❌ Index build error: {e}", exc_info=True)
            return
    else:
        stats = rag_system.get_stats()
        logger.info(f"✅ Using existing indices:")
        logger.info(f"   Layer 1: {stats.get('layer1', {}).get('paper_count', 0)} papers")
        logger.info(f"   Layer 2: {stats.get('layer2', {}).get('chunk_count', 0)} chunks")
    
    # 3. 執行測試查詢（使用會觸發Layer 2的具體問題）
    test_queries = [
        "LSTM模型的具體架構和參數設定是什麼？",  # 具體實作細節
        "論文中使用了哪些數據集進行實驗？",  # 具體細節
        "mRBF和LSTM如何結合使用？請詳細說明實作方法"  # 實作細節
    ]
    
    for i, query in enumerate(test_queries, 1):
        logger.info("\n" + "=" * 70)
        logger.info(f"📝 Test Query {i}: {query}")
        logger.info("=" * 70)
        
        try:
            # 執行查詢（動態提取會在內部自動觸發）
            answer = rag_system.query(query)
            
            logger.info(f"\n✅ Query completed")
            logger.info(f"Answer preview: {answer[:200]}...")
            
        except Exception as e:
            logger.error(f"❌ Query failed: {e}", exc_info=True)
    
    # 4. 檢查 Neo4j 中的結果
    logger.info("\n" + "=" * 70)
    logger.info("🔍 Verifying Neo4j Results")
    logger.info("=" * 70)
    
    try:
        # 統計 Entity nodes
        entity_query = "MATCH (e:Entity) RETURN count(e) as count"
        result = graph_manager.graph.query(entity_query)
        entity_count = result[0]['count'] if result else 0
        
        logger.info(f"✅ Total entities in Neo4j: {entity_count}")
        
        # 統計 MENTIONED_IN relationships
        rel_query = "MATCH (e:Entity)-[r:MENTIONED_IN]->(c:Chunk) RETURN count(r) as count"
        result = graph_manager.graph.query(rel_query)
        rel_count = result[0]['count'] if result else 0
        
        logger.info(f"✅ MENTIONED_IN relationships: {rel_count}")
        
        # 顯示幾個實體例子
        sample_query = """
        MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk)
        RETURN e.name as name, e.type as type, count(c) as mentions
        ORDER BY mentions DESC
        LIMIT 5
        """
        samples = graph_manager.graph.query(sample_query)
        
        if samples:
            logger.info("\n📊 Top 5 entities by mentions:")
            for sample in samples:
                logger.info(f"  - {sample['name']} ({sample['type']}): {sample['mentions']} mentions")
        
    except Exception as e:
        logger.error(f"❌ Verification failed: {e}")
    
    logger.info("\n" + "=" * 70)
    logger.info("🎉 Test Complete!")
    logger.info("=" * 70)
    logger.info("💡 Dynamic extraction only processes retrieved chunks")
    logger.info("💡 Results are cached to avoid re-extraction")
    logger.info("💡 Entities are stored in Neo4j for future queries")
    logger.info("=" * 70)


if __name__ == "__main__":
    import sys
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n⚠️  Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}", exc_info=True)
        sys.exit(1)
