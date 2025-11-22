from flask import Flask, request, jsonify, render_template, Response
from langchain_ollama import OllamaLLM
from langchain_core.tools import Tool
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_classic.prompts import PromptTemplate
import numpy as np
import logging
import json
import os
import tempfile
import threading
from datetime import datetime
from typing import List, Dict, Optional, Any
from werkzeug.utils import secure_filename
from system_api.search_engine import SearchEngine
from system_api.rag_support import RAGChatbot
from system_api.web_researcher import WebSearcher
from system_api.node_support import ResearchInheritanceAnalyzer
from system_api.pdf_validator import PDFValidator
from system_api.paper_extractor import PaperExtractor
from system_api.database_updater import DatabaseUpdater
from system_api.pdf_storage import PDFStorage
from system_api.hierarchical_rag_system import HierarchicalRAGSystem
from system_api.graph_manager import GraphManager
from system_api.graph_extractor import GraphDataExtractor
from collections import Counter
model_name = "jcai/llama-3-taiwan-8b-instruct:q4_k_m"

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize Ollama LLM for agent (Moved up for Graph RAG dependency)
agent_llm = None
try:
    agent_llm = OllamaLLM(
        model=model_name, # agent model
        temperature=0,
        base_url="http://localhost:11434"
    )
    # Test the connection
    test_response = agent_llm.invoke("Hello")
    logger.info("Agent Ollama connection successful")
except Exception as e:
    logger.error(f"Failed to connect to Ollama for agent: {e}")
    logger.warning("Agent will use fallback routing")

# ============================================================================
# Hierarchical RAG System Configuration
# ============================================================================
logger.info("Initializing Hierarchical RAG System...")
rag_system = None

try:
    rag_system = HierarchicalRAGSystem(
        pdf_directory="./test_data",
        model_name=model_name,
        embedding_model="quentinz/bge-large-zh-v1.5:latest",
        vectorstore_path="./vectorstore",   
        chunk_size=800,
        chunk_overlap=100,
        config={
            'layer1': {
                'k_documents': int(os.getenv('LAYER1_K_DOCUMENTS', '10')),
                'confidence_threshold': float(os.getenv('LAYER1_THRESHOLD', '0.6')),
                'similarity_threshold': 0.49,  # 降低閾值（純語義策略下需要更寬鬆）
            },
            'layer2': {
                'k_documents': int(os.getenv('LAYER2_K_DOCUMENTS', '3')),  # 限制為 3 個 chunks
                'confidence_threshold': float(os.getenv('LAYER2_THRESHOLD', '0.6'))
            },
            'expansion': {
                'enabled': os.getenv('ENABLE_EXPANSION', 'true').lower() == 'true'
            },
            'hybrid_search': {
                'enabled': True,  # 啟用混合檢索
                'semantic_weight': 0.9,  # 語義搜尋權重（主要依靠語義理解）
                'keyword_weight': 0.1,   # 關鍵詞搜尋權重（輔助參考）
                'use_jieba': True,       # 使用 jieba 中文分詞
                'enable_smart_weighting': True,  # 啟用字典式智能權重（fallback）
                'use_llm_analyzer': True,  # ✨ 啟用 LLM 自動分析詞彙（純語義策略）
            },
            'query_expansion': {
                'enabled': False,  # 停用查詢擴展（LLM 分析器已足夠）
                'min_word_count': 15,
            },
            'adaptive_weights': {
                'enabled': False,  # 停用自適應權重（LLM 分析器已包含）
                'default_semantic_weight': 0.9,
                'default_keyword_weight': 0.1,
            },
            'query_routing': {
                'enabled': True,  # ✨ 啟用智能查詢路由（自動判斷 Graph 或 RAG）
                'default_route': 'rag',  # 預設路由（當無法判斷時）
                'confidence_threshold': 0.7,  # 路由決策的最低信心度
            },
            'performance': {
                'cache_size': int(os.getenv('CACHE_SIZE', '10'))
            },
            'layer2_reranking': {
                'enabled': True,  # 永久啟用 Cross-Encoder Re-ranking（捨棄 FAISS 模式）
                'model': 'qllama/bce-reranker-base_v1:latest',
                'ollama_base_url': 'http://localhost:11434',
                'max_length': 512,
                'timeout': 30,
                'max_candidates': 300,
                'cache_limit_mb': 100,
            },
            'chunking': {
                # 分塊策略：'naive' (傳統固定大小) 或 'summarization' (段落摘要)
                'mode': os.getenv('CHUNKING_MODE', 'summarization'),  # 預設: naive (向後兼容)
                'summarization': {
                    'model': os.getenv('SUMMARIZATION_MODEL', 'llama3.2:latest'),  # 摘要使用的 LLM 
                    'section_parser': 'pymupdf_regex',  # PDF 段落解析方法
                    'min_sections': int(os.getenv('MIN_SECTIONS', '3')),  # 最少段落數
                    'map_reduce_threshold': int(os.getenv('MAP_REDUCE_THRESHOLD', '3000')),  # Map-Reduce 閾值
                    'target_summary_length': int(os.getenv('TARGET_SUMMARY_LENGTH', '300')),  # 目標摘要長度
                    'ollama_base_url': 'http://localhost:11434',
                    'store_original': False  # 是否存儲原始文本
                }
            }
        }
    )
    
    # 顯示分塊模式資訊
    chunking_mode = os.getenv('CHUNKING_MODE', 'naive')
    logger.info("✓ Hierarchical RAG System initialized successfully")
    logger.info(f"  Chunking Mode: {chunking_mode.upper()}")
    if chunking_mode == 'summarization':
        logger.info(f"  Summarization Model: {os.getenv('SUMMARIZATION_MODEL', 'llama3:8b')}")
        logger.info(f"  Target Summary Length: {os.getenv('TARGET_SUMMARY_LENGTH', '300')} chars")
        
except Exception as e:
    logger.error(f"Failed to initialize Hierarchical RAG System: {e}", exc_info=True)
    rag_system = None

# Initialize Graph RAG Components (Moved up to attach before index check)
logger.info("Initializing Graph RAG components...")
graph_manager = None
graph_extractor = None

try:
    # Reuse the existing 'agent_llm'
    if agent_llm:
        graph_manager = GraphManager(llm=agent_llm)
        graph_extractor = GraphDataExtractor(llm=agent_llm)
        logger.info("✓ Graph RAG components initialized successfully")
        # Attach graph components to IndexManager if available so incremental adds also index to Graph
        try:
            if rag_system and hasattr(rag_system, 'index_manager') and graph_manager:
                rag_system.index_manager.set_graph_components(graph_manager, graph_extractor)
                logger.info("✓ Graph components attached to rag_system.index_manager")
                
                # Initialize GraphRetriever and GraphIntegrator for rag_system
                from system_api.graph_retriever import GraphRetriever
                from system_api.graph_integrator import GraphIntegrator
                
                rag_system.graph_retriever = GraphRetriever(graph_manager)
                rag_system.graph_integrator = GraphIntegrator(rag_system.llm)
                logger.info("✓ GraphRetriever and GraphIntegrator initialized for rag_system")
                
        except Exception as e:
            logger.warning(f"Could not attach graph components to IndexManager: {e}")
    else:
        logger.warning("Agent LLM not available, skipping Graph RAG initialization")
except Exception as e:
    logger.error(f"✗ Failed to initialize Graph RAG components: {e}")
    logger.warning("Graph capabilities will be disabled.")
    graph_manager = None
    graph_extractor = None

# Now check and update indices (after Graph components are attached)
if rag_system:
    try:
        # 智能檢查並更新索引
        logger.info("Checking index status...")
        check_result = rag_system.check_and_update_indices()
        
        if check_result['status'] == 'success':
            if check_result['action_taken']:
                logger.info(f"✓ Action taken: {check_result['action_taken']}")
                logger.info(f"✓ Processed {check_result['details'].get('papers_processed', 0)} papers")
            else:
                logger.info("✓ Indices are up-to-date")
        elif check_result['status'] == 'up_to_date':
            logger.info("✓ Indices are healthy and up-to-date")
        else:
            logger.error(f"✗ Index check/update failed: {check_result.get('details', {}).get('error', 'Unknown error')}")
    except Exception as e:
        logger.error(f"Error during index check: {e}", exc_info=True)

# Add this after initializing other components
logger.info("Loading data categories...")
categories_data = None
try:
    with open('./json_files/data_categories.json', 'r', encoding='utf-8') as f:
        categories_data = json.load(f)
    logger.info(f"Loaded {len(categories_data)} paper categories")
except Exception as e:
    logger.error(f"Failed to load data categories: {e}")
    categories_data = []

# Initialize ResearchInheritanceAnalyzer with separate LLM
logger.info("Initializing ResearchInheritanceAnalyzer...")
inheritance_analyzer = None
inheritance_llm = None
try:
    inheritance_llm = OllamaLLM(
        model=model_name,
        temperature=0.7,
        base_url="http://localhost:11434"
    )
    # Test the connection
    test_response = inheritance_llm.invoke("Hello")
    logger.info("Inheritance Ollama connection successful")
    
    inheritance_analyzer = ResearchInheritanceAnalyzer(
        json_path="./json_files/relationships.json",
        model=model_name,
        verbose=False
    )
    logger.info("ResearchInheritanceAnalyzer initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize ResearchInheritanceAnalyzer or its LLM: {e}")
    inheritance_analyzer = None

print('Initializing WebSearcher...')
web_searcher = WebSearcher(agent_llm)

# Global sync for RAG streaming
_rag_completion_event = None
_rag_result_buffer = None

def assistant_call(user_input: str) -> str:
    """
    AI assistant for paper summary, interpretation, translation, and retrieval
    
    In streaming mode, this waits for RAG to complete before returning.
    The actual RAG retrieval is handled by query_stream endpoint.
    """
    global _rag_completion_event, _rag_result_buffer
    
    logger.info(f"Assistant Call invoked with: '{user_input}'")
    
    # Create event if it doesn't exist (streaming mode detection)
    if _rag_completion_event is None:
        logger.info("Creating RAG completion event for streaming mode")
        _rag_completion_event = threading.Event()
        _rag_result_buffer = ""
    
    # Wait for RAG to complete
    logger.info("⏳ Waiting for RAG streaming to complete...")
    success = _rag_completion_event.wait(timeout=300)  # Wait up to 5 minutes (LLM can be slow)
    
    if success:
        logger.info("✓ RAG streaming completed, returning status message")
        # Return a short status message instead of full content
        # The actual content was already streamed to the user
        result = "[RAG retrieval completed - answer has been streamed to user]"
    else:
        logger.warning("⚠️ RAG streaming timeout after 300s")
        result = "處理超時，請重試。"
    
    # Clear for next call
    _rag_completion_event = None
    _rag_result_buffer = None
    
    return result


def web_search_call(user_input: str) -> str:
    """AI web scraper"""
    
    logger.info(f"Web Search Call invoked with: '{user_input}'")
    result = web_searcher.search(user_input)

    return result


def graph_analysis_call(query: str) -> str:
    """
    Use this tool for MACRO-level questions: relationships, trends, aggregations across papers.
    Examples: "Which papers use LSTM?", "Compare research goals", "List all datasets".
    
    This tool implements YOUR 3-STAGE GRAPHRAG FLOW:
    1. Graph query to find relevant papers/techniques
    2. Integrate known information and answer
    3. Determine if more details are needed → descend to Layer2 with filtering
    """
    if not graph_manager:
        return "Graph RAG is not available."
    
    logger.info(f"🔍 [Graph Tool] Querying: '{query}'")
    
    try:
        # ========================================================================
        # STAGE 1: Graph Query - Find Related Papers/Techniques
        # ========================================================================
        logger.info("📊 Stage 1: Executing Graph Query...")
        graph_answer = graph_manager.query_graph(query)
        
        logger.info(f"  ✓ Graph query completed")
        logger.info(f"  Answer preview: {graph_answer[:150]}...")
        
        # ========================================================================
        # STAGE 2: Integrate Known Information
        # ========================================================================
        logger.info("🔗 Stage 2: Integrating Graph Results...")
        
        # Extract paper information from Neo4j based on original query
        # This extracts paper_id and relevant node types for filtering
        paper_extraction_result = _extract_papers_from_query(query)
        
        if not paper_extraction_result['success']:
            logger.info("  ✓ Graph answer complete (no papers found)")
            return f"[Graph Analysis Result]\n{graph_answer}\n\n[✓ Answer from Knowledge Graph]"
        
        paper_ids = paper_extraction_result['paper_ids']
        detected_nodes = paper_extraction_result['node_types']  # e.g., ['Method', 'Dataset']
        
        logger.info(f"  ✓ Found {len(paper_ids)} relevant papers")
        logger.info(f"  ✓ Detected node types: {detected_nodes}")
        
        # ========================================================================
        # STAGE 3: Determine if Details Needed → Descend to Layer2
        # ========================================================================
        logger.info("🎯 Stage 3: Checking if details are needed...")
        
        needs_details = _should_descend_to_layer2(query, graph_answer)
        
        if not needs_details:
            # Query only asks macro-level info (e.g., "哪些論文使用YOLO")
            logger.info("  ✓ Macro-only query - Graph answer sufficient")
            return f"ANSWER READY:\n{graph_answer}\n\n[This is the complete answer. Return it as your Final Answer now.]"
        
        # User wants details (e.g., "如何使用", "個別介紹")
        logger.info("  → Details requested - descending to Layer2")
        
        # Determine which section_names/chunk_types to filter
        filter_sections = _determine_section_filters(query, detected_nodes)
        logger.info(f"  → Filtering Layer2 by sections: {filter_sections}")
        
        # Retrieve chunks from Layer2 with filtering
        layer2_docs = []
        for paper_id in paper_ids[:5]:  # Limit to top 5 papers to avoid overload
            paper_results = rag_system.layer2.search_with_scores(
                query=query,
                k=2,  # Top 2 chunks per paper (reduced to save time with multiple papers)
                filter_paper_ids=[paper_id],
                filter_chunk_types=filter_sections  # ✨ Use section_name filtering
            )
            layer2_docs.extend([doc for doc, score in paper_results])
        
        logger.info(f"  ✓ Retrieved {len(layer2_docs)} chunks from Layer2")
        
        # Generate detailed answer with Graph + Layer2 content
        if layer2_docs:
            # Step 1: 清理並格式化Graph答案
            cleaned_graph_answer = _clean_graph_answer(graph_answer)
            
            # Step 2: 生成結構化的Layer2詳細答案
            detailed_answer = _generate_structured_answer(
                query=query,
                paper_ids=paper_ids,
                layer2_docs=layer2_docs,
                rag_system=rag_system
            )
            
            # Step 3: 組合最終答案（結構化輸出）
            combined_answer = _format_final_answer(
                graph_summary=cleaned_graph_answer,
                detailed_content=detailed_answer,
                paper_count=len(paper_ids)
            )
            
            # ✨ 返回完整答案，使用 ANSWER READY 格式（LLM 更熟悉）
            return f"""ANSWER READY:
{combined_answer}

[This is the complete answer with all details. Return it as your Final Answer NOW.]"""
        else:
            logger.warning("  ⚠️ No chunks found in Layer2 - returning Graph answer only")
            cleaned_answer = _clean_graph_answer(graph_answer)
            return f"""ANSWER READY:
{cleaned_answer}

[This is the complete answer. Return it as your Final Answer NOW.]"""
        
    except Exception as e:
        logger.error(f"Graph tool error: {e}", exc_info=True)
        return f"Error querying graph: {str(e)}"


def _extract_papers_from_query(query: str) -> dict:
    """
    Extract paper_ids and node types from the original query by executing a Graph query.
    
    Returns:
        {
            'success': bool,
            'paper_ids': List[str],
            'node_types': List[str]  # e.g., ['Method', 'Dataset', 'Domain']
        }
    """
    try:
        # Detect node types from query keywords
        node_types = []
        query_lower = query.lower()
        
        if any(kw in query_lower for kw in ['方法', 'method', '演算法', 'algorithm', 'model', '模型', 'yolo', 'lstm', 'cnn']):
            node_types.append('Method')
        if any(kw in query_lower for kw in ['資料集', 'dataset', 'data', '數據']):
            node_types.append('Dataset')
        if any(kw in query_lower for kw in ['領域', 'domain', '應用', 'application', '交通', 'transportation']):
            node_types.append('Domain')
        if any(kw in query_lower for kw in ['指標', 'metric', 'evaluation', '評估']):
            node_types.append('Metric')
        
        # Build Cypher query based on detected node types
        if 'Method' in node_types:
            cypher = """
            MATCH (p:Paper)-[:USES_METHOD]->(m:Method)
            WHERE toLower(m.name) CONTAINS $keyword OR toLower(m.name_zh) CONTAINS $keyword
            RETURN DISTINCT p.paper_id as paper_id, p.title as title, 'Method' as node_type
            LIMIT 10
            """
            # Extract keyword from query (simple approach)
            for kw in ['yolo', 'lstm', 'cnn', 'transformer', 'bert']:
                if kw in query_lower:
                    results = graph_manager.graph.query(cypher, {'keyword': kw})
                    if results:
                        return {
                            'success': True,
                            'paper_ids': [r['paper_id'] for r in results],
                            'node_types': node_types
                        }
        
        if 'Domain' in node_types:
            cypher = """
            MATCH (p:Paper)-[:APPLIED_IN]->(d:Domain)
            WHERE toLower(d.name) CONTAINS $keyword OR toLower(d.name_zh) CONTAINS $keyword
            RETURN DISTINCT p.paper_id as paper_id, p.title as title, 'Domain' as node_type
            LIMIT 10
            """
            for kw in ['智慧交通', 'smart transportation', '醫療', 'medical']:
                if kw in query_lower or kw in query:
                    results = graph_manager.graph.query(cypher, {'keyword': kw})
                    if results:
                        return {
                            'success': True,
                            'paper_ids': [r['paper_id'] for r in results],
                            'node_types': node_types
                        }
        
        # Fallback: use generic query
        cypher = """
        MATCH (p:Paper)
        RETURN p.paper_id as paper_id, p.title as title
        LIMIT 10
        """
        results = graph_manager.graph.query(cypher)
        
        if results:
            return {
                'success': True,
                'paper_ids': [r['paper_id'] for r in results],
                'node_types': node_types or ['Paper']
            }
        
        return {'success': False, 'paper_ids': [], 'node_types': []}
        
    except Exception as e:
        logger.error(f"Failed to extract papers from query: {e}")
        return {'success': False, 'paper_ids': [], 'node_types': []}


def _should_descend_to_layer2(query: str, graph_answer: str) -> bool:
    """
    Determine if the query requires detailed information from Layer2.
    
    Returns True if query contains keywords requesting details.
    """
    query_lower = query.lower()
    
    # ========================================================================
    # Category 1: 明確要求詳細內容
    # ========================================================================
    explicit_detail_keywords = [
        # 中文
        '如何', '怎麼', '怎样', '怎么',
        '詳細', '详细', '具體', '具体', '細節', '细节',
        '介紹', '介绍', '說明', '说明', '描述', '解釋', '解释',
        '個別', '个别', '分別', '分别', '各自',
        
        # English  
        'how', 'detail', 'specific', 'describe', 'explain', 'elaborate',
        'summarize', 'summary', 'each', 'individual', 'individually'
    ]
    
    # ========================================================================
    # Category 2: 實作細節查詢
    # ========================================================================
    implementation_keywords = [
        # 中文
        '訓練', '训练', '實驗', '实验', '測試', '测试',
        '參數', '参数', '超參數', '超参数', '配置', '設定', '设定',
        '步驟', '步骤', '流程', '過程', '过程', '架構', '架构',
        '實現', '实现', '實作', '实作',
        
        # English
        'training', 'experiment', 'testing', 'evaluation',
        'parameter', 'hyperparameter', 'configuration', 'setting',
        'step', 'process', 'procedure', 'architecture',
        'implementation', 'approach', 'technique'
    ]
    
    # ========================================================================
    # Category 3: 具體內容/數據查詢（重要！）
    # ========================================================================
    concrete_content_keywords = [
        # 中文 - 資料集相關
        '什麼資料集', '什么资料集', '哪個資料集', '哪个资料集',
        '資料集', '资料集', '數據集', '数据集', '資料', '数据',
        
        # 中文 - 評估指標
        '準確率', '准确率', '精確度', '精确度', '召回率', 'F1',
        '指標', '指标', '評估', '评估', '效能', '性能',
        
        # 中文 - 模型/方法
        '什麼方法', '什么方法', '什麼模型', '什么模型',
        '採用', '采用', '使用什麼', '使用什么',
        
        # English - Dataset
        'what dataset', 'which dataset', 'dataset used',
        
        # English - Metrics
        'accuracy', 'precision', 'recall', 'metric', 'evaluation',
        'performance', 'score', 'result',
        
        # English - Model/Method  
        'what method', 'which method', 'what model', 'which model',
        'use what', 'adopt', 'employ'
    ]
    
    # ========================================================================
    # Category 4: 比較/對比查詢
    # ========================================================================
    comparison_keywords = [
        # 中文
        '比較', '比较', '對比', '对比', '差異', '差异',
        '不同', '區別', '区别', '優缺點', '优缺点',
        '優勢', '优势', '劣勢', '劣势', '特點', '特点',
        
        # English
        'compare', 'comparison', 'contrast', 'difference',
        'versus', 'vs', 'distinguish', 'advantage', 'disadvantage'
    ]
    
    # ========================================================================
    # Category 5: 深入理解查詢
    # ========================================================================
    deep_understanding_keywords = [
        # 中文
        '為什麼', '为什么', '原理', '機制', '机制',
        '貢獻', '贡献', '創新', '创新', '突破',
        '意義', '意义', '價值', '价值',
        
        # English
        'why', 'reason', 'principle', 'mechanism',
        'contribution', 'innovation', 'novelty',
        'significance', 'value'
    ]
    
    # ========================================================================
    # Category 6: 問題解決查詢
    # ========================================================================
    problem_solving_keywords = [
        # 中文
        '解決', '解决', '改進', '改进', '克服',
        '問題', '问题', '挑戰', '挑战', '困難', '困难',
        
        # English
        'solve', 'address', 'tackle', 'overcome',
        'problem', 'challenge', 'issue', 'difficulty'
    ]
    
    # 合併所有需要細節的關鍵詞
    detail_keywords = (
        explicit_detail_keywords + 
        implementation_keywords + 
        concrete_content_keywords +
        comparison_keywords +
        deep_understanding_keywords +
        problem_solving_keywords
    )
    
    # ========================================================================
    # 判斷邏輯
    # ========================================================================
    
    # 1. 檢查是否有明確的細節請求
    has_detail_request = any(kw in query_lower for kw in detail_keywords)
    
    # 2. 檢查是否為Macro-only關鍵詞
    macro_only_keywords = [
        '哪些', '哪个', '哪幾', '有什麼', '有哪些', '有幾',
        'which', 'what', 'list', 'list all'
    ]
    has_macro_keyword = any(kw in query_lower for kw in macro_only_keywords)
    
    # 3. 特殊規則：如果同時有macro關鍵詞和detail關鍵詞，優先detail
    # 例如："哪些論文使用YOLO，並且他們如何使用" → 需要detail
    if has_detail_request:
        logger.info(f"  ✓ Details requested (keyword match): {query[:50]}...")
        return True
    
    # 4. 如果只有macro關鍵詞，判定為macro-only
    if has_macro_keyword:
        logger.info(f"  ✓ Macro-only detected (listing query): {query[:50]}...")
        return False
    
    # 5. 智能判斷：檢查query結構
    # 如果query包含"並且"、"而且"、"and"等連接詞，通常表示複雜查詢，需要細節
    conjunction_keywords = ['並且', '并且', '而且', '以及', '還有', '还有', 'and', 'also', 'plus']
    has_conjunction = any(kw in query_lower for kw in conjunction_keywords)
    
    if has_conjunction and len(query) > 20:
        logger.info(f"  ✓ Complex query with conjunction: {query[:50]}...")
        return True
    
    # 6. 智能判斷：問句類型
    # "為什麼"、"怎麼"、"如何" 開頭的問題通常需要詳細解釋
    question_starters = ['為什麼', '为什么', '怎麼', '怎么', '如何', 'why', 'how']
    starts_with_question = any(query_lower.startswith(kw) for kw in question_starters)
    
    if starts_with_question:
        logger.info(f"  ✓ Question requiring explanation: {query[:50]}...")
        return True
    
    # 7. 預設：短query且只提到論文 → macro-only
    if len(query) < 15:
        logger.info(f"  ✓ Short macro query: {query[:50]}...")
        return False
    
    # 8. 最終預設：長query通常需要更多細節
    logger.info(f"  ✓ Default: long query likely needs details: {query[:50]}...")
    return True
    

def _determine_section_filters(query: str, detected_nodes: List[str]) -> List[str]:
    """
    Determine which section_names to filter based on query intent and detected node types.
    
    Returns:
        List of section names (e.g., ['Method', 'Experimental Setup', 'Results'])
    """
    query_lower = query.lower()
    sections = []
    
    # Map query keywords to section names
    # Note: section_name in chunks come from PDF parsing (e.g., "Method", "Results", "Dataset")
    
    if any(kw in query_lower for kw in ['方法', 'method', '如何使用', '演算法', 'algorithm']):
        sections.extend(['Method', 'Methodology', 'method', 'methodology'])
    
    if any(kw in query_lower for kw in ['資料集', 'dataset', 'data', '數據']):
        sections.extend(['Dataset', 'Data', 'Experimental Setup', 'dataset', 'data'])
    
    if any(kw in query_lower for kw in ['結果', 'result', '效果', 'performance', '表現']):
        sections.extend(['Results', 'Evaluation', 'Performance', 'results', 'evaluation'])
    
    if any(kw in query_lower for kw in ['實驗', 'experiment', '訓練', 'training']):
        sections.extend(['Experimental Setup', 'Experiments', 'Training', 'experiments'])
    
    if any(kw in query_lower for kw in ['介紹', '說明', 'introduction', 'background', 'overview']):
        sections.extend(['Introduction', 'Background', 'Abstract', 'introduction'])
    
    # If node types detected, add corresponding sections
    if 'Method' in detected_nodes:
        sections.extend(['Method', 'Methodology', 'method'])
    if 'Dataset' in detected_nodes:
        sections.extend(['Dataset', 'Data', 'dataset'])
    
    # Remove duplicates and return
    sections = list(set(sections))
    
    # If no specific sections, return core sections (avoid empty filter)
    if not sections:
        sections = ['Method', 'Results', 'method', 'results']
    
    return sections


# ============================================================================
# Answer Formatting and Post-processing Functions
# ============================================================================

def _clean_graph_answer(graph_answer: str) -> str:
    """
    清理Graph答案，移除不必要的信息和格式問題
    
    Args:
        graph_answer: 原始的Graph答案
        
    Returns:
        清理後的答案
    """
    import re
    
    # 移除 "(Year unknown)" 等無用信息
    cleaned = re.sub(r'\(Year unknown\)', '', graph_answer)
    cleaned = re.sub(r'\(year unknown\)', '', cleaned)
    cleaned = re.sub(r'\(Unknown\)', '', cleaned)
    
    # 移除多餘的空格和空行
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)
    
    # 修正標點符號後的空格
    cleaned = re.sub(r'\s+([,。，、])', r'\1', cleaned)
    
    return cleaned.strip()


def _generate_structured_answer(
    query: str,
    paper_ids: List[str],
    layer2_docs: List[Any],
    rag_system: Any
) -> str:
    """
    生成結構化的詳細答案，避免直接暴露Context標記
    
    Args:
        query: 用戶查詢
        paper_ids: 相關論文ID列表
        layer2_docs: Layer2檢索的文檔
        rag_system: RAG系統實例
        
    Returns:
        結構化的詳細答案
    """
    # 按paper_id分組chunks
    papers_chunks = {}
    for doc in layer2_docs:
        paper_id = doc.metadata.get('paper_id', 'unknown')
        title = doc.metadata.get('title', paper_id)
        
        if paper_id not in papers_chunks:
            papers_chunks[paper_id] = {
                'title': title,
                'chunks': []
            }
        papers_chunks[paper_id]['chunks'].append(doc.page_content)
    
    # 檢測語言
    def is_chinese(text: str) -> bool:
        chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        return chinese_chars > len(text) * 0.3
    
    is_chinese_query = is_chinese(query)
    
    # 為每篇論文生成答案
    paper_answers = []
    for paper_id, data in papers_chunks.items():
        title = data['title']
        chunks_text = "\n\n".join(data['chunks'])
        
        # 構建prompt（不暴露Context標記）
        if is_chinese_query:
            prompt = f"""請根據以下論文內容回答問題。

問題：{query}

論文標題：{title}

論文內容：
{chunks_text}

【重要指引】
1. 請直接回答問題，不要提及"Context"或"背景資料"等字眼
2. 使用繁體中文回答
3. 如果論文中有具體數據（資料集、準確率、參數等），請明確列出
4. 回答要結構化，使用列點或段落清楚呈現
5. 如果內容中沒有相關資訊，請說明「論文中未詳細說明此部分」

請針對「{title}」這篇論文回答："""
        else:
            prompt = f"""Based on the following paper content, answer the question.

Question: {query}

Paper Title: {title}

Paper Content:
{chunks_text}

【Important Guidelines】
1. Answer directly without mentioning "Context" or "background materials"
2. If there are specific data (datasets, accuracy, parameters, etc.), list them clearly
3. Structure your answer with bullet points or clear paragraphs
4. If information is not available, state "This is not detailed in the paper"

Please answer for the paper "{title}":"""
        
        try:
            # 設定max_tokens避免截斷
            answer = rag_system.llm.invoke(prompt)
            paper_answers.append({
                'title': title,
                'answer': answer.strip()
            })
        except Exception as e:
            logger.error(f"Failed to generate answer for {title}: {e}")
            paper_answers.append({
                'title': title,
                'answer': f"生成答案時發生錯誤：{str(e)}"
            })
    
    # 組合所有論文的答案（使用新的結構化格式）
    if is_chinese_query:
        # 1. 先列出論文清單
        paper_list = "論文列表：\n"
        for i, pa in enumerate(paper_answers, 1):
            paper_list += f"\t{i}\t《{pa['title']}》\n"
        
        # 2. 再逐篇提供詳細內容
        detailed_answers = ""
        for pa in paper_answers:
            # 提取論文簡稱（取標題的前15個字或第一個下劃線前的部分）
            title_short = pa['title'].split('_')[0] if '_' in pa['title'] else pa['title'][:15]
            detailed_answers += f"\n【來源：{title_short}】\n{pa['answer']}\n"
        
        structured_answer = paper_list + detailed_answers
    else:
        # English format
        paper_list = "Paper List:\n"
        for i, pa in enumerate(paper_answers, 1):
            paper_list += f"\t{i}\t《{pa['title']}》\n"
        
        detailed_answers = ""
        for pa in paper_answers:
            title_short = pa['title'].split('_')[0] if '_' in pa['title'] else pa['title'][:15]
            detailed_answers += f"\n【Source: {title_short}】\n{pa['answer']}\n"
        
        structured_answer = paper_list + detailed_answers
    
    return structured_answer.strip()


def _format_final_answer(
    graph_summary: str,
    detailed_content: str,
    paper_count: int
) -> str:
    """
    格式化最終答案，提供清晰的結構
    
    Args:
        graph_summary: Graph查詢的摘要（不使用，因為詳細內容已包含論文列表）
        detailed_content: 詳細內容（已包含論文列表和逐篇說明）
        paper_count: 論文數量
        
    Returns:
        格式化的最終答案
    """
    # 直接返回詳細內容（已經是結構化格式：論文列表 + 逐篇說明）
    return detailed_content.strip()


# Create tools for the agent
tools = [
    Tool(
        name="AssistantCall",
        func=assistant_call,
        description="Use for specific details, summaries, or content retrieval from SINGLE or specific papers."
    ),
    
    Tool(
        name="GraphAnalysisCall",
        func=graph_analysis_call,
        description="Use for MACRO-level questions: finding papers by method/dataset, counting, trends, or cross-paper relationships. Keywords: 'which papers', 'list all', 'compare', 'how many', methods, datasets, domains, metrics."
    ),

    Tool(
        name="WebSearchCall",
        func=web_search_call,
        description="Use ONLY when the user EXPLICITLY requests web/internet search with keywords like 'search online', 'search the web', 'find on internet', 'search internet', or 'web search'."
    )
]

# Create agent only if LLM is available
agent_executor = None
if agent_llm:
    try:
        template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

CRITICAL FORMAT RULES:
1. Each "Action:" MUST appear EXACTLY ONCE per turn on a NEW LINE
2. "Action Input:" MUST appear EXACTLY ONCE immediately after "Action:" on the NEXT LINE
3. Do NOT write multiple "Action:" lines or explanatory text between them
4. Do NOT write "Action 1:", "Action 2:", or numbered actions

Use the following format STRICTLY:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be exactly one of [{tool_names}]
Action Input: the COMPLETE original question (DO NOT simplify, translate, or shorten)
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

CRITICAL: When providing Action Input:
1. Copy the ENTIRE original question EXACTLY as asked
2. Do NOT simplify or shorten the query
3. Do NOT split complex queries into multiple actions
4. PRESERVE all connecting words: "並且", "而且", "以及", "and", "also"
5. PRESERVE all detail requests: "如何使用", "個別介紹", "how they use", "explain each"

WRONG Examples:
❌ Question: "哪些論文使用YOLO，並且他們如何使用" → Action Input: "哪些論文使用YOLO"  (WRONG - lost detail request)
❌ Question: "哪些論文和智慧交通有關，並且個別介紹" → Action Input: "哪些論文和智慧交通有關"  (WRONG - lost "個別介紹")

CORRECT Examples:
✅ Question: "哪些論文使用YOLO，並且他們如何使用" → Action Input: "哪些論文使用YOLO，並且他們如何使用"  (CORRECT)
✅ Question: "哪些論文和智慧交通有關，並且個別介紹" → Action Input: "哪些論文和智慧交通有關，並且個別介紹"  (CORRECT)

CRITICAL: When you see "ANSWER READY:" in an Observation:
1. IMMEDIATELY write "Thought: I now know the final answer"
2. IMMEDIATELY write "Final Answer:" and copy EXACTLY EVERYTHING after "ANSWER READY:" (including ALL 📊摘要, 📝詳細說明, and all論文 details)
3. Do NOT summarize, shorten, or modify ANY content
4. Do NOT add your own commentary
5. Include ALL sections, formatting, and separators
6. Do NOT call any more actions

EXAMPLE:
Observation: 
ANSWER READY:
📊 **摘要**
Content here with multiple papers...
────────
📝 **詳細說明** (共 5 篇論文)
### Paper 1: ...
### Paper 2: ...
[All content]

[This is the complete answer. Return it as your Final Answer NOW.]

Thought: I now know the final answer
Final Answer: 
ANSWER READY:
📊 **摘要**
Content here with multiple papers...
────────
📝 **詳細說明** (共 5 篇論文)
### Paper 1: ...
### Paper 2: ...
[All content]

IMPORTANT ROUTING RULES:

1. **GraphAnalysisCall** (Cross-Paper Query) - CHECK THIS FIRST:
   - USE WHEN: Query contains "Which papers", "哪些論文", "List all", "How many", "Compare".
   - PRIORITY RULE: If query contains "Which papers" or "哪些論文", ALWAYS use GraphAnalysisCall.
   - CRITICAL: GraphAnalysisCall handles BOTH finding papers AND getting details in one call.
   - KEYWORDS: "Which papers...", "哪些論文", "List all...", "列出所有", "How many...", "多少", "Compare...", "比較", "papers that use X", "使用X的論文".
   - Examples: 
     * "Which papers use LSTM?" → GraphAnalysisCall (will return list only)
     * "哪些論文用到YOLO，並且他們如何使用？" → GraphAnalysisCall (will find papers AND get details automatically)
     * "哪些論文和智慧交通有關，並且個別介紹他們在做什麼" → GraphAnalysisCall (will find papers AND provide introductions automatically)
     * "List all datasets" → GraphAnalysisCall
     * "Compare research goals" → GraphAnalysisCall
   - NOTE: GraphAnalysisCall is SMART - it automatically descends to Layer2 when the query requests details (e.g., "如何使用", "個別介紹", "並且").

2. **AssistantCall** (Single Paper Detail Query):
   - USE WHEN: User asks about content/details of ONE SPECIFIC paper (NOT multiple papers).
   - KEYWORDS: "this paper", "the paper about X", specific paper title, "what does paper X say", "explain X in paper".
   - Examples: 
     * "Summarize the paper about traffic flow prediction" → AssistantCall
     * "What methods does the LSTM paper use?" → AssistantCall
     * "這篇關於交通流量預測的論文用什麼方法？" → AssistantCall
   - NOTE: AssistantCall will identify the specific paper, check if abstract is sufficient, then retrieve detailed chunks if needed.

3. **WebSearchCall**:
   - USE ONLY for explicit internet search requests.
   - KEYWORDS: "search internet", "search online", "web search", "find on internet"

Here are some examples (FOLLOW THESE FORMATS EXACTLY):

Example 1 - Single Paper Query (Use AssistantCall):
Question: Summarize the paper about traffic flow prediction
Thought: This asks for content from a SPECIFIC paper. This is a single-paper detail query.
Action: AssistantCall
Action Input: Summarize the paper about traffic flow prediction
Observation: [Paper identified: "Paper123_Traffic_Flow". Abstract provides overview. Retrieving detailed chunks...]
Thought: I now know the final answer
Final Answer: The paper on traffic flow prediction uses LSTM networks to predict traffic patterns. It employs the PeMSD7 dataset and achieves 92% accuracy...

Example 2 - Single Paper Detail Query (Use AssistantCall):
Question: 芒果分類那篇論文用什麼方法？
Thought: This asks for methods in a SPECIFIC paper about mango classification. Single-paper query.
Action: AssistantCall
Action Input: 芒果分類那篇論文用什麼方法？
Observation: [Paper: "基礎5_應用集成式深度學習模型進行芒果分類辨識". Methods: Mask R-CNN, CNN...]
Thought: I now know the final answer
Final Answer: 芒果分類論文使用 Mask R-CNN 和卷積神經網路進行影像分類，訓練參數為 batch_size=32...

Example 3 - Single Paper Dataset Query (Use AssistantCall):
Question: What dataset does the LSTM paper use?
Thought: This asks about dataset in a SPECIFIC paper (LSTM paper). Single-paper query.
Action: AssistantCall
Action Input: What dataset does the LSTM paper use?
Observation: [Paper identified. Dataset chunks retrieved: PeMSD7, training size 10000 samples...]
Thought: I now know the final answer
Final Answer: The LSTM paper uses the PeMSD7 dataset with 10,000 training samples...

Example 4 - Cross-Paper Query (Use GraphAnalysisCall):
Question: Which papers use LSTM?
Thought: This asks "which papers" use LSTM. This is a cross-paper query.
Action: GraphAnalysisCall
Action Input: Which papers use LSTM?
Observation: ANSWER READY:
Papers using LSTM: 
1) Paper1 (Traffic Flow Prediction)
2) Paper2 (Time Series Analysis)
...

[This is the complete answer. Return it as your Final Answer NOW.]
Thought: I now know the final answer
Final Answer: ANSWER READY:
Papers using LSTM: 
1) Paper1 (Traffic Flow Prediction)
2) Paper2 (Time Series Analysis)
...

Example 5 - Cross-Paper Query with Detail Request (Use GraphAnalysisCall):
Question: 哪些論文用到YOLO，並且他們如何使用？
Thought: This asks "which papers" and requests details ("如何使用"). Use GraphAnalysisCall with COMPLETE query.
Action: GraphAnalysisCall
Action Input: 哪些論文用到YOLO，並且他們如何使用？
Observation: ANSWER READY:
📊 **摘要**
使用 YOLO 的論文包括：論文1、論文2...

────────────────────────────────────────────────────────────────────────────────
📝 **詳細說明** (共 2 篇論文)

### Paper 1: 論文1
此論文使用 YOLO 進行物體偵測，採用 YOLOv3 架構...

### Paper 2: 論文2
此論文將 YOLO 應用於醫學影像分析，使用 YOLOv4...

[This is the complete answer with all details. Return it as your Final Answer NOW.]
Thought: I now know the final answer
Final Answer: ANSWER READY:
📊 **摘要**
使用 YOLO 的論文包括：論文1、論文2...

────────────────────────────────────────────────────────────────────────────────
📝 **詳細說明** (共 2 篇論文)

### Paper 1: 論文1
此論文使用 YOLO 進行物體偵測，採用 YOLOv3 架構...

### Paper 2: 論文2
此論文將 YOLO 應用於醫學影像分析，使用 YOLOv4...

Example 6 - Cross-Paper Comparison (Use GraphAnalysisCall):
Question: Compare research goals in Smart Transportation domain
Thought: This asks to compare across multiple papers. Cross-paper comparison query.
Action: GraphAnalysisCall
Action Input: Compare research goals in Smart Transportation domain
Observation: [Knowledge Graph comparison results...]
Thought: I now know the final answer
Final Answer: Research goals in Smart Transportation include: Traffic prediction (3 papers), Route optimization (2 papers)...

Example 7 - Count Query (Use GraphAnalysisCall):
Question: How many papers use the PeMSD7 dataset?
Thought: This asks for a count across papers. Cross-paper aggregation.
Action: GraphAnalysisCall
Action Input: How many papers use the PeMSD7 dataset?
Observation: [Knowledge Graph count: 5 papers]
Thought: I now know the final answer
Final Answer: 5 papers use the PeMSD7 dataset.

Example 8 - Web Search (Use WebSearchCall):
Question: Search the internet for latest LSTM papers
Thought: User explicitly said "search the internet".
Action: WebSearchCall
Action Input: latest LSTM papers
Observation: [web search results]
Thought: I now know the final answer
Final Answer: Here are the latest papers from internet search...

If the query looks like previous agent output or contains terms like Thought: or Action:, treat it as text to summarize and use AssistantCall.

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

        prompt = PromptTemplate.from_template(template)
        
        agent = create_react_agent(agent_llm, tools, prompt)
        
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=20,  # Increased to allow complex queries with multiple papers
            max_execution_time=300,  # Increased to 5 minutes for queries with many papers (each paper needs LLM call)
            return_intermediate_steps=True
            # Note: early_stopping_method removed as 'generate' is not supported
        )
        logger.info("Agent created successfully")
    except Exception as e:
        logger.error(f"Failed to create agent: {e}")
        agent_executor = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    status_info = {
        'agent_ready': agent_executor is not None,
        'inheritance_analyzer_ready': inheritance_analyzer is not None,
        'inheritance_llm_ready': inheritance_llm is not None,
        'rag_system_ready': rag_system is not None
    }
    
    if all(status_info.values()):
        return jsonify({
            'status': 'ok', 
            'message': 'All systems ready',
            'details': status_info
        })
    else:
        message = []
        if not status_info['agent_ready']:
            message.append('Agent not initialized')
        if not status_info['inheritance_analyzer_ready']:
            message.append('ResearchInheritanceAnalyzer not initialized')
        if not status_info['inheritance_llm_ready']:
            message.append('Inheritance LLM not initialized')
        if not status_info['rag_system_ready']:
            message.append('RAG System not initialized')
        
        return jsonify({
            'status': 'warning', 
            'message': ', '.join(message),
            'details': status_info
        }), 200

@app.route('/graph/data', methods=['GET'])
def get_graph_data():
    """
    Endpoint for frontend graph visualization
    Returns nodes and edges from Neo4j graph database
    """
    if not graph_manager:
        return jsonify({'error': 'Graph system not active'}), 503
    
    try:
        limit = request.args.get('limit', default=100, type=int)
        data = graph_manager.get_visualization_data(limit=limit)
        return jsonify(data), 200
    except Exception as e:
        logger.error(f"Graph visualization error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/query_stream', methods=['POST'])
def query_stream():
    """Streaming endpoint for real-time Agent + LLM responses with reasoning process"""
    import queue
    import threading
    from system_api.agent_callbacks import AgentStreamCallback
    
    try:
        user_input = request.json.get('input', '')
        
        if not user_input:
            return jsonify({'error': 'No input provided'}), 400
        
        logger.info(f"Processing streaming query with agent: {user_input}")
        
        # Check for agent format in input (avoid loops)
        user_input_lower = user_input.lower()
        if all(keyword in user_input_lower for keyword in ['question:', 'thought:', 'action:']):
            logger.warning("Detected agent format, using direct RAG")
            use_agent = False
        elif not agent_executor:
            logger.warning("Agent not available, using direct RAG")
            use_agent = False
        else:
            use_agent = True
        
        def generate():
            try:
                if not use_agent:
                    # Direct RAG streaming (no agent)
                    yield f"data: {json.dumps({'type': 'start', 'action': 'DirectRAG'})}\n\n"
                    
                    if rag_system:
                        for chunk in rag_system.query_stream(user_input):
                            if chunk:
                                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
                    else:
                        yield f"data: {json.dumps({'type': 'error', 'message': 'RAG system not available'})}\n\n"
                    
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                    return
                
                # Use Agent with streaming callbacks
                event_queue = queue.Queue()
                stop_event = threading.Event()  # Signal to stop agent execution
                callback = AgentStreamCallback(event_queue, stop_event)
                
                # Run agent in a separate thread
                def run_agent():
                    try:
                        agent_executor.invoke(
                            {"input": user_input},
                            config={"callbacks": [callback]}
                        )
                        # Signal completion
                        event_queue.put({'type': 'agent_done'})
                    except Exception as e:
                        if stop_event.is_set():
                            logger.info(f"Agent execution stopped by user")
                            event_queue.put({'type': 'stopped'})
                        else:
                            logger.error(f"Agent execution error: {e}")
                            event_queue.put({'type': 'error', 'content': str(e)})
                
                agent_thread = threading.Thread(target=run_agent, daemon=True)
                agent_thread.start()
                
                # Send initial metadata
                yield f"data: {json.dumps({'type': 'start', 'mode': 'agent'})}\n\n"
                
                # Stream events from queue
                final_answer_started = False
                final_answer_buffer = ""
                
                while True:
                    try:
                        # Wait for events with timeout
                        event = event_queue.get(timeout=0.1)
                        
                        if event['type'] == 'agent_done':
                            # Agent finished, send final answer if we have buffer
                            if final_answer_buffer and not final_answer_started:
                                yield f"data: {json.dumps({'type': 'final_answer', 'content': final_answer_buffer})}\n\n"
                            yield f"data: {json.dumps({'type': 'done'})}\n\n"
                            break
                        
                        elif event['type'] == 'thought':
                            yield f"data: {json.dumps(event)}\n\n"
                        
                        elif event['type'] == 'action':
                            yield f"data: {json.dumps(event)}\n\n"
                            
                            # If action is AssistantCall, start streaming the RAG output
                            if event['tool'] == 'AssistantCall' and rag_system:
                                try:
                                    # assistant_call has already created the event and is waiting
                                    # We just need to fill the buffer and signal completion
                                    global _rag_completion_event, _rag_result_buffer
                                    
                                    logger.info(f"🔄 Intercepting AssistantCall - starting RAG streaming for: {event.get('input', user_input)}")
                                    yield f"data: {json.dumps({'type': 'tool_start', 'tool': 'AssistantCall'})}\n\n"
                                    
                                    action_input = event.get('input', user_input)
                                    # Use single paper query mode for AssistantCall
                                    rag_stream = rag_system.query_single_paper_stream(action_input)
                                    
                                    chunk_count = 0
                                    for chunk in rag_stream:
                                        # Check stop signal during RAG streaming
                                        if stop_event.is_set():
                                            logger.info("Stop signal detected during RAG streaming")
                                            # Try to close the generator
                                            try:
                                                rag_stream.close()
                                            except:
                                                pass
                                            break
                                        if chunk:
                                            yield f"data: {json.dumps({'type': 'tool_token', 'tool': 'AssistantCall', 'content': chunk})}\n\n"
                                            final_answer_buffer += chunk
                                            # Check if buffer still exists (not cleared by timeout)
                                            if _rag_result_buffer is not None:
                                                _rag_result_buffer += chunk
                                            chunk_count += 1
                                    
                                    logger.info(f"✓ RAG streaming completed: {chunk_count} chunks streamed")
                                    
                                    # Signal completion to assistant_call (if event still exists)
                                    if _rag_completion_event is not None:
                                        _rag_completion_event.set()
                                    else:
                                        logger.warning("⚠️ RAG completion event was already cleared (timeout)")
                                    
                                    final_answer_started = True
                                except GeneratorExit:
                                    # Client disconnected during RAG streaming
                                    logger.info("Client disconnected during RAG streaming")
                                    stop_event.set()
                                    # Try to close the RAG stream
                                    try:
                                        if 'rag_stream' in locals():
                                            rag_stream.close()
                                    except:
                                        pass
                                    raise
                        
                        elif event['type'] == 'observation':
                            # Skip observation if we already streamed the tool output
                            if not final_answer_started:
                                yield f"data: {json.dumps(event)}\n\n"
                        
                        elif event['type'] == 'final_answer':
                            if not final_answer_started:
                                # Stream final answer character by character if not already streamed
                                content = event.get('content', '')
                                for char in content:
                                    yield f"data: {json.dumps({'type': 'chunk', 'content': char})}\n\n"
                            else:
                                # Already streamed via tool, just signal completion
                                pass
                        
                        elif event['type'] == 'stopped':
                            # Agent was stopped by user
                            yield f"data: {json.dumps({'type': 'stopped'})}\n\n"
                            break
                        
                        elif event['type'] == 'error':
                            yield f"data: {json.dumps(event)}\n\n"
                            yield f"data: {json.dumps({'type': 'done'})}\n\n"
                            break
                        
                        else:
                            # Forward other event types
                            yield f"data: {json.dumps(event)}\n\n"
                    
                    except queue.Empty:
                        # Check if thread is still alive
                        if not agent_thread.is_alive():
                            # Thread died without sending agent_done
                            logger.warning("Agent thread died unexpectedly")
                            if final_answer_buffer:
                                yield f"data: {json.dumps({'type': 'final_answer', 'content': final_answer_buffer})}\n\n"
                            yield f"data: {json.dumps({'type': 'done'})}\n\n"
                            break
                        continue
                
                # Wait for thread to complete
                agent_thread.join(timeout=5)
                
            except GeneratorExit:
                # Client disconnected, signal the agent to stop
                logger.info("Client disconnected, stopping agent...")
                if 'stop_event' in locals():
                    stop_event.set()
                raise  # Re-raise to properly close the generator
            except Exception as e:
                logger.error(f"Error in streaming: {e}", exc_info=True)
                if 'stop_event' in locals():
                    stop_event.set()
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        
        return Response(generate(), mimetype='text/event-stream')
        
    except Exception as e:
        logger.error(f"Error in query_stream: {e}")
        return jsonify({'error': str(e)}), 500

# Note: Non-streaming /query endpoint has been removed. 
# All queries now use streaming mode via /query_stream for better UX.

@app.route('/inheritance_catalog', methods=['GET'])
def inheritance_catalog():
    try:
        # Reinitialize analyzer to get latest data from relationships.json
        current_analyzer = ResearchInheritanceAnalyzer(
            json_path="./json_files/relationships.json",
            model=model_name,
            verbose=False
        )
        
        catalog = current_analyzer.list_selection_catalog(
            sort="year",
            include_year=True
        )
        return jsonify(catalog)
    except Exception as e:
        logger.error(f"Error retrieving inheritance catalog: {e}")
        return jsonify({'error': str(e)}), 500
@app.route('/categories_data', methods=['GET'])
def categories_data_endpoint():
    """Return all categories data and available filter options"""
    try:
        # Reload categories data from file to get latest updates
        current_categories_data = []
        try:
            with open('./json_files/data_categories.json', 'r', encoding='utf-8') as f:
                current_categories_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to reload categories data: {e}")
            return jsonify({'error': 'Failed to load categories data'}), 500
        
        # Extract unique values for each filter
        purposes = set()
        years = set()
        datasets = set()
        models = set()
        
        for paper in current_categories_data:
            purposes.update(paper.get('研究目的', []))
            years.update(paper.get('年份', []))
            datasets.update(paper.get('資料集', []))
            models.update(paper.get('建模', []))
        
        return jsonify({
            'papers': current_categories_data,
            'options': {
                'purposes': sorted(list(purposes)),
                'years': sorted(list(years), reverse=True),
                'datasets': sorted(list(datasets)),
                'models': sorted(list(models))
            }
        })
    except Exception as e:
        logger.error(f"Error retrieving categories data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/categories_query', methods=['POST'])
def categories_query():
    """Natural language query for categories search"""
    try:
        user_input = request.json.get('input', '')
        
        if not user_input:
            return jsonify({'error': 'No input provided'}), 400
        
        # Reload categories data from file to get latest updates
        current_categories_data = []
        try:
            with open('./json_files/data_categories.json', 'r', encoding='utf-8') as f:
                current_categories_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to reload categories data: {e}")
            return jsonify({'error': 'Failed to load categories data'}), 500
        
        logger.info(f"Processing categories query: {user_input}")
        
        # Simple keyword matching (can be enhanced with LLM)
        user_lower = user_input.lower()
        
        filtered_papers = []
        for paper in current_categories_data:
            # Check if any field contains the search terms
            searchable_text = ' '.join([
                paper.get('論文標題', ''),
                ' '.join(paper.get('研究目的', [])),
                ' '.join(paper.get('年份', [])),
                ' '.join(paper.get('資料集', [])),
                ' '.join(paper.get('建模', [])),
                ' '.join(paper.get('資料前處理', [])),
                ' '.join(paper.get('評估指標', []))
            ]).lower()
            
            if any(term in searchable_text for term in user_lower.split()):
                filtered_papers.append(paper)
        
        return jsonify({
            'papers': filtered_papers,
            'count': len(filtered_papers)
        })
    
    except Exception as e:
        logger.error(f"Error in categories query: {e}")
        return jsonify({'error': str(e)}), 500
    
@app.route('/inheritance_query', methods=['POST'])
def inheritance_query():
    try:
        user_input = request.json.get('input', '')
        
        if not user_input:
            return jsonify({'error': 'No input provided'}), 400
        
        # Reinitialize analyzer to get latest data from relationships.json
        current_analyzer = ResearchInheritanceAnalyzer(
            json_path="./json_files/relationships.json",
            model=model_name,
            verbose=False
        )
        
        logger.info(f"Processing inheritance query: {user_input}")
        
        # Handle list authors request
        if any(keyword in user_input.lower() for keyword in ['list authors', 'available researchers', 'who are the researchers']):
            authors = current_analyzer.list_authors(sort="year", include_year=True)
            if authors:
                return jsonify({
                    'output': "Available researchers:\n" + "\n".join([f"- {author}" for author in authors])
                })
            else:
                return jsonify({'output': "No researchers found in the database."})
        
        # Handle lineage analysis
        result = current_analyzer.analyze_research_lineage(user_input)
        return jsonify({
            'output': result
        })
    
    except Exception as e:
        logger.error(f"Error in inheritance query: {e}")
        return jsonify({'error': str(e)}), 500

def fallback_routing(user_input):
    """
    Bag-of-words based routing as fallback using numpy for cosine similarity
    
    Note: This function is no longer actively used since /query endpoint was removed.
    It's kept for backward compatibility and potential emergency fallback scenarios.
    In normal operation, all requests go through /query_stream with streaming.
    """
    
    def compute_bow_similarity(query, docs):
        def tokenize(text):
            return text.lower().split()
        
        query_tokens = tokenize(query)
        doc_tokens = [tokenize(doc) for doc in docs]
        
        all_tokens = set(query_tokens)
        for dt in doc_tokens:
            all_tokens.update(dt)
        
        vocab = list(all_tokens)
        
        # No IDF, just TF
        query_tf = Counter(query_tokens)
        query_vec = np.array([query_tf.get(token, 0) for token in vocab], dtype=float)
        query_norm = np.linalg.norm(query_vec)
        if query_norm > 0:
            query_vec /= query_norm
        
        doc_vecs = []
        for dt in doc_tokens:
            tf = Counter(dt)
            vec = np.array([tf.get(token, 0) for token in vocab], dtype=float)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            doc_vecs.append(vec)
        
        similarities = [np.dot(query_vec, dv) if np.linalg.norm(dv) > 0 else 0.0 for dv in doc_vecs]
        return similarities
    
    tool_map = {
        'AssistantCall': {
            'desc': "Primary tool for ALL paper-related tasks including searching papers, summarization, translation, interpretation, analysis, finding papers by topic or year, explaining methodologies, and any academic content queries from local database. Use for general paper questions, research queries, and document analysis.",
            'func': assistant_call
        },

        'WebSearchCall': {
            'desc': "Only for explicit web/internet search requests with keywords: search online, search web, find on internet, search internet, web search, online search. Do NOT use for general paper queries or year-based searches unless explicitly requested to search online.",
            'func': web_search_call
        }
    }
    
    user_input_lower = user_input.lower()
    if any(keyword in user_input_lower for keyword in ['action:', 'thought:', 'question:', 'entering new agentexecutor chain', 'paperssearch', 'retrievalcall']):
        logger.warning("Detected potential loop input, routing to AssistantCall")
        action = 'AssistantCall'
        output = assistant_call(user_input)
    else:
        docs = [info['desc'] for info in tool_map.values()]
        sims = compute_bow_similarity(user_input, docs)
        similarities = dict(zip(tool_map.keys(), sims))
        

        action = max(similarities, key=similarities.get)
        output = tool_map[action]['func'](user_input)
    
    return jsonify({
        'action': action + ' (Fallback)',
        'output': output
    })

@app.route('/upload_paper', methods=['POST'])
def upload_paper():
    """
    Handle paper PDF upload, validation, extraction, and database update
    """
    temp_file_path = None
    
    try:
        # Step 1: Receive file upload
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'Invalid file type. Only PDF files are accepted.'}), 400
        
        # Save to temporary location
        temp_dir = tempfile.gettempdir()
        filename = secure_filename(file.filename)
        temp_file_path = os.path.join(temp_dir, f"upload_{os.getpid()}_{filename}")
        file.save(temp_file_path)
        
        logger.info(f"File uploaded: {filename} -> {temp_file_path}")
        
        # Step 2: Validate PDF
        logger.info("Starting PDF validation...")
        validator = PDFValidator(llm=agent_llm)
        success, message, pdf_text = validator.validate(temp_file_path)
        
        if not success:
            # Clean up temp file
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            return jsonify({'error': f'Validation failed: {message}'}), 400
        
        logger.info("PDF validation successful")
        
        # Step 3: Extract paper metadata
        logger.info("Starting paper extraction...")
        extractor = PaperExtractor(model=model_name)
        paper_data = extractor.extract(pdf_text)
        extractor.validate_extraction(paper_data)
        
        logger.info(f"Paper extraction successful: {paper_data.get('論文標題')}")
        
        # Step 4: Classify and update database
        logger.info("Starting database update...")
        updater = DatabaseUpdater(model=model_name)
        success, db_message, classification = updater.update_database(paper_data)
        
        if not success:
            # Clean up temp file
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            return jsonify({'error': f'Database update failed: {db_message}'}), 500
        
        logger.info("Database update successful")
        
        # Step 5: Store PDF file
        logger.info("Storing PDF file...")
        storage = PDFStorage()
        
        # Get paper title and handle both string and list cases
        paper_title = paper_data.get('論文標題', 'untitled')
        if isinstance(paper_title, list):
            paper_title = paper_title[0] if paper_title else 'untitled'
        
        stored_filename = storage.store(temp_file_path, paper_title)
        stored_pdf_path = os.path.join('./data', stored_filename)
        
        # Clean up temp file
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        logger.info(f"PDF stored successfully: {stored_filename}")
        
        # Step 6: Add to RAG system
        message = 'Paper uploaded and processed successfully'
        if rag_system and rag_system.vectorstore:
            logger.info("Adding new PDF to RAG system...")
            result = rag_system.add_document(stored_pdf_path)
            
            if result['status'] == 'success':
                chunks_added = result['chunks_added']
                duration = result['duration_seconds']
                logger.info(f"PDF indexed: {chunks_added} chunks in {duration:.1f}s")
                message = f'Paper uploaded and indexed successfully ({chunks_added} chunks added in {duration:.1f}s)'
            else:
                logger.warning(f"PDF upload succeeded but indexing failed: {result.get('error')}")
                message = 'Paper uploaded successfully, but indexing failed. Please restart service to update search index.'
        
        # Step 7: Synchronous Graph Indexing
        if graph_manager and graph_extractor:
            try:
                logger.info(f"🚀 Starting Graph extraction for: {stored_filename}")
                
                # 1. Extract structured data
                # Use 'pdf_text' which was extracted in Step 2
                graph_data = graph_extractor.extract(pdf_text)
                
                # 2. Write to Neo4j (with bilingual domain support)
                # Use stored_filename as paper_id to match vector store
                success = graph_manager.add_paper_metadata(
                    paper_id=stored_filename,
                    title=paper_data.get('論文標題', 'Untitled'),
                    year=str(paper_data.get('年份', 'Unknown')),
                    research_goal=graph_data.get('research_goal', ''),
                    methods=graph_data.get('methods', []),
                    datasets=graph_data.get('datasets', []),
                    domain=graph_data.get('domain', 'Unknown'),
                    metrics=graph_data.get('metrics', []),
                    domain_zh=graph_data.get('domain_zh', '未知領域')
                )
                
                if success:
                    logger.info(f"✅ Graph ingestion successful for {stored_filename}")
                else:
                    logger.error(f"❌ Graph ingestion failed for {stored_filename}")
                    
            except Exception as e:
                # Log error but don't fail the whole upload if graph part fails
                logger.error(f"❌ Error during graph processing: {e}", exc_info=True)
        
        # Return success with extracted metadata and classification
        return jsonify({
            'success': True,
            'message': message,
            'paper_data': paper_data,
            'classification': classification,
            'stored_filename': stored_filename
        }), 200
    
    except Exception as e:
        logger.error(f"Upload processing failed: {e}", exc_info=True)
        
        # Clean up temp file on error
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass
        
        return jsonify({
            'error': f'An error occurred: {str(e)}'
        }), 500

@app.route('/cancel_upload', methods=['POST'])
def cancel_upload():
    """Cancel an upload and rollback database changes"""
    try:
        data = request.get_json()
        paper_title = data.get('paper_title')
        stored_filename = data.get('stored_filename')
        
        if not paper_title:
            return jsonify({'error': 'Paper title is required'}), 400
        
        logger.info(f"Canceling upload for paper: {paper_title}")
        
        # Step 1: Remove from database
        updater = DatabaseUpdater(model=model_name)
        success, message = updater.remove_paper(paper_title)
        
        if not success:
            logger.warning(f"Failed to remove paper from database: {message}")
        
        # Step 2: Delete PDF file if it exists
        if stored_filename:
            pdf_path = os.path.join('./data', stored_filename)
            if os.path.exists(pdf_path):
                try:
                    os.remove(pdf_path)
                    logger.info(f"Deleted PDF file: {pdf_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete PDF file: {e}")
        
        return jsonify({
            'success': True,
            'message': 'Upload canceled and data removed successfully'
        }), 200
    
    except Exception as e:
        logger.error(f"Cancel upload failed: {e}", exc_info=True)
        return jsonify({
            'error': f'Cancel failed: {str(e)}'
        }), 500

@app.route('/update_paper', methods=['POST'])
def update_paper():
    """Update paper metadata after user edits"""
    try:
        data = request.get_json()
        paper_data = data.get('paper_data')
        original_title = data.get('original_title')
        stored_filename = data.get('stored_filename')
        classification = data.get('classification')
        
        if not paper_data or not original_title:
            return jsonify({'error': 'Paper data and original title are required'}), 400
        
        logger.info(f"Updating paper: {original_title} -> {paper_data.get('論文標題')}")
        
        # Step 1: Remove old data from database
        updater = DatabaseUpdater(model=model_name)
        success, message = updater.remove_paper(original_title)
        
        if not success:
            logger.warning(f"Failed to remove old paper data: {message}")
            return jsonify({'error': f'Failed to remove old data: {message}'}), 500
        
        # Step 2: Add updated data to database
        success, db_message, new_classification = updater.update_database(paper_data)
        
        if not success:
            logger.error(f"Failed to update database: {db_message}")
            return jsonify({'error': f'Failed to update database: {db_message}'}), 500
        
        logger.info(f"Updated database for: {paper_data['論文標題']}")
        
        # Step 3: Rename PDF file if title changed
        if original_title != paper_data['論文標題'] and stored_filename:
            try:
                old_pdf_path = os.path.join('./data', stored_filename)
                # Generate new filename
                safe_title = "".join(c for c in paper_data['論文標題'] if c.isalnum() or c in (' ', '-', '_')).strip()
                safe_title = safe_title[:50]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_filename = f"{safe_title}_{timestamp}.pdf"
                new_pdf_path = os.path.join('./data', new_filename)
                
                if os.path.exists(old_pdf_path):
                    os.rename(old_pdf_path, new_pdf_path)
                    logger.info(f"Renamed PDF: {stored_filename} -> {new_filename}")
                    stored_filename = new_filename
            except Exception as e:
                logger.warning(f"Failed to rename PDF file: {e}")
        
        return jsonify({
            'success': True,
            'message': 'Paper updated successfully',
            'stored_filename': stored_filename,
            'classification': new_classification
        }), 200
    
    except Exception as e:
        logger.error(f"Update paper failed: {e}", exc_info=True)
        return jsonify({
            'error': f'Update failed: {str(e)}'
        }), 500

# ============================================================================
# Hierarchical RAG Monitoring Endpoints
# ============================================================================

@app.route('/rag/stats', methods=['GET'])
def rag_stats():
    """
    獲取 RAG 系統統計資訊
    
    Returns:
        - mode: RAG 模式 (legacy/hierarchical)
        - system_stats: 系統統計（文檔數、索引狀態等）
        - query_stats: 查詢統計（僅 hierarchical 模式）
    """
    try:
        if not rag_system:
            return jsonify({'error': 'RAG system not initialized'}), 503
        
        # 基本資訊
        stats = {
            'mode': 'hierarchical',
            'system_ready': rag_system.is_ready() if hasattr(rag_system, 'is_ready') else True
        }
        
        # 獲取系統統計
        system_stats = rag_system.get_stats()
        stats['system_stats'] = system_stats
        
        # 獲取查詢統計
        if hasattr(rag_system, 'query_logger'):
            try:
                query_stats = rag_system.query_logger.get_statistics()
                stats['query_stats'] = query_stats
                
                # 性能摘要
                perf_summary = rag_system.query_logger.get_performance_summary()
                stats['performance_summary'] = perf_summary
            except Exception as e:
                logger.warning(f"Failed to get query stats: {e}")
                stats['query_stats'] = None
        
        return jsonify(stats), 200
        
    except Exception as e:
        logger.error(f"Failed to get RAG stats: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/rag/health', methods=['GET'])
def rag_health():
    """
    RAG 系統健康檢查
    
    Returns:
        - status: healthy/degraded/unhealthy
        - components: 各組件狀態
        - issues: 問題列表（如有）
    """
    try:
        health = {
            'status': 'healthy',
            'components': {},
            'issues': []
        }
        
        # 檢查 RAG 系統
        if not rag_system:
            health['status'] = 'unhealthy'
            health['issues'].append('RAG system not initialized')
            health['components']['rag_system'] = 'down'
        else:
            health['components']['rag_system'] = 'up'
            
            # 檢查是否就緒
            if hasattr(rag_system, 'is_ready'):
                is_ready = rag_system.is_ready()
                if not is_ready:
                    health['status'] = 'degraded'
                    health['issues'].append('RAG system not ready (indices not built)')
                    health['components']['indices'] = 'not_ready'
                else:
                    health['components']['indices'] = 'ready'
        
        # 檢查階層式索引
        if rag_system:
            try:
                stats = rag_system.get_stats()
                
                # 檢查 Layer 1
                if stats.get('layer1', {}).get('is_initialized'):
                    health['components']['layer1'] = 'up'
                else:
                    health['components']['layer1'] = 'down'
                    health['status'] = 'degraded'
                    health['issues'].append('Layer 1 not initialized')
                
                # 檢查 Layer 2
                if stats.get('layer2', {}).get('is_initialized'):
                    health['components']['layer2'] = 'up'
                else:
                    health['components']['layer2'] = 'down'
                    health['status'] = 'degraded'
                    health['issues'].append('Layer 2 not initialized')
                    
            except Exception as e:
                health['status'] = 'degraded'
                health['issues'].append(f'Failed to check components: {str(e)}')
        
        return jsonify(health), 200
        
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500


@app.route('/rag/mode', methods=['GET'])
def rag_mode():
    """
    獲取當前 RAG 模式資訊
    
    Returns:
        - mode: 當前模式
        - features: 支援的功能
        - config: 配置資訊
    """
    try:
        mode_info = {
            'mode': 'hierarchical',
            'type': 'HierarchicalRAGSystem',
            'features': ['two-layer', 'confidence-evaluation', 'context-expansion'],
            'config': {
                'layer1_threshold': float(os.getenv('LAYER1_THRESHOLD', '0.7')),
                'layer2_threshold': float(os.getenv('LAYER2_THRESHOLD', '0.8')),
                'expansion_enabled': os.getenv('ENABLE_EXPANSION', 'true').lower() == 'true',
                'cache_size': int(os.getenv('CACHE_SIZE', '10'))
            }
        }
        
        return jsonify(mode_info), 200
        
    except Exception as e:
        logger.error(f"Failed to get mode info: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/rag/rebuild', methods=['POST'])
def rag_rebuild():
    """
    手動觸發索引重建或更新
    
    Query Parameters:
        - force: true/false (是否強制完全重建)
    
    Returns:
        - status: success/error
        - action_taken: 執行的操作
        - details: 詳細資訊
    """
    try:
        if not rag_system:
            return jsonify({'error': 'RAG system not initialized'}), 503
        
        force = request.args.get('force', 'false').lower() == 'true'
        
        logger.info(f"Manual index rebuild requested (force={force})")
        
        # 執行智能檢查並更新
        result = rag_system.check_and_update_indices(force_rebuild=force)
        
        if result['status'] in ['success', 'up_to_date']:
            return jsonify(result), 200
        else:
            return jsonify(result), 500
        
    except Exception as e:
        logger.error(f"Rebuild failed: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 AI Agent Router Starting...")
    print("="*70)
    print("\n📊 System Status:")
    print(f"  RAG Mode: HIERARCHICAL (HierarchicalRAGSystem)")
    print(f"  RAG System: {'✓ Ready' if rag_system else '✗ Failed'}")
    
    # 顯示分塊模式
    chunking_mode = os.getenv('CHUNKING_MODE', 'naive')
    print(f"  Chunking Mode: {chunking_mode.upper()}")
    if chunking_mode == 'summarization':
        print(f"    └─ Model: {os.getenv('SUMMARIZATION_MODEL', 'llama3:8b')}")
        print(f"    └─ Summary Length: {os.getenv('TARGET_SUMMARY_LENGTH', '300')} chars")
    
    if rag_system:
        try:
            stats = rag_system.get_stats()
            is_ready = rag_system.is_ready()
            print(f"  System Ready: {'✓ Yes' if is_ready else '⚠ No (indices not built)'}")
            if is_ready:
                layer1_stats = stats.get('layer1', {})
                layer2_stats = stats.get('layer2', {})
                print(f"  Layer 1: {layer1_stats.get('paper_count', 0)} papers")
                print(f"  Layer 2: {layer2_stats.get('chunk_count', 0)} chunks")
        except Exception as e:
            print(f"  ⚠ Warning: Could not get stats: {e}")
    
    print(f"  ResearchInheritanceAnalyzer: {'✓ Ready' if inheritance_analyzer else '✗ Failed'}")
    print(f"  Inheritance LLM: {'✓ Ready' if inheritance_llm else '✗ Failed'}")
    print(f"  Agent LLM: {'✓ Ready' if agent_llm else '✗ Failed'}")
    
    print("\n🌐 Monitoring Endpoints:")
    print("  GET  /rag/stats   - System statistics")
    print("  GET  /rag/health  - Health check")
    print("  GET  /rag/mode    - Current RAG mode info")
    print("  POST /rag/rebuild - Rebuild indices (use ?force=true to force)")
    
    print("\n" + "="*70)
    print("Server running on http://localhost:4000")
    print("="*70 + "\n")
    
    app.run(debug=False, port=4000)