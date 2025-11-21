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
    
    This tool will:
    1. Query Knowledge Graph with Text2Cypher for relevant papers
    2. Integrate Graph results with LLM
    3. **Automatically descend to Layer2 if details are needed** (YOUR IDEAL FLOW!)
    """
    if not graph_manager:
        return "Graph RAG is not available."
    
    logger.info(f"🔍 [Graph Tool] Querying: '{query}'")
    try:
        # Step 1: Use GraphManager.query_graph() for Text2Cypher
        # This generates Cypher from natural language and executes it
        graph_answer = graph_manager.query_graph(query)
        
        logger.info(f"  Graph query executed")
        logger.info(f"  Answer preview: {graph_answer[:150]}...")
        
        # Check if the answer requests more details (contains keywords like "細節", "詳細", "details")
        needs_details = any(keyword in query.lower() for keyword in [
            '細節', '详细', 'details', 'detail', '具體', '具体', 'specific',
            '訓練', '训练', 'training', '參數', '参数', 'parameter',
            '實驗', '实验', 'experiment', '結果', '结果', 'result'
        ])
        
        if not needs_details:
            # Query only asks "which papers" - Graph answer is sufficient
            logger.info("  ✓ Macro-only query - Graph answer sufficient")
            return f"[Graph Analysis Result]\n{graph_answer}\n\n[✓ Answer from Knowledge Graph - sufficient confidence]"
        
        # Step 2: Extract paper IDs from graph_answer to descend to Layer2
        logger.info("  → Query requests details - descending to Layer2")
        
        # Try to find papers in Neo4j based on the query
        # Use a simple Cypher to get paper IDs mentioned in the answer
        paper_id_query = """
        MATCH (p:Paper)-[:USES_METHOD]->(m:Method)
        WHERE toLower(m.name) CONTAINS "ensemble" OR toLower(m.name) CONTAINS "集成"
        RETURN p.paper_id as paper_id, p.title as title
        LIMIT 5
        """
        
        try:
            paper_results = graph_manager.graph.query(paper_id_query)
            paper_ids = [r['paper_id'] for r in paper_results]
            logger.info(f"  Found {len(paper_ids)} papers for Layer2 retrieval")
        except Exception as e:
            logger.warning(f"  Could not extract paper IDs: {e}")
            # Return Graph answer only
            return f"[Graph Analysis Result]\n{graph_answer}\n\n[✓ Answer from Knowledge Graph]"
        
        if not paper_ids:
            return f"[Graph Analysis Result]\n{graph_answer}\n\n[✓ Answer from Knowledge Graph]"
        
        # Step 3: Retrieve chunks from Layer2 (no chunk_type filtering for now)
        # Note: Current system uses 'section_summary' chunk type, not semantic types like 'method'/'results'
        # So we retrieve all chunk types and rely on semantic search to find relevant content
        logger.info(f"  Retrieving top-k chunks per paper (no chunk_type filter)")
        
        # Step 4: Retrieve chunks from Layer2
        layer2_docs = []
        for paper_id in paper_ids[:5]:  # Limit to top 5 papers
            paper_results = rag_system.layer2.search_with_scores(
                query=query,
                k=3,  # Top 3 chunks per paper (increased since no filtering)
                filter_paper_ids=[paper_id],
                filter_chunk_types=None  # Don't filter by chunk_type
            )
            layer2_docs.extend([doc for doc, score in paper_results])
        
        logger.info(f"  Retrieved {len(layer2_docs)} chunks from Layer2")
        
        # Step 5: Generate final answer with Graph + Layer2 content
        if layer2_docs:
            # Combine graph answer with detailed chunks
            answer = rag_system._generate_answer(query, {
                'final_docs': layer2_docs,
                'status': 'success'
            })
            
            # Prepend graph answer for context
            combined_answer = f"{graph_answer}\n\n詳細資訊如下：\n\n{answer}"
            return f"[Graph Analysis Result]\n{combined_answer}\n\n[✓ Answer from Knowledge Graph → Layer2 - included detailed chunks]"
        else:
            return f"[Graph Analysis Result]\n{graph_answer}\n\n[✓ Answer from Knowledge Graph]"
        
    except Exception as e:
        logger.error(f"Graph tool error: {e}", exc_info=True)
        return f"Error querying graph: {str(e)}"

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
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

IMPORTANT ROUTING RULES:

1. **GraphAnalysisCall** (Macro-View):
   - USE WHEN: User asks "Which papers...", "List all...", "How many...", "Compare...", "What are the common methods...".
   - KEYWORDS: Methods, Datasets, Domain, Metrics, Relationship, Trends, Count.
   - Examples: "Which papers use LSTM?", "List all datasets", "How many papers in Smart Transportation?"

2. **AssistantCall** (Micro-View):
   - USE WHEN: User asks about content/details of a specific paper or wants summaries.
   - KEYWORDS: Summarize, Explain, Abstract, Introduction, Methodology, Content.
   - Examples: "Summarize this paper", "Explain the methodology in paper X"

3. **WebSearchCall**:
   - USE ONLY for explicit internet search requests.
   - KEYWORDS: "search internet", "search online", "web search", "find on internet"

Here are some examples (FOLLOW THESE FORMATS EXACTLY):

Example 1 - Macro Query (Use GraphAnalysisCall):
Question: Which papers use LSTM?
Thought: This asks for papers by method. This is a macro-level query across papers.
Action: GraphAnalysisCall
Action Input: Which papers use LSTM?
Observation: [list of papers using LSTM from graph]
Thought: I now know the final answer
Final Answer: Papers using LSTM include...

Example 2 - Chinese Macro Query (Use GraphAnalysisCall):
Question: 哪些論文用到集成式學習？
Thought: This asks "which papers" use ensemble learning. Macro-level query.
Action: GraphAnalysisCall
Action Input: 哪些論文用到集成式學習？
Observation: [list of papers from graph]
Thought: I now know the final answer
Final Answer: 使用集成式學習的論文包括...

Example 3 - Multi-Part Query (GraphAnalysisCall handles BOTH automatically):
Question: 哪些論文用到集成式學習，細節又是什麼？
Thought: This asks "which papers" (macro) AND "details" (micro). GraphAnalysisCall will automatically handle BOTH: it queries the Knowledge Graph, and if details are needed, it descends to Layer2 for chunk-level information.
Action: GraphAnalysisCall
Action Input: 哪些論文用到集成式學習，細節又是什麼？
Observation: [Graph Analysis Result]
使用集成式學習的論文包括「基礎5_應用集成式深度學習模型進行芒果分類辨識」。

細節如下：
- 方法: 使用 Mask R-CNN 和卷積神經網路進行影像分類
- 訓練參數: batch_size=32, learning_rate=0.001
- 結果: 準確率達到 95.3%

[✓ Answer from Knowledge Graph → Layer2 - included detailed chunks]
Thought: GraphAnalysisCall automatically retrieved both the paper list from Graph AND the details from Layer2. I now have the complete answer.
Final Answer: 使用集成式學習的論文包括「基礎5_應用集成式深度學習模型進行芒果分類辨識」。此論文使用 Mask R-CNN 和卷積神經網路進行芒果分類，訓練參數為 batch_size=32, learning_rate=0.001，準確率達到 95.3%。

Example 4 - Count Query (Use GraphAnalysisCall):
Question: How many papers use the PeMSD7 dataset?
Thought: This asks for a count across papers. This is a macro-level aggregation query.
Action: GraphAnalysisCall
Action Input: How many papers use the PeMSD7 dataset?
Observation: [count from graph]
Thought: I now know the final answer
Final Answer: X papers use the PeMSD7 dataset.

Example 4 - Comparison Query (Use GraphAnalysisCall):
Question: Compare research goals in Smart Transportation domain
Thought: This asks to compare across multiple papers. This is a macro-level query.
Action: GraphAnalysisCall
Action Input: Compare research goals in Smart Transportation domain
Observation: [comparison from graph]
Thought: I now know the final answer
Final Answer: Research goals in Smart Transportation include...

Example 5 - Specific Paper Summary (Use AssistantCall):
Question: Summarize the paper about traffic flow prediction
Thought: This asks for content from a specific paper. This is a micro-level query.
Action: AssistantCall
Action Input: Summarize the paper about traffic flow prediction
Observation: [summary from vector database]
Thought: I now know the final answer
Final Answer: The paper discusses...

Example 6 - Content Explanation (Use AssistantCall):
Question: Explain the methodology in the first paper
Thought: This asks for detailed content. This is a micro-level query.
Action: AssistantCall
Action Input: Explain the methodology in the first paper
Observation: [methodology details from vector database]
Thought: I now know the final answer
Final Answer: The methodology includes...

Example 7 - Web Search (Use WebSearchCall):
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
            max_iterations=5,  # Allow multiple tool calls and reasoning steps
            max_execution_time=60,  # Maximum 60 seconds per query
            return_intermediate_steps=True
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
                                    rag_stream = rag_system.query_stream(action_input)
                                    
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