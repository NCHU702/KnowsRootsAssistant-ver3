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
from system_api.rag_system import AcademicRAGSystem
from collections import Counter
# 
model_name = "jcai/llama-3-taiwan-8b-instruct:q4_k_m"

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize Rag system
logger.info("Initializing RAG System...")
rag_system = None
try:
    rag_system = AcademicRAGSystem(
        pdf_directory="./data",
        model_name=model_name,
        embedding_model="nomic-embed-text",  # or another embedding model
        chunk_size=800,  # 減小以避免 Ollama embedding 限制
        chunk_overlap=100,  # 減少重疊
        k_documents=4
    )
    logger.info("RAG System initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RAG System: {e}")
    rag_system = None
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
# Initialize Ollama LLM for agent
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

def assistant_call(user_input: str) -> str:
    """AI assistant for paper summary, interpretation, translation, and retrieval"""
    logger.info(f"Assistant Call invoked with: '{user_input}'")
    
    # Use RAG system for queries
    if rag_system:
        logger.info("Using RAG system for query")
        return rag_system.query(user_input)
    else:
        return "RAG system not available. Please check system status."


def web_search_call(user_input: str) -> str:
    """AI web scraper"""
    
    logger.info(f"Web Search Call invoked with: '{user_input}'")
    result = web_searcher.search(user_input)

    return result

# Create tools for the agent
tools = [
    Tool(
        name="AssistantCall",
        func=assistant_call,
        description="Use this for paper summary, interpretation, or translation tasks. Best for analyzing or processing academic content, such as summarizing, translating, or explaining documents."
    ),

    Tool(
        name="WebSearchCall",
        func=web_search_call,
        description="Use this for web scraping, searching the internet for recent or latest information, including top academic papers on specific topics from 2024 or 2025, or finding online resources. Do not use invented tools like PaperSearch; use this for any online paper searches."
    )
]

# Create agent only if LLM is available
agent_executor = None
if agent_llm:
    try:
        template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be exactly one of [{tool_names}]. Do not invent new tool names like PaperSearch.
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Here are some examples:

Example 1:
Question: Summarize a research paper on AI
Thought: This requires summarizing academic content, so I should use AssistantCall.
Action: AssistantCall
Action Input: Summarize the AI paper
Observation: [some summary]
Thought: I now know the final answer
Final Answer: The summary is...

Example 2:
Question: Find old papers on machine learning
Thought: This is for searching local database for older papers, so use RetrievalCall.
Action: RetrievalCall
Action Input: machine learning
Observation: [list of papers]
Thought: I now know the final answer
Final Answer: Here are the papers...

Example 3:
Question: Search for 2024 papers on remote sensing
Thought: This requires internet search for recent papers, so use WebSearchCall. Do not use RetrievalCall for recent years.
Action: WebSearchCall
Action Input: top remote sensing papers 2024
Observation: [search results]
Thought: I now know the final answer
Final Answer: Here are the top papers...

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
                                    yield f"data: {json.dumps({'type': 'tool_start', 'tool': 'AssistantCall'})}\n\n"
                                    
                                    action_input = event.get('input', user_input)
                                    rag_stream = rag_system.query_stream(action_input)
                                    
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

@app.route('/query', methods=['POST'])
def query():
    try:
        user_input = request.json.get('input', '')
        
        if not user_input:
            return jsonify({'error': 'No input provided'}), 400
        
        user_input_lower = user_input.lower()
        if all(keyword in user_input_lower for keyword in ['question:', 'thought:', 'action:']):
            logger.warning("Detected agent format in input, using fallback routing")
            return fallback_routing(user_input)
        
        if not agent_executor:
            # Fallback to direct routing if agent not available
            logger.warning("Agent not available, using fallback routing")
            return fallback_routing(user_input)
        
        logger.info(f"Processing query: {user_input}")
        
        # Execute the agent with error handling
        try:
            response = agent_executor.invoke({"input": user_input})
            
            # Robustly extract action taken and action input from intermediate steps
            action_taken = "Direct Response"
            action_input = None
            try:
                steps = response.get('intermediate_steps', []) if isinstance(response, dict) else []
                for step in steps:
                    if not step:
                        continue
                    first = step[0]
                    # support both objects and dict-like entries
                    tool_name = None
                    tool_input = None
                    if hasattr(first, 'tool'):
                        tool_name = getattr(first, 'tool', None)
                    elif isinstance(first, dict):
                        tool_name = first.get('tool') or first.get('name')
                    
                    if hasattr(first, 'tool_input'):
                        tool_input = getattr(first, 'tool_input', None)
                    elif isinstance(first, dict):
                        tool_input = first.get('tool_input') or first.get('input')
                    
                    if tool_name:
                        action_taken = tool_name
                        action_input = tool_input
                        break
            except Exception:
                logger.debug("Failed to parse intermediate_steps", exc_info=True)
            
            output_text = response.get('output', '') if isinstance(response, dict) else str(response)
            
            # If agent used AssistantCall tool, ensure we return assistant_call's result (more consistent format)
            if action_taken == 'AssistantCall':
                logger.info("Agent used AssistantCall - invoking assistant_call to get canonical assistant output")
                try:
                    # prefer the tool's input if available, otherwise use the original user input
                    assist_input = action_input if action_input else user_input
                    assistant_result = assistant_call(assist_input)
                    output_text = assistant_result
                except Exception as e:
                    logger.warning(f"assistant_call failed while honoring Agent's AssistantCall: {e}")
            
            # If agent used WebSearchCall but user asked for summarization/analysis, prefer AssistantCall
            # elif action_taken == 'WebSearchCall':
            #     summarize_keywords = ['summarize', 'summary', 'explain', 'interpret', 'translate', 'abstract', 'conclude']
            #     if any(k in user_input.lower() for k in summarize_keywords):
            #         logger.info("User intent appears to be summarization/analysis; calling assistant_call instead of web search output")
            #         try:
            #             assistant_result = assistant_call(user_input)
            #             output_text = assistant_result
            #             action_taken = 'AssistantCall (Forced)'
            #         except Exception as e:
            #             logger.warning(f"assistant_call failed when forced after WebSearchCall: {e}")
            
            # Detect clearly invalid agent outputs and fallback if necessary
            if not output_text or "stopped due to" in output_text.lower() or "invalid format" in output_text.lower():
                logger.warning("Agent output indicates failure; falling back to routing")
                return fallback_routing(user_input)
            
            logger.info(f"Action taken: {action_taken}")
            logger.info(f"Output length: {len(output_text)} chars")
            
            return jsonify({
                'action': action_taken,
                'output': output_text
            })
            
        except Exception as agent_error:
            logger.error(f"Agent execution error: {agent_error}")
            # Fallback to simple routing based on keywords
            return fallback_routing(user_input)
    
    except Exception as e:
        logger.error(f"Request error: {e}")
        return jsonify({'error': str(e)}), 500

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
    """Bag-of-words based routing as fallback using numpy for cosine similarity"""
    
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
            'desc': "Handle tasks involving summarization, translation, interpretation, or detailed analysis of academic content, papers, or documents. Use when the query asks to explain, summarize, or process existing information.",
            'func': assistant_call
        },

        'WebSearchCall': {
            'desc': "Perform web searches for recent information, latest or top academic papers on specific topics, online resources, news, or real-time data from the internet. Use for queries about current events, new research in 2024 or 2025, or anything requiring up-to-date web content.",
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

if __name__ == '__main__':
    print("\n" + "="*50)
    print("AI Agent Router Starting...")
    print("="*50)
    print("\nSystem Status:")
    print(f"  RAG System: {'✓ Ready' if rag_system else '✗ Failed'}")
    print(f"  ResearchInheritanceAnalyzer: {'✓ Ready' if inheritance_analyzer else '✗ Failed'}")
    print(f"  Inheritance LLM: {'✓ Ready' if inheritance_llm else '✗ Failed'}")
    print(f"  Agent LLM: {'✓ Ready' if agent_llm else '✗ Failed'}")
    
    if rag_system:
        stats = rag_system.get_stats()
        print(f"  RAG Documents: {stats.get('total_documents', 0)}")
    
    app.run(debug=False, port=4000)