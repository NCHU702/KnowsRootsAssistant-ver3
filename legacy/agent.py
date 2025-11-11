from flask import Flask, request, jsonify, render_template
from langchain_ollama import OllamaLLM
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
import logging
from system_api.search_engine import SearchEngine
search_engine = SearchEngine('./paper_entries.csv')
# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize Ollama LLM with proper configuration
try:
    llm = OllamaLLM(
        model="gemma3:270m",  # Make sure this model is pulled in Ollama
        temperature=0,
        base_url="http://localhost:11434"  # Default Ollama URL
    )
    # Test the connection
    test_response = llm.invoke("Hello")
    logger.info("Ollama connection successful")
except Exception as e:
    logger.error(f"Failed to connect to Ollama: {e}")
    logger.error("Make sure Ollama is running and the model is pulled")

# Define functions with more detailed responses
def assistant_call(user_input: str) -> str:
    """AI assistant for paper summary, interpretation, or translation"""
    logger.info(f"Assistant Call invoked with: '{user_input}'")
    return f"Assistant is analyzing the paper content for: '{user_input}'. This tool would normally process academic papers for summary, interpretation, or translation tasks."

def retrieval_call(user_input: str) -> str:
    search_engine.search(user_input)
    """AI assistant to retrieve papers' relationships"""
    logger.info(f"Retrieval Call invoked with: '{user_input}'")
    return f"Searching for paper relationships and citations related to: '{user_input}'. This tool would analyze citation networks and paper connections."

def web_search_call(user_input: str) -> str:
    """AI web scraper"""
    logger.info(f"Web Search Call invoked with: '{user_input}'")
    return f"Performing web search for: '{user_input}'. This tool would scrape and search the internet for relevant information."

def history_finder_call(user_input: str) -> str:
    """AI assistant to organize history"""
    logger.info(f"History Finder Call invoked with: '{user_input}'")
    return f"Searching through historical records for: '{user_input}'. This tool would organize and search through past records and historical information."

# Create tools for the agent
tools = [
    Tool(
        name="AssistantCall",
        func=assistant_call,
        description="Use this for paper summary, interpretation, or translation tasks. Best for analyzing academic content."
    ),
    Tool(
        name="RetrievalCall",
        func=retrieval_call,
        description="Use this to retrieve papers' relationships, find connections between papers, or analyze paper citations and references."
    ),
    Tool(
        name="WebSearchCall",
        func=web_search_call,
        description="Use this for web scraping, searching the internet for information, or finding online resources."
    ),
    Tool(
        name="HistoryFinderCall",
        func=history_finder_call,
        description="Use this to organize or search through history, past records, or historical information."
    )
]

# Create a more robust prompt template
template = """You are a helpful assistant that routes user queries to the appropriate tool.

Available tools:
{tools}

Tool Names: {tool_names}

To use a tool, you must follow this EXACT format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, must be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Remember to use the EXACT format above.

Question: {input}
Thought: I need to determine which tool is most appropriate for this query.
{agent_scratchpad}"""

prompt = PromptTemplate(
    template=template,
    input_variables=["input", "tools", "tool_names", "agent_scratchpad"]
)

try:
    # Create the agent with error handling
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
        output_parser=ReActSingleInputOutputParser(),
        stop_sequence=["\nObservation:", "\n\tObservation:"]
    )
    
    # Create an agent executor with better error handling
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3,
        return_intermediate_steps=True,
        max_execution_time=30  # Add timeout
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
    if agent_executor:
        return jsonify({'status': 'ok', 'message': 'Agent is ready'})
    else:
        return jsonify({'status': 'error', 'message': 'Agent not initialized. Check Ollama connection.'}), 500

@app.route('/query', methods=['POST'])
def query():
    try:
        user_input = request.json.get('input', '')
        
        if not user_input:
            return jsonify({'error': 'No input provided'}), 400
        
        if not agent_executor:
            return jsonify({'error': 'Agent not initialized. Please check Ollama is running.'}), 500
        
        logger.info(f"Processing query: {user_input}")
        
        # Execute the agent with error handling
        try:
            response = agent_executor.invoke({"input": user_input})
            
            # Extract action taken from intermediate steps
            action_taken = "Direct Response"
            if 'intermediate_steps' in response and response['intermediate_steps']:
                for step in response['intermediate_steps']:
                    if len(step) > 0 and hasattr(step[0], 'tool'):
                        action_taken = step[0].tool
                        break
            
            output_text = response.get('output', 'No response generated')
            
            logger.info(f"Action taken: {action_taken}")
            logger.info(f"Output: {output_text[:100]}...")
            
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

def fallback_routing(user_input):
    """Simple keyword-based routing as fallback"""
    user_input_lower = user_input.lower()
    
    if any(word in user_input_lower for word in ['summarize', 'summary', 'translate', 'interpret', 'paper']):
        action = 'AssistantCall'
        output = assistant_call(user_input)
    elif any(word in user_input_lower for word in ['relationship', 'citation', 'reference', 'connection']):
        action = 'RetrievalCall'
        output = retrieval_call(user_input)
    elif any(word in user_input_lower for word in ['search', 'web', 'internet', 'online', 'find']):
        action = 'WebSearchCall'
        output = web_search_call(user_input)
    elif any(word in user_input_lower for word in ['history', 'past', 'record', 'archive']):
        action = 'HistoryFinderCall'
        output = history_finder_call(user_input)
    else:
        action = 'AssistantCall'
        output = assistant_call(user_input)
    
    return jsonify({
        'action': action + ' (Fallback)',
        'output': output
    })

if __name__ == '__main__':
    print("\n" + "="*50)
    print("AI Agent Router Starting...")
    print("="*50)
    print("\nIMPORTANT: Make sure Ollama is running!")
    print("1. Start Ollama: 'ollama serve' (if not already running)")
    print("2. Pull the model: 'ollama pull llama3.2:latest'")
    print("3. Test Ollama: 'ollama run llama3.2:latest'")
    print("\n" + "="*50 + "\n")
    
    app.run(debug=True, port=5000)