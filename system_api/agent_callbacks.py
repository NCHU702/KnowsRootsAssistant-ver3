"""
Custom callback handler for streaming agent execution
Captures Thought → Action → Observation process in real-time
"""

from typing import Any, Dict, List, Optional
from langchain_core.callbacks import BaseCallbackHandler
import queue
import logging

logger = logging.getLogger(__name__)


class AgentStreamCallback(BaseCallbackHandler):
    """Callback handler that streams agent thinking process"""
    
    def __init__(self, queue: queue.Queue, stop_event=None):
        """
        Initialize the callback handler
        
        Args:
            queue: Thread-safe queue to send events to the stream generator
            stop_event: Threading event to signal when to stop execution
        """
        self.queue = queue
        self.stop_event = stop_event
        self.current_step = 0
        
    def on_agent_action(self, action, **kwargs) -> None:
        """Called when agent takes an action"""
        try:
            # Check if stop signal is set
            if self.stop_event and self.stop_event.is_set():
                logger.info("Stop signal detected, interrupting agent...")
                raise KeyboardInterrupt("Agent execution stopped by user")
            
            tool_name = action.tool if hasattr(action, 'tool') else str(action)
            tool_input = action.tool_input if hasattr(action, 'tool_input') else ''
            
            # Send action event
            self.queue.put({
                'type': 'thought',
                'step': self.current_step,
                'content': f"I should use the {tool_name} tool."
            })
            
            self.queue.put({
                'type': 'action',
                'step': self.current_step,
                'tool': tool_name,
                'input': str(tool_input)
            })
            
            logger.info(f"🎯 Agent action detected: {tool_name} with input: {str(tool_input)[:100]}")
            
        except Exception as e:
            logger.error(f"Error in on_agent_action: {e}")
            raise  # Re-raise to stop execution
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """Called when a tool starts running"""
        try:
            # Check if stop signal is set
            if self.stop_event and self.stop_event.is_set():
                logger.info("Stop signal detected, interrupting tool execution...")
                raise KeyboardInterrupt("Tool execution stopped by user")
            
            tool_name = serialized.get('name', 'Unknown')
            
            self.queue.put({
                'type': 'tool_start',
                'step': self.current_step,
                'tool': tool_name
            })
            
            logger.debug(f"Tool start: {tool_name}")
            
        except Exception as e:
            logger.error(f"Error in on_tool_start: {e}")
            raise  # Re-raise to stop execution
    
    def on_tool_end(self, output: str, **kwargs) -> None:
        """Called when a tool finishes running"""
        try:
            # Check if stop signal is set
            if self.stop_event and self.stop_event.is_set():
                logger.info("Stop signal detected after tool execution...")
                raise KeyboardInterrupt("Execution stopped by user after tool")
            
            # Send observation (truncate if too long)
            observation = output[:500] + "..." if len(output) > 500 else output
            
            self.queue.put({
                'type': 'observation',
                'step': self.current_step,
                'content': observation
            })
            
            self.current_step += 1
            
            logger.debug(f"Tool end, observation length: {len(output)}")
            
        except Exception as e:
            logger.error(f"Error in on_tool_end: {e}")
            raise  # Re-raise to stop execution
    
    def on_tool_error(self, error: Exception, **kwargs) -> None:
        """Called when a tool errors"""
        try:
            self.queue.put({
                'type': 'error',
                'step': self.current_step,
                'content': str(error)
            })
            
            logger.error(f"Tool error: {error}")
            
        except Exception as e:
            logger.error(f"Error in on_tool_error: {e}")
    
    def on_agent_finish(self, finish, **kwargs) -> None:
        """Called when agent finishes"""
        try:
            output = finish.return_values.get('output', '') if hasattr(finish, 'return_values') else str(finish)
            
            self.queue.put({
                'type': 'final_answer',
                'content': output
            })
            
            logger.debug("Agent finished")
            
        except Exception as e:
            logger.error(f"Error in on_agent_finish: {e}")
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Called when LLM generates a new token (for final answer streaming)"""
        try:
            # Check if stop signal is set
            if self.stop_event and self.stop_event.is_set():
                logger.info("Stop signal detected, interrupting token generation...")
                raise KeyboardInterrupt("Token generation stopped by user")
            
            # Only stream tokens for final answer, not for internal reasoning
            if kwargs.get('tags') and 'final_answer' in kwargs['tags']:
                self.queue.put({
                    'type': 'token',
                    'content': token
                })
        except Exception as e:
            logger.error(f"Error in on_llm_new_token: {e}")
            raise  # Re-raise to stop execution
    
    def on_chain_error(self, error: Exception, **kwargs) -> None:
        """Called when chain errors"""
        try:
            self.queue.put({
                'type': 'error',
                'content': str(error)
            })
            
            logger.error(f"Chain error: {error}")
            
        except Exception as e:
            logger.error(f"Error in on_chain_error: {e}")


class ToolStreamCallback(BaseCallbackHandler):
    """Callback handler for streaming tool outputs (AssistantCall, WebSearchCall)"""
    
    def __init__(self, queue: queue.Queue, tool_name: str):
        """
        Initialize the callback handler for a specific tool
        
        Args:
            queue: Thread-safe queue to send events to
            tool_name: Name of the tool being executed
        """
        self.queue = queue
        self.tool_name = tool_name
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Called when LLM generates a new token"""
        try:
            self.queue.put({
                'type': 'tool_token',
                'tool': self.tool_name,
                'content': token
            })
        except Exception as e:
            logger.error(f"Error in ToolStreamCallback.on_llm_new_token: {e}")
