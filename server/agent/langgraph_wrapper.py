"""
Backward compatibility wrapper for LangGraph agents.
Provides the same interface as the original LangChain agent.
"""
from langchain_core.messages import HumanMessage
from .langgraph_agent import create_langgraph_agent
from .agent_factory import create_agent as create_langchain_agent
import logging

logger = logging.getLogger(__name__)


class LangGraphAgentWrapper:
    """
    Wrapper to make LangGraph agent compatible with existing interface.
    """
    
    def __init__(self, langgraph_agent, user_id=None):
        self.agent = langgraph_agent
        self.user_id = user_id
        
    def invoke(self, inputs):
        """
        Invoke the LangGraph agent with proper input handling.
        Maintains compatibility with LangChain agent interface.
        """
        try:
            # Extract the input query
            if isinstance(inputs, dict):
                query = inputs.get('input', inputs.get('query', str(inputs)))
            else:
                query = str(inputs)
            
            # Prepare state for LangGraph agent with proper config
            state = {
                "messages": [HumanMessage(content=query)]
            }
            
            # Configure the agent execution with thread_id for checkpointer
            config = {
                "configurable": {
                    "thread_id": self.user_id or "default_thread"
                }
            }
            
            # Execute the agent
            result = self.agent.invoke(state, config=config)
            
            # Format response to match LangChain agent interface
            response = {
                "input": query,
                "output": result.get("messages", [])[-1].content if result.get("messages") else "No response generated",
                "intermediate_steps": []
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error in LangGraph agent wrapper: {e}")
            return {
                "input": str(inputs),
                "output": f"I encountered an error processing your request: {e}",
                "intermediate_steps": []
            }


def create_compatible_langgraph_agent(user_id=None, query=None):
    """
    Create a LangGraph agent with LangChain-compatible interface.
    
    Args:
        user_id (str): The user ID for personalized support
        query (str): Optional initial query (for context)
    
    Returns:
        LangGraphAgentWrapper: Agent with compatible interface
    """
    try:
        # Create the LangGraph agent
        langgraph_agent = create_langgraph_agent(user_id=user_id, query=query)
        
        # Wrap it for compatibility
        wrapper = LangGraphAgentWrapper(langgraph_agent, user_id=user_id)
        
        logger.info(f"Successfully created compatible LangGraph agent for user {user_id}")
        return wrapper
        
    except Exception as e:
        logger.error(f"Failed to create LangGraph agent, falling back to LangChain: {e}")
        # Fallback to original implementation
        return create_langchain_agent(user_id=user_id, query=query)
