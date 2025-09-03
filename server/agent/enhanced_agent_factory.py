"""
Enhanced agent factory with LangGraph React agent option.
Provides both original LangChain and new LangGraph implementations.
"""
from .agent_factory import create_agent as create_langchain_agent
from .langgraph_wrapper import create_compatible_langgraph_agent
from .langgraph_agent import create_langgraph_agent_graph
import logging
import os

logger = logging.getLogger(__name__)

def create_agent(user_id=None, query=None, use_langgraph=None):
    """
    Create an agent with optional LangGraph backend.
    
    Args:
        user_id (str): The user ID for personalized support
        query (str): Optional initial query (for context)
        use_langgraph (bool): Whether to use LangGraph implementation. 
                             If None, checks environment variable AGENT_BACKEND
    
    Returns:
        Agent executor (LangChain or LangGraph based)
    """
    # Determine which backend to use
    if use_langgraph is None:
        use_langgraph = os.getenv("AGENT_BACKEND", "langchain").lower() == "langgraph"
    
    if use_langgraph:
        logger.info(f"Creating LangGraph React agent for user {user_id}")
        try:
            return create_compatible_langgraph_agent(user_id=user_id, query=query)
        except Exception as e:
            logger.error(f"Failed to create LangGraph agent, falling back to LangChain: {e}")
            return create_langchain_agent(user_id=user_id, query=query)
    else:
        logger.info(f"Creating LangChain agent for user {user_id}")
        return create_langchain_agent(user_id=user_id, query=query)


def create_agent_with_validation(user_id=None, query=None):
    """
    Create a LangGraph agent with built-in validation workflow.
    This uses the full StateGraph implementation for maximum control.
    
    Args:
        user_id (str): The user ID for personalized support
        query (str): Optional initial query (for context)
    
    Returns:
        Compiled LangGraph StateGraph with validation
    """
    return create_langgraph_agent_graph(user_id=user_id, query=query)


def create_hybrid_agent(user_id=None, query=None):
    """
    Create an agent that intelligently chooses between implementations
    based on the query type and user context.
    
    Args:
        user_id (str): The user ID for personalized support
        query (str): Optional initial query (for context)
    
    Returns:
        Most appropriate agent for the query
    """
    # Analyze query to determine best agent type
    if query and any(keyword in query.lower() for keyword in [
        "how to", "what is", "explain", "guide", "tutorial", "help", "support"
    ]):
        # FAQ-type queries benefit from validation
        logger.info(f"Using LangGraph agent with validation for FAQ query: {query[:50]}...")
        try:
            return create_agent(user_id=user_id, query=query, use_langgraph=True)
        except Exception as e:
            logger.warning(f"Failed to create LangGraph agent for FAQ, using standard: {e}")
            return create_agent(user_id=user_id, query=query, use_langgraph=False)
    else:
        # Personal queries or simple interactions use standard agent
        logger.info(f"Using standard agent for query: {query[:50] if query else 'No query'}...")
        return create_agent(user_id=user_id, query=query, use_langgraph=False)
