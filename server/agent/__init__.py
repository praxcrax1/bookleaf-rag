# agent package init
from .agent_factory import create_agent as create_langchain_agent
from .enhanced_agent_factory import create_agent, create_agent_with_validation, create_hybrid_agent
from .langgraph_agent import create_langgraph_agent, create_langgraph_agent_graph
from .langgraph_wrapper import create_compatible_langgraph_agent

# Backward compatibility - default export
__all__ = [
    'create_agent',  # Enhanced factory with LangGraph support
    'create_langchain_agent',  # Original LangChain implementation
    'create_langgraph_agent',  # Direct LangGraph React agent
    'create_compatible_langgraph_agent',  # LangGraph with LangChain interface
    'create_agent_with_validation',  # Full LangGraph StateGraph with validation
    'create_hybrid_agent',  # Intelligent agent selection
    'create_langgraph_agent_graph'  # Full graph implementation
]
