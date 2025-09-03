"""
LangGraph-based React Agent with self-validation for RAG responses.
This implementation uses LangGraph's create_react_agent with enhanced FAQ validation.
"""
from typing import TypedDict, List, Literal, Optional, Any, Dict
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent, ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langchain.tools import Tool
from pydantic import BaseModel, Field
from config import Config
from .tools import user_book_summary_tool, document_retrieval_tool
import logging
from typing_extensions import Annotated

logger = logging.getLogger(__name__)
config = Config()


class ValidationResult(BaseModel):
    """Pydantic model for validation results"""
    is_valid: bool = Field(description="Whether the response adequately answers the question")
    confidence_score: float = Field(description="Confidence in the response quality (0.0-1.0)")
    needs_retry: bool = Field(description="Whether to search again with a different query")
    suggested_query: Optional[str] = Field(description="Suggested alternative query if retry is needed")
    feedback: str = Field(description="Detailed feedback about the response quality")


class EnhancedAgentState(MessagesState):
    """Enhanced state for the RAG agent with validation"""
    user_id: Optional[str] = None
    search_attempts: int = 0
    max_search_attempts: int = 2
    last_retrieval_result: Optional[str] = None
    validation_history: List[Dict[str, Any]] = []


def create_langgraph_agent(user_id: str = None, query: str = None):
    """
    Create a LangGraph-based React agent with self-validation for FAQ responses.
    
    Args:
        user_id (str): The user ID for personalized support
        query (str): Optional initial query (for context)
    
    Returns:
        Compiled LangGraph agent with validation capabilities
    """
    try:
        # Initialize the language model
        model = ChatGoogleGenerativeAI(
            model=config.model_name,
            google_api_key=config.google_api_key,
            temperature=0.3
        )
        
        logger.info(f"Using Google Gemini model for LangGraph agent: {config.model_name}")

        # Enhanced FAQ tool with validation context
        @tool
        def faq_and_support(query: str) -> str:
            """
            Search FAQ database and support documentation for customer queries. 
            Use for:
            - General questions about services, features, pricing
            - How-to guides and tutorials
            - Technical support issues
            - Policy and terms questions
            - Troubleshooting assistance
            - Account-related general questions
            """
            return document_retrieval_tool.invoke({"query": query, "tab_filter": None})

        @tool 
        def user_book_lookup(user_id: str) -> str:
            """
            Look up specific user's book information and account details using their user ID.
            Use for:
            - "My books" or "my account" queries
            - Book status inquiries
            - Writing progress questions
            - Personal book recommendations
            - Account-specific issues
            - Book-related support for this specific user
            """
            return user_book_summary_tool.invoke({"user_id": user_id})

        # Define tools for the agent
        tools = [faq_and_support, user_book_lookup]
        
        # Create the main React agent with enhanced prompt
        react_agent = create_react_agent(
            model=model,
            tools=tools,
            checkpointer=MemorySaver(),  # For conversation persistence
            prompt=f"""You are a smart, helpful customer care representative for a book/writing platform. You provide excellent customer support by combining conversation history with real-time information lookup.

**CUSTOMER INFORMATION:**
- Current Customer ID: {user_id}
- Session: Customer Care Support with Enhanced FAQ Validation

**YOUR CAPABILITIES:**
1. **FAQ & General Support**: Answer questions about services, features, policies, and troubleshooting
2. **Personal Account Support**: Look up specific user's books, account status, and personalized information. Always use the provided user_id: {user_id}

**INTELLIGENT DECISION MAKING:**

**Use `faq_and_support` tool for:**
- General questions: "How does this work?", "What are your prices?", "How do I..."
- Technical issues: "I can't log in", "The app is slow", "Feature not working"
- Policy questions: "What's your refund policy?", "Terms of service"
- General troubleshooting and how-to guides

**Use `user_book_lookup` tool for:**
- Personal queries: "My books", "My account", "My writing progress"
- Specific user issues: "Where is my book?", "My book status"
- Account-specific support: "My subscription", "My purchases"
- Use the user_id: {user_id}

**CUSTOMER CARE EXCELLENCE:**
1. **Be Empathetic**: Acknowledge customer concerns and frustrations
2. **Be Proactive**: Anticipate follow-up questions and provide comprehensive answers
3. **Be Personal**: Use the customer's information when relevant
4. **Be Solution-Oriented**: Always try to provide actionable solutions
5. **Be Clear**: Use simple, easy-to-understand language

**TONE & STYLE:**
- Friendly, professional, and helpful
- Use "I'll help you with that" rather than "I can help"
- Express genuine care for resolving their issues
- Be conversational but maintain professionalism
- Use positive language and avoid negative phrases

**ERROR HANDLING:**
- If no information is found, apologize and offer alternatives
- Escalate complex issues when appropriate
- Always maintain a helpful, solution-focused attitude
- Suggest contacting human support if needed

Remember: You're providing an excellent customer experience by being intelligent, contextual, and genuinely helpful."""
        )

        logger.info(f"Successfully created LangGraph React agent for user {user_id}")
        return react_agent

    except Exception as e:
        logger.error(f"Error creating LangGraph React agent for user {user_id}: {e}")
        raise


def create_langgraph_agent_graph(user_id: str = None, query: str = None):
    """
    Create a full LangGraph StateGraph with validation workflow.
    This provides more control over the validation process.
    """
    try:
        # Initialize models
        model = ChatGoogleGenerativeAI(
            model=config.model_name,
            google_api_key=config.google_api_key,
            temperature=0.3
        )
        
        validation_model = model.with_structured_output(ValidationResult)
        
        # Define tools
        @tool
        def faq_and_support(query: str) -> str:
            """Search FAQ database and support documentation for customer queries."""
            return document_retrieval_tool.invoke({"query": query, "tab_filter": None})

        @tool 
        def user_book_lookup(user_id: str) -> str:
            """Look up specific user's book information and account details."""
            return user_book_summary_tool.invoke({"user_id": user_id})

        tools = [faq_and_support, user_book_lookup]
        model_with_tools = model.bind_tools(tools)
        
        # Define nodes
        def agent_node(state: EnhancedAgentState) -> EnhancedAgentState:
            """Main agent reasoning node"""
            response = model_with_tools.invoke(state["messages"])
            return {"messages": [response]}
        
        def tools_node(state: EnhancedAgentState) -> EnhancedAgentState:
            """Execute tools"""
            tool_node = ToolNode(tools)
            result = tool_node.invoke(state)
            return result
        
        def validate_node(state: EnhancedAgentState) -> EnhancedAgentState:
            """Validate FAQ responses"""
            if state.get("search_attempts", 0) >= state.get("max_search_attempts", 2):
                return state
                
            messages = state["messages"]
            if len(messages) < 2:
                return state
                
            # Find last FAQ tool usage
            last_tool_result = None
            original_question = None
            
            for i in range(len(messages) - 1, -1, -1):
                msg = messages[i]
                if isinstance(msg, ToolMessage) and hasattr(msg, 'name') and msg.name == "faq_and_support":
                    last_tool_result = msg.content
                    break
                elif isinstance(msg, HumanMessage):
                    original_question = msg.content
                    
            if not last_tool_result or not original_question:
                return state
                
            # Perform validation
            validation_prompt = f"""
            Original Question: {original_question}
            Retrieved Information: {last_tool_result}
            
            Evaluate if the retrieved information adequately answers the original question.
            Consider it insufficient if it says "couldn't find" or doesn't match the topic.
            """
            
            try:
                validation_result = validation_model.invoke([HumanMessage(content=validation_prompt)])
                
                current_attempts = state.get("search_attempts", 0)
                max_attempts = state.get("max_search_attempts", 2)
                
                if validation_result.needs_retry and current_attempts < max_attempts:
                    state["search_attempts"] = current_attempts + 1
                    retry_query = validation_result.suggested_query or original_question
                    
                    # Add retry message
                    retry_message = HumanMessage(
                        content=f"Search again with: {retry_query}"
                    )
                    state["messages"].append(retry_message)
                    
            except Exception as e:
                logger.error(f"Validation error: {e}")
                
            return state
        
        def should_continue(state: EnhancedAgentState) -> Literal["tools", "validate", "end"]:
            """Router function"""
            last_message = state["messages"][-1]
            
            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                return "tools"
            elif (isinstance(last_message, ToolMessage) and 
                  hasattr(last_message, 'name') and 
                  last_message.name == "faq_and_support" and
                  state.get("search_attempts", 0) < state.get("max_search_attempts", 2)):
                return "validate"
            else:
                return "end"
        
        # Build graph
        workflow = StateGraph(EnhancedAgentState)
        
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tools_node)
        workflow.add_node("validate", validate_node)
        
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                "validate": "validate", 
                "end": END
            }
        )
        workflow.add_edge("tools", "validate")
        workflow.add_edge("validate", "agent")
        
        # Compile with checkpointer for persistence
        graph = workflow.compile(checkpointer=MemorySaver())
        
        logger.info(f"Successfully created LangGraph StateGraph for user {user_id}")
        return graph
        
    except Exception as e:
        logger.error(f"Error creating LangGraph StateGraph for user {user_id}: {e}")
        raise
