"""
Clean LangGraph React Agent implementation for RAG system.
This is the single agent implementation using LangGraph with React pattern.
"""
from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from .tools import user_book_summary_tool, document_retrieval_tool
from config import Config
import logging

logger = logging.getLogger(__name__)
config = Config()

def create_agent(user_id: str = None) -> Any:
    """
    Create a LangGraph React agent for customer support.
    
    Args:
        user_id (str): The user ID for personalized support
    
    Returns:
        LangGraph React agent
    """
    try:
        # Initialize the language model
        model = ChatGoogleGenerativeAI(
            model=config.model_name,
            google_api_key=config.google_api_key,
            temperature=0.3
        )
        logger.info(f"Using Google Gemini model: {config.model_name}")

        def faq_and_support(query: str) -> str:
            """
            Search FAQ database and support documentation for customer queries.
            Use for:
            - General questions about services, features, pricing
            - How-to guides and tutorials
            - Technical support issues
            - Policy and terms questions
            - Troubleshooting assistance
            """
            return document_retrieval_tool.invoke({"query": query, "tab_filter": None})
        
        def user_book_lookup(user_id: str) -> str:
            """
            Look up specific user's book information and account details.
            Use for:
            - "My books" or "my account" queries
            - Book status inquiries
            - Writing progress questions
            - Personal book recommendations
            - Account-specific issues
            """
            return user_book_summary_tool.invoke({"user_id": user_id})

        # Define tools for the agent
        tools = [faq_and_support, user_book_lookup]
        
        # Create system prompt
        system_prompt = f"""You are a smart, helpful customer care representative for a book/writing platform. You provide excellent customer support using real-time information lookup.

                **CUSTOMER INFORMATION:**
                - Current Customer ID: {user_id}
                - Session: Customer Care Support

                **YOUR CAPABILITIES:**
                1. **FAQ & General Support**: Answer questions about services, features, policies, and troubleshooting
                2. **Personal Account Support**: Look up specific user's books, account status, and personalized information

                **TOOL USAGE GUIDELINES:**

                **Use `faq_and_support` for:**
                - General questions: "How does this work?", "What are your prices?", "How do I..."
                - Technical issues: "I can't log in", "The app is slow", "Feature not working"
                - Policy questions: "What's your refund policy?", "Terms of service"
                - General troubleshooting and how-to guides

                **Use `user_book_lookup` for:**
                - Personal queries: "My books", "My account", "My writing progress"
                - Specific user issues: "Where is my book?", "My book status"
                - Account-specific support: "My subscription", "My purchases"
                - Always use the provided user_id: {user_id}

                **CUSTOMER CARE EXCELLENCE:**
                1. **Be Empathetic**: Acknowledge customer concerns and frustrations
                2. **Be Proactive**: Anticipate follow-up questions and provide comprehensive answers
                3. **Be Personal**: Use the customer's information when relevant
                4. **Be Solution-Oriented**: Always try to provide actionable solutions
                5. **Be Clear**: Use simple, easy-to-understand language

                **RESPONSE STRUCTURE:**
                1. **Acknowledge**: Show you understand the customer's request
                2. **Research**: Use appropriate tools to gather information
                3. **Provide**: Give comprehensive, helpful answers
                4. **Follow-up**: Suggest next steps or ask if they need more help

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

                Remember: You're providing an excellent customer experience by being intelligent, helpful, and genuinely caring about resolving customer issues."""

        # Create the React agent
        agent = create_react_agent(
            model=model,
            tools=tools,
            prompt=system_prompt
        )

        logger.info(f"Successfully created LangGraph React agent for user {user_id}")
        return agent

    except Exception as e:
        logger.error(f"Error creating LangGraph agent for user {user_id}: {e}")
        raise


class AgentWrapper:
    """
    Wrapper to maintain API compatibility while using LangGraph agents.
    """
    
    def __init__(self, agent, user_id: str = None):
        self.agent = agent
        self.user_id = user_id
    
    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke the LangGraph agent with proper input handling.
        Maintains compatibility with the expected interface.
        """
        try:
            # Extract the input query
            if isinstance(inputs, dict):
                query = inputs.get('input', inputs.get('query', str(inputs)))
            else:
                query = str(inputs)
            
            # Execute the agent
            result = self.agent.invoke({"messages": [("user", query)]})
            
            # Extract the final response
            messages = result.get("messages", [])
            final_response = messages[-1].content if messages else "No response generated"
            
            # Format response to match expected interface
            response = {
                "input": query,
                "output": final_response,
                "intermediate_steps": []  # LangGraph handles this internally
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error in LangGraph agent wrapper: {e}")
            return {
                "input": str(inputs),
                "output": f"I encountered an error processing your request: {e}",
                "intermediate_steps": []
            }


def create_agent_executor(user_id: str = None) -> AgentWrapper:
    """
    Create a wrapped LangGraph agent that maintains API compatibility.
    
    Args:
        user_id (str): The user ID for personalized support
    
    Returns:
        AgentWrapper: Wrapped LangGraph agent with compatible interface
    """
    agent = create_agent(user_id=user_id)
    return AgentWrapper(agent, user_id=user_id)
