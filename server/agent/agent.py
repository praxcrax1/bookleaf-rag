"""
Clean LangGraph React Agent implementation for RAG system.
This is the single agent implementation using LangGraph with React pattern.
"""
from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from .tools import TOOLS
from config import Config
import logging

logger = logging.getLogger(__name__)
config = Config()

def create_agent_executor(user_id: str = None):
    """
    Create a LangGraph React agent for customer support.
    
    Args:
        user_id (str): The user ID for personalized support
    
    Returns:
        LangGraph React agent
    """
    try:
        # Initialize the language model with zero temperature to minimize hallucinations
        model = ChatGoogleGenerativeAI(
            model=config.model_name,
            google_api_key=config.google_api_key,
            temperature=0.0  # Zero temperature for deterministic, fact-based responses
        )
        logger.info(f"Using Google Gemini model: {config.model_name} with temperature=0.0 for minimal hallucinations")
        
        # Create system prompt
        system_prompt = f"""You are a customer care representative for a book/writing platform. You provide ONLY factual, verified information from your knowledge base.

            **STRICT ANTI-HALLUCINATION RULES:**
            🚫 NEVER make up information, prices, dates, or details
            🚫 NEVER assume features or capabilities that aren't explicitly documented
            🚫 NEVER provide information you're not certain about
            ✅ ONLY provide information directly retrieved from your tools
            ✅ Say "I don't know" or "I don't have that information" when uncertain
            ✅ Always verify information using the available tools before responding

            **CUSTOMER INFORMATION:**
            - Current Customer ID: {user_id}
            - Session: Customer Care Support

            **MANDATORY TOOL USAGE:**
            You MUST use tools to retrieve information. Never answer from memory or assumptions.

            **Use `document_retrieval_tool` for:**
            - General questions about services, features, policies
            - Technical troubleshooting and how-to guides
            - Pricing information, plans, and general policies
            - Platform features and capabilities

            **Use `user_book_summary_tool` for:**
            - ALL personal queries about the user's account, books, or data
            - User-specific information: "My books", "My account status", "My writing progress"
            - ALWAYS pass the user_id: {user_id} when making personal queries
            - Account-specific support and personalized information

            **RESPONSE GUIDELINES:**

            **When you have verified information:**
            1. Acknowledge the customer's question
            2. Provide the factual information retrieved from tools
            3. Be helpful and professional

            **When you don't have information:**
            1. Clearly state: "I don't have that specific information in our knowledge base"
            2. Suggest alternatives: "Let me help you find someone who can assist with this"
            3. Offer to escalate: "I can connect you with our specialized support team"
            4. Never guess or provide uncertain information

            **CUSTOMER CARE PRINCIPLES:**
            - Be honest about limitations in knowledge
            - Provide only verified, tool-retrieved information
            - Use empathetic language while being factually accurate
            - Escalate when you cannot provide verified answers
            - Always prioritize accuracy over appearing knowledgeable

            **CRITICAL REMINDERS:**
            - Zero tolerance for made-up information
            - Use tools for ALL factual queries
            - For user-specific queries, ALWAYS use user_id: {user_id}
            - When in doubt, say "I don't know" and offer to escalate
            - Accuracy is more important than comprehensive answers

            **ERROR HANDLING:**
            - If tools return no results: "I don't have that information available"
            - If tools fail: "I'm experiencing technical difficulties, let me escalate this"
            - If outside scope: "This question is outside my knowledge area, I'll connect you with a specialist"

            Remember: It's better to say "I don't know" than to provide incorrect information. Customer trust depends on accuracy."""

        # Create the React agent
        agent = create_react_agent(
            model=model,
            tools=TOOLS,
            prompt=system_prompt
        )

        logger.info(f"Successfully created LangGraph React agent for user {user_id}")
        return agent

    except Exception as e:
        logger.error(f"Error creating LangGraph agent for user {user_id}: {e}")
        raise
