"""
Smart Customer Care Agent with FAQ support and user book lookup capabilities.
Provides intelligent customer support with conversation history awareness.
"""
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from .tools import user_book_summary_tool, document_retrieval_tool
from config import Config
from langchain.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
import logging

logger = logging.getLogger(__name__)
config = Config()

def create_agent(user_id=None, query=None):
    """
    Create a smart customer care agent with FAQ support and user book lookup.
    The agent maintains conversation history and provides contextual support.
    
    Args:
        user_id (str): The user ID for personalized support
        query (str): Optional initial query (for context)
    
    Returns:
        AgentExecutor: Configured customer care agent
    """
    try:
        # Initialize the language model with slightly higher temperature for more natural responses
        model = ChatGoogleGenerativeAI(
            model=config.model_name,
            google_api_key=config.google_api_key,
            temperature=0.4  # Slightly higher for more conversational responses
        )
        logger.info(f"Using Google Gemini model for customer care: {config.model_name}")

        def faq_support_tool(query: str, tab_filter: str = None):
            """Handle FAQ and general customer support queries"""
            return document_retrieval_tool.invoke({"query": query, "tab_filter": tab_filter})
        
        def user_books_lookup_tool(user_id: str):
            """Look up user's books and account information"""
            return user_book_summary_tool.invoke({"user_id": user_id})

        # Define customer care tools
        tools = [
            Tool(
                name="faq_and_support",
                func=faq_support_tool,
                description="""
                Search FAQ database and support documentation for customer queries. 
                Use for:
                - General questions about services, features, pricing
                - How-to guides and tutorials
                - Technical support issues
                - Policy and terms questions
                - Troubleshooting assistance
                - Account-related general questions
                """
            ),
            Tool(
                name="user_book_lookup",
                func=user_books_lookup_tool,
                description="""
                Look up specific user's book information and account details using their user ID.
                Use for:
                - "My books" or "my account" queries
                - Book status inquiries
                - Writing progress questions
                - Personal book recommendations
                - Account-specific issues
                - Book-related support for this specific user
                """
            )
        ]

        # Bind tools to the model
        llm_with_tools = model.bind_tools(tools)

        # Set up MongoDB message history for conversation continuity
        message_history = MongoDBChatMessageHistory(
            connection_string=config.mongo_uri,
            session_id=user_id,
            database_name=config.db_name,
            collection_name="customer_care_histories"
        )

        # Conversation memory with message history
        memory = ConversationBufferMemory(
            chat_memory=message_history,
            memory_key="chat_history",
            return_messages=True
        )

        # Enhanced customer care prompt template
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                f"""You are a smart, helpful customer care representative for a book/writing platform. You provide excellent customer support by combining conversation history with real-time information lookup.

                **CUSTOMER INFORMATION:**
                - Current Customer ID: {user_id}
                - Session: Customer Care Support

                **YOUR CAPABILITIES:**
                1. **FAQ & General Support**: Answer questions about services, features, policies, and troubleshooting
                2. **Personal Account Support**: Look up specific user's books, account status, and personalized information. Always use the provided user_id to fetch their data.

                **INTELLIGENT DECISION MAKING:**

                **Use `faq_and_support` tool for:**
                - General questions: "How does this work?", "What are your prices?", "How do I..."
                - Technical issues: "I can't log in", "The app is slow", "Feature not working"
                - Policy questions: "What's your refund policy?", "Terms of service"
                - General troubleshooting and how-to guides

                **Use `user_book_lookup` tool for:**
                - Use the user_id provided to look up their specific books and account info
                - Personal queries: "My books", "My account", "My writing progress"
                - Specific user issues: "Where is my book?", "My book status"
                - Account-specific support: "My subscription", "My purchases"

                **CONVERSATION CONTEXT AWARENESS:**
                - Always review chat history to understand the full context
                - Reference previous interactions: "As we discussed earlier..."
                - Build on previous questions and answers
                - Remember customer's specific issues and preferences
                - If a customer asks a follow-up question, connect it to the previous conversation

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

                **CONVERSATION CONTINUITY:**
                - If this is a continuing conversation, reference what you know from history
                - Connect new questions to previous context
                - Remember solutions you've already provided
                - Build rapport by remembering customer details

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

                Remember: You're not just answering questions - you're providing an excellent customer experience by being intelligent, contextual, and genuinely helpful."""
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ])

        # Create the customer care agent
        agent = create_tool_calling_agent(llm_with_tools, tools, prompt=prompt)

        # Build the agent executor with customer care optimizations
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=True,
            return_intermediate_steps=True,
            max_iterations=4,  # Slightly fewer iterations for faster response
            early_stopping_method="generate",
            handle_parsing_errors=True
        )

        logger.info(f"Successfully created customer care agent for user {user_id}")
        return agent_executor

    except Exception as e:
        logger.error(f"Error creating customer care agent for user {user_id}: {e}")
        raise