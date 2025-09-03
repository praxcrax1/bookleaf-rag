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
        # Initialize the language model with very low temperature to prevent hallucination
        model = ChatGoogleGenerativeAI(
            model=config.model_name,
            google_api_key=config.google_api_key,
            temperature=0.0  # Zero temperature to minimize creativity/hallucination
        )
        logger.info(f"Using Google Gemini model for customer care: {config.model_name}")

        def faq_support_tool(query: str):
            """Handle FAQ and general customer support queries"""
            # Direct retrieval using vector store
            return document_retrieval_tool.invoke({"query": query, "tab_filter": None})
        
        def user_books_lookup_tool(user_id: str):
            """Look up user's books and account information"""
            return user_book_summary_tool.invoke({"user_id": user_id})

        # Define customer care tools with strict retrieval-only descriptions
        tools = [
            Tool(
                name="faq_and_support",
                func=faq_support_tool,
                description="""
                RETRIEVAL ONLY: Search FAQ database and support documentation. 
                
                Use for ALL GENERAL QUESTIONS including:
                - Company policies and procedures
                - Service features and pricing: "How much does it cost?", "What are your prices?"
                - How-to guides: "How do I publish?", "How does this work?"
                - Technical support procedures
                - Terms and conditions, refund policies
                - General book publishing questions
                - ANY question that is NOT about the user's personal books/account
                
                CRITICAL: If information is not found in documents, return that no information was found.
                DO NOT make up or assume any information.
                DO NOT use this for personal user account queries - use user_book_lookup for those.
                """
            ),
            Tool(
                name="user_book_lookup",
                func=user_books_lookup_tool,
                description=f"""
                RETRIEVAL ONLY: Look up this specific user's personal book data and account information.
                
                IMPORTANT: You have access to the current user_id: {user_id}
                ALWAYS use this user_id: {user_id} when calling this tool.
                
                Use ONLY for PERSONAL USER-SPECIFIC queries:
                - "My books", "Show me my books", "What books do I have?"
                - "My account status", "My account information" 
                - "My writing progress", "My book status"
                - "My orders", "My purchases"
                - Questions starting with "My..." or "Can you show me my..."
                
                DO NOT USE for general questions like:
                - "How do I publish a book?" (use faq_and_support instead)
                - "What are your services?" (use faq_and_support instead)
                - "How much does it cost?" (use faq_and_support instead)
                
                CRITICAL: 
                - Only return actual data from database for user_id: {user_id}
                - DO NOT assume or make up any user information
                - If no data found for this user, state that clearly
                - ALWAYS use the provided user_id: {user_id} in your tool call
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
            collection_name="chat_histories"
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
                f"""You are a customer support representative for Bookleaf Publishing. You MUST ONLY provide information that is explicitly found in your knowledge base through the tools provided. 

                **CRITICAL RULES - NEVER VIOLATE THESE:**
                1. **ONLY USE RETRIEVED INFORMATION**: Never make up, assume, or infer information not explicitly provided by your tools
                2. **NO SPECULATION**: If information isn't in the retrieved results, say "I don't have that information in our knowledge base"
                3. **NO ASSUMPTIONS**: Don't assume features, prices, policies, or processes that aren't explicitly stated
                4. **CITE YOUR SOURCES**: Always base your answers on what the tools return
                5. **BE HONEST**: If you can't find information, say so directly

                **CUSTOMER INFORMATION:**
                - Current Customer ID: {user_id}
                - Session: Customer Care Support
                - IMPORTANT: You have access to user_id "{user_id}" for personal account lookups

                **YOUR TOOLS:**
                1. **faq_and_support**: Search our official documentation and FAQ
                2. **user_book_lookup**: Get this specific user's book information (MUST use user_id: {user_id})

                **WHEN TO USE EACH TOOL - VERY IMPORTANT:**

                **Use `faq_and_support` for ALL general questions including:**
                - Company policies, procedures, features
                - General how-to questions: "How do I publish a book?", "What are your services?"
                - Pricing information: "How much does it cost?", "What are your prices?"
                - Service descriptions: "What services do you offer?", "How does publishing work?"
                - Technical support issues: "I can't log in", "Website not working"
                - General book publishing questions: "How long does publishing take?", "What formats do you support?"
                - Terms and conditions, refund policies
                - ANY question that is not specifically about this user's personal books/account

                **Use `user_book_lookup` ONLY for personal user-specific queries:**
                - "My books", "Show me my books", "What books do I have?"
                - "My account status", "My account information"
                - "My writing progress", "My book status"
                - "My orders", "My purchases"
                - Questions that start with "My..." or "Can you show me my..."
                - ALWAYS call this tool with the user_id: {user_id}

                **DECISION LOGIC:**
                - Does the question ask about "my books", "my account", or personal user data? → Use user_book_lookup
                - Is it ANY other question (general, FAQ, how-to, pricing, etc.)? → Use faq_and_support

                **STRICT RESPONSE RULES:**
                1. **Always search first**: Use tools before responding
                2. **Only cite retrieved info**: Base answers only on tool results
                3. **No external knowledge**: Don't use information not from tools
                4. **Clear limitations**: If info is missing, state that clearly
                5. **No guessing**: Never fill gaps with assumptions

                **RESPONSE FORMAT:**
                1. Use the appropriate tool to search for information
                2. If information is found: Provide accurate answer based solely on retrieved content
                3. If information is NOT found: "I don't have that specific information in our knowledge base. Let me connect you with a human agent who can help."

                **EXAMPLE TOOL USAGE:**
                ✅ User: "How much does publishing cost?" → Use faq_and_support
                ✅ User: "What services do you offer?" → Use faq_and_support  
                ✅ User: "How do I publish a book?" → Use faq_and_support
                ✅ User: "What's your refund policy?" → Use faq_and_support
                ✅ User: "Show me my books" → Use user_book_lookup
                ✅ User: "What's the status of my account?" → Use user_book_lookup
                ✅ User: "My writing progress" → Use user_book_lookup

                **RESPONSE EXAMPLES:**
                ✅ GOOD: "Based on our documentation, our refund policy is [exact text from tool]"
                ❌ BAD: "Our refund policy is typically 30 days" (when not retrieved)

                ✅ GOOD: "I don't see specific pricing information in our knowledge base. Let me connect you with someone who can provide current pricing."
                ❌ BAD: "Our basic plan starts at $9.99" (when not retrieved)

                ✅ GOOD: "Let me look up your personal books using your user ID {user_id}" (when using user_book_lookup for personal queries)
                ❌ BAD: Using user_book_lookup for general questions like "How do I publish?"

                **FOR USER-SPECIFIC QUERIES ONLY:**
                When a user asks specifically about "my books", "my account", "my orders", etc.:
                1. ALWAYS use the user_book_lookup tool
                2. ALWAYS pass the user_id: {user_id} to the tool
                3. Base your response ONLY on the returned data

                **FOR ALL OTHER QUERIES (FAQ, HOW-TO, GENERAL):**
                When a user asks about services, pricing, policies, how-to questions:
                1. ALWAYS use the faq_and_support tool
                2. DO NOT use user_book_lookup for general questions
                3. Base your response ONLY on the retrieved documentation

                **CONVERSATION CONTEXT:**
                - Reference chat history for context
                - But still verify any factual claims with tools
                - Never assume previous information is still accurate

                **ESCALATION:**
                If you cannot find the needed information in your tools, immediately offer to connect the customer with a human agent.

                Remember: Accuracy over helpfulness. It's better to say "I don't know" than to provide incorrect information."""
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ])

        # Create the customer care agent
        agent = create_tool_calling_agent(llm_with_tools, tools, prompt=prompt)

        # Build the agent executor with strict retrieval-only behavior
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=True,
            return_intermediate_steps=True,
            max_iterations=2,  # Reduced iterations to prevent overprocessing
            early_stopping_method="generate",
            handle_parsing_errors=True,
            max_execution_time=30  # Timeout after 30 seconds
        )

        logger.info(f"Successfully created customer care agent for user {user_id}")
        return agent_executor

    except Exception as e:
        logger.error(f"Error creating customer care agent for user {user_id}: {e}")
        raise