"""
Agent creation logic for conversational AI with document search and memory.
Creates intelligent agents that can reason about when to use different tools and data sources.
"""
from langchain.agents import AgentExecutor, Tool, create_tool_calling_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from .tools import TOOLS
from config import Config
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import logging

logger = logging.getLogger(__name__)

config = Config()

def create_agent(user_id=None, llm_provider="gemini"):
    """
    Create a conversational AI agent with access to multiple data sources and reasoning capabilities.
    The agent can intelligently decide when to use the validation agent (RAG) and other tools.
    """
    try:
        # Initialize the language model
        model = ChatGoogleGenerativeAI(
            model=config.model_name,
            google_api_key=config.google_api_key,
            temperature=0.3
        )
        logger.info(f"Using Google Gemini model: {config.model_name}")

        # Bind tools to the model
        llm_with_tools = model.bind_tools(TOOLS)

        # Set up in-memory conversation history (can be replaced with MongoDB)
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Prompt template for intelligent reasoning and tool selection
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are an advanced AI assistant specializing in document search and validation. You have access to a RAG validation agent and other tools.\n\n"
                "**CORE DECISION FRAMEWORK:**\n"
                "1. **ANALYZE THE QUERY TYPE:**\n"
                "   - **Validation/Document Questions** → Use `validation_rag`\n"
                "   - **Other tools** → Use as appropriate\n"
                "2. **INTELLIGENT TOOL USAGE:**\n"
                "   - Use tools for relevant queries\n"
                "   - Combine tools intelligently\n"
                "3. **QUALITY STANDARDS:**\n"
                "   - Honesty Principle: If no relevant info found, say \"I don't know about this\"\n"
                "   - Step-by-step reasoning\n"
                "   - Comprehensive responses\n"
                "4. **RESPONSE FORMATTING:**\n"
                "   - Use clear Markdown formatting\n"
                "   - Distinguish between sources\n"
                "   - Provide actionable insights\n"
                "5. **MEMORY UTILIZATION:**\n"
                "   - Reference previous conversations\n"
                "   - Build context over multiple interactions\n"
                "   - Remember user preferences\n"
                "**ERROR HANDLING:**\n"
                "- If a tool fails, try alternatives\n"
                "- If no information is found, be honest and suggest alternatives\n"
                "- Always maintain a helpful tone\n"
                "**CONVERSATION STYLE:**\n"
                "- Professional yet friendly\n"
                "- Focus on being genuinely helpful\n"
                "- Provide specific, actionable information\n"
                "- Ask clarifying questions when needed\n"
                """
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ])

        # Create the agent with tool calling capability
        agent = create_tool_calling_agent(llm_with_tools, TOOLS, prompt=prompt)

        # Build and return the agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=TOOLS,
            memory=memory,
            verbose=True,
            return_intermediate_steps=True,
            max_iterations=5,
            early_stopping_method="generate",
            handle_parsing_errors=True
        )

        logger.info(f"Successfully created agent for user {user_id}")
        return agent_executor

    except Exception as e:
        logger.error(f"Error creating agent for user {user_id}: {e}")
        raise
