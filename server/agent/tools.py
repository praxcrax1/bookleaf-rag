"""
Tool wrappers for LangChain agent integration.
Wraps the validation agent and other tools for use in the main agent.
"""
from typing_extensions import Annotated
from langchain_core.tools import tool, InjectedToolArg
from validation.validation_agent import ValidationAgent
from vector_store.manager import VectorStoreManager
from database.mongo import get_user_book_summary
from config import Config
from langchain_pinecone import PineconeVectorStore
import asyncio
# Instantiate config and vector store manager
config = Config()
vector_store_manager = VectorStoreManager(config)

# Create or connect to the Pinecone vector store (LangChain object)
vector_store = PineconeVectorStore.from_existing_index(
    index_name=config.index_name,
    embedding=vector_store_manager.embeddings
)

# Explicit retriever function matching previous working logic

def retriever(query, tab_filter=None):
    filter_dict = {}
    if tab_filter:
        filter_dict["tab"] = {"$eq": tab_filter}
    return vector_store.similarity_search(query, k=config.top_k, filter=filter_dict)

# Instantiate validation agent (RAG tool)
validation_agent = ValidationAgent(config, retriever, vector_store)

@tool
def document_retrieval_tool(query: str, tab_filter: str = None) -> str:
    """Search and validate documents using the RAG validation agent. Use for queries requiring document retrieval and validation."""
    return validation_agent.invoke(query)

@tool
def user_book_summary_tool(
    user_id: Annotated[str, InjectedToolArg]
) -> str:
    """Get a comprehensive summary of all books for a specific user ID. Returns book details including titles, statuses, and stage notes. Use when users ask about their books, writing progress, or book status."""
    try:
        # Since this is an async function, we need to run it in an event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(get_user_book_summary(user_id))
        loop.close()
        
        # Format the result for the agent
        if "error" in result:
            return f"Error retrieving book summary for user {user_id}: {result['error']}"
        
        summary = result['summary']
        books_info = []
        
        for book in result['books']:
            book_info = f"- **{book['title']}** (ID: {book['book_id']}) - Status: {book['status']}"
            if book['stage_notes']:
                book_info += f"\n  Notes: {book['stage_notes']}"
            books_info.append(book_info)
        
        response = f"## Book Summary for User {user_id}\n\n"
        response += f"{summary}\n\n"
        
        if books_info:
            response += "### Book Details:\n"
            response += "\n".join(books_info)
        
        return response
        
    except Exception as e:
        return f"Failed to retrieve book summary for user {user_id}: {str(e)}"


# Tools list for the agent
TOOLS = [document_retrieval_tool, user_book_summary_tool]
