"""
Tool wrappers for LangChain agent integration.
Wraps the validation agent and other tools for use in the main agent.
"""
from typing_extensions import Annotated
from langchain_core.tools import tool, InjectedToolArg
from vector_store.manager import VectorStoreManager
from database.mongo import get_user_book_summary
from config import Config
from langchain_pinecone import PineconeVectorStore

# Instantiate config and vector store manager
config = Config()
vector_store_manager = VectorStoreManager(config)


@tool
def document_retrieval_tool(query: str, tab_filter: str | None = None) -> str:
    """Search documents directly from the vector store. Use for queries requiring document retrieval."""
    # Create or connect to the Pinecone vector store
    vector_store = PineconeVectorStore.from_existing_index(
        index_name=config.index_name,
        embedding=vector_store_manager.embeddings
    )
    
    # Direct retrieval from vector store
    filter_dict = {}
    if tab_filter:
        filter_dict["tab"] = {"$eq": tab_filter}
    
    # Get relevant documents
    docs = vector_store.similarity_search(
        query, 
        k=config.top_k,
        filter=filter_dict if filter_dict else None
    )
    
    # Format the retrieved documents for the agent
    if not docs:
        return "I couldn't find any relevant information about that topic."
    
    result = "Here's what I found in our knowledge base:\n\n"
    for i, doc in enumerate(docs, 1):
        result += f"Document {i}:\n{doc.page_content}\n\n"
    
    return result
@tool
def user_book_summary_tool(
    user_id: Annotated[str, InjectedToolArg]
) -> str:
    """Get a comprehensive summary of all books for a specific user ID. Returns book details including titles, statuses, and stage notes. Use when users ask about their books, writing progress, or book status."""
    try:
        # Simple synchronous call - no async complications!
        result = get_user_book_summary(user_id)
        
        # Handle the case where user has no books
        if "error" in result:
            return f"I encountered an issue retrieving your book information: {result['error']}"
        
        # Check if user has no books
        if result['total_books'] == 0:
            return "You currently don't have any books in your account. You can start by uploading a document or creating a new book project!"
        
        summary = result['summary']
        books_info = []
        
        for book in result['books']:
            book_info = f"- **{book['title']}** (ID: {book['book_id']}) - Status: {book['status']}"
            if book['stage_notes']:
                book_info += f"\n  Notes: {book['stage_notes']}"
            books_info.append(book_info)
        
        response = f"## Your Book Summary\n\n"
        response += f"{summary}\n\n"
        
        if books_info:
            response += "### Your Books:\n"
            response += "\n".join(books_info)
        
        return response
        
    except Exception as e:
        return f"I'm sorry, but I couldn't retrieve your book information at the moment. Please try again later or contact support if the issue persists."

