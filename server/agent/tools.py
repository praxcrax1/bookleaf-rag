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
from typing import List
from collections import defaultdict

# Instantiate config and vector store manager
config = Config()
vector_store_manager = VectorStoreManager(config)

def _multi_size_retrieval(vector_store, query: str, tab_filter: str | None) -> List:
    all_docs = []
    base_filter = {}
    if tab_filter:
        base_filter["tab"] = {"$eq": tab_filter}
    large_filter = {**base_filter, "chunk_type": {"$eq": "large_context"}}
    large_docs = vector_store.similarity_search(query, k=3, filter=large_filter if large_filter != base_filter else None)
    detailed_filter = {**base_filter, "chunk_type": {"$eq": "detailed"}}
    detailed_docs = vector_store.similarity_search(query, k=5, filter=detailed_filter if detailed_filter != base_filter else None)
    medium_filter = {**base_filter, "chunk_type": {"$eq": "medium_context"}}
    medium_docs = vector_store.similarity_search(query, k=4, filter=medium_filter if medium_filter != base_filter else None)
    return large_docs + medium_docs + detailed_docs

def _entity_cross_reference(vector_store, docs: List, tab_filter: str | None) -> List:
    entities_found = set()
    topics_found = set()
    for doc in docs:
        if hasattr(doc, 'metadata'):
            entities_found.update(doc.metadata.get('entities', []))
            topics_found.update(doc.metadata.get('topics', []))
    related_docs = []
    for entity in list(entities_found)[:5]:
        entity_docs = vector_store.similarity_search(
            entity, k=2,
            filter={"entities": {"$in": [entity]}} if not tab_filter else {"entities": {"$in": [entity]}, "tab": {"$eq": tab_filter}}
        )
        related_docs.extend(entity_docs)
    for topic in list(topics_found)[:3]:
        topic_docs = vector_store.similarity_search(
            topic, k=2,
            filter={"topics": {"$in": [topic]}} if not tab_filter else {"topics": {"$in": [topic]}, "tab": {"$eq": tab_filter}}
        )
        related_docs.extend(topic_docs)
    return docs + related_docs

def _deduplicate_and_rank(docs: List) -> List:
    seen_content = set()
    unique_docs = []
    for doc in docs:
        content_hash = hash(doc.page_content[:200])
        if content_hash not in seen_content:
            seen_content.add(content_hash)
            unique_docs.append(doc)
    def sort_key(doc):
        chunk_type = doc.metadata.get('chunk_type', 'unknown')
        type_priority = {'large_context': 1, 'medium_context': 2, 'detailed': 3}
        return (type_priority.get(chunk_type, 4), -len(doc.page_content))
    return sorted(unique_docs, key=sort_key)[:8]

def _format_scattered_content(docs: List, query: str) -> str:
    if not docs:
        return "No relevant information found in our knowledge base."
    
    content_groups = defaultdict(list)
    overview_content = []
    
    for doc in docs:
        chunk_type = doc.metadata.get('chunk_type', 'unknown')
        topics = doc.metadata.get('topics', ['general'])
        if chunk_type == 'large_context':
            overview_content.append(doc)
        primary_topic = topics[0] if topics else 'general'
        content_groups[primary_topic].append(doc)
    
    result = "Based on our knowledge base, here's the information I found:\n\n"
    
    if overview_content:
        result += "**Overview:**\n"
        for doc in overview_content[:2]:
            result += f"{doc.page_content.strip()}\n\n"
    
    for topic, topic_docs in content_groups.items():
        if topic != 'general' and len(topic_docs) > 1:
            result += f"**{topic.title()}:**\n"
            for doc in topic_docs[:3]:
                result += f"{doc.page_content.strip()}\n\n"
    
    # Add a disclaimer
    result += "\n*This information is based on our current documentation. For the most up-to-date information or if you need clarification, please contact our support team.*"
    
    return result

@tool
def document_retrieval_tool(query: str, tab_filter: str | None = None) -> str:
    """Advanced retrieval for scattered knowledge across FAQ documents. ONLY returns information found in documents."""
    vector_store = PineconeVectorStore.from_existing_index(
        index_name=config.index_name,
        embedding=vector_store_manager.embeddings
    )
    results = _multi_size_retrieval(vector_store, query, tab_filter)
    enhanced_results = _entity_cross_reference(vector_store, results, tab_filter)
    final_docs = _deduplicate_and_rank(enhanced_results)
    if not final_docs:
        return "I couldn't find any relevant information about that topic in our knowledge base. Please contact our support team for assistance."
    return _format_scattered_content(final_docs, query)

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

