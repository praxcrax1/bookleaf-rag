"""
Tool wrappers for LangChain agent integration.
Wraps the validation agent and other tools for use in the main agent.
"""
from langchain.tools import Tool
from validation_agent import ValidationAgent
from vector_store import VectorStoreManager
from config import Config
from langchain_pinecone import PineconeVectorStore

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

def validation_rag_tool(query: str, tab_filter: str = None):
    """Use the validation agent to retrieve documents."""
    return validation_agent.invoke(query)

# Define LangChain Tool
validation_rag = Tool(
    name="validation_rag",
    func=validation_rag_tool,
    description=(
        "Search and validate documents using the RAG validation agent. "
        "Use for queries requiring document retrieval and validation."
    )
)

# Add more tools here as needed
TOOLS = [validation_rag]
