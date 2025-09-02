from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from typing import List, Optional, Callable
from config import Config


class VectorStoreManager:
    def __init__(self, config: Config):
        self.config = config
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=config.embedding_model,
            google_api_key=config.google_api_key
        )
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=config.pinecone_api_key)
        
        # Create or connect to index
        existing_indexes = self.pc.list_indexes().names()
        if config.index_name not in existing_indexes:
            self.pc.create_index(
                name=config.index_name,
                dimension=config.dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
        
        self.index = self.pc.Index(config.index_name)
    
    def store_documents(self, documents: List[Document]) -> PineconeVectorStore:
        """Store documents in Pinecone with enhanced metadata"""
        # Filter metadata to ensure it's compatible with Pinecone
        filtered_docs = filter_complex_metadata(documents)
        
        # Create vector store
        vector_store = PineconeVectorStore.from_documents(
            filtered_docs,
            self.embeddings,
            index_name=self.config.index_name
        )
        
        return vector_store
    
    def get_retriever(self, vector_store: PineconeVectorStore) -> Callable:
        """Create a custom retriever with metadata filtering"""
        def tab_aware_retriever(query: str, tab_filter: Optional[str] = None) -> List[Document]:
            # Build filter
            filter_dict = {}
            if tab_filter:
                filter_dict["tab"] = {"$eq": tab_filter}
            
            # Retrieve documents
            docs = vector_store.similarity_search(
                query, 
                k=self.config.top_k,
                filter=filter_dict
            )
            
            # If no results with filter, try without filter
            if not docs and tab_filter:
                docs = vector_store.similarity_search(query, k=self.config.top_k)
            
            return docs
        
        return tab_aware_retriever


class EnhancedRetriever:
    def __init__(self, vector_store: PineconeVectorStore, config: Config):
        self.vector_store = vector_store
        self.config = config
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.output_parsers import JsonOutputParser
        from langchain.prompts import ChatPromptTemplate
        
        self.llm = ChatGoogleGenerativeAI(
            model=config.model_name,
            google_api_key=config.google_api_key
        )
    
    def multi_query_retrieval(self, query: str, tab_filter: Optional[str] = None) -> List[Document]:
        """Generate multiple query variants for better retrieval"""
        from langchain.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import JsonOutputParser
        
        prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant. Generate multiple different ways to express the following query
        for searching a document database. Create 3 different variations that might retrieve
        relevant information.
        
        Original query: {query}
        
        Return a JSON array of query variations.
        """)
        
        chain = prompt | self.llm | JsonOutputParser()
        variations = chain.invoke({"query": query})
        
        # Retrieve documents for each variation
        all_docs = []
        for variation in variations:
            docs = self.vector_store.similarity_search(
                variation, 
                k=self.config.top_k // 2,  # Get fewer docs per variation
                filter={"tab": tab_filter} if tab_filter else None
            )
            all_docs.extend(docs)
        
        # Remove duplicates
        seen_content = set()
        unique_docs = []
        for doc in all_docs:
            if doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                unique_docs.append(doc)
        
        return unique_docs[:self.config.top_k]  # Return top K documents
    
    def hybrid_retrieval(self, query: str, tab_filter: Optional[str] = None) -> List[Document]:
        """Combine semantic search with keyword-based search"""
        # Semantic search
        semantic_docs = self.vector_store.similarity_search(
            query, 
            k=self.config.top_k,
            filter={"tab": tab_filter} if tab_filter else None
        )
        
        # Keyword-based search (simple implementation)
        keyword_docs = []
        if tab_filter:
            # Get all documents from the tab and filter by keyword matches
            all_tab_docs = self.vector_store.similarity_search(
                "",  # Empty query to get all docs
                k=100,  # Limit for performance
                filter={"tab": tab_filter}
            )
            
            # Simple keyword matching
            query_keywords = set(query.lower().split())
            for doc in all_tab_docs:
                content = doc.page_content.lower()
                if any(keyword in content for keyword in query_keywords):
                    keyword_docs.append(doc)
        
        # Combine and deduplicate
        combined_docs = semantic_docs + keyword_docs
        seen_content = set()
        unique_docs = []
        for doc in combined_docs:
            if doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                unique_docs.append(doc)
        
        return unique_docs[:self.config.top_k]
