from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Optional, List
import uvicorn
import logging
from langchain_pinecone import PineconeVectorStore
from config import config
from document_processor import DocumentProcessor
from vector_store import VectorStoreManager
from validation_agent import ValidationAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentQASystem:
    def __init__(self, config):
        self.config = config
        self.processor = DocumentProcessor(config)
        self.vector_store_manager = VectorStoreManager(config)
        self.vector_store = None
        self.retriever = None
        self.agent = None
        self._initialized = False
    
    def check_existing_documents(self) -> bool:
        """Check if documents already exist in the vector store"""
        try:
            # Query the index to see if it has any documents
            index_stats = self.vector_store_manager.index.describe_index_stats()
            total_vector_count = index_stats.get('total_vector_count', 0)
            logger.info(f"Found {total_vector_count} existing documents in vector store")
            return total_vector_count > 0
        except Exception as e:
            logger.error(f"Error checking existing documents: {str(e)}")
            return False
    
    def initialize_from_existing_store(self):
        """Initialize system using existing documents in the vector store"""
        try:
            logger.info("Initializing from existing vector store...")
            # Create vector store connection
            self.vector_store = PineconeVectorStore.from_existing_index(
                index_name=self.config.index_name,
                embedding=self.vector_store_manager.embeddings
            )
            self.retriever = self.vector_store_manager.get_retriever(self.vector_store)
            
            logger.info("Initializing agent...")
            self.agent = ValidationAgent(config, self.retriever, self.vector_store)
            
            self._initialized = True
            logger.info("System initialized successfully from existing store!")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing from existing store: {str(e)}")
            return False
    
    def initialize_with_document(self, doc_url: str):
        """Initialize the complete system with a new document"""
        try:
            logger.info("Loading and processing new document...")
            documents = self.processor.extract_google_doc(doc_url)
            
            logger.info("Processing and chunking document...")
            chunks = self.processor.hierarchical_chunking(documents)
            enhanced_chunks = self.processor.enhance_metadata(chunks)
            
            logger.info("Storing in vector database...")
            self.vector_store = self.vector_store_manager.store_documents(enhanced_chunks)
            self.retriever = self.vector_store_manager.get_retriever(self.vector_store)
            
            logger.info("Initializing agent...")
            self.agent = ValidationAgent(config, self.retriever, self.vector_store)
            
            self._initialized = True
            logger.info("System initialized successfully with new document!")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing system with document: {str(e)}")
            return False
    
    def auto_initialize(self) -> bool:
        """Auto-initialize the system if possible"""
        if self._initialized:
            return True
            
        # First, check if documents already exist in the store
        if self.check_existing_documents():
            return self.initialize_from_existing_store()
        
        return False
    
    def query(self, question: str) -> str:
        """Query the system - auto-initialize if needed"""
        # Try to auto-initialize if not already initialized
        if not self._initialized:
            if not self.auto_initialize():
                return "System not initialized. Please initialize with a document first using the /initialize endpoint."
        
        if not self.agent:
            return "System initialization failed. Please try initializing with a document."
        
        try:
            return self.agent.invoke(question)
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"I apologize, but I encountered an error while processing your query: {str(e)}"
    
    @property
    def is_initialized(self) -> bool:
        """Check if the system is properly initialized"""
        return self._initialized and self.agent is not None

app = FastAPI(
    title="RAG Document Q&A System",
    description="A robust RAG system with Pinecone, Gemini, and LangGraph for document Q&A",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global QA system instance
qa_system = DocumentQASystem(config)

# Pydantic models for API
class InitializeRequest(BaseModel):
    doc_url: HttpUrl
    
class QueryRequest(BaseModel):
    question: str
    retrieval_method: Optional[str] = "standard"  # standard, multi_query, hybrid, enhanced, debug

class QueryResponse(BaseModel):
    answer: str
    success: bool
    error: Optional[str] = None

class InitializeResponse(BaseModel):
    success: bool
    message: str
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    initialized: bool


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    # Auto-initialize if possible
    qa_system.auto_initialize()
    
    return HealthResponse(
        status="healthy",
        initialized=qa_system.is_initialized
    )


@app.post("/initialize", response_model=InitializeResponse)
async def initialize_system(request: InitializeRequest):
    """Initialize the RAG system with a Google Document"""
    try:
        logger.info(f"Initializing system with document: {request.doc_url}")
        
        success = qa_system.initialize_with_document(str(request.doc_url))
        
        if success:
            return InitializeResponse(
                success=True,
                message="System initialized successfully! You can now query the document."
            )
        else:
            return InitializeResponse(
                success=False,
                message="Failed to initialize system",
                error="Document processing failed"
            )
        
    except Exception as e:
        logger.error(f"Error initializing system: {str(e)}")
        return InitializeResponse(
            success=False,
            message="Failed to initialize system",
            error=str(e)
        )


@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """Query the document with a question"""
    try:
        logger.info(f"Processing query: {request.question}")
        
        if request.retrieval_method == "enhanced":
            # Use enhanced retrieval for debugging/analysis
            from vector_store import EnhancedRetriever
            # Auto-initialize if not already done
            if not qa_system.is_initialized:
                qa_system.auto_initialize()
                
            if qa_system.vector_store:
                enhanced_retriever = EnhancedRetriever(qa_system.vector_store, config)
                docs = enhanced_retriever.multi_query_retrieval(request.question)
                answer = f"Retrieved {len(docs)} documents using {request.retrieval_method} method"
            else:
                answer = "Enhanced retrieval requires vector store to be initialized"
                
        elif request.retrieval_method == "debug":
            # Debug mode: show retrieval details
            from enhanced_retriever import PackageAwareRetriever
            # Auto-initialize if not already done
            if not qa_system.is_initialized:
                qa_system.auto_initialize()
                
            if qa_system.vector_store:
                enhanced_retriever = PackageAwareRetriever(qa_system.vector_store, config)
                docs = enhanced_retriever.retrieve_with_context(request.question)
                explanation = enhanced_retriever.get_retrieval_explanation(request.question, docs)
                answer = f"Debug Info:\n{explanation}"
            else:
                answer = "Debug mode requires vector store to be initialized"
        else:
            # Standard query processing with auto-initialization
            answer = qa_system.query(request.question)
        
        return QueryResponse(
            answer=answer,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return QueryResponse(
            answer="",
            success=False,
            error=str(e)
        )


@app.get("/status")
async def get_status():
    """Get system status and configuration"""
    # Auto-initialize if possible
    qa_system.auto_initialize()
    
    return {
        "initialized": qa_system.is_initialized,
        "has_existing_documents": qa_system.check_existing_documents(),
        "config": {
            "index_name": config.index_name,
            "chunk_size": config.chunk_size,
            "top_k": config.top_k,
            "confidence_threshold": config.confidence_threshold
        }
    }


@app.post("/reset")
async def reset_system():
    """Reset the system (clear initialization)"""
    qa_system._initialized = False
    qa_system.vector_store = None
    qa_system.retriever = None
    qa_system.agent = None
    
    return {
        "success": True,
        "message": "System reset successfully. Re-initialization will happen automatically on next query."
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
