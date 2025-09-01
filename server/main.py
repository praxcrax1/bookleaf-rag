from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Optional, List
import uvicorn
import logging
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
    
    def initialize_system(self, doc_url: str):
        """Initialize the complete system with a document"""
        try:
            logger.info("Loading document...")
            documents = self.processor.extract_google_doc(doc_url)
            
            logger.info("Processing and chunking document...")
            chunks = self.processor.hierarchical_chunking(documents)
            enhanced_chunks = self.processor.enhance_metadata(chunks)
            
            logger.info("Storing in vector database...")
            self.vector_store = self.vector_store_manager.store_documents(enhanced_chunks)
            self.retriever = self.vector_store_manager.get_retriever(self.vector_store)
            
            logger.info("Initializing agent...")
            self.agent = ValidationAgent(config, self.retriever, self.vector_store)
            
            logger.info("System initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing system: {str(e)}")
            return False
    
    def query(self, question: str) -> str:
        """Query the system"""
        if not self.agent:
            raise ValueError("System not initialized. Call initialize_system() first.")
        
        try:
            return self.agent.invoke(question)
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"I apologize, but I encountered an error while processing your query: {str(e)}"

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
system_initialized = False

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
    return HealthResponse(
        status="healthy",
        initialized=system_initialized
    )


@app.post("/initialize", response_model=InitializeResponse)
async def initialize_system(request: InitializeRequest, background_tasks: BackgroundTasks):
    """Initialize the RAG system with a Google Document"""
    global system_initialized, qa_system
    
    try:
        logger.info(f"Initializing system with document: {request.doc_url}")
        
        # Initialize system in background
        def init_system():
            global system_initialized
            try:
                success = qa_system.initialize_system(str(request.doc_url))
                system_initialized = success
                if success:
                    logger.info("System initialization completed successfully")
                else:
                    logger.error("System initialization failed")
            except Exception as e:
                logger.error(f"Background initialization error: {str(e)}")
                system_initialized = False
        
        background_tasks.add_task(init_system)
        
        return InitializeResponse(
            success=True,
            message="System initialization started. Please wait a few moments before querying."
        )
        
    except Exception as e:
        logger.error(f"Error starting initialization: {str(e)}")
        return InitializeResponse(
            success=False,
            message="Failed to start system initialization",
            error=str(e)
        )


@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """Query the document with a question"""
    global system_initialized
    
    if not system_initialized:
        raise HTTPException(
            status_code=400, 
            detail="System not initialized. Please initialize with a document first."
        )
    
    try:
        logger.info(f"Processing query: {request.question}")
        
        if request.retrieval_method == "enhanced":
            # Use enhanced retrieval for debugging/analysis
            from vector_store import EnhancedRetriever
            if qa_system.vector_store:
                enhanced_retriever = EnhancedRetriever(qa_system.vector_store, config)
                docs = enhanced_retriever.multi_query_retrieval(request.question)
                answer = f"Retrieved {len(docs)} documents using {request.retrieval_method} method"
            else:
                answer = "Enhanced retrieval requires vector store to be initialized"
        elif request.retrieval_method == "debug":
            # Debug mode: show retrieval details
            from enhanced_retriever import PackageAwareRetriever
            if qa_system.vector_store:
                enhanced_retriever = PackageAwareRetriever(qa_system.vector_store, config)
                docs = enhanced_retriever.retrieve_with_context(request.question)
                explanation = enhanced_retriever.get_retrieval_explanation(request.question, docs)
                answer = f"Debug Info:\n{explanation}"
            else:
                answer = "Debug mode requires vector store to be initialized"
        else:
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
    return {
        "initialized": system_initialized,
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
    global system_initialized
    system_initialized = False
    
    return {
        "success": True,
        "message": "System reset successfully. Re-initialization required."
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
