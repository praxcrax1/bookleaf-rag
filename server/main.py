from typing import Optional
import logging
import uvicorn

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from langchain_pinecone import PineconeVectorStore

from config import config
from document_processor import DocumentProcessor
from vector_store import VectorStoreManager
from validation_agent import ValidationAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleRAGSystem:
    def __init__(self):
        """Initialize the RAG system with required components"""
        self.processor = DocumentProcessor(config)
        self.vector_store_manager = VectorStoreManager(config)
        # Initialize system state
        self.vector_store = None
        self.retriever = None
        self.agent = None
        # Connect to existing index on startup
        self._setup_system()
    
    def _initialize_components(self, vector_store: PineconeVectorStore) -> bool:
        """Initialize or update retriever and agent components from vector store"""
        try:
            self.vector_store = vector_store
            self.retriever = self.vector_store_manager.get_retriever(self.vector_store)
            self.agent = ValidationAgent(config, self.retriever, self.vector_store)
            return True
        except Exception as e:
            logger.error(f"Component initialization error: {e}")
            return False
    
    def _setup_system(self) -> None:
        """Setup the system by connecting to existing Pinecone index"""
        try:
            vector_store = PineconeVectorStore.from_existing_index(
                index_name=config.index_name,
                embedding=self.vector_store_manager.embeddings
            )
            if self._initialize_components(vector_store):
                logger.info("System ready - connected to Pinecone index")
        except Exception as e:
            logger.error(f"Setup error: {e}")
    
    def upload_document(self, doc_url: str) -> bool:
        """Process and upload a new document to Pinecone"""
        try:
            # Process document through pipeline
            documents = self.processor.extract_google_doc(doc_url)
            chunks = self.processor.hierarchical_chunking(documents)
            enhanced_chunks = self.processor.enhance_metadata(chunks)
            
            # Store in Pinecone and initialize components
            vector_store = self.vector_store_manager.store_documents(enhanced_chunks)
            if self._initialize_components(vector_store):
                logger.info("Document uploaded successfully")
                return True
            return False
        except Exception as e:
            logger.error(f"Upload error: {e}")
            return False
    
    def query(self, question: str) -> str:
        """Query the documents in Pinecone using the validation agent"""
        if not self.agent:
            return "No documents found. Please upload a document first."
        
        try:
            return self.agent.invoke(question)
        except Exception as e:
            logger.error(f"Query error: {e}")
            return f"Error processing query: {str(e)}"

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

# Global system instance - connects to Pinecone on startup
rag_system = SimpleRAGSystem()

# Pydantic models for API requests and responses
class UploadRequest(BaseModel):
    """Request model for document upload endpoint"""
    doc_url: HttpUrl
    
class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    question: str

class UploadResponse(BaseModel):
    """Response model for document upload endpoint"""
    success: bool
    message: str

class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    answer: str
    success: bool


@app.get("/")
async def root():
    """Simple health check"""
    return {"status": "RAG System Ready"}


@app.post("/upload", response_model=UploadResponse)
async def upload_document(request: UploadRequest):
    """Upload a Google Doc to the Pinecone vector store"""
    success = rag_system.upload_document(str(request.doc_url))
    
    return UploadResponse(
        success=success, 
        message="Document uploaded successfully!" if success else "Failed to upload document"
    )


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the RAG system with a question"""
    try:
        answer = rag_system.query(request.question)
        return QueryResponse(answer=answer, success=True)
    except Exception as e:
        logger.error(f"API query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Start the FastAPI server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Set to False in production
        log_level="info"
    )
