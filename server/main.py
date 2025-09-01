from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Optional
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

class SimpleRAGSystem:
    def __init__(self):
        self.processor = DocumentProcessor(config)
        self.vector_store_manager = VectorStoreManager(config)
        # Initialize components once on startup
        self._setup_system()
    
    def _setup_system(self):
        """Setup the system - connect to existing Pinecone index"""
        try:
            self.vector_store = PineconeVectorStore.from_existing_index(
                index_name=config.index_name,
                embedding=self.vector_store_manager.embeddings
            )
            self.retriever = self.vector_store_manager.get_retriever(self.vector_store)
            self.agent = ValidationAgent(config, self.retriever, self.vector_store)
            logger.info("System ready - connected to Pinecone index")
        except Exception as e:
            logger.error(f"Setup error: {e}")
            self.vector_store = None
            self.agent = None
    
    def upload_document(self, doc_url: str) -> bool:
        """Upload a new document to Pinecone"""
        try:
            # Process document
            documents = self.processor.extract_google_doc(doc_url)
            chunks = self.processor.hierarchical_chunking(documents)
            enhanced_chunks = self.processor.enhance_metadata(chunks)
            
            # Store in Pinecone
            self.vector_store = self.vector_store_manager.store_documents(enhanced_chunks)
            self.retriever = self.vector_store_manager.get_retriever(self.vector_store)
            self.agent = ValidationAgent(config, self.retriever, self.vector_store)
            
            logger.info("Document uploaded successfully")
            return True
        except Exception as e:
            logger.error(f"Upload error: {e}")
            return False
    
    def query(self, question: str) -> str:
        """Query the documents in Pinecone"""
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

# Pydantic models
class UploadRequest(BaseModel):
    doc_url: HttpUrl
    
class QueryRequest(BaseModel):
    question: str

class UploadResponse(BaseModel):
    success: bool
    message: str

class QueryResponse(BaseModel):
    answer: str
    success: bool


@app.get("/")
async def root():
    """Simple health check"""
    return {"status": "RAG System Ready"}


@app.post("/upload", response_model=UploadResponse)
async def upload_document(request: UploadRequest):
    """Upload a document to Pinecone"""
    success = rag_system.upload_document(str(request.doc_url))
    
    if success:
        return UploadResponse(success=True, message="Document uploaded successfully!")
    else:
        return UploadResponse(success=False, message="Failed to upload document")


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the documents"""
    answer = rag_system.query(request.question)
    
    return QueryResponse(
        answer=answer,
        success=True
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
