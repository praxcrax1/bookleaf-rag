import logging
import uvicorn

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from langchain_pinecone import PineconeVectorStore

from config import config
from document.processor import DocumentProcessor
from vector_store.manager import VectorStoreManager
from agent.agent_factory import create_agent
from models.schemas import UploadRequest, QueryRequest, UploadResponse, QueryResponse, RegisterRequest, LoginRequest, AuthResponse
from auth.auth import get_current_user
from database.mongo import get_user_books, register_user, authenticate_user
from jose import jwt
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleRAGSystem:
    def __init__(self):
        """Initialize the RAG system with required components"""
        self.processor = DocumentProcessor(config)
        self.vector_store_manager = VectorStoreManager(config)
        self.agent_executor = create_agent()
        # Initialize system state
        self.vector_store = None
        self.retriever = None
        # Connect to existing index on startup
        self._setup_system()
    
    def _initialize_components(self, vector_store: PineconeVectorStore) -> bool:
        """Initialize or update retriever and agent components from vector store"""
        try:
            self.vector_store = vector_store
            self.retriever = self.vector_store_manager.get_retriever(self.vector_store)
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
        """Query the documents using the new agent executor"""
        if not self.agent_executor:
            return "Agent not initialized. Please upload a document first."
        try:
            result = self.agent_executor.invoke({"input": question})
            return result["output"] if "output" in result else str(result)
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
async def query_documents(request: QueryRequest, current_user: dict = Depends(get_current_user)):
    """Query the RAG system with a question (auth required)"""
    query = request.question.strip()
    if not query:
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )
    author_id = current_user.get("user_id")
    user_books = await get_user_books(author_id)
    try:
        logger.info(f"Query request: {query[:100]} for user {author_id}")
        response = await asyncio.to_thread(
            rag_system.agent_executor.invoke,
            {"input": query}
        )
        answer = response.get("output", "I apologize, but I couldn't generate a response.")
        intermediate_steps = response.get("intermediate_steps", []) if getattr(request, "verbose", False) else []
        reasoning_steps = []
        if request.verbose and intermediate_steps:
            for step in intermediate_steps:
                if len(step) >= 2:
                    action = step[0]
                    observation = step[1]
                    reasoning_steps.append({
                        "tool": getattr(action, 'tool', "unknown"),
                        "input": getattr(action, 'tool_input', {}),
                        "output": observation[:500] + "..." if len(str(observation)) > 500 else str(observation)
                    })
        logger.info("Successfully processed query request")
        return QueryResponse(
            answer=answer,
            query=query,
            reasoning_steps=reasoning_steps if request.verbose else None,
            success=True
        )
    except Exception as e:
        logger.error(f"API query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/register", response_model=AuthResponse)
async def register_endpoint(request: RegisterRequest):
    success, message = await register_user(request.email, request.password, request.author_id)
    return AuthResponse(success=success, message=message)

@app.post("/login", response_model=AuthResponse)
async def login_endpoint(request: LoginRequest):
    user = await authenticate_user(request.email, request.password)
    if not user:
        return AuthResponse(success=False, message="Invalid credentials")
    # Issue JWT token
    payload = {"user_id": str(user.get("author_id", "")), "email": user["email"]}
    token = jwt.encode(payload, config.jwt_secret, algorithm=config.jwt_algorithm)
    return AuthResponse(success=True, message="Login successful", token=token)


if __name__ == "__main__":
    # Start the FastAPI server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Set to False in production
        log_level="info"
    )
