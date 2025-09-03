import logging
import uvicorn

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware

from config import config
from document.processor import DocumentProcessor
from vector_store.manager import VectorStoreManager
from agent.agent_factory import create_agent
from models.schemas import UploadRequest, QueryRequest, UploadResponse, QueryResponse, RegisterRequest, LoginRequest, AuthResponse
from auth.auth import get_current_user
from database.mongo import register_user, authenticate_user, db
from jose import jwt
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components
processor = DocumentProcessor(config)
vector_store_manager = VectorStoreManager(config)

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


@app.get("/")
async def root():
    """Simple health check"""
    return {"status": "RAG System Ready"}


@app.post("/upload", response_model=UploadResponse)
async def upload_document(request: UploadRequest):
    """Upload a Google Doc to the Pinecone vector store"""
    try:
        # Process document through pipeline
        documents = processor.extract_google_doc(str(request.doc_url))
        chunks = processor.hierarchical_chunking(documents)
        enhanced_chunks = processor.enhance_metadata(chunks)
        
        # Store in Pinecone
        vector_store_manager.store_documents(enhanced_chunks)
        logger.info("Document uploaded successfully")
        
        return UploadResponse(success=True, message="Document uploaded successfully!")
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return UploadResponse(success=False, message=f"Failed to upload document: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest, current_user: dict = Depends(get_current_user)):
    """Query the RAG system with a question (auth required)"""
    query = request.question.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    user_id = current_user.get("user_id")

    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found in token")
    
    try:
        logger.info(f"Query request: {query[:100]} for user {user_id}")
        
        # Create agent with user_id for this specific query
        agent_executor = create_agent(user_id=user_id, query=query)
        
        response = await asyncio.to_thread(
            agent_executor.invoke,
            {"input": query}
        )
        
        answer = response.get("output", "I apologize, but I couldn't generate a response.")
        intermediate_steps = response.get("intermediate_steps", [])
        
        # Safety check: If no tools were used but answer is provided, flag as potential hallucination
        tools_used = len(intermediate_steps) > 0
        if not tools_used and answer and "I don't" not in answer and "knowledge base" not in answer:
            logger.warning(f"Potential hallucination detected for query: {query[:50]}")
            answer = "I don't have specific information about that in our knowledge base. Please contact our support team for accurate information."
        
        reasoning_steps = None
        if intermediate_steps:
            reasoning_steps = []
            for step in intermediate_steps:
                if len(step) >= 2:
                    action = step[0]
                    observation = step[1]
                    reasoning_steps.append({
                        "tool": action.tool if hasattr(action, 'tool') else "unknown",
                        "input": action.tool_input if hasattr(action, 'tool_input') else {},
                        "output": observation[:500] + "..." if len(str(observation)) > 500 else str(observation)
                    })
        
        logger.info("Successfully processed query request")

        return QueryResponse(
            answer=answer,
            query=query,
            reasoning_steps=reasoning_steps,
            success=True
        )
    except Exception as e:
        logger.error(f"API query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/register", response_model=AuthResponse)
async def register_endpoint(request: RegisterRequest):
    success, message, user_id = await register_user(request.email, request.password)
    
    if success:
        # Issue JWT token immediately upon registration
        payload = {"user_id": user_id, "email": request.email}
        token = jwt.encode(payload, config.jwt_secret, algorithm=config.jwt_algorithm)
        return AuthResponse(success=True, message=message, token=token)
    else:
        return AuthResponse(success=False, message=message)

@app.post("/login", response_model=AuthResponse)
async def login_endpoint(request: LoginRequest):
    user = await authenticate_user(request.email, request.password)
    if not user:
        return AuthResponse(success=False, message="Invalid credentials")
    
    user_id = str(user.get("_id"))
    
    payload = {"user_id": user_id, "email": user["email"]}
    token = jwt.encode(payload, config.jwt_secret, algorithm=config.jwt_algorithm)
    return AuthResponse(success=True, message="Login successful", token=token)


@app.delete("/query/delete")
async def delete_chat_history(current_user: dict = Depends(get_current_user)):
    """Delete all chat history for the authenticated user."""
    try:
        user_id = current_user.get("user_id")
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID not found in token")
        
        # Delete chat history from MongoDB collection
        chat_histories_collection = db["chat_histories"]
        result = await chat_histories_collection.delete_many({"SessionId": str(user_id)})
        
        deleted_count = result.deleted_count
        if deleted_count == 0:
            return {"status": "not_found", "message": "No chat history found for user."}
        
        logger.info(f"Deleted {deleted_count} chat history records for user {user_id}")
        return {"status": "success", "message": f"Deleted {deleted_count} chat history records."}
        
    except Exception as e:
        logger.error(f"Error deleting chat history: {e}")
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
