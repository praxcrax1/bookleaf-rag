from pydantic import BaseModel, HttpUrl

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
