from pydantic import BaseModel, HttpUrl, EmailStr
from typing import List, Optional, Dict, Any

class UploadRequest(BaseModel):
    """Request model for document upload endpoint"""
    doc_url: HttpUrl

class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    question: str
    verbose: Optional[bool] = False

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class AuthResponse(BaseModel):
    success: bool
    message: str
    token: Optional[str] = None

class UploadResponse(BaseModel):
    """Response model for document upload endpoint"""
    success: bool
    message: str

class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    answer: str
    reasoning_steps: Optional[List[Dict[str, Any]]] = None
    query: str
    success: bool
