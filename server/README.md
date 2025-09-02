# Simple RAG Document Q&A System

A streamlined RAG (Retrieval Augmented Generation) system for Google Documents using Pinecone vector database, Google Gemini AI, and LangGraph for intelligent document querying.

## 🚀 Features

- **Simple Upload & Query**: Upload Google Docs once, query anytime
- **Smart Document Processing**: Extracts and processes Google Docs with hierarchical chunking
- **Enhanced Metadata Extraction**: Automatically extracts packages, benefits, numerical data, and content types
- **Package-Aware Retrieval**: Intelligent retrieval that understands specific packages and benefits mentioned in queries
- **LangGraph Validation Pipeline**: Multi-stage document grading and answer validation
- **Auto-Initialization**: Automatically connects to existing Pinecone documents on startup
- **REST API**: Simple FastAPI server with just 3 endpoints

## 📋 System Requirements

- Python 3.8+
- Pinecone account and API key
- Google AI API key (for Gemini)
- Internet connection for document access

## 🛠️ Installation

1. **Clone and navigate to the project:**
   ```bash
   cd /path/to/test-rag/server
   ```

2. **Create virtual environment and install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   Create a `.env` file with your API keys:
   ```
   PINECONE_API_KEY=your_pinecone_api_key_here
   GOOGLE_API_KEY=your_google_api_key_here
   INDEX_NAME=google-doc-rag
   CHUNK_SIZE=1000
   TOP_K=5
   ```

## 🔧 Configuration

Configure the system through environment variables in the `.env` file:

**Required:**
- `PINECONE_API_KEY`: Your Pinecone API key
- `GOOGLE_API_KEY`: Your Google AI API key for Gemini

**Optional (with defaults):**
- `INDEX_NAME`: Pinecone index name (default: "google-doc-rag")
- `DIMENSION`: Embedding dimension (default: 768)
- `CHUNK_SIZE`: Text chunk size for processing (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `TOP_K`: Number of documents to retrieve (default: 5)
- `CONFIDENCE_THRESHOLD`: Validation confidence threshold (default: 0.8)
- `SCORE_THRESHOLD`: Document relevance threshold (default: 0.7)

## 🏃‍♂️ Usage

### Start the API Server

```bash
# Activate virtual environment
source venv/bin/activate

# Start the server
python main.py
```

The server will start at `http://localhost:8000`. Visit `http://localhost:8000/docs` for interactive API documentation.

## 📚 API Documentation

The RAG system provides a comprehensive REST API with detailed documentation available at `http://localhost:8000/docs` (Swagger UI) and `http://localhost:8000/redoc` (ReDoc).

### Authentication

Most endpoints require JWT authentication. Include the token in the Authorization header:
```bash
Authorization: Bearer <your_jwt_token>
```

### API Endpoints

#### 1. **Health Check**
**`GET /`**

Simple health check to verify the system is running.

**Request:**
```bash
curl http://localhost:8000/
```

**Response:**
```json
{
  "status": "RAG System Ready"
}
```

---

#### 2. **User Registration**
**`POST /register`**

Register a new user account.

**Request Payload:**
```json
{
  "email": "user@example.com",
  "password": "securepassword123",
}
```

**Example Request:**
```bash
curl -X POST "http://localhost:8000/register" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "john.doe@example.com",
    "password": "mySecurePassword123",
  }'
```

**Response:**
```json
{
  "success": true,
  "message": "User registered successfully",
  "token": "JWT"
}
```

**Error Response:**
```json
{
  "success": false,
  "message": "User already exists",
  "token": null
}
```

---

#### 3. **User Login**
**`POST /login`**

Authenticate user and receive JWT token.

**Request Payload:**
```json
{
  "email": "user@example.com",
  "password": "securepassword123"
}
```

**Example Request:**
```bash
curl -X POST "http://localhost:8000/login" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "john.doe@example.com",
    "password": "mySecurePassword123"
  }'
```

**Success Response:**
```json
{
  "success": true,
  "message": "Login successful",
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Error Response:**
```json
{
  "success": false,
  "message": "Invalid credentials",
  "token": null
}
```

---

#### 4. **Document Upload**
**`POST /upload`**

Upload a Google Document to the vector store for processing.

**Request Payload:**
```json
{
  "doc_url": "https://docs.google.com/document/d/DOCUMENT_ID/edit"
}
```

**Example Request:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -H "Content-Type: application/json" \
  -d '{
    "doc_url": "https://docs.google.com/document/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit"
  }'
```

**Success Response:**
```json
{
  "success": true,
  "message": "Document uploaded successfully!",
  "document_id": null,
  "chunks_processed": null
}
```

**Error Response:**
```json
{
  "success": false,
  "message": "Failed to upload document: Document not accessible",
  "document_id": null,
  "chunks_processed": null
}
```

---

#### 5. **Query Documents**
**`POST /query`** (🔐 Auth Required)

Query the RAG system with intelligent document retrieval and AI-powered responses.

**Request Payload:**
```json
{
  "question": "Your question here",
  "verbose": false
}
```

**Example Request:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..." \
  -d '{
    "question": "What are the benefits of the Bestseller Breakthrough Package?",
  }'
```

**Success Response**
```json
{
  "answer": "The Bestseller Breakthrough Package includes 5 free author copies, free shipping for Indian authors, a 20% discount coupon code for future orders, and priority customer support.",
  "query": "What are the benefits of the Bestseller Breakthrough Package?",
  "reasoning_steps": [
        {
            "tool": "faq_and_support",
            "input": "benefits of Bestseller Breakthrough Package",
            "output": "The Bestseller Breakthrough Package includes the following benefits:\n\n*   **Author Copies:** Dispatched 15–20 business days after your book goes live on all platforms, shipped directly to your registered address (complimentary copies not provided to International authors).\n*   **21st Century Emily Dickinson Award:** This prestigious award is included.\n*   **Global Publication & Distribution:** Your book is published globally and made available across multiple premium platforms, including all 13 ..."
        }
    ],
  "success": true,
  "confidence_score": null,
  "sources_used": null
}
```

**Error Response:**
```json
{
  "detail": "Query cannot be empty"
}
```

---

#### 6. **Delete Chat History**
**`DELETE /query/delete`** (🔐 Auth Required)

Delete all chat history for the authenticated user.

**Example Request:**
```bash
curl -X DELETE "http://localhost:8000/query/delete" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

**Success Response:**
```json
{
  "status": "success",
  "message": "Deleted 15 chat history records."
}
```

**No History Response:**
```json
{
  "status": "not_found",
  "message": "No chat history found for user."
}
```

**Error Response:**
```json
{
  "detail": "User ID not found in token"
}
```

---

### HTTP Status Codes

The API uses standard HTTP status codes:

| Status Code | Description |
|-------------|-------------|
| `200` | Success - Request completed successfully |
| `400` | Bad Request - Invalid request payload or parameters |
| `401` | Unauthorized - Invalid or missing authentication token |
| `403` | Forbidden - User doesn't have permission for this resource |
| `404` | Not Found - Requested resource doesn't exist |
| `422` | Unprocessable Entity - Request validation failed |
| `500` | Internal Server Error - Server encountered an error |

---

### Request/Response Headers

**Required Headers for Protected Endpoints:**
```http
Content-Type: application/json
Authorization: Bearer <jwt_token>
```

**Response Headers:**
```http
Content-Type: application/json
X-Process-Time: <processing_time_ms>
```

---

### Error Response Format

All error responses follow this consistent format:

```json
{
  "detail": "Error message describing what went wrong",
  "error_type": "validation_error|authentication_error|server_error",
  "timestamp": "2025-09-02T10:30:00Z",
  "path": "/query",
  "request_id": "req_abc123"
}
```

**Example Error Responses:**

**400 Bad Request:**
```json
{
  "detail": "Query cannot be empty"
}
```

**401 Unauthorized:**
```json
{
  "detail": "Could not validate credentials"
}
```

**422 Validation Error:**
```json
{
  "detail": [
    {
      "loc": ["body", "email"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

---

### Advanced Usage Examples

#### Full Authentication Flow

```python
import requests
import json

base_url = "http://localhost:8000"

# 1. Register a new user
register_response = requests.post(f"{base_url}/register", json={
    "email": "author@example.com",
    "password": "securepassword123",
    "author_id": "author_001"
})

if register_response.json()["success"]:
    print("User registered successfully")

# 2. Login to get JWT token
login_response = requests.post(f"{base_url}/login", json={
    "email": "author@example.com",
    "password": "securepassword123"
})

token = login_response.json()["token"]
headers = {"Authorization": f"Bearer {token}"}

# 3. Upload a document
upload_response = requests.post(f"{base_url}/upload", json={
    "doc_url": "https://docs.google.com/document/d/YOUR_DOCUMENT_ID/edit"
})

# 4. Query with authentication
query_response = requests.post(f"{base_url}/query", 
    headers=headers,
    json={
        "question": "What's my book status?",
        "verbose": True
    }
)

result = query_response.json()
print(f"Answer: {result['answer']}")
```

#### Batch Document Processing

```python
import requests
import asyncio
import aiohttp

async def upload_documents(doc_urls):
    """Upload multiple documents concurrently"""
    async with aiohttp.ClientSession() as session:
        tasks = []
        for url in doc_urls:
            task = session.post("http://localhost:8000/upload", 
                              json={"doc_url": url})
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        return [await r.json() for r in responses]

# Usage
doc_urls = [
    "https://docs.google.com/document/d/doc1/edit",
    "https://docs.google.com/document/d/doc2/edit",
    "https://docs.google.com/document/d/doc3/edit"
]

results = asyncio.run(upload_documents(doc_urls))
```

#### Advanced Query Examples

```python
# Book status inquiry
response = requests.post("http://localhost:8000/query",
    headers=headers,
    json={
        "question": "What's the current status of my book project?",
        "verbose": True
    }
)

# Package comparison
response = requests.post("http://localhost:8000/query",
    headers=headers,
    json={
        "question": "Compare the benefits between Bestseller Breakthrough and Premium packages",
        "verbose": False
    }
)

# Specific feature inquiry
response = requests.post("http://localhost:8000/query",
    headers=headers,
    json={
        "question": "How many author copies do I get and what are the shipping costs?",
        "verbose": True
    }
)
```

---

### Rate Limiting

Current rate limits per user:
- **Upload endpoint**: 10 requests per minute
- **Query endpoint**: 100 requests per minute  
- **Auth endpoints**: 20 requests per minute

Rate limit headers included in responses:
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1693737600
```

---

### Webhook Integration (Future Feature)

Coming soon - webhook support for real-time notifications:

```json
{
  "webhook_url": "https://your-app.com/webhooks/rag-updates",
  "events": ["document.processed", "query.completed"],
  "secret": "your_webhook_secret"
}
```

## 📊 System Architecture

```
Google Doc URL → PDF Download → Text Extraction → Hierarchical Chunking → Enhanced Metadata → Pinecone Storage
                                      ↓
User Query → PackageAwareRetriever → LangGraph ValidationAgent → Validated Response
```

### Core Components:

1. **SimpleRAGSystem**: Main orchestrator that coordinates all components
2. **DocumentProcessor**: Extracts Google Docs, converts to PDF, performs hierarchical chunking
3. **VectorStoreManager**: Manages Pinecone vector database operations and retrieval
4. **ValidationAgent**: LangGraph-based agent with multi-stage document processing pipeline
5. **PackageAwareRetriever**: Advanced retrieval system that understands package-specific queries

### LangGraph Pipeline:

The ValidationAgent uses a sophisticated multi-stage pipeline:

1. **Document Retrieval**: Package-aware semantic search with multi-stage retrieval
2. **Document Grading**: AI-powered relevance scoring of retrieved documents  
3. **Answer Generation**: Context-aware response generation with package specificity
4. **Answer Validation**: Verification against source documents with confidence scoring
5. **Final Answer**: Delivers validated response or requests clarification

### Enhanced Metadata Extraction:

The system automatically extracts and indexes:
- **Package Information**: Bestseller Breakthrough, Limited Publishing, Premium packages
- **Benefits**: Author copies, shipping details, coupon codes, bulk orders
- **Numerical Data**: Copy counts, prices, percentages, timeframes
- **Content Types**: Numeric data, bullet points, descriptive content
- **Location Context**: Indian vs International author distinctions

## 🧪 Testing

Test the system to verify functionality:

```bash
# Start the server
python main.py

# Test health endpoint
curl http://localhost:8000/

# Test upload (replace with your Google Doc URL)
curl -X POST "http://localhost:8000/upload" \
  -H "Content-Type: application/json" \
  -d '{"doc_url": "https://docs.google.com/document/d/YOUR_DOCUMENT_ID/edit"}'

# Test query
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What packages are available?"}'
```

## 🔍 Key Features Explained

### Package-Aware Querying

The system understands specific package queries:

```python
# Specific package query
"How many author copies do I get with Bestseller Breakthrough Package?"

# The system will:
# 1. Detect "Bestseller Breakthrough Package" context
# 2. Search for specific information about that package
# 3. Return precise information, not generic details
```

### Smart Metadata Extraction

When processing documents, the system automatically extracts:

- **Packages**: "Bestseller Breakthrough Package", "Limited Publishing Package"
- **Benefits**: "5 author copies", "free shipping", "coupon code"  
- **Numerical Data**: Copy counts, prices, delivery days
- **Content Classification**: Numeric data vs descriptive content

### Multi-Stage Retrieval

The PackageAwareRetriever performs:

1. **Direct Semantic Search**: Standard similarity search
2. **Package-Specific Search**: Enhanced search with package context
3. **Benefit-Specific Search**: Targeted search for specific benefits
4. **Exact Phrase Matching**: Finds precise terms and phrases
5. **Re-ranking**: Scores and ranks results by relevance

## 📝 Example Queries

The system excels at answering specific, package-aware questions:

**Package-Specific Questions:**
```
"How many free author copies do I get with Bestseller Breakthrough Package?"
"What is the delivery charge for Indian authors in Limited Publishing Package?"
"Do I get a coupon code with Premium Package?"
```

**Benefit-Specific Questions:**
```
"What are the shipping costs?"
"How do bulk orders work?"
"What's included in the author copies benefit?"
```

**Numerical Questions:**
```
"How many copies are included?"
"What's the delivery time?"
"What percentage discount do I get?"
```

## 🚨 Error Handling

The system includes comprehensive error handling:

- **Invalid Google Doc URLs**: Clear error messages for inaccessible documents
- **API Key Issues**: Helpful guidance for configuration problems
- **Network Problems**: Graceful handling of connectivity issues
- **Empty Results**: Requests clarification when no relevant information is found
- **Validation Failures**: Falls back to best-effort answers with disclaimers

## 📈 Performance & Scalability

**Optimized for Accuracy:**
- Hierarchical chunking preserves document context
- Package-aware retrieval reduces false positives  
- Multi-stage validation ensures answer quality
- Enhanced metadata improves search precision

**Efficient Design:**
- Vector embeddings cached in Pinecone
- One-time document processing
- Persistent connections to vector database
- Configurable chunk sizes and retrieval limits

## 🔐 Security & Production Notes

**Environment Variables:**
- Store API keys securely in `.env` file (never commit to version control)
- Use environment-specific configurations for different deployments

**Production Considerations:**
- Configure CORS appropriately (`allow_origins=["*"]` is for development only)
- Implement rate limiting for the API endpoints
- Add authentication/authorization as needed
- Use HTTPS in production environments
- Monitor Pinecone usage and costs

**Google Doc Access:**
- Ensure Google Docs are publicly accessible or properly shared
- Document URLs must be in shareable format
- Large documents may take time to process initially

## 🆘 Troubleshooting

**Common Issues & Solutions:**

1. **"No documents found" error**
   - Check if documents were uploaded successfully to Pinecone
   - Verify Pinecone API key and index configuration
   - Try restarting the server to reconnect to existing index

2. **Google Doc access errors**
   - Ensure the document is publicly accessible or properly shared
   - Check the document URL format
   - Verify network connectivity

3. **API key errors**
   - Confirm your `.env` file has correct API keys
   - Check Pinecone API key permissions
   - Verify Google AI API key is active

4. **Empty or generic responses**
   - The query might be too vague - try being more specific
   - Check if relevant information exists in the uploaded document
   - Try mentioning specific package names in your query

5. **Server startup issues**
   - Check all dependencies are installed: `pip install -r requirements.txt`
   - Verify Python version compatibility (3.8+)
   - Check port 8000 is not already in use

**Debug Mode:**
Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Architecture Benefits

**Why This Design Works:**

1. **Simplicity**: Just upload once, query forever
2. **Accuracy**: Package-aware retrieval reduces hallucinations
3. **Reliability**: LangGraph validation ensures quality responses
4. **Scalability**: Pinecone handles vector storage efficiently
5. **Maintainability**: Clean separation of concerns across components

## 📄 Dependencies

Core technologies used:
- **LangChain**: Document processing and retrieval framework
- **LangGraph**: Multi-stage validation pipeline
- **Pinecone**: Vector database for semantic search
- **Google Gemini**: AI model for embeddings and text generation
- **FastAPI**: REST API framework
- **PyPDF2**: PDF processing for Google Docs

---

**Built for**: Accurate document Q&A with package-specific intelligence
**Perfect for**: Customer support, product documentation, policy documents
