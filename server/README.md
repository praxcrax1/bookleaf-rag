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

### API Endpoints

**Simple 3-endpoint system:**

1. **`GET /`** - Health check
   ```bash
   curl http://localhost:8000/
   ```

2. **`POST /upload`** - Upload a Google Document (do this once)
   ```bash
   curl -X POST "http://localhost:8000/upload" \
     -H "Content-Type: application/json" \
     -d '{"doc_url": "https://docs.google.com/document/d/YOUR_DOCUMENT_ID/edit"}'
   ```

3. **`POST /query`** - Query the uploaded documents (use anytime)
   ```bash
   curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "What are the benefits of the Bestseller Breakthrough Package?"}'
   ```

### Python Client Example

```python
import requests

# Upload document once
upload_response = requests.post("http://localhost:8000/upload", json={
    "doc_url": "https://docs.google.com/document/d/YOUR_DOCUMENT_ID/edit"
})

# Query anytime
query_response = requests.post("http://localhost:8000/query", json={
    "question": "How many author copies do I get?"
})

answer = query_response.json()["answer"]
print(answer)
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
