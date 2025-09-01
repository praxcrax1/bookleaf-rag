# RAG System with Pinecone, Gemini, and LangGraph

A comprehensive document Q&A system that addresses the challenges of scattered information across multiple tabs using advanced RAG techniques with validation.

## 🚀 Features

- **Hierarchical Chunking**: Preserves document structure and tab information
- **Enhanced Metadata**: Adds content type, key phrases, and context requirements
- **Tab-Aware Retrieval**: Filters and prioritizes content based on detected tab relevance
- **Multi-Stage Validation**: Includes document grading and answer validation with LangGraph
- **Self-Correction**: The agent can revise queries and retrieval based on validation results
- **Fallback Mechanisms**: Handles cases where information is insufficient
- **REST API**: Complete FastAPI server with interactive documentation
- **Multiple Retrieval Methods**: Standard, multi-query, and hybrid retrieval

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

2. **Run the setup script:**
   ```bash
   ./setup.sh
   ```

3. **Configure environment variables:**
   - Copy `.env.example` to `.env`
   - Add your API keys:
     ```
     PINECONE_API_KEY=your_pinecone_api_key_here
     PINECONE_ENVIRONMENT=your_pinecone_environment_here
     GOOGLE_API_KEY=your_google_api_key_here
     ```

## 🔧 Configuration

The system can be configured through environment variables in the `.env` file:

- `PINECONE_API_KEY`: Your Pinecone API key
- `PINECONE_ENVIRONMENT`: Your Pinecone environment (e.g., "us-east1-gcp")
- `GOOGLE_API_KEY`: Your Google AI API key for Gemini
- `INDEX_NAME`: Pinecone index name (default: "google-doc-rag")
- `CHUNK_SIZE`: Text chunk size for processing (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `TOP_K`: Number of documents to retrieve (default: 5)
- `CONFIDENCE_THRESHOLD`: Validation confidence threshold (default: 0.8)

## 🏃‍♂️ Usage

### Option 1: Command Line Interface

```bash
# Activate virtual environment
source venv/bin/activate

# Run the main script
python main.py
```

### Option 2: REST API Server

```bash
# Start the API server
python api_server.py
```

The server will start at `http://localhost:8000`. Visit `http://localhost:8000/docs` for interactive API documentation.

#### API Endpoints:

- `GET /health` - Check server health
- `POST /initialize` - Initialize system with a Google Document
- `POST /query` - Query the document
- `GET /status` - Get system status
- `POST /reset` - Reset the system

### Option 3: Client Example

```bash
# Run the client example for interactive querying
python client_example.py
```

## 📊 System Architecture

```
Google Doc → Preprocessing → Hierarchical Chunking → Embedding → Pinecone Storage
     ↓
Query → LangGraph Agent → Retrieval → Validation → Response Generation
```

### Core Components:

1. **DocumentProcessor**: Extracts and processes Google Docs with hierarchical chunking
2. **VectorStoreManager**: Manages Pinecone vector storage with enhanced metadata
3. **ValidationAgent**: LangGraph-based agent with multi-stage validation
4. **EnhancedRetriever**: Advanced retrieval methods (multi-query, hybrid)

## 🧪 Testing

Run the test suite to verify system functionality:

```bash
python test_system.py
```

The test suite includes:
- Unit tests for individual components
- Integration tests for system workflow
- Performance tests for processing speed

## 🔍 Advanced Features

### Multi-Stage Validation

The system uses LangGraph to implement a sophisticated validation workflow:

1. **Document Retrieval**: Smart retrieval with tab awareness
2. **Document Grading**: AI-powered relevance scoring
3. **Answer Generation**: Context-aware response generation
4. **Answer Validation**: Verification against source documents
5. **Self-Correction**: Automatic retry with improved queries

### Enhanced Retrieval Methods

- **Standard**: Basic semantic similarity search
- **Multi-Query**: Generates query variations for better coverage
- **Hybrid**: Combines semantic and keyword-based search

### Tab-Aware Processing

The system automatically detects document sections/tabs and can filter queries based on detected content areas:
- Financial sections
- Technical specifications
- Marketing content
- General summaries

## 📝 Example Usage

```python
from main import DocumentQASystem
from config import config

# Initialize system
qa_system = DocumentQASystem(config)

# Load a Google Document
doc_url = "https://docs.google.com/document/d/YOUR_DOCUMENT_ID"
qa_system.initialize_system(doc_url)

# Query the document
answer = qa_system.query("What are the main financial projections?")
print(answer)
```

## 🚨 Error Handling

The system includes comprehensive error handling for:
- Invalid document URLs
- API key issues
- Network connectivity problems
- Malformed queries
- Insufficient document content

## 📈 Performance Considerations

- **Chunking Strategy**: Optimized for both context preservation and retrieval efficiency
- **Caching**: Vector embeddings are cached in Pinecone
- **Batch Processing**: Documents are processed in configurable batches
- **Background Processing**: API server supports background initialization

## 🔐 Security Notes

- Store API keys securely in environment variables
- Configure CORS appropriately for production use
- Consider rate limiting for production deployments
- Validate input URLs to prevent malicious document access

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Troubleshooting

### Common Issues:

1. **"System not initialized" error**: Make sure to call `initialize_system()` first
2. **API key errors**: Verify your keys in the `.env` file
3. **Pinecone connection issues**: Check your environment and API key
4. **Document access issues**: Ensure the Google Doc is publicly accessible
5. **Memory issues**: Reduce `CHUNK_SIZE` or `TOP_K` for large documents

### Debug Mode:

Enable debug logging by setting the log level in your code:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the test output
3. Check API documentation at `/docs` endpoint
4. Verify environment configuration

---

**Built with**: LangChain, LangGraph, Pinecone, Google Gemini, FastAPI
