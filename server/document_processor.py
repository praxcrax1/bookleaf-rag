from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import PyPDF2
import io
import requests
from typing import List, Dict
import re
from config import Config


class DocumentProcessor:
    """Simple document processor for RAG system"""
    def __init__(self, config: Config):
        self.config = config
        # Initialize embedding model
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=config.embedding_model,
            google_api_key=config.google_api_key
        )
        # Text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
        )
    
    def extract_google_doc(self, doc_url: str) -> List[Document]:
        """Extract content from Google Doc by converting to PDF"""
        try:
            # Extract document ID from URL
            document_id = doc_url.split("/d/")[1].split("/")[0]
            export_url = f"https://docs.google.com/document/d/{document_id}/export?format=pdf"
            
            # Download and extract text from PDF
            response = requests.get(export_url)
            pdf_file = io.BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Process each page into a document
            documents = []
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text.strip():
                    documents.append(Document(
                        page_content=text,
                        metadata={
                            "page": page_num + 1,
                            "source": doc_url
                        }
                    ))
            
            return documents
        except Exception as e:
            print(f"Error extracting Google Doc: {e}")
            return []
    
    def hierarchical_chunking(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks for better retrieval"""
        all_chunks = []
        
        for doc in documents:
            # Split document into chunks
            chunks = self.text_splitter.split_text(doc.page_content)
            
            # Create document objects for each chunk
            for i, chunk in enumerate(chunks):
                chunk_doc = Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_id": i
                    }
                )
                all_chunks.append(chunk_doc)
            
            # Add a shortened version of the full document for context
            if len(doc.page_content) > 1000:  # Only add if document is substantial
                full_doc = Document(
                    page_content=doc.page_content[:1000],  # First 1000 chars for context
                    metadata={
                        **doc.metadata,
                        "chunk_id": "summary"
                    }
                )
                all_chunks.append(full_doc)
        
        return all_chunks
    
    def enhance_metadata(self, documents: List[Document]) -> List[Document]:
        """Add basic metadata to chunks for better retrieval"""
        enhanced_docs = []
        
        for doc in documents:
            content = doc.page_content
            metadata = doc.metadata.copy()
            
            # Extract key words from content
            words = re.findall(r'\b[A-Za-z]{3,}\b', content)
            from collections import Counter
            word_freq = Counter(words)
            key_words = [word for word, count in word_freq.most_common(5) if count > 1]
            metadata["key_words"] = key_words
            
            # Basic content type classification
            if re.search(r'\d+\.\d+|\d+%|\$\d+', content):  # Numbers, percentages, currency
                content_type = "numeric"
            elif re.search(r'^[\s]*[-•*]', content, re.MULTILINE):  # Bullet points
                content_type = "list"
            else:
                content_type = "text"
            
            metadata["content_type"] = content_type
            
            enhanced_docs.append(Document(
                page_content=content,
                metadata=metadata
            ))
        
        return enhanced_docs
    

