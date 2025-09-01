from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
import PyPDF2
import io
import requests
from typing import List, Optional, Dict
import re
from config import Config


class DocumentProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=config.embedding_model,
            google_api_key=config.google_api_key
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
        )
    
    def extract_google_doc(self, doc_url: str) -> List[Document]:
        """Extract content from Google Doc"""
        # Convert Google Doc to a downloadable format
        document_id = doc_url.split("/d/")[1].split("/")[0]
        export_url = f"https://docs.google.com/document/d/{document_id}/export?format=pdf"
        
        # Download and extract text from PDF
        response = requests.get(export_url)
        pdf_file = io.BytesIO(response.content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        documents = []
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text.strip():
                # Try to detect tab/section headers
                tab_name = self._detect_tab_name(text, page_num)
                documents.append(Document(
                    page_content=text,
                    metadata={
                        "page": page_num + 1,
                        "tab": tab_name,
                        "source": doc_url
                    }
                ))
        
        return documents
    
    def _detect_tab_name(self, text: str, page_num: int) -> str:
        """Heuristic to detect tab/section names from content"""
        # Look for common patterns that might indicate tab names
        lines = text.split('\n')
        if lines:
            first_line = lines[0].strip()
            # Check if first line looks like a header
            if (len(first_line) < 50 and 
                (first_line.isupper() or 
                 re.match(r'^(#+\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$', first_line))):
                return first_line
            
            # Check for tab indicators in text
            tab_patterns = [
                r'tab:\s*([^\n]+)',
                r'section:\s*([^\n]+)',
                r'worksheet:\s*([^\n]+)',
            ]
            
            for pattern in tab_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
        
        return f"Page_{page_num + 1}"
    
    def hierarchical_chunking(self, documents: List[Document]) -> List[Document]:
        """Create hierarchical chunks preserving tab structure"""
        all_chunks = []
        
        for doc in documents:
            tab_name = doc.metadata.get("tab", "Unknown")
            chunks = self.text_splitter.split_text(doc.page_content)
            
            for i, chunk in enumerate(chunks):
                chunk_doc = Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_id": i,
                        "tab": tab_name,
                        "full_document": False
                    }
                )
                all_chunks.append(chunk_doc)
            
            # Also add the full document content for context (with lower weight)
            full_doc = Document(
                page_content=doc.page_content[:3000],  # Limit full document size
                metadata={
                    **doc.metadata,
                    "chunk_id": "full",
                    "full_document": True
                }
            )
            all_chunks.append(full_doc)
        
        return all_chunks
    
    def enhance_metadata(self, documents: List[Document]) -> List[Document]:
        """Add additional metadata to chunks for better retrieval"""
        enhanced_docs = []
        
        for doc in documents:
            content = doc.page_content
            metadata = doc.metadata.copy()
            
            # Extract key phrases
            key_phrases = self._extract_key_phrases(content)
            metadata["key_phrases"] = key_phrases
            
            # Determine content type
            metadata["content_type"] = self._classify_content_type(content)
            
            # Extract package information
            metadata["packages_mentioned"] = self._extract_package_info(content)
            
            # Extract specific benefits/features
            metadata["benefits"] = self._extract_benefits(content)
            
            # Extract numerical information
            metadata["numerical_data"] = self._extract_numerical_data(content)
            
            # Add semantic metadata
            metadata["requires_context"] = self._requires_context(content)
            
            enhanced_docs.append(Document(
                page_content=content,
                metadata=metadata
            ))
        
        return enhanced_docs
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text"""
        # Simple implementation - could be enhanced with NLP
        words = re.findall(r'\b[A-Za-z]{3,}\b', text)
        from collections import Counter
        word_freq = Counter(words)
        return [word for word, count in word_freq.most_common(5) if count > 1]
    
    def _classify_content_type(self, text: str) -> str:
        """Classify content type"""
        if re.search(r'\d+\.\d+|\d+%|\$\d+', text):  # Numbers, percentages, currency
            return "numeric_data"
        elif re.search(r'^[\s]*[-•*]', text, re.MULTILINE):  # Bullet points
            return "bullet_points"
        elif re.search(r'.{50,}', text):  # Long sentences
            return "descriptive"
        else:
            return "general"
    
    def _requires_context(self, text: str) -> bool:
        """Determine if this chunk likely requires context from other chunks"""
        # Heuristic: if text contains references to other sections or incomplete thoughts
        patterns = [
            r'as mentioned above',
            r'as discussed in section',
            r'see (?:page|tab)',
            r'however,?',
            r'although',
            r'in conclusion',
            r'in summary'
        ]
        
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _extract_package_info(self, text: str) -> List[str]:
        """Extract package names and types from text"""
        packages = []
        
        # Common package patterns
        package_patterns = [
            r'bestseller?\s+breakthrough\s+package',
            r'limited\s+publishing\s+package',
            r'premium\s+package',
            r'basic\s+package',
            r'standard\s+package',
            r'deluxe\s+package',
            r'professional\s+package',
            r'economy\s+package'
        ]
        
        text_lower = text.lower()
        for pattern in package_patterns:
            if re.search(pattern, text_lower):
                match = re.search(pattern, text_lower)
                packages.append(match.group().title())
        
        return packages
    
    def _extract_benefits(self, text: str) -> List[str]:
        """Extract specific benefits or features mentioned"""
        benefits = []
        
        # Look for benefit patterns
        benefit_patterns = [
            r'(\d+)\s+(author\s+copies?)',
            r'(\d+)\s+(complimentary\s+copies?)',
            r'(\d+)\s+(free\s+copies?)',
            r'(coupon\s+code)',
            r'(free\s+shipping)',
            r'(bulk\s+order)',
            r'(printing\s+cost)',
            r'(delivery\s+charge)',
            r'(international\s+authors?)',
            r'(indian\s+authors?)'
        ]
        
        for pattern in benefit_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    benefits.append(' '.join(match))
                else:
                    benefits.append(match)
        
        return benefits
    
    def _extract_numerical_data(self, text: str) -> Dict[str, List[str]]:
        """Extract numerical data with context"""
        numerical_data = {
            "copies": [],
            "prices": [],
            "percentages": [],
            "days": []
        }
        
        # Extract copy numbers with context
        copy_patterns = [
            r'(\d+)\s+(author\s+copies?|complimentary\s+copies?|free\s+copies?)',
            r'(one|two|three|four|five|\d+)\s+(free\s+author\s+copy|author\s+copy)'
        ]
        
        for pattern in copy_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                numerical_data["copies"].append(' '.join(match))
        
        # Extract prices
        price_matches = re.findall(r'\$\d+(?:\.\d{2})?|\₹\d+(?:\.\d{2})?', text)
        numerical_data["prices"].extend(price_matches)
        
        # Extract percentages
        percentage_matches = re.findall(r'\d+%', text)
        numerical_data["percentages"].extend(percentage_matches)
        
        # Extract days/timeframes
        day_matches = re.findall(r'(\d+)\s+(business\s+days?|days?|weeks?)', text, re.IGNORECASE)
        for match in day_matches:
            numerical_data["days"].append(' '.join(match))
        
        return numerical_data
