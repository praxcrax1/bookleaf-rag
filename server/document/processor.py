from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import PyPDF2
import io
import requests
from typing import List
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
        """Multi-layered chunking for scattered knowledge across documents"""
        all_chunks = []
        # Multi-size chunking
        for doc in documents:
            large_chunks = self._create_chunks(doc, RecursiveCharacterTextSplitter(
                chunk_size=self.config.LARGE_CHUNK_SIZE,
                chunk_overlap=self.config.LARGE_CHUNK_OVERLAP,
                length_function=len,
            ), 'large_context')
            medium_chunks = self._create_chunks(doc, RecursiveCharacterTextSplitter(
                chunk_size=self.config.MEDIUM_CHUNK_SIZE,
                chunk_overlap=self.config.MEDIUM_CHUNK_OVERLAP,
                length_function=len,
            ), 'medium_context')
            small_chunks = self._create_chunks(doc, RecursiveCharacterTextSplitter(
                chunk_size=self.config.SMALL_CHUNK_SIZE,
                chunk_overlap=self.config.SMALL_CHUNK_OVERLAP,
                length_function=len,
            ), 'detailed')
            all_chunks.extend(large_chunks + medium_chunks + small_chunks)
        # Add cross-references
        enhanced_chunks = self._add_cross_references(all_chunks)
        return enhanced_chunks

    def _create_chunks(self, doc: Document, text_splitter, chunk_type: str) -> List[Document]:
        chunks = text_splitter.split_text(doc.page_content)
        chunk_docs = []
        for i, chunk in enumerate(chunks):
            chunk_doc = Document(
                page_content=chunk,
                metadata={
                    **doc.metadata,
                    "chunk_id": f"{chunk_type}_{i}",
                    "chunk_type": chunk_type,
                    "chunk_size": len(chunk),
                    "chunk_index": i
                }
            )
            chunk_docs.append(chunk_doc)
        return chunk_docs

    def _add_cross_references(self, chunks: List[Document]) -> List[Document]:
        doc_groups = {}
        for chunk in chunks:
            source = chunk.metadata.get('source', 'unknown')
            if source not in doc_groups:
                doc_groups[source] = []
            doc_groups[source].append(chunk)
        enhanced_chunks = []
        for source, source_chunks in doc_groups.items():
            for i, chunk in enumerate(source_chunks):
                nearby_chunks = []
                window = getattr(self.config, 'CROSS_REF_WINDOW', 2)
                for j in range(max(0, i - window), min(len(source_chunks), i + window + 1)):
                    if i != j:
                        nearby_chunk = source_chunks[j]
                        nearby_chunks.append({
                            'chunk_id': nearby_chunk.metadata['chunk_id'],
                            'chunk_type': nearby_chunk.metadata['chunk_type']
                        })
                chunk.metadata['nearby_chunks'] = nearby_chunks
                enhanced_chunks.append(chunk)
        return enhanced_chunks

    def enhance_metadata(self, documents: List[Document]) -> List[Document]:
        enhanced_docs = []
        for doc in documents:
            content = doc.page_content
            metadata = doc.metadata.copy()
            entities = self._extract_key_entities(content)
            topics = self._extract_topics(content)
            section_type = self._classify_section_type(content)
            content_summary = self._summarize_chunk(content)
            metadata.update({
                'entities': entities,
                'topics': topics,
                'section_type': section_type,
                'content_summary': content_summary,
                'word_count': len(content.split()),
                'has_numbers': bool(re.search(r'\d+', content)),
                'has_lists': bool(re.search(r'^[\s]*[-•*]', content, re.MULTILINE)),
                'has_headings': bool(re.search(r'^#+\s', content, re.MULTILINE))
            })
            enhanced_docs.append(Document(
                page_content=content,
                metadata=metadata
            ))
        return enhanced_docs

    def _extract_key_entities(self, text: str) -> List[str]:
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        common_words = {'The', 'This', 'That', 'When', 'Where', 'How', 'What', 'Why'}
        entities = [word for word in set(words) if word not in common_words and len(word) > 2]
        return entities[:10]

    def _extract_topics(self, text: str) -> List[str]:
        words = re.findall(r'\b[a-z]{4,}\b', text.lower())
        from collections import Counter
        word_freq = Counter(words)
        stop_words = {'with', 'have', 'this', 'that', 'they', 'them', 'were', 'been', 'their', 'would', 'could', 'should', 'about', 'other', 'which', 'when', 'where', 'what', 'how', 'why', 'who'}
        topics = [word for word, count in word_freq.most_common(5) if word not in stop_words and count > 1]
        return topics

    def _classify_section_type(self, text: str) -> str:
        if re.search(r'^\s*#', text, re.MULTILINE):
            return 'heading'
        elif re.search(r'^\s*[-•*]\s', text, re.MULTILINE):
            return 'list'
        elif re.search(r'\d+\.\s', text):
            return 'numbered_list'
        elif re.search(r'[?]\s*$', text, re.MULTILINE):
            return 'faq'
        elif re.search(r'\$\d+|\d+%|\d+\.\d+', text):
            return 'pricing_numeric'
        elif len(text.split('.')) > 3:
            return 'paragraph'
        else:
            return 'general'

    def _summarize_chunk(self, text: str) -> str:
        sentences = text.split('.')
        if sentences and len(sentences[0]) < 150:
            return sentences[0].strip() + '.'
        else:
            return text[:100].strip() + '...'


