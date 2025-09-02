import re
import time
import logging
from typing import List
from langchain.schema import Document
from config import Config

# Configure logging
logger = logging.getLogger('EnhancedRetriever')

class PackageAwareRetriever:
    """A simple retriever that enhances semantic search with quoted phrases"""
    def __init__(self, vector_store, config: Config):
        self.vector_store = vector_store
        self.config = config
    
    def retrieve_with_context(self, query: str) -> List[Document]:
        """Retrieve documents with enhanced search for quoted phrases"""
        start_time = time.time()
        logger.info(f"Starting enhanced retrieval for query: {query[:100]}...")
        
        # Get documents using simple search approach
        documents = self._search_with_quoted_phrases(query)
        
        logger.info(f"Enhanced retrieval completed in {time.time() - start_time:.2f}s, found {len(documents[:self.config.top_k])} documents")
        return documents[:self.config.top_k]
    
    def _search_with_quoted_phrases(self, query: str) -> List[Document]:
        """Simple search strategy that also searches for quoted phrases"""
        all_docs = []
        
        try:
            # Standard search
            standard_search_start = time.time()
            logger.info("Performing standard vector search")
            standard_docs = self.vector_store.similarity_search(
                query, 
                k=self.config.top_k
            )
            logger.info(f"Standard search completed in {time.time() - standard_search_start:.2f}s, found {len(standard_docs)} documents")
            all_docs.extend(standard_docs)
            
            # Look for quoted phrases
            quoted_phrases = re.findall(r'"([^"]+)"', query)
            if quoted_phrases:
                logger.info(f"Found {len(quoted_phrases)} quoted phrases, performing additional searches")
                
                for i, phrase in enumerate(quoted_phrases):
                    phrase_search_start = time.time()
                    logger.info(f"Searching for quoted phrase {i+1}/{len(quoted_phrases)}: '{phrase}'")
                    
                    # Search for exact phrases
                    try:
                        phrase_docs = self.vector_store.similarity_search(
                            phrase, 
                            k=3
                        )
                        logger.info(f"Phrase search completed in {time.time() - phrase_search_start:.2f}s, found {len(phrase_docs)} documents")
                        all_docs.extend(phrase_docs)
                    except Exception as e:
                        logger.error(f"Error searching for phrase '{phrase}': {str(e)}")
            
            # Remove duplicates while preserving order
            dedup_start = time.time()
            seen_content = set()
            unique_docs = []
            for doc in all_docs:
                if doc.page_content not in seen_content:
                    seen_content.add(doc.page_content)
                    unique_docs.append(doc)
            
            logger.info(f"Deduplication completed in {time.time() - dedup_start:.2f}s")
            logger.info(f"Total unique documents: {len(unique_docs)} (from original {len(all_docs)})")
            
            return unique_docs
            
        except Exception as e:
            logger.error(f"Error in search_with_quoted_phrases: {str(e)}", exc_info=True)
            # Return whatever documents we have so far or empty list
            return all_docs if all_docs else []
    
    def get_retrieval_explanation(self, query: str, retrieved_docs: List[Document]) -> str:
        """Simple explanation of documents retrieved"""
        start_time = time.time()
        logger.info("Generating retrieval explanation")
        
        explanation = f"Retrieved {len(retrieved_docs)} documents for query: '{query}'\n"
        
        quoted_phrases = re.findall(r'"([^"]+)"', query)
        if quoted_phrases:
            explanation += f"- Found {len(quoted_phrases)} quoted phrases for exact matching\n"
            for phrase in quoted_phrases:
                explanation += f"  - \"{phrase}\"\n"
        
        logger.info(f"Retrieval explanation generated in {time.time() - start_time:.2f}s")
        return explanation
    

