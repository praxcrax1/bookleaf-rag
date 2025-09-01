from langgraph.graph import StateGraph, END
from langchain.schema import BaseMessage, HumanMessage, AIMessage, Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from typing import TypedDict, Annotated, Sequence, List, Optional, Callable
import operator
import time
import logging
from config import Config
from enhanced_retriever import PackageAwareRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ValidationAgent')


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    retrieved_docs: List[Document]
    confirmed_docs: List[Document]
    reasoning: str
    needs_confirmation: bool
    confirmation_result: bool
    query_attempts: int


class ValidationAgent:
    def __init__(self, config: Config, retriever: Callable, vector_store=None):
        self.config = config
        self.retriever = retriever
        self.vector_store = vector_store
        self.llm = ChatGoogleGenerativeAI(
            model=config.model_name,
            google_api_key=config.google_api_key,
            temperature=0.1
        )
        
        # Initialize enhanced retriever if vector_store is provided
        if vector_store:
            self.enhanced_retriever = PackageAwareRetriever(vector_store, config)
        else:
            self.enhanced_retriever = None
            
        self.graph = self._create_graph()
    
    def _create_graph(self) -> StateGraph:
        """Create the LangGraph state machine"""
        builder = StateGraph(AgentState)
        
        # Define nodes
        builder.add_node("retrieve_documents", self.retrieve_documents)
        builder.add_node("grade_documents", self.grade_documents)
        builder.add_node("generate_initial_answer", self.generate_initial_answer)
        builder.add_node("validate_answer", self.validate_answer)
        builder.add_node("request_clarification", self.request_clarification)
        builder.add_node("provide_final_answer", self.provide_final_answer)
        
        # Define edges
        builder.set_entry_point("retrieve_documents")
        builder.add_edge("retrieve_documents", "grade_documents")
        builder.add_conditional_edges(
            "grade_documents",
            self.decide_after_grading,
            {
                "sufficient": "generate_initial_answer",
                "insufficient": "request_clarification"
            }
        )
        builder.add_edge("generate_initial_answer", "validate_answer")
        builder.add_conditional_edges(
            "validate_answer",
            self.decide_after_validation,
            {
                "confirmed": "provide_final_answer",
                "needs_revision": "retrieve_documents"
            }
        )
        builder.add_edge("request_clarification", END)
        builder.add_edge("provide_final_answer", END)
        
        return builder.compile()
    
    def retrieve_documents(self, state: AgentState) -> dict:
        """Retrieve relevant documents based on the query"""
        start_time = time.time()
        logger.info("Starting document retrieval")
        
        # Extract the latest user message
        user_query = state["messages"][-1].content if state["messages"] else ""
        logger.info(f"Query: {user_query[:100]}...")
        
        # Use enhanced retriever if available, otherwise fall back to standard retriever
        try:
            if self.enhanced_retriever:
                logger.info("Using enhanced retriever")
                retrieval_start = time.time()
                documents = self.enhanced_retriever.retrieve_with_context(user_query)
                logger.info(f"Enhanced retrieval completed in {time.time() - retrieval_start:.2f}s, found {len(documents)} documents")
            else:
                # Try to detect which tab the query might belong to
                tab_filter = self._detect_tab_from_query(user_query)
                logger.info(f"Using standard retriever with tab filter: {tab_filter}")
                retrieval_start = time.time()
                documents = self.retriever(user_query, tab_filter)
                logger.info(f"Standard retrieval completed in {time.time() - retrieval_start:.2f}s, found {len(documents)} documents")
            
            logger.info(f"Document retrieval completed in {time.time() - start_time:.2f}s")
            return {
                "retrieved_docs": documents,
                "query_attempts": state.get("query_attempts", 0) + 1
            }
        except Exception as e:
            logger.error(f"Error during document retrieval: {str(e)}", exc_info=True)
            # Return empty list to allow the process to continue with error handling
            return {
                "retrieved_docs": [],
                "query_attempts": state.get("query_attempts", 0) + 1
            }
    
    def _detect_tab_from_query(self, query: str) -> Optional[str]:
        """Heuristic to detect which tab a query might belong to"""
        # This could be enhanced with a trained classifier
        tab_keywords = {
            "summary": ["summary", "overview", "introduction"],
            "financial": ["financial", "revenue", "cost", "profit", "budget"],
            "technical": ["technical", "specification", "implementation", "code"],
            "marketing": ["marketing", "promotion", "campaign", "audience"],
        }
        
        query_lower = query.lower()
        for tab, keywords in tab_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return tab
        
        return None
    
    def grade_documents(self, state: AgentState) -> dict:
        """Grade the retrieved documents for relevance using optimized batching"""
        start_time = time.time()
        logger.info("Starting document grading with optimized batch approach")
        
        user_query = state["messages"][-1].content
        documents = state["retrieved_docs"]
        
        if not documents:
            logger.warning("No documents retrieved for grading")
            return {
                "reasoning": "No documents retrieved. Need to request clarification.",
                "confirmed_docs": []
            }
        
        logger.info(f"Grading {len(documents)} documents for relevance")
        
        # Step 1: First filter documents with quick keyword-based pre-filtering
        prefiltered_docs = self._keyword_prefilter(user_query, documents)
        
        # If we have enough documents after pre-filtering, skip the LLM grading
        if len(prefiltered_docs) >= 3:
            logger.info(f"Pre-filtering found {len(prefiltered_docs)} relevant documents, skipping LLM grading")
            reasoning = ["Documents were evaluated based on keyword relevance"]
            return {
                "confirmed_docs": prefiltered_docs[:5],  # Limit to top 5 docs if we have more
                "reasoning": "\n".join(reasoning),
                "needs_confirmation": False
            }
            
        # Step 2: Use batched LLM grading if we have fewer documents than a threshold
        # or pre-filtering didn't find enough documents
        if len(documents) <= 5:
            graded_docs, reasoning = self._batch_grade_documents(user_query, documents)
        else:
            # For larger document sets, do a two-stage approach
            # First apply TF-IDF or BM25 filtering to get top candidates
            filtered_docs = self._statistical_filter(user_query, documents, top_k=5)
            # Then use LLM to grade this smaller set
            graded_docs, reasoning = self._batch_grade_documents(user_query, filtered_docs)
        
        # Check if we have enough relevant documents
        has_sufficient_docs = len(graded_docs) >= 2 or (
            len(graded_docs) == 1 and 
            len(graded_docs[0].page_content) > 200  # Substantial single document
        )
        
        logger.info(f"Document grading completed in {time.time() - start_time:.2f}s")
        logger.info(f"Found {len(graded_docs)}/{len(documents)} relevant documents")
        
        return {
            "confirmed_docs": graded_docs,
            "reasoning": "\n".join(reasoning),
            "needs_confirmation": not has_sufficient_docs
        }
        
    def _keyword_prefilter(self, query: str, documents: List[Document]) -> List[Document]:
        """Pre-filter documents based on keyword matching to avoid LLM calls"""
        if not documents:
            return []
            
        logger.info("Applying keyword-based pre-filtering")
        start_time = time.time()
        
        try:
            # Use NLTK's stopwords instead of hardcoded list
            import nltk
            from nltk.corpus import stopwords
            
            # Download stopwords if not already downloaded
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords', quiet=True)
            
            stop_words = set(stopwords.words('english'))
            
        except ImportError:
            # Fallback to a minimal set if NLTK is not available
            logger.warning("NLTK not available, using minimal stopwords set")
            stop_words = {'a', 'an', 'the', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'with', 'is', 'are', 'was'}
        
        # Get significant query terms (non-stopwords with length > 2)
        query_terms = set(term.lower() for term in query.split() 
                        if term.lower() not in stop_words and len(term) > 2)
        
        # Add exact phrases as important terms (terms in quotes)
        import re
        quoted_phrases = re.findall(r'"([^"]*)"', query)
        for phrase in quoted_phrases:
            if len(phrase) > 2:
                query_terms.add(phrase.lower())
        
        # Score documents based on term frequency and metadata
        scored_docs = []
        for doc in documents:
            doc_content = doc.page_content.lower()
            
            # Basic relevance scoring
            score = 0
            
            # Score individual terms
            for term in query_terms:
                # Count exact matches
                term_count = doc_content.count(term)
                score += term_count
                
                # Give higher weight to terms in document title or headings if available
                if doc.metadata.get('title') and term in doc.metadata.get('title', '').lower():
                    score += 3
            
            # Boost score for package matches
            packages_mentioned = doc.metadata.get("packages_mentioned", [])
            if packages_mentioned:
                for package in packages_mentioned:
                    if package.lower() in query.lower():
                        score += 5  # Strong boost for package match
            
            # Give higher weight to documents from relevant tabs
            if 'tab' in doc.metadata:
                tab = doc.metadata['tab']
                # Check if query keywords are related to the tab
                tab_relevance = 0
                if 'summary' in tab and any(w in query.lower() for w in ['overview', 'summary', 'introduction']):
                    tab_relevance = 2
                elif 'financial' in tab and any(w in query.lower() for w in ['price', 'cost', 'financial']):
                    tab_relevance = 2
                # Add more tab relevance checks as needed
                score += tab_relevance
                
            scored_docs.append((doc, score))
            
        # Sort by score and return top documents
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        filtered_docs = [doc for doc, score in scored_docs if score > 0]
        
        logger.info(f"Keyword pre-filtering completed in {time.time() - start_time:.2f}s")
        logger.info(f"Pre-filtered down to {len(filtered_docs)}/{len(documents)} documents")
        
        return filtered_docs
    
    def _statistical_filter(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """Filter documents using statistical methods like cosine similarity"""
        if not documents:
            return []
            
        logger.info(f"Applying statistical filtering to select top {top_k} documents")
        start_time = time.time()
        
        try:
            # Simple TF-IDF approach
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Extract document contents and combine with metadata where available
            doc_contents = []
            for doc in documents:
                # Combine document content with metadata for better matching
                content = doc.page_content
                
                # Add metadata to enhance relevance scoring
                metadata_text = ""
                if doc.metadata.get("title"):
                    metadata_text += f" {doc.metadata['title']} "
                if doc.metadata.get("tab"):
                    metadata_text += f" {doc.metadata['tab']} "
                if doc.metadata.get("packages_mentioned"):
                    metadata_text += f" {' '.join(doc.metadata['packages_mentioned'])} "
                
                # Combine content with metadata (give metadata higher weight by repeating it)
                enhanced_content = f"{content} {metadata_text} {metadata_text}"
                doc_contents.append(enhanced_content)
            
            # Create TF-IDF matrix with more sophisticated parameters
            vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2),  # Use both unigrams and bigrams
                max_df=0.85,         # Ignore terms that appear in >85% of docs
                min_df=1             # Only keep terms that appear in at least 1 doc
            )
            
            # Process the documents and query
            tfidf_matrix = vectorizer.fit_transform(doc_contents + [query])
            
            # Calculate similarity between query and documents
            query_vec = tfidf_matrix[-1]
            doc_vecs = tfidf_matrix[:-1]
            similarities = cosine_similarity(query_vec, doc_vecs)[0]
            
            # Sort documents by similarity
            doc_scores = [(doc, sim) for doc, sim in zip(documents, similarities)]
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Return top-k documents
            filtered_docs = [doc for doc, _ in doc_scores[:top_k]]
            logger.info(f"TF-IDF filtering completed successfully, selected {len(filtered_docs)} documents")
            
        except Exception as e:
            logger.error(f"Error in statistical filtering: {str(e)}", exc_info=True)
            # Fallback to keyword-based filtering if TF-IDF fails
            logger.info("Falling back to simple filtering method")
            
            # Simple keyword match as fallback
            query_terms = set(term.lower() for term in query.split() if len(term) > 2)
            scored_docs = []
            
            for doc in documents:
                doc_content = doc.page_content.lower()
                score = sum(doc_content.count(term) for term in query_terms)
                scored_docs.append((doc, score))
                
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            filtered_docs = [doc for doc, score in scored_docs[:top_k] if score > 0]
            
            # If no matches, just return first top_k documents
            if not filtered_docs and documents:
                filtered_docs = documents[:top_k]
        
        logger.info(f"Statistical filtering completed in {time.time() - start_time:.2f}s")
        
        return filtered_docs
    
    def _batch_grade_documents(self, query: str, documents: List[Document]) -> tuple[List[Document], List[str]]:
        """Grade multiple documents in a single LLM call for efficiency"""
        if not documents:
            return [], []
            
        logger.info(f"Batch grading {len(documents)} documents")
        start_time = time.time()
        
        # Prepare batch input for grading
        doc_snippets = []
        for i, doc in enumerate(documents):
            # Truncate content to avoid excessive token usage
            content = doc.page_content[:800]
            doc_snippets.append(f"Document {i+1}: {content}")
        
        # Combine into a single context
        batch_docs = "\n\n---\n\n".join(doc_snippets)
        
        # Create batch grading prompt
        batch_grading_prompt = ChatPromptTemplate.from_template("""
        You are a document grader. Evaluate whether the following documents are relevant to the user's query.
        Consider if each document contains information that can help answer the query.
        
        User Query: {query}
        
        {documents}
        
        For each document, respond with a JSON object containing:
        - document_id: The document number (1, 2, etc.)
        - score: A number between 0 and 1 indicating relevance
        - reason: A brief explanation of your scoring (1-2 sentences)
        - is_relevant: Boolean indicating if the document is relevant (score >= 0.7)
        
        Format your response as a JSON array with one object per document:
        [
            {"document_id": 1, "score": 0.8, "reason": "Contains specific information about X", "is_relevant": true},
            {"document_id": 2, "score": 0.2, "reason": "Not related to the query", "is_relevant": false},
            ...
        ]
        """)
        
        graded_docs = []
        reasoning = []
        
        try:
            logger.info("Sending batch to LLM for grading")
            llm_start_time = time.time()
            
            chain = batch_grading_prompt | self.llm | JsonOutputParser()
            grades = chain.invoke({
                "query": query,
                "documents": batch_docs
            })
            
            logger.info(f"Batch LLM grading completed in {time.time() - llm_start_time:.2f}s")
            
            # Process the results
            if isinstance(grades, list):
                for grade in grades:
                    doc_id = grade.get("document_id")
                    is_relevant = grade.get("is_relevant", False)
                    
                    # Adjust for 0 vs 1-based indexing
                    if doc_id is not None and 1 <= doc_id <= len(documents):
                        doc = documents[doc_id - 1]
                        
                        if is_relevant:
                            graded_docs.append(doc)
                            logger.info(f"Document {doc_id} scored {grade.get('score', 0)}, marked as relevant")
                        else:
                            logger.info(f"Document {doc_id} scored {grade.get('score', 0)}, marked as not relevant")
                        
                        reasoning.append(f"Document {doc_id} from {doc.metadata.get('tab', 'unknown')}: "
                                        f"Score {grade.get('score', 0)} - {grade.get('reason', '')}")
            else:
                logger.error("Invalid response format from batch grading")
                # Fallback to accepting all documents
                graded_docs = documents
                reasoning = ["Used fallback grading due to invalid LLM response"]
                
        except Exception as e:
            logger.error(f"Error in batch document grading: {str(e)}", exc_info=True)
            # If batch grading fails, include all documents
            graded_docs = documents
            reasoning = [f"Used fallback grading due to error: {str(e)}"]
        
        logger.info(f"Batch document grading completed in {time.time() - start_time:.2f}s")
        
        return graded_docs, reasoning
    
    def decide_after_grading(self, state: AgentState) -> str:
        """Decide next step after document grading"""
        if state["needs_confirmation"] or not state["confirmed_docs"]:
            return "insufficient"
        return "sufficient"
    
    def generate_initial_answer(self, state: AgentState) -> dict:
        """Generate an initial answer based on retrieved documents"""
        start_time = time.time()
        logger.info("Starting answer generation")
        
        user_query = state["messages"][-1].content
        documents = state["confirmed_docs"]
        
        logger.info(f"Generating answer from {len(documents)} confirmed documents")
        
        # Format context from documents with enhanced metadata
        context_parts = []
        total_context_length = 0
        
        for doc in documents:
            context_part = f"From {doc.metadata.get('tab', 'document')}:\n{doc.page_content}"
            
            # Add package context if available
            if doc.metadata.get("packages_mentioned"):
                context_part += f"\nPackages mentioned: {', '.join(doc.metadata['packages_mentioned'])}"
            
            # Add benefits if available
            if doc.metadata.get("benefits"):
                context_part += f"\nBenefits: {', '.join(doc.metadata['benefits'])}"
            
            # Add numerical data if available
            if doc.metadata.get("numerical_data"):
                numerical_data = doc.metadata["numerical_data"]
                for key, values in numerical_data.items():
                    if values:
                        context_part += f"\n{key.title()}: {', '.join(values)}"
            
            context_parts.append(context_part)
            total_context_length += len(context_part)
        
        context = "\n\n".join(context_parts)
        logger.info(f"Total context length: {total_context_length} characters")
        
        # Check if context is too large and truncate if necessary
        max_context = 10000  # Adjust based on model's context window
        if total_context_length > max_context:
            logger.warning(f"Context too large ({total_context_length}), truncating to {max_context} characters")
            context = context[:max_context] + "\n\n[Context truncated due to length]"
        
        # Enhanced answer prompt with package awareness
        answer_prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant that provides PRECISE and SPECIFIC answers based on the provided context.
        
        IMPORTANT INSTRUCTIONS:
        1. Focus on the EXACT package or situation mentioned in the question
        2. If the user mentions a specific package (like "Bestseller Breakthrough Package"), provide information ONLY for that package
        3. Don't provide generic information that covers all packages unless specifically asked
        4. If you find conflicting information, prioritize the most specific and relevant details
        5. Be direct and concise - avoid listing all possible scenarios unless asked
        6. If the context doesn't contain specific information for the mentioned package, say so clearly
        
        Context:
        {context}
        
        Question: {question}
        
        Provide a precise, specific answer focusing on exactly what was asked:
        """)
        
        try:
            logger.info("Sending to LLM for answer generation")
            llm_start_time = time.time()
            
            chain = answer_prompt | self.llm | StrOutputParser()
            answer = chain.invoke({"context": context, "question": user_query})
            
            logger.info(f"Answer generation LLM call completed in {time.time() - llm_start_time:.2f}s")
            logger.info(f"Answer length: {len(answer)} characters")
        except Exception as e:
            logger.error(f"Error during answer generation: {str(e)}", exc_info=True)
            answer = "I'm sorry, I encountered an error while generating an answer. Please try a more specific question or rephrase your query."
        
        logger.info(f"Answer generation completed in {time.time() - start_time:.2f}s")
        
        return {
            "messages": [AIMessage(content=answer)],
            "reasoning": f"Generated specific answer based on {len(documents)} documents with enhanced metadata"
        }
    
    def validate_answer(self, state: AgentState) -> dict:
        """Validate the generated answer against the source documents"""
        start_time = time.time()
        logger.info("Starting answer validation")
        
        user_query = state["messages"][-2].content  # User query is second to last
        answer = state["messages"][-1].content  # Latest message is the answer
        documents = state["confirmed_docs"]
        
        logger.info(f"Validating answer of length {len(answer)} against {len(documents)} documents")
        
        # Check if answer is grounded in documents
        validation_prompt = ChatPromptTemplate.from_template("""
        Verify whether the following answer is fully supported by the provided context documents.
        Check for any hallucinations, unsupported claims, or missing information.
        
        Question: {question}
        Answer: {answer}
        
        Context Documents:
        {context}
        
        Respond with a JSON object containing:
        - "is_grounded": boolean indicating if the answer is fully supported
        - "confidence": number between 0 and 1 indicating confidence in validation
        - "issues": list of any identified issues or missing information
        - "suggestion": suggestion for improving the answer if needed
        """)
        
        # Prepare context, but limit size to avoid timeouts
        context_parts = []
        total_context_length = 0
        max_context_for_validation = 8000  # Lower than for generation to speed up validation
        
        for doc in documents:
            doc_content = doc.page_content
            # Skip if adding this document would exceed the limit
            if total_context_length + len(doc_content) > max_context_for_validation:
                logger.info(f"Skipping document for validation due to context length limit")
                continue
            
            context_parts.append(doc_content)
            total_context_length += len(doc_content)
        
        context = "\n\n".join(context_parts)
        
        if len(context) > max_context_for_validation:
            context = context[:max_context_for_validation] + "\n\n[Context truncated for validation]"
            logger.warning(f"Context for validation truncated to {max_context_for_validation} characters")
        
        logger.info(f"Validation context length: {len(context)} characters")
        
        try:
            logger.info("Sending to LLM for answer validation")
            llm_start_time = time.time()
            
            chain = validation_prompt | self.llm | JsonOutputParser()
            validation = chain.invoke({
                "question": user_query,
                "answer": answer,
                "context": context
            })
            
            logger.info(f"Validation LLM call completed in {time.time() - llm_start_time:.2f}s")
            
            is_confirmed = (
                validation.get("is_grounded", False) and 
                validation.get("confidence", 0) >= self.config.confidence_threshold
            )
            
            logger.info(f"Validation result: is_grounded={validation.get('is_grounded')}, confidence={validation.get('confidence')}")
            if validation.get("issues"):
                logger.info(f"Validation issues: {validation.get('issues')}")
            
            logger.info(f"Answer validation completed in {time.time() - start_time:.2f}s")
            
            return {
                "confirmation_result": is_confirmed,
                "reasoning": f"Validation: {validation.get('suggestion', 'No issues found')}",
                "needs_confirmation": not is_confirmed
            }
        except Exception as e:
            logger.error(f"Error during answer validation: {str(e)}", exc_info=True)
            # If validation fails, assume the answer is acceptable to avoid getting stuck
            return {
                "confirmation_result": True,
                "reasoning": f"Validation failed: {str(e)}",
                "needs_confirmation": False
            }
    
    def decide_after_validation(self, state: AgentState) -> str:
        """Decide next step after validation"""
        if state["needs_confirmation"] and state["query_attempts"] < 3:
            return "needs_revision"
        return "confirmed"
    
    def request_clarification(self, state: AgentState) -> dict:
        """Request clarification when documents are insufficient"""
        clarification_prompt = """
        I couldn't find enough specific information in the document to answer your question accurately. 
        Could you please:
        1. Provide more context about what you're looking for?
        2. Specify which section or tab of the document you're referring to?
        3. Rephrase your question?
        
        This will help me provide a more accurate answer.
        """
        
        return {
            "messages": [AIMessage(content=clarification_prompt)]
        }
    
    def provide_final_answer(self, state: AgentState) -> dict:
        """Provide the final answer, potentially with caveats"""
        answer = state["messages"][-1].content
        
        if not state["confirmation_result"]:
            # Add disclaimer for unconfirmed answers
            disclaimer = "\n\nNote: I couldn't verify all details in the document. This answer is based on the available information but may be incomplete."
            answer += disclaimer
        
        return {
            "messages": [AIMessage(content=answer)]
        }
    
    def invoke(self, query: str) -> str:
        """Invoke the agent with a query"""
        overall_start_time = time.time()
        logger.info(f"Starting validation agent with query: {query[:100]}...")
        
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "retrieved_docs": [],
            "confirmed_docs": [],
            "reasoning": "",
            "needs_confirmation": False,
            "confirmation_result": False,
            "query_attempts": 0
        }
        
        try:
            result = self.graph.invoke(initial_state)
            answer = result["messages"][-1].content
            logger.info(f"Validation agent completed in {time.time() - overall_start_time:.2f}s")
            logger.info(f"Final answer length: {len(answer)} characters")
            return answer
        except Exception as e:
            logger.error(f"Error in validation agent graph: {str(e)}", exc_info=True)
            # Provide a fallback response
            return "I'm sorry, I encountered a timeout or error while processing your question. Please try a more specific or shorter question."
