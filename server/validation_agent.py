from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage, Document
from langchain.tools import Tool, tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from typing import TypedDict, Annotated, Sequence, List, Optional, Callable
import operator
from config import Config
from enhanced_retriever import PackageAwareRetriever


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
        # Extract the latest user message
        user_query = state["messages"][-1].content if state["messages"] else ""
        
        # Use enhanced retriever if available, otherwise fall back to standard retriever
        if self.enhanced_retriever:
            documents = self.enhanced_retriever.retrieve_with_context(user_query)
        else:
            # Try to detect which tab the query might belong to
            tab_filter = self._detect_tab_from_query(user_query)
            documents = self.retriever(user_query, tab_filter)
        
        return {
            "retrieved_docs": documents,
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
        """Grade the retrieved documents for relevance"""
        user_query = state["messages"][-1].content
        documents = state["retrieved_docs"]
        
        if not documents:
            return {
                "reasoning": "No documents retrieved. Need to request clarification.",
                "confirmed_docs": []
            }
        
        # Grade each document
        grading_prompt = ChatPromptTemplate.from_template("""
        You are a document grader. Evaluate whether the following document is relevant to the user's query.
        Consider if the document contains information that can help answer the query.
        
        Query: {query}
        Document: {document}
        
        Respond with a JSON object containing:
        - "score": a number between 0 and 1 indicating relevance
        - "reason": a brief explanation of your scoring
        - "is_relevant": boolean indicating if the document is relevant (score >= 0.7)
        """)
        
        graded_docs = []
        reasoning = []
        
        for doc in documents:
            chain = grading_prompt | self.llm | JsonOutputParser()
            try:
                grade = chain.invoke({
                    "query": user_query,
                    "document": doc.page_content[:1000]  # Limit size for grading
                })
                
                if grade.get("is_relevant", False):
                    graded_docs.append(doc)
                
                reasoning.append(f"Document from {doc.metadata.get('tab', 'unknown')}: Score {grade.get('score', 0)} - {grade.get('reason', '')}")
            except Exception as e:
                # If grading fails, include the document anyway
                graded_docs.append(doc)
                reasoning.append(f"Document from {doc.metadata.get('tab', 'unknown')}: Unable to grade - {str(e)}")
        
        # Check if we have enough relevant documents
        has_sufficient_docs = len(graded_docs) >= 2 or (
            len(graded_docs) == 1 and 
            len(graded_docs[0].page_content) > 200  # Substantial single document
        )
        
        return {
            "confirmed_docs": graded_docs,
            "reasoning": "\n".join(reasoning),
            "needs_confirmation": not has_sufficient_docs
        }
    
    def decide_after_grading(self, state: AgentState) -> str:
        """Decide next step after document grading"""
        if state["needs_confirmation"] or not state["confirmed_docs"]:
            return "insufficient"
        return "sufficient"
    
    def generate_initial_answer(self, state: AgentState) -> dict:
        """Generate an initial answer based on retrieved documents"""
        user_query = state["messages"][-1].content
        documents = state["confirmed_docs"]
        
        # Format context from documents with enhanced metadata
        context_parts = []
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
        
        context = "\n\n".join(context_parts)
        
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
        
        chain = answer_prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": user_query})
        
        return {
            "messages": [AIMessage(content=answer)],
            "reasoning": f"Generated specific answer based on {len(documents)} documents with enhanced metadata"
        }
    
    def validate_answer(self, state: AgentState) -> dict:
        """Validate the generated answer against the source documents"""
        user_query = state["messages"][-2].content  # User query is second to last
        answer = state["messages"][-1].content  # Latest message is the answer
        documents = state["confirmed_docs"]
        
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
        
        context = "\n\n".join([doc.page_content for doc in documents])
        chain = validation_prompt | self.llm | JsonOutputParser()
        
        try:
            validation = chain.invoke({
                "question": user_query,
                "answer": answer,
                "context": context
            })
            
            is_confirmed = (
                validation.get("is_grounded", False) and 
                validation.get("confidence", 0) >= self.config.confidence_threshold
            )
            
            return {
                "confirmation_result": is_confirmed,
                "reasoning": f"Validation: {validation.get('suggestion', 'No issues found')}",
                "needs_confirmation": not is_confirmed
            }
        except Exception as e:
            # If validation fails, assume the answer is acceptable
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
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "retrieved_docs": [],
            "confirmed_docs": [],
            "reasoning": "",
            "needs_confirmation": False,
            "confirmation_result": False,
            "query_attempts": 0
        }
        
        result = self.graph.invoke(initial_state)
        return result["messages"][-1].content
