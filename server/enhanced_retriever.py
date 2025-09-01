import re
from typing import List, Dict, Any, Optional, Tuple
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from config import Config


class PackageAwareRetriever:
    def __init__(self, vector_store, config: Config):
        self.vector_store = vector_store
        self.config = config
        self.llm = ChatGoogleGenerativeAI(
            model=config.model_name,
            google_api_key=config.google_api_key,
            temperature=0.1
        )
    
    def retrieve_with_context(self, query: str) -> List[Document]:
        """Retrieve documents with package context awareness"""
        
        # 1. Extract package information from query
        package_context = self._extract_package_context(query)
        
        # 2. Perform multi-stage retrieval
        candidates = self._multi_stage_retrieval(query, package_context)
        
        # 3. Re-rank based on package relevance
        ranked_docs = self._rerank_by_package_relevance(candidates, package_context, query)
        
        return ranked_docs[:self.config.top_k]
    
    def _extract_package_context(self, query: str) -> Dict[str, Any]:
        """Extract package context from the query"""
        context = {
            "package_type": None,
            "specific_benefit": None,
            "location": None,
            "numerical_query": False
        }
        
        query_lower = query.lower()
        
        # Extract package type
        if "bestseller" in query_lower or "breakthrough" in query_lower:
            context["package_type"] = "bestseller breakthrough package"
        elif "limited" in query_lower:
            context["package_type"] = "limited publishing package"
        elif "premium" in query_lower:
            context["package_type"] = "premium package"
        
        # Extract specific benefits being asked about
        if "author copies" in query_lower or "free copies" in query_lower:
            context["specific_benefit"] = "author_copies"
        elif "shipping" in query_lower or "delivery" in query_lower:
            context["specific_benefit"] = "shipping"
        elif "coupon" in query_lower:
            context["specific_benefit"] = "coupon_code"
        elif "bulk order" in query_lower:
            context["specific_benefit"] = "bulk_order"
        
        # Extract location context
        if "indian" in query_lower or "india" in query_lower:
            context["location"] = "india"
        elif "international" in query_lower:
            context["location"] = "international"
        
        # Check if it's a numerical query
        if re.search(r'\b(how many|how much|\d+)\b', query_lower):
            context["numerical_query"] = True
        
        return context
    
    def _multi_stage_retrieval(self, query: str, package_context: Dict) -> List[Document]:
        """Perform multi-stage retrieval"""
        all_candidates = []
        
        # Stage 1: Direct semantic search
        semantic_docs = self.vector_store.similarity_search(query, k=self.config.top_k * 2)
        all_candidates.extend(semantic_docs)
        
        # Stage 2: Package-specific search if package context is detected
        if package_context["package_type"]:
            package_query = f"{query} {package_context['package_type']}"
            package_docs = self.vector_store.similarity_search(package_query, k=self.config.top_k)
            all_candidates.extend(package_docs)
        
        # Stage 3: Benefit-specific search
        if package_context["specific_benefit"]:
            benefit_query = f"{package_context['specific_benefit']} {query}"
            benefit_docs = self.vector_store.similarity_search(benefit_query, k=self.config.top_k)
            all_candidates.extend(benefit_docs)
        
        # Stage 4: Search for exact phrases
        exact_phrases = self._extract_exact_phrases(query)
        for phrase in exact_phrases:
            phrase_docs = self.vector_store.similarity_search(phrase, k=3)
            all_candidates.extend(phrase_docs)
        
        # Remove duplicates while preserving order
        seen_content = set()
        unique_candidates = []
        for doc in all_candidates:
            if doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                unique_candidates.append(doc)
        
        return unique_candidates
    
    def _extract_exact_phrases(self, query: str) -> List[str]:
        """Extract important exact phrases from the query"""
        phrases = []
        
        # Look for quoted phrases
        quoted_phrases = re.findall(r'"([^"]+)"', query)
        phrases.extend(quoted_phrases)
        
        # Look for important compound terms
        important_terms = [
            "author copies", "bestseller breakthrough", "limited publishing",
            "coupon code", "bulk order", "delivery charge", "international authors",
            "indian authors", "complimentary copies", "free copies"
        ]
        
        query_lower = query.lower()
        for term in important_terms:
            if term in query_lower:
                phrases.append(term)
        
        return phrases
    
    def _rerank_by_package_relevance(self, candidates: List[Document], 
                                   package_context: Dict, query: str) -> List[Document]:
        """Re-rank documents based on package relevance"""
        scored_docs = []
        
        for doc in candidates:
            score = self._calculate_relevance_score(doc, package_context, query)
            scored_docs.append((doc, score))
        
        # Sort by score in descending order
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in scored_docs]
    
    def _calculate_relevance_score(self, doc: Document, 
                                 package_context: Dict, query: str) -> float:
        """Calculate relevance score for a document"""
        score = 0.0
        content_lower = doc.page_content.lower()
        metadata = doc.metadata
        
        # Base semantic relevance (this would be the original similarity score)
        score += 1.0
        
        # Package type matching
        if package_context["package_type"]:
            if package_context["package_type"] in content_lower:
                score += 2.0
            # Check metadata
            if metadata.get("packages_mentioned") and any(
                package_context["package_type"] in pkg.lower() 
                for pkg in metadata["packages_mentioned"]
            ):
                score += 1.5
        
        # Specific benefit matching
        if package_context["specific_benefit"]:
            benefit = package_context["specific_benefit"].replace("_", " ")
            if benefit in content_lower:
                score += 1.5
            # Check benefits in metadata
            if metadata.get("benefits") and any(
                benefit in b.lower() for b in metadata["benefits"]
            ):
                score += 1.0
        
        # Location matching
        if package_context["location"]:
            if package_context["location"] in content_lower:
                score += 1.0
        
        # Numerical query bonus
        if package_context["numerical_query"]:
            # Look for numbers in content
            if re.search(r'\d+', content_lower):
                score += 0.5
            # Check numerical metadata
            if metadata.get("numerical_data"):
                numerical_data = metadata["numerical_data"]
                if any(numerical_data.values()):  # Has any numerical data
                    score += 1.0
        
        # Exact phrase matching
        query_lower = query.lower()
        exact_phrases = self._extract_exact_phrases(query_lower)
        for phrase in exact_phrases:
            if phrase.lower() in content_lower:
                score += 1.5
        
        # Content type bonus
        content_type = metadata.get("content_type", "")
        if content_type == "numeric_data" and package_context["numerical_query"]:
            score += 0.5
        
        # Penalize generic content
        generic_phrases = [
            "depends on your package", "varies by location", "contact support",
            "terms and conditions apply"
        ]
        for phrase in generic_phrases:
            if phrase in content_lower:
                score -= 0.5
        
        return score
    
    def get_retrieval_explanation(self, query: str, retrieved_docs: List[Document]) -> str:
        """Provide explanation of why these documents were retrieved"""
        package_context = self._extract_package_context(query)
        
        explanation = f"Retrieved {len(retrieved_docs)} documents for query: '{query}'\n"
        
        if package_context["package_type"]:
            explanation += f"- Detected package type: {package_context['package_type']}\n"
        
        if package_context["specific_benefit"]:
            explanation += f"- Specific benefit requested: {package_context['specific_benefit']}\n"
        
        if package_context["location"]:
            explanation += f"- Location context: {package_context['location']}\n"
        
        if package_context["numerical_query"]:
            explanation += "- Numerical information requested\n"
        
        explanation += "\nDocument relevance breakdown:\n"
        for i, doc in enumerate(retrieved_docs, 1):
            score = self._calculate_relevance_score(doc, package_context, query)
            explanation += f"{i}. Score: {score:.2f} - {doc.page_content[:100]}...\n"
        
        return explanation
