from typing import List, Tuple, Dict, Optional
import torch
from ..utils.data_types import Document, QueryResult, RAGExample, RAGConfig
from ..models.embedding_model import GeckoEmbeddingModel
from ..retrieval.document_store import DocumentStore
from ..models.language_model import LanguageModel
from .base import BaseRAG

class IterDRAG(BaseRAG):
    """
    Implementation of Iterative Demonstration-based RAG (IterDRAG)
    as described in the paper for handling complex multi-hop queries
    """
    
    def __init__(self,
                 document_store: DocumentStore,
                 embedding_model: GeckoEmbeddingModel,
                 language_model: LanguageModel,
                 config: RAGConfig):
        super().__init__(document_store, embedding_model, language_model, config)
        
    def _format_prompt(self,
                      examples: List[RAGExample],
                      documents: List[Document],
                      query: str,
                      sub_queries: List[str] = None,
                      intermediate_answers: List[str] = None) -> str:
        """
        Format prompt for iterative processing with examples and current state
        """
        prompt = self.language_model.format_system_prompt(
            "You are an expert in breaking down and answering complex questions. "
            "For multi-hop questions, decompose them into simpler sub-questions "
            "and answer them step by step. Follow the Self-Ask format:\n"
            "1. Generate a follow-up question starting with 'Follow up:'\n"
            "2. Provide an intermediate answer starting with 'Intermediate answer:'\n"
            "3. When ready, provide the final answer starting with 'So the final answer is:'\n"
        )
        
        # Add demonstrations
        for ex in examples:
            prompt += self.language_model.format_user_prompt(
                "Context:\n" + "\n".join(f"[Doc {i+1}]: {doc.content}" 
                for i, doc in enumerate(ex.documents))
            )
            prompt += f"\nQuestion: {ex.query}\n"
            
            if ex.sub_queries and ex.intermediate_answers:
                for sq, ia in zip(ex.sub_queries, ex.intermediate_answers):
                    prompt += f"Follow up: {sq}\n"
                    prompt += f"Intermediate answer: {ia}\n"
            
            prompt += f"So the final answer is: {ex.answer}\n\n"
        
        # Add current query context
        prompt += self.language_model.format_user_prompt(
            "Context:\n" + "\n".join(f"[Doc {i+1}]: {doc.content}" 
            for i, doc in enumerate(documents))
        )
        prompt += f"\nQuestion: {query}\n"
        
        # Add current progress if available
        if sub_queries and intermediate_answers:
            for sq, ia in zip(sub_queries, intermediate_answers):
                prompt += f"Follow up: {sq}\n"
                prompt += f"Intermediate answer: {ia}\n"
        
        return prompt
    
    def _generate_next_step(self, prompt: str) -> Tuple[str, str]:
        """
        Generate the next step in the iterative process:
        - sub-query for additional information
        - intermediate answer
        - final answer
        """
        response = self.language_model.generate(
            prompt,
            max_new_tokens=100,
            temperature=0.7
        )[0]
        
        if "Follow up:" in response:
            # Extract sub-query
            sub_query = response.split("Follow up:")[1].split("\n")[0].strip()
            return "sub_query", sub_query
        elif "So the final answer is:" in response:
            # Extract final answer
            final_answer = response.split("So the final answer is:")[1].strip()
            return "final", final_answer
        else:
            # Extract intermediate answer
            answer = response.split("Intermediate answer:")[1].split("\n")[0].strip()
            return "intermediate", answer
    
    def _retrieve_for_sub_query(self, 
                              sub_query: str, 
                              existing_docs: List[Document]) -> List[Document]:
        """
        Retrieve additional documents for a sub-query while avoiding duplicates
        """
        sub_query_embedding = self.embedding_model.embed(sub_query)
        
        # Retrieve new documents
        new_docs = self.document_store.retrieve(
            sub_query_embedding,
            k=self.config.num_documents // 2  # Retrieve fewer docs for sub-queries
        )
        
        # Filter out duplicates
        existing_ids = {doc.doc_id for doc in existing_docs}
        filtered_docs = [doc for doc in new_docs if doc.doc_id not in existing_ids]
        
        return filtered_docs
    
    def process_query(self, 
                     query: str, 
                     examples: List[RAGExample]) -> QueryResult:
        """
        Process a query using IterDRAG approach:
        1. Start with initial retrieval
        2. Iteratively:
            - Generate sub-queries
            - Retrieve additional context
            - Generate intermediate answers
            - Produce final answer
        """
        # Initial retrieval
        query_embedding = self.embedding_model.embed(query)
        documents = self.document_store.retrieve(
            query_embedding,
            k=self.config.num_documents
        )
        
        sub_queries = []
        intermediate_answers = []
        total_context_length = 0
        
        # Keep track of highest relevance score
        max_confidence = max(doc.score for doc in documents)
        
        # Iterative processing
        for iteration in range(self.config.max_iterations):
            # Format current state as prompt
            prompt = self._format_prompt(
                examples[:self.config.num_shots],
                documents,
                query,
                sub_queries,
                intermediate_answers
            )
            
            # Update context length tracking
            step_length = self.language_model.get_token_count(prompt)
            total_context_length += step_length
            
            # Check if we've exceeded max context length
            if total_context_length > self.config.max_context_length:
                # Force final answer generation
                final_prompt = self._format_prompt(
                    examples[:1],  # Use fewer examples to fit context
                    documents[-10:],  # Use most recent documents
                    query,
                    sub_queries,
                    intermediate_answers
                )
                _, final_answer = self._generate_next_step(final_prompt + "\nSo the final answer is:")
                break
            
            # Generate next step
            step_type, content = self._generate_next_step(prompt)
            
            if step_type == "final":
                final_answer = content
                break
                
            elif step_type == "sub_query":
                sub_queries.append(content)
                
                # Retrieve additional documents for sub-query
                new_docs = self._retrieve_for_sub_query(content, documents)
                documents.extend(new_docs)
                
                # Update confidence if we found more relevant documents
                if new_docs:
                    max_confidence = max(max_confidence, 
                                      max(doc.score for doc in new_docs))
                
            else:  # intermediate answer
                intermediate_answers.append(content)
            
            # Force final answer on last iteration
            if iteration == self.config.max_iterations - 1:
                final_prompt = prompt + "\nSo the final answer is:"
                _, final_answer = self._generate_next_step(final_prompt)
        
        return QueryResult(
            query=query,
            documents=documents,
            answer=final_answer,
            confidence=max_confidence,
            sub_queries=sub_queries,
            intermediate_answers=intermediate_answers,
            effective_context_length=total_context_length
        )
    
    def evaluate_decomposition(self, 
                             query_result: QueryResult,
                             ground_truth: str) -> Dict[str, float]:
        """
        Evaluate the quality of query decomposition and intermediate steps
        """
        metrics = {
            'num_steps': len(query_result.sub_queries),
            'avg_docs_per_step': len(query_result.documents) / max(1, len(query_result.sub_queries)),
            'context_efficiency': query_result.confidence / query_result.effective_context_length
        }
        
        return metrics


              




                 
   
