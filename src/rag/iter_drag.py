from typing import List ,Tuple,Dict ,Optional
import torch
from ..utils.data_types import Document, QueryResult , RAGExample, RAGConfig
from ..models.embedding_model import GeckoEmbeddingModel
from ..retrieval.document_store import DocumentStore
from ..models.language_model import LanguageModel
from .base import BaseRAG


class IterDRAG(BaseRAG):
    def __init__(self,
                    document_store: DocumentStore,
                    embedding_model: GeckoEmbeddingModel,
                    language_model :LanguageModel,
                    config :RAGConfig):
        super().__init__(document_store.embedding_model,language_model,config)
    
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

                 
   
