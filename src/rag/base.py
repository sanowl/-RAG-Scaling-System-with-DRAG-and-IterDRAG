from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from ..utils.data_types import QueryResult, RAGExample, RAGConfig, Document
from ..models.embedding_model import GeckoEmbeddingModel
from ..retrieval.document_store import DocumentStore
from ..models.language_model import LanguageModel

class BaseRAG(ABC):
    """Base class for RAG implementations"""
    
    def __init__(self,
                 document_store: DocumentStore,
                 embedding_model: GeckoEmbeddingModel,
                 language_model: LanguageModel,
                 config: RAGConfig):
        self.document_store = document_store
        self.embedding_model = embedding_model
        self.language_model = language_model
        self.config = config
        
    @abstractmethod
    def process_query(self, query: str, examples: List[RAGExample]) -> QueryResult:
        """Process a query and return results"""
        pass
    
    def _truncate_documents(self, documents: List[Document]) -> List[Document]:
        """Truncate documents to max length while preserving order"""
        truncated_docs = []
        current_length = 0
        
        for doc in documents:
            doc_length = len(doc.content.split())
            if current_length + doc_length <= self.config.max_doc_length:
                truncated_docs.append(doc)
                current_length += doc_length
            else:
                break
                
        return truncated_docs
    
    def _calculate_effective_length(self, prompt: str) -> int:
        """Calculate effective context length of a prompt"""
        return len(self.language_model.tokenize(prompt))
    
    @abstractmethod
    def _format_prompt(self, *args, **kwargs) -> str:
        """Format prompt for the language model"""
        pass