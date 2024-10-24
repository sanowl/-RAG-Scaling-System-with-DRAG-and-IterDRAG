from  dataclasses import dataclass, field
from  typing import List, Optional ,Dict
import torch

@dataclass
class Document:
    content :str
    doc_id: str
    socre: float


@dataclass
class QueryResult:
    """" this will store the results for both drag and iterDrag processing"""
    query: str
    documents: List[Document]
    answer: str
    confidence: float
    sub_queries: List[str] = field(default_factory=list)
    intermediate_answers: List[str] = field(default_factory=list)
    effective_context_length: int = 0

@dataclass
class RAGExample:
    """Stores demonstration examples for in-context learning"""
    documents: List[Document]
    query :str
    answer:str 
    sub_queries: Optional[List[str]] = None
    intermediate_answers: Optional[List[str]] = None

@dataclass
class RAGConfig:
    """Configuration for RAG processing"""
    num_documents: int = 50  # k in paper
    num_shots: int = 8      # m in paper
    max_iterations: int = 5  # n in paper
    max_doc_length: int = 1024
    max_context_length: int = 1_000_000  # Gemini 1.5 Flash context window

    

    

