from transformers import AutoTokenizer, AutoModel
import torch
from typing import List, Union
import numpy as np

class GeckoEmbeddingModel:
    """Implements the Gecko embedding model for document and query encoding"""
    
    def __init__(self, model_name: str = "google/gecko-1b"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
    def embed(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """Generate embeddings for single text or batch of texts"""
        if isinstance(texts, str):
            texts = [texts]
            
        # Tokenize with padding and truncation
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        )
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]  # Use CLS token
            
        return embeddings
    
    def compute_similarity(self, query_emb: torch.Tensor, doc_embs: torch.Tensor) -> torch.Tensor:
        """Compute similarity scores between query and documents"""
        return torch.matmul(query_emb, doc_embs.T)
