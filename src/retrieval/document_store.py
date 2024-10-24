from typing import List ,Dict ,Optional
import numpy as np
import torch
from ..utils.data_types import Document
from..models.embedding_model import GeckoEmbeddingModel

class DocumentStore:
    def __init__(self,embedding_model: GeckoEmbeddingModel):
        self.documents:Dict[str,Document]={}
        self.embedding_model = embedding_model
        self.documents_embeddings:Optional[torch.Tensor] = None
        self.doc_ids:List[str]=[]

    def add_documents(self,documents:List[Dict [str,str]]):
        """this will add doc to store and compute embd"""
        contents = [doc['content'] for doc in documents]
        embeddinggs = self.embedding_model(contents)

        #store the doc and embd
        for idx,doc in enumerate(documents):
            doc_id = str(len(self.documents))
            documents = Document(
                content=doc['content'],
                doc_id=doc_id,
                socre=0.0
            )
            self.documents[doc_id] = documents 
            self.doc_ids.append(doc_id)

            # update the doc embd
            if self.documents_embeddings is None:
                self.documents_embeddings = embeddinggs
            else:
                self.document_embeddings = torch.cat([self.document_embeddings, embeddinggs])

    def retrieve(self, query_embedding: torch.Tensor, k: int = 50) -> List[Document]:

         #  compute sim scores
         scores = self.embedding_model.compute_similarity(
             query_embedding,
             self.document_embeddings  
         )
         # get top-k documents indices
         top_k_scores ,top_k_indices=torch.topk(scores,k=min(k,len(self.doc_ids)))

         # getting the doc
         retrieved_docs = []
         for idx,score in zip [top_k_indices[0],top_k_scores[0]]:
             doc_id = self .doc_ids[idx]
             doc = self.documents[doc_id]
             doc.socre  = score.item()
             retrieved_docs.append(doc)

 


