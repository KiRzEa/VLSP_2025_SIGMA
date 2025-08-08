import torch
from typing import Union, List, Dict

from langchain_huggingface import HuggingFaceEmbeddings

from src.services.embedding_service.models.text_embedding.base import BaseTextEmbeddingModel

class E5TextEmbeddingModel(BaseTextEmbeddingModel):
    def __init__(
            self,
            model_name: str = "tranguyen/halong_embedding-legal-document-finetune",
            model_kwargs: Dict = {"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs: Dict = {"normalize_embeddings": False}
    ):
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.encode_kwargs = encode_kwargs

        self.load_model()

    @property
    def embed_dim(self) -> int:
        return 768
    
    def load_model(self) -> HuggingFaceEmbeddings:
        self.model = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs=self.model_kwargs,
            encode_kwargs=self.encode_kwargs,
        )

    def preprocess(self, batch):
        pass

    def encode(self, texts: Union[str, List[str]]):
        if isinstance(texts, str):
            return self.model.embed_query(text=texts)
        elif isinstance(texts, list):
            return self.model.embed_documents(texts=texts)