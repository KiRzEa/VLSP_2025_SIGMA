from typing import Union, List

from langchain_huggingface import HuggingFaceEmbeddings

from src.services.embedding_service.models.text_embedding.base import BaseTextEmbeddingModel

class E5TextEmbeddingModel(BaseTextEmbeddingModel):
    def __init__(
            self,
            model_name: str = "tranguyen/halong_embedding-legal-document-finetune"
    ):
        self.model = self.load_model(model_name)

    @property
    def embed_dim(self) -> int:
        return 768
    
    def load_model(self, model_name: str) -> HuggingFaceEmbeddings:
        model = HuggingFaceEmbeddings(model_name=model_name)
        return model
    def preprocess(self, batch):
        pass

    def encode(self, texts: Union[str, List[str]]):
        if isinstance(texts, str):
            return self.model.embed_query(text=texts)
        elif isinstance(texts, list):
            return self.model.embed_documents(texts=texts)