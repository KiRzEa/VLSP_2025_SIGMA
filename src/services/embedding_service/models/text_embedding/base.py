import torch
from abc import abstractmethod
from typing import Union, List
import numpy as np
from src.services.embedding_service.models.base import BaseEmbeddingModel

class BaseTextEmbeddingModel(BaseEmbeddingModel):

    def __init__(self):
        super().__init__()
        self.model, self.tokenizer = self.load_model()

    @abstractmethod
    def preprocess(self, batch: List[str]):
        pass

    @abstractmethod
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        pass

    def encode_file(self, file_path: str, batch_size: int = 32) -> np.ndarray:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        return self.encode(lines, batch_size=batch_size)