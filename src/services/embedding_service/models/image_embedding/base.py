import numpy as np
from PIL import Image
from pathlib import Path
from abc import abstractmethod
from typing import Any, Union, Sequence, List

from src.services.embedding_service.models.base import BaseEmbeddingModel

class BaseImageEmbeddingModel(BaseEmbeddingModel):
    
    def __init__(self):
        super().__init__()
        self.model, self.processor = self.load_model()
    
    @abstractmethod
    def encode(self, images: Sequence[Image.Image], batch_size: int = 32) -> np.ndarray:
        pass
    
    def encode_paths(self, paths: Sequence[Union[str, Path]], batch_size: int = 32) -> np.ndarray:
        images = [Image.open(path).convert("RGB") for path in paths]
        return self.encode(image=images, batch_size=batch_size)