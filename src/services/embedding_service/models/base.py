import torch
import torch.nn as nn 
from abc import ABC, abstractmethod
from typing import Union, Tuple

class BaseEmbeddingModel(ABC):
    
    @abstractmethod
    def load_model(self) -> Tuple[nn.Module, any]:
        """Load and return model"""
        pass
    
    @abstractmethod
    def preprocess(self, batch):
        """Preprocess input batch"""
        pass
    
    @property
    @abstractmethod
    def embed_dim(self) -> int:
        """Return embedding dimension"""
        pass
    
    def to_device(self, device: Union[str, torch.device]):
        """Move model to a new device in-place"""
        self.device = torch.device(device)
        self.model.to(self.device)
        return self