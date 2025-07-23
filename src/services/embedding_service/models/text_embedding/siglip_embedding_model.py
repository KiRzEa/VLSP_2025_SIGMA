import torch
import numpy as np
from pathlib import Path
from typing import List, Union, Any, Sequence
from transformers import AutoTokenizer, AutoModel
from src.services.embedding_service.text_embedding.base import BaseTextEmbeddingModel

class HuggingFaceSigLIPTextEmbeddingModel(BaseTextEmbeddingModel):
    
    def __init__(
        self,
        pretrained_model_name_or_path: str = "google/siglip-so400m-patch14-384",
        device: Union[str, torch.device] = "cpu",
        **kwargs
    ):
        self.device = device
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        super().__init__(**kwargs)
    
    def load_model(self) -> Any:
        model = AutoModel.from_pretrained(self.pretrained_model_name_or_path).to(self.device).eval()
        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path)
        return model, tokenizer
    
    def preprocess(self, texts: List[str]):
        return self.tokenizer(texts, truncate=True, padding=True, return_tensors="pt")
    
    @property
    def embed_dim(self):
        return self.model.token_embedding.embedding_dim
    
    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True
    ):
        all_text_embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                batch_tensors = self.preprocess(batch_texts).to(self.device)
                features = self.model.get_text_features(**batch_tensors)
                if normalize:
                    features = torch.nn.functional.normalize(features)
                all_text_embeddings.append(features)
        return torch.cat(all_text_embeddings, dim=0).cpu().numpy()