import torch
import open_clip
import numpy as np
from typing import List, Any, Union
from transformers import AutoTokenizer, CLIPModel
from src.services.embedding_service.text_embedding.base import BaseTextEmbeddingModel

class OpenCLIPTextEmbeddingModel(BaseTextEmbeddingModel):

    def __init__(
        self, 
        model_name: str = "ViT-B-32", 
        pretrained: str = "laion2b_s34b_b79k", 
        device: Union[str, torch.device] = "cpu", 
        **kwargs
    ):
        self.device = device
        self.model_name = model_name
        self.pretrained = pretrained
        super().__init__(**kwargs)

    def load_model(self) -> Any:
        model, _, _ = open_clip.create_model_and_transforms(
            model_name=self.model_name,
            pretrained=self.pretrained
        )
        tokenizer = open_clip.get_tokenizer(self.model_name)
        return model.to(self.device).eval(), tokenizer
    
    def preprocess(self, texts: List[str]) -> Any:
        return self.tokenizer(texts)
    
    @property
    def embed_dim(self) -> int:
        return self.model.token_embedding.embedding_dim
    
    def encode(
        self, 
        texts: List[str], 
        batch_size: int = 32,
        normalize: bool = True
    ) -> np.ndarray:
        all_text_embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                batch_tensors = self.preprocess(batch_texts).to(self.device)
                features = self.model.encode_text(batch_tensors)
                if normalize:
                    features = torch.nn.functional.normalize(features)
                all_text_embeddings.append(features)
        return torch.cat(all_text_embeddings, dim=0).cpu().numpy()
    
class HuggingFaceCLIPTextEmbeddingModel(BaseTextEmbeddingModel):

    def __init__(
        self,
        pretrained_model_name_or_path: str = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", 
        device: Union[str, torch.device] = "cpu", 
        **kwargs
    ):
        self.device = device
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        super().__init__(**kwargs)

    def load_model(self) -> Any:
        model = CLIPModel.from_pretrained(self.pretrained_model_name_or_path).to(self.device).eval()
        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path)
        return model, tokenizer
    
    def preprocess(self, texts: str):
        return self.tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
    
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