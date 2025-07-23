import torch
import open_clip
import numpy as np
from PIL import Image
from typing import List, Any, Sequence, Union
from transformers import AutoProcessor, CLIPModel
from src.services.embedding_service.image_embedding.base import BaseImageEmbeddingModel

class OpenCLIPImageEmbeddingModel(BaseImageEmbeddingModel):
    
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
        model, _, processor = open_clip.create_model_and_transforms(
            model_name=self.model_name,
            pretrained=self.pretrained
        )
        return model.to(self.device).eval(), processor
    
    def preprocess(self, images) -> Any:
        return torch.stack(
            [self.processor(image) for image in images]
        )
    
    @property
    def embed_dim(self):
        return self.model.token_embedding.embedding_dim
    
    def encode(
        self, 
        images: Sequence[Image.Image], 
        batch_size: int = 32,
        normalize: bool = True
    ) -> np.ndarray:
        all_image_embeddings: List[torch.Tensor] = []
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch_images = images[i : i + batch_size]
                batch_tensors = self.preprocess(batch_images).to(self.device)
                features = self.model.encode_image(batch_tensors)
                if normalize:
                    features = torch.nn.functional.normalize(features)
                all_image_embeddings.append(features)
        return torch.cat(all_image_embeddings, dim=0).cpu().numpy()
    
class HuggingFaceCLIPImageEmbeddingModel(BaseImageEmbeddingModel):
    
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
        model = CLIPModel.from_pretrained(
            pretrained_model_name_or_path=self.pretrained_model_name_or_path
        ).to(self.device).eval()
        
        processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path=self.pretrained_model_name_or_path
        )
        return model, processor
    
    def preprocess(self, images: List[Image.Image]):
        return self.processor(images=images, return_tensors="pt")
        
    @property
    def embed_dim(self):
        return self.model.token_embedding.embedding_dim
    
    def encode(
        self, 
        images: Sequence[Image.Image], 
        batch_size: int = 32, 
        normalize: bool = True
    ) -> np.ndarray:
        all_image_embeddings: List[torch.Tensor] = []
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch_images = images[i : i + batch_size]
                batch_tensors = self.preprocess(batch_images).to(self.device)
                features = self.model.get_image_features(**batch_tensors)
                if normalize:
                    features = torch.nn.functional.normalize(features)
                all_image_embeddings.append(features)
        return torch.cat(all_image_embeddings, dim=0).cpu().numpy()