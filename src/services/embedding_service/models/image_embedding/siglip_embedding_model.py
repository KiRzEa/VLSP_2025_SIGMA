import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Union, Any, Sequence
from transformers import AutoProcessor, AutoModel
from src.services.embedding_service.image_embedding.base import BaseImageEmbeddingModel

class HuggingFaceSigLIPImageEmbeddingModel(BaseImageEmbeddingModel):
    
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
        processor = AutoProcessor.from_pretrained(self.pretrained_model_name_or_path)
        return model, processor
    
    def preprocess(self, images):
        return self.processor(images=images, return_tensors="pt")
    
    def encode(
        self,
        images: Sequence[Image.Image],
        batch_size: int = 32,
        normalize: bool = True
    ):
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