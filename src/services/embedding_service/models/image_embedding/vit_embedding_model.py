import torch
import numpy as np

from PIL import Image
from tqdm.auto import tqdm
from typing import Sequence, Tuple, Any
from transformers import AutoModel, AutoImageProcessor

from src.services.embedding_service.models.image_embedding.base import BaseImageEmbeddingModel


class ViTImageEmbeddingModel(BaseImageEmbeddingModel):
    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224-in21k",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model_name = model_name
        self.device = device
        super().__init__()
    
    @property
    def embed_dim(self):
        return self.model.config.hidden_size
    
    def preprocess(self, batch: Sequence[Image.Image]):
        return self.processor(batch, return_tensors="pt")

    def load_model(self) -> Tuple[torch.nn.Module, Any]:
        processor = AutoImageProcessor.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name).to(self.device)
        model.eval()
        return model, processor

    def encode(self, images: Sequence[Image.Image], batch_size: int = 32) -> np.ndarray:
        all_features = []

        with torch.no_grad():
            for i in tqdm(range(0, len(images), batch_size)):
                batch = images[i:i + batch_size]
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                features = outputs.last_hidden_state[:, 0, :]  # CLS token
                all_features.append(features.cpu())

        return torch.cat(all_features, dim=0).numpy()
