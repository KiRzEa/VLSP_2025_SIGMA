import os
import torch
from pathlib import Path
from typing import List, Dict, Union
from PIL import Image

from src.services.retrieval_service.base import BaseRetriever
from src.services.embedding_service import ViTImageEmbeddingModel

IMAGE_INDEX_STORAGE = os.environ.get("IMAGE_INDEX_STORAGE")

class ImageRetriever(BaseRetriever):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.db_path = Path(IMAGE_INDEX_STORAGE)
        self.encoder = ViTImageEmbeddingModel()

        self._load_db()

    def _load_db(self):
        db = torch.load(self.db_path, weights_only=False)
        self.features = torch.tensor(db["features"])  # (N, D)
        self.filenames = db["filenames"]

    def search(self, image: Union[str, Path, Image.Image], top_k=5) -> List[Dict]:
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        query_feat = torch.tensor(self.encoder.encode([image]))

        sim = torch.matmul(self.features, query_feat.T).squeeze(1)
        top_scores, top_indices = torch.topk(sim, top_k)

        return [
            {"filename": self.filenames[i], "score": float(top_scores[j])}
            for j, i in enumerate(top_indices)
        ]
