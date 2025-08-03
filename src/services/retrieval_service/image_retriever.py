import os
import torch
from PIL import Image
from pathlib import Path
from typing import List, Union

from src.services.retrieval_service.base import BaseRetriever
from src.services.embedding_service import ViTImageEmbeddingModel
from src.models.image_retrieval_node import (
    ImageEntry,
    SearchResultEntry
)

IMAGE_INDEX_STORAGE = os.environ.get("IMAGE_INDEX_STORAGE")

class ImageRetriever(BaseRetriever):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.db_path = Path(IMAGE_INDEX_STORAGE)
        self.encoder = ViTImageEmbeddingModel()

        self._load_db()

    def _load_db(self):
        db = torch.load(self.db_path, weights_only=False)
        features = db["features"]
        filenames = db["filenames"]
        metadata = db["metadata"]
        descriptions = db["descriptions"]

        self.entries: List[ImageEntry] = [
            ImageEntry(
                filename=filenames[i],
                metadata=metadata[i],
                description=descriptions[i],
                feature=torch.tensor(features[i])
            )
            for i in range(len(filenames))
        ]

    def search(
        self,
        image: Union[str, Path, Image.Image],
        top_k: int = 5,
        traffic_sign_type: str = None
    ) -> List[ImageEntry]:
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        query_feat = torch.tensor(self.encoder.encode([image])).squeeze(0)  # (D,)

        # Optionally filter entries by traffic sign type
        if traffic_sign_type:
            candidate_entries = [
                entry for entry in self.entries
                if "reference_sign_type" in entry.metadata and
                traffic_sign_type in entry.metadata["reference_sign_type"]
            ]
        else:
            candidate_entries = self.entries

        if not candidate_entries:
            return []
        # Stack features into a matrix: (N, D)
        candidate_features = torch.stack([entry.feature for entry in candidate_entries]) 
        # Compute similarity
        # Compute similarity: (N,)
        similarities = torch.matmul(candidate_features, query_feat)

        top_scores, top_indices = torch.topk(similarities, min(top_k, len(similarities)))

        # Return top-k entries with similarity scores
        return [
            SearchResultEntry(
                **candidate_entries[i].model_dump(),
                score=float(top_scores[j])
            )
            for j, i in enumerate(top_indices)
        ]

    def filter_based_on_traffic_sign_type(self, traffic_sign_type: str) -> List[ImageEntry]:
        return [
            entry for entry in self.entries
            if "reference_sign_type" in entry.metadata
            and traffic_sign_type in entry.metadata["reference_sign_type"]
        ]
        

