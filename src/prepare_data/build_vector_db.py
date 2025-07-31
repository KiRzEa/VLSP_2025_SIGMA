import os
import json
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from src.services.embedding_service import ViTImageEmbeddingModel

vit = ViTImageEmbeddingModel()

# CONFIG
IMAGE_FOLDER = Path("./data/processed/law_db/images")
ARTICLE_JSON = Path('./data/processed/law_db/articles/traffic_sign_description.json')
VEC_DB_PATH = Path("./assets/vector_db.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

# === LOAD IMAGE FILENAMES FROM ARTICLE JSON ===
with open(ARTICLE_JSON, "r", encoding="utf-8") as f:
    articles = json.load(f)

image_filenames = []
image_paths = []
image_metadata = []
for article in articles:
    for img in article["images"]:
        image_filenames.append(img)
        image_paths.append(IMAGE_FOLDER / img)
        image_metadata.append(article)

# === FEATURE EXTRACTION ===
all_features = []
print("Extracting features...")
all_features = vit.encode_paths(image_paths)

# === SAVE VECTOR DB TO DISK ===
print("Saving vector database to disk...")
torch.save({
    "features": all_features,
    "filenames": image_filenames,
    "metadata": image_metadata
}, VEC_DB_PATH)
print(f"Saved to {VEC_DB_PATH}")

