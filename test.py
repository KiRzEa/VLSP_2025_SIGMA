# search_vector_db.py
import torch

from PIL import Image
from pathlib import Path

from src.services.service_manager import get_service

service = get_service()

VEC_DB_PATH = Path("./assets/vector_db.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === LOAD VECTOR DB ===
db = torch.load(VEC_DB_PATH, weights_only=False)
features = torch.tensor(db["features"])  # shape: (N, D)
filenames = db["filenames"]


# === SEARCH FUNCTION ===
def search_image(image_path: Image.Image, top_k=5):
    query_feat = torch.tensor(service.vit.encode_paths([image_path]))

    # Compute cosine similarity
    sim = torch.matmul(features, query_feat.T).squeeze(1)
    top_scores, top_indices = torch.topk(sim, top_k)

    results = [
        {"filename": filenames[i], "score": float(top_scores[j])}
        for j, i in enumerate(top_indices)
    ]
    return results

# === EXAMPLE USAGE ===
if __name__ == "__main__":
    query_path = "./data/processed/law_db/images/image1222.jpg"
    results = search_image(query_path)

    print("Top matches:")
    for res in results:
        print(f"{res['filename']} - Score: {res['score']:.4f}")
