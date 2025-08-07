import json
import torch
from pathlib import Path
from tqdm.auto import tqdm

from src.services.llm_service import FPTChatModel
from src.services.embedding_service import ViTImageEmbeddingModel
from src.prompts import SIGN_ATTRIBUTES_EXTRACTION_PROMPT
from src.utils import extract_json_from_deepseek_response

# CONFIG
VEC_DB_PATH = Path("./assets/vector_db.pt")
IMAGE_FOLDER = Path("./data/processed/law_db/images")
ARTICLE_JSON = Path('./data/traffic_sign_db/traffic_sign_descriptions.json')

BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    vit = ViTImageEmbeddingModel()
    gemma = FPTChatModel(model_name="gemma-3-27b-it")
    # === LOAD IMAGE FILENAMES FROM ARTICLE JSON ===
    with open(ARTICLE_JSON, "r", encoding="utf-8") as f:
        articles = json.load(f)

    image_filenames = []
    image_paths = []
    image_metadata = []
    image_descriptions = []
    for article in tqdm(articles, desc="[INFO] Generating image descriptions..."):
        for img in article["images"]:
            response = gemma.generate(
                system_prompt=SIGN_ATTRIBUTES_EXTRACTION_PROMPT,
                image_paths=[IMAGE_FOLDER / img]
            )

            image_description = extract_json_from_deepseek_response(
                response=response,
                return_json=True
            )

            image_filenames.append(img)
            image_paths.append(IMAGE_FOLDER / img)
            image_metadata.append(article)
            image_descriptions.append(image_description)
            

    # === FEATURE EXTRACTION ===
    all_features = []
    print("Extracting features...")
    all_features = vit.encode_paths(image_paths)
    # === SAVE VECTOR DB TO DISK ===
    print("Saving vector database to disk...")
    torch.save({
        "features": all_features,
        "filenames": image_filenames,
        "metadata": image_metadata,
        "descriptions": image_descriptions

    }, VEC_DB_PATH)
    print(f"Saved to {VEC_DB_PATH}")

if __name__ == '__main__':
    main()

