import os
import json
import glob
from tqdm.auto import tqdm

from src.core.logger import setup_logger
from src.services.vector_store import ElasticVectorStore
from src.prepare_data.semantic_chunking import LegalDocumentChunker
# === CONFIG ===
DATA_FOLDER="data/processed/law_db/articles"

logger = setup_logger("Build Law Vector DB")

def load_documents(folder_path):
    paths = glob.glob(os.path.join(folder_path, "*.json"))
    documents = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            try:
                data = json.load(f)
                documents.extend(data)
            except Exception as e:
                print(f"Failed to load {path}: {e}")
    return documents

def build_vector_db():
    logger.info("[INFO] Start build Law Vector DB...")

    chunker = LegalDocumentChunker()
    vector_store = ElasticVectorStore()

    documents = load_documents(DATA_FOLDER)
    
    logger.info("[INFO] Documents loaded.")

    for doc in tqdm(documents, desc="[INFO] Indexing..."):
        try:
            chunks = chunker.chunk(doc, as_dict=False)
            vector_store.add(chunks)
        except Exception as e:
            logger.error(f"[INFO] Failed to indexing document {doc.get('law_id', '???')}: {e}")

    logger.info("[INFO] Vector DB build completed.")

if __name__ == "__main__":
    build_vector_db()