import os
import numpy as np
from typing import List, Dict

from pymongo import MongoClient
from pymongo.server_api import ServerApi

MONGODB_URI = os.environ.get("MONGODB_URI")

class MongoVectorStore:
    def __init__(self, db_name: str = "legal_db", collection_name: str = "chunks"):
        self.client = MongoClient(MONGODB_URI, server_api=ServerApi("1"))
        try:
            self.client.admin.command("ping")
            print("Connected to MongoDB.")
        except Exception as e:
            print("MongoDB connection failed:", e)
            raise

        self.collection = self.client[db_name][collection_name]

    def add(self, chunks: List[Dict], embeddings: List[List[float]]):
        """
        Store chunks + their embeddings in MongoDB.
        """
        documents = []
        for chunk, emb in zip(chunks, embeddings):
            doc = {
                **chunk,
                "embedding": emb
            }
            documents.append(doc)

        if documents:
            self.collection.insert_many(documents)
            print(f"Inserted {len(documents)} documents.")

    def search(self, query_vector: List[float], law_id: str, article_id: str, top_k: int = 5) -> List[Dict]:
        """
        Perform brute-force cosine similarity search.
        """
        query_vec = np.array(query_vector)

        # Retrieve candidates with the same law_id and article_id
        candidates = list(self.collection.find({"law_id": law_id, "article_id": article_id}))

        # Compute cosine similarity manually
        def cosine_sim(v1, v2):
            v1, v2 = np.array(v1), np.array(v2)
            if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
                return -1.0
            return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

        scored = [
            {**doc, "score": cosine_sim(query_vec, doc["embedding"])}
            for doc in candidates
        ]

        # Sort by similarity and return top_k
        return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]
