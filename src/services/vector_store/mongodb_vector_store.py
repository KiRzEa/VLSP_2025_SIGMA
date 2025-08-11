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

    def get_candidates(self, candidate_ids: List[Dict]) -> List[Dict]:
        """
        Retrieve all chunks with the same law_id and article_id.
        """
        candidate_ids = [
            {
                "metadata.law_id": pair["law_id"],
                "metadata.article_id": pair["article_id"]
            }
            for pair in candidate_ids
        ]
        query = {"$or": candidate_ids} if candidate_ids else {}
        return list(self.collection.find(query))
