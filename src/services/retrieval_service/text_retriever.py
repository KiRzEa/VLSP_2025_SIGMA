import numpy as np
from typing import List, Dict, Optional, Literal

from src.core.logger import setup_logger
from src.services.vector_store import MongoVectorStore, ElasticVectorStore
from src.services.retrieval_service.base import BaseRetriever
from src.services.embedding_service import E5TextEmbeddingModel

logger = setup_logger("Text Retriever")
class TextRetriever(BaseRetriever):
    def __init__(self, vector_store_type: Literal["mongo", "elastic"]):
        self.vector_store_type = vector_store_type

    def _init_vector_store(self):
        if self.vector_store_type == "mongo":
            self.embedder = E5TextEmbeddingModel()
            self.vector_store = MongoVectorStore()
            logger.info("Mongo vector store initialized")
        elif self.vector_store_type == "elastic":
            self.vector_store = ElasticVectorStore()
            logger.info("Elastic vector store initialized")

    @staticmethod
    def cosine_sim(v1: List[float], v2: List[float]) -> float:
        v1, v2 = np.array(v1), np.array(v2)
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return -1.0
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    def search(self, query_str: str, candidate_ids: Optional[List[Dict]] = None, top_k: int = 5) -> List[Dict]:
        if isinstance(self.vector_store, MongoVectorStore):
            query_vector = self.embedder.encode(query_str)
            query_embedding = np.array(query_vector)

            # Get candidates from the vector store
            candidates = self.vector_store.get_candidates(candidate_ids)

            # Score candidates
            scored = [
                {**doc, "score": self.cosine_sim(query_embedding, doc["embedding"])}
                for doc in candidates
            ]

            # Sort by score & return top_k
            return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]
        else:
            return self.vector_store.retrieve(query_str, top_k=top_k)
    

if __name__ == '__main__':

    retriever = TextRetriever()

    law_article_pairs = [
        {"law_id": "QCVN 41:2024/BGTVT", "article_id": "26"},
        {"law_id": "QCVN 41:2024/BGTVT", "article_id": "14"}
    ]

    results = retriever.search("biển báo nguy hiểm", candidate_ids=law_article_pairs, top_k=3)

    for r in results:
        print(r["score"], r.get("text", "<no text>"))