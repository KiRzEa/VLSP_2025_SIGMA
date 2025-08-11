import os
from typing import List, Tuple

from langchain_elasticsearch import ElasticsearchStore, DenseVectorStrategy
from langchain_core.documents import Document

from src.services.embedding_service import E5TextEmbeddingModel

ELASTIC_SEARCH_URL = os.environ.get("ELASTIC_SEARCH_URL")
ELASTIC_SEARCH_INDEX_NAME = os.environ.get("ELASTIC_SEARCH_INDEX_NAME")
ELASTIC_SEARCH_API_KEY = os.environ.get("ELASTIC_SEARCH_API_KEY")

class ElasticVectorStore:
    def __init__(self):
        self.vector_store = ElasticsearchStore(
            es_url=ELASTIC_SEARCH_URL,
            index_name=ELASTIC_SEARCH_INDEX_NAME,
            es_api_key=ELASTIC_SEARCH_API_KEY,
            embedding=E5TextEmbeddingModel().model,
            strategy=DenseVectorStrategy(hybrid=True)
        )

    def add(self, documents: List[Document]):
        self.vector_store.add_documents(documents)

    def retrieve(self, query: str, top_k=10):
        results: List[Tuple[Document, float]] = self.vector_store.similarity_search_with_score(
            query=query,
            top_k=top_k
        )

        return [
            {
                "text": doc.page_content,
                "score": score,
                "metadata": doc.metadata
            }
            for doc, score in results
        ]
