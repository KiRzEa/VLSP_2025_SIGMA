import re
from typing import List, Dict, Union

from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents.base import Document

from src.services.embedding_service import E5TextEmbeddingModel
from src.utils import format_article

class LegalDocumentChunker:
    def __init__(self):
        self.text_splitter = SemanticChunker(E5TextEmbeddingModel().model)

    def chunk(
        self, 
        document: Dict, 
        as_dict: bool = True
    ) -> Union[List[Dict], List[Document]]:
        """
        Splits the document['text'] field into semantically meaningful chunks.

        Args:
            document (dict): A legal article with fields like 'text', 'law_id', etc.
            as_dict (bool): If True, returns a list of dictionaries. 
                            If False, returns a list of Document objects.

        Returns:
            List[Dict] | List[Document]
        """
        document, metadata = format_article(document)
        chunks: List[Document] = self._semantic_split(document, metadata)

        if as_dict:
            return [
                {("text" if k == "page_content" else k): v for k, v in chunk.model_dump().items()}
                for chunk in chunks
            ]
        return chunks

    def _semantic_split(self, document: str, metadata: Dict) -> List[Document]:
        """
        Splits the text by semantic boundaries (e.g., using paragraph or sentence).
        """
        # Basic placeholder: split by sentences (you can improve with LLM/Vietnamese NLP tools)
        chunks = self.text_splitter.create_documents(texts=[document], metadatas=[metadata])
        chunks = [doc for doc in chunks if doc.page_content.strip()]

        return chunks
