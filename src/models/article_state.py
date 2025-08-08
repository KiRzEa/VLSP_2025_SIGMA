from pydantic import BaseModel
from typing import List, Dict, Optional

from langchain_core.messages import AnyMessage

from src.models.base import QuestionType
from src.models.image_state import ImageOutputState

class ArticleMetadata(BaseModel):
    law_id: str
    article_id: str
    score: Optional[float] = None

class ProcessedArticle(BaseModel):
    metadata: ArticleMetadata
    raw_text: str
    filtered_info: Optional[str] = None 

class ArticleInputState(BaseModel):
    question: str
    choices: Optional[Dict] = None
    image_analysis: Optional[ImageOutputState]
    article_ids: Optional[List[Dict]] = None

class ArticleOutputState(BaseModel):
    processed_articles: List[ProcessedArticle] = []

class ArticleProcessingState(ArticleInputState, ArticleOutputState):
    articles: List[ProcessedArticle] = []
    relevant_articles: List[ProcessedArticle] = []
    messages: List[AnyMessage] = []