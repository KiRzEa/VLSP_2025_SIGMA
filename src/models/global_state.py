from pathlib import Path
from pydantic import BaseModel
from typing import Optional, List, Dict, Union

from src.models.base import QuestionType
from src.models.image_state import ImageOutputState
from src.models.article_state import ArticleOutputState


class GlobalInputState(BaseModel):
    question: str
    choices: Optional[Dict] = None
    question_type: QuestionType
    input_image_path: Union[Path, str]
    article_ids: Optional[List[Dict]] = None

class GlobalOutputState(BaseModel):
    predicted_answer: Optional[str] = None

class GlobalProcessingState(GlobalInputState, GlobalOutputState):
    image_output: Optional[ImageOutputState] = None
    article_output: Optional[ArticleOutputState] = None
