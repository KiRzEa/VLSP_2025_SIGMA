from pathlib import Path
from typing import Dict, Any, Optional

from src.models.global_state import GlobalInputState
from src.models.base import QuestionType

def transform_raw_sample_to_input_state(
    sample: Dict[str, Any],
    image_root: Optional[str] = Path("./data/train_data/train_images/train_images")
) -> GlobalInputState:
    question = sample["question"]
    question_type = QuestionType(sample["question_type"])
    input_image_path = image_root / f"{sample['image_id']}.jpg"
    article_ids = sample["relevant_articles"]

    if question_type == QuestionType.YES_NO:
        choices = None
    else:
        choices = sample["choices"]
    
    return GlobalInputState(
        question=question,
        question_type=question_type,
        choices=choices,
        input_image_path=input_image_path,
        article_ids=article_ids
    )
