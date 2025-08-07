from PIL import Image
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class InputImageState(BaseModel):
    input_image_path: str

class ImageOutputState(BaseModel):
    sign_interpretation: Optional[Dict[str, Any]] = None
    scene_text: Optional[str] = None

class ImageProcessingState(InputImageState, ImageOutputState):
    sign_bboxes: Optional[List[Dict]] = None
    detected_signs: Optional[List[Image.Image]] = None
    sign_descriptions: Optional[List[Dict[str, Any]]] = None

    class Config:
        arbitrary_types_allowed=True