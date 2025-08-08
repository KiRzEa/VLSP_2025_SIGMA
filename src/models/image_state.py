from PIL import Image
from pathlib import Path
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union

class ImageInputState(BaseModel):
    input_image_path: Union[str, Path]

class ImageOutputState(BaseModel):
    sign_interpretation: Optional[Dict[str, Any]] = None
    scene_text: Optional[str] = None

class ImageProcessingState(ImageInputState, ImageOutputState):
    sign_bboxes: Optional[List[Dict]] = None
    detected_signs: Optional[List[Image.Image]] = None
    sign_descriptions: Optional[List[Dict[str, Any]]] = None

    class Config:
        arbitrary_types_allowed=True