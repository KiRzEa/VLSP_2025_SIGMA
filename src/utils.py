import re
import base64
from typing import List, Dict, Tuple, Optional
import requests
from io import BytesIO
from PIL import Image


def extract_images_and_tables(text: str) -> Dict[str, List[str]]:
    """
    Extract content from custom tags <<IMAGE:.../IMAGE>> and <<TABLE:.../TABLE>>.
    """
    image_pattern = r"<<IMAGE:\s*(.*?)\s*/IMAGE>>"
    table_pattern = r"<<TABLE:\s*(.*?)\s*/TABLE>>"

    return {
        "images": re.findall(image_pattern, text, re.DOTALL),
        "tables": re.findall(table_pattern, text, re.DOTALL)
    }


def get_image_format(file_path: str) -> str:
    ext = file_path.lower().split('.')[-1]
    if ext in ("jpg", "jpeg"):
        return "JPEG"
    elif ext == "png":
        return "PNG"
    raise ValueError(f"Unsupported image format: {ext}")


def resize_image(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    return img.resize(size)


def encode_image_from_pil(img: Image.Image, format: str) -> str:
    buffered = BytesIO()
    img.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def encode_image(image_path: str, resize: bool = False, size: Tuple[int, int] = (1280, 1280)) -> Tuple[str, str]:
    """
    Encode an image file to base64 string.
    
    Returns:
        Tuple[str, str]: (base64_string, image_format)
    """
    format = get_image_format(image_path)
    
    with Image.open(image_path) as img:
        if resize:
            img = resize_image(img, size)
        encoded_string = encode_image_from_pil(img, format)
    
    return encoded_string, format.lower()


def encode_image_content_from_url(
    image_url: str,
    resize: bool = False,
    size: Tuple[int, int] = (1280, 1280)
) -> str:
    """
    Encode an image from a URL to base64 string.

    Returns:
        str: base64 encoded image.
    """
    response = requests.get(image_url, stream=True, timeout=10)
    response.raise_for_status()

    if resize:
        with Image.open(BytesIO(response.content)) as img:
            img = resize_image(img, size)
            return encode_image_from_pil(img, "PNG")
    return base64.b64encode(response.content).decode("utf-8")