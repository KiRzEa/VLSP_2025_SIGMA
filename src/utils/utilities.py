import os
import re
import json
import base64
import requests
from PIL import Image
from io import BytesIO
from typing import List, Dict, Tuple, Union

from langchain_core.messages import BaseMessage

from src.models.article import Article

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
    ext = os.path.splitext(file_path)[-1].lower().strip(".")
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


def encode_image(image: Union[Image.Image, str], resize: bool = False, size: Tuple[int, int] = (1280, 1280)) -> Tuple[str, str]:
    """
    Encode an image (file path or PIL.Image) to a base64 string and get its format.

    Args:
        image (Union[Image.Image, str]): Image file path or PIL Image.
        resize (bool): Whether to resize image.
        size (Tuple[int, int]): Resize dimensions.

    Returns:
        Tuple[str, str]: (base64-encoded string, image format like "jpeg" or "png")
    """
    if isinstance(image, str):
        img = Image.open(image)
        img_format = get_image_format(image)
    elif isinstance(image, Image.Image):
        img = image
        img_format = img.format if img.format else "PNG"  # fallback
    else:
        raise ValueError("Input must be a file path or a PIL.Image.Image instance.")

    if resize:
        img = img.resize(size)

    buffered = BytesIO()
    img.save(buffered, format=img_format)
    encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return encoded_string, img_format.lower()


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

def extract_json_from_deepseek_response(response: Union[BaseMessage, str], return_json=False) -> dict:
    """
    Extract and parse the JSON content from a DeepSeek model response.

    Args:
        response (BaseMessage): The output message returned by the DeepSeek model.

    Returns:
        dict: A parsed JSON dictionary from the content following the </think> tag.

    """
    if isinstance(response, BaseMessage):
        content = response.content
    else:
        content = response

    content = content.replace("json", "").replace("```", "").strip()

    # Attempt to find the content after </think>
    if "</think>" in content:
        json_str = content.split("</think>", maxsplit=1)[-1].strip()
    else:
        # If <think> tags are not present, assume entire content is JSON
        json_str = content

    try:
        return json.loads(json_str) if return_json else json_str
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from DeepSeek response: {e}\nRaw content: {json_str}")
    

### PROMPT INPUT FORMAT
def format_choices(choices: Dict) -> str:
    return "\n".join(f"- {key}. {value}" for key, value in choices.items())


def get_article_text(articles: List[Article], index: int) -> str:
    article = articles[index]
    return f"Tiêu đề: {article.title}\nNội dung: {article.text}"

