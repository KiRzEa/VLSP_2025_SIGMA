import base64
import requests
import io
from io import BytesIO
from PIL import Image

def encode_image(image_path: str, resize=False, size=(1280, 1280)) -> str:
    """
    Encodes an image file to a base64 string.
    Args:
        image_path (str): The path to the image file.
    """
    if resize:
        with Image.open(image_path) as img:
            img = img.resize(size)
            # conver img to base64
            buffered = io.BytesIO()
            if image_path.endswith(".jpg") or image_path.endswith(".jpeg"):
                img.save(buffered, format="JPEG")
                format = "jpeg"
            elif image_path.endswith(".png"):
                img.save(buffered, format="PNG")
                format = "png"
            else:
                raise ValueError("Unsupported image format")
            encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
    else:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

        if image_path.endswith(".jpg") or image_path.endswith(".jpeg"):
            format = "jpeg"
        elif image_path.endswith(".png"):
            format = "png"
        else:
            raise ValueError("Unsupported image format")
    
    return encoded_string, format

def encode_image_content_from_url(image_url: str, resize=False, size=(1280, 1280)) -> str:
    """
    Encode an image from a URL to a base64 string.
    Args:
        image_url (str): The URL of the image.
    Returns:
        str: The base64 encoded string of the image.
    """
    if resize:
        with requests.get(image_url, stream=True) as response:
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            img = img.resize(size)
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return encoded_string
    else:
        with requests.get(image_url) as response:
            response.raise_for_status()
            encoded_string = base64.b64encode(response.content).decode("utf-8")
    return encoded_string