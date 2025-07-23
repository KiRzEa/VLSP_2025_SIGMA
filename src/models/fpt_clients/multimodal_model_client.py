import os
from openai import OpenAI

from src.models.base_client import BaseChatModelClient
from src.utils.utilities import encode_image

FPT_CLOUD_API_KEY = os.environ.get("FPT_CLOUD_API_KEY")
FPT_CLOUD_BASE_URL = os.environ.get("FPT_CLOUD_BASE_URL")

MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 2048))
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.9))

class FPTMultimodalModelClient(BaseChatModelClient):
    def __init__(self, model_name: str):
        self.client = OpenAI(
            api_key=FPT_CLOUD_API_KEY,
            base_url=FPT_CLOUD_BASE_URL
        )
        self.model_name = model_name
    
    def chat(self, prompt: str, image_path: str = None, system_prompt: str = "You are a helpful assistant.") -> str:
        """
        Run the VLM model with optional image input.
        Args:
            prompt (str): The user's text prompt.
            image_path (str, optional): Path to the image file (if any).
        """
        user_content = []

        if image_path:
            encoded_img, format = encode_image(image_path, resize=True, size=(1280, 1280))
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{format};base64,{encoded_img}"
                }
            })

        user_content.append({
            "type": "text",
            "text": prompt,
        })

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )

        return response.choices[0].message.content
    
    def infer(self, **kwargs):
        return self.chat(**kwargs)