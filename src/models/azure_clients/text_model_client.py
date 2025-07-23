import os

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

from src.models.base_client import BaseChatModelClient

AZURE_API_KEY = os.environ.get("AZURE_API_KEY")
AZURE_ENDPOINT = os.environ.get("AZURE_ENDPOINT")
AZURE_API_VERSION = os.environ.get("AZURE_API_VERSION")

MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 2048))
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.9))

class AzureTextModelClient(BaseChatModelClient):
    def __init__(self, model_name):
        self.client = ChatCompletionsClient(
            endpoint=AZURE_ENDPOINT,
            credential=AzureKeyCredential(AZURE_API_KEY),
            api_version=AZURE_API_VERSION
        )
        self.model_name = model_name

    def chat(self, prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
        messages = [
            SystemMessage(content=system_prompt),
            UserMessage(content=prompt)
        ]
        response = self.client.complete(
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            model=self.model_name,
        )
        return response.choices[0].message.content

    def infer(self, **kwargs):
        return self.chat(**kwargs)

