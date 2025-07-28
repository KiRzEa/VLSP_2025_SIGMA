from abc import ABC, abstractmethod
from typing import Optional, List, Dict

from langchain_core.language_models.chat_models import BaseChatModel as LangChainChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.utils import encode_image

class BaseChatModel(ABC):
    """
    Abstract base class for interacting with LangChain chat models.
    """

    def __init__(self):
        self.client: Optional[LangChainChatModel] = None

    @abstractmethod
    def _connect(self) -> None:
        """
        Initialize the LangChain LLM client. This must set `self.client`.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Clean up the LLM client if necessary (e.g., closing session or releasing resources).
        """
        pass

    def ensure_connection(self) -> None:
        """
        Ensure that the LLM client is initialized before using it.
        """
        if self.client is None:
            self._connect()
    
    def get_client(self):
        """
        Get the LLM Client
        """
        self.ensure_connection()
        return self.client

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        image_paths: Optional[List[str]] = None
    ) -> str:
        """
        Generate a response from the LLM using optional images and user prompt.

        Args:
            system_prompt (str): Context or behavior definition.
            user_prompt (str): User input prompt.
            image_paths (Optional[List[str]]): List of image paths to include.

        Returns:
            str: The LLM's response.
        """
        self.ensure_connection()
        try:
            user_content = self._build_user_content(user_prompt, image_paths)

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_content),
            ]
            response = self.client.invoke(messages)
            return response.content
        except Exception as e:
            raise RuntimeError(f"Failed to generate response: {e}") from e

    def _build_user_content(
        self,
        prompt: str,
        image_paths: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Build user message content with optional multiple images.

        Args:
            prompt (str): The user query.
            image_paths (Optional[List[str]]): Paths to image files.

        Returns:
            List[Dict]: A list of content dictionaries for HumanMessage.
        """
        content = []

        if image_paths:
            for path in image_paths:
                encoded_image, img_format = encode_image(path, resize=True, size=(1280, 1280))
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{img_format};base64,{encoded_image}"
                    }
                })

        content.append({"type": "text", "text": prompt})
        return content