from langchain_openai import ChatOpenAI
from src.services.llm_service.base import BaseChatModel
from src.core.config import Settings, settings


class FPTChatModel(BaseChatModel):
    """
    FPT Chat Model implementation using ChatOpenAI.
    """

    def __init__(
        self,
        model_name: str = "Qwen2.5-VL-7B-Instruct",
        settings: Settings = settings,
        temperature: float = 0.0,
        max_tokens: int = 16384,
        max_retries: int = 3
    ):
        super().__init__()
        self.settings = settings
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.client = None
        self._connect()

    def _connect(self) -> None:
        """Initialize the Azure AI chat model client."""
        fpt_cfg = self.settings.fpt

        model_config = {
            "model": self.model_name,
            "base_url": fpt_cfg.FPT_CLOUD_BASE_URL,
            "api_key": fpt_cfg.FPT_CLOUD_API_KEY,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_retries": self.max_retries,
        }

        self.client = ChatOpenAI(**model_config)

    def close(self) -> None:
        """Clean up the client."""
        self.client = None
