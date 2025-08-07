from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from src.services.llm_service.base import BaseChatModel
from src.core.config import Settings, settings


class AzureChatModel(BaseChatModel):
    """
    Azure Chat Model implementation using AzureAIChatCompletionsModel.
    """

    def __init__(
        self,
        model_name: str = "DeepSeek-R1-0528",
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
        azure_cfg = self.settings.azure

        model_config = {
            "model": self.model_name,
            "api_version": azure_cfg.AZURE_API_VERSION,
            "endpoint": azure_cfg.AZURE_ENDPOINT,
            "credential": azure_cfg.AZURE_API_KEY,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_retries": self.max_retries,
        }

        self.client = AzureAIChatCompletionsModel(**model_config)

    def close(self) -> None:
        """Clean up the client."""
        self.client = None
