from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import cached_property


class BaseConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore"
    )


class AzureConfig(BaseConfig):
    AZURE_API_VERSION: str = Field(default="2024-05-01-preview", description="Azure OpenAI API version")
    AZURE_API_KEY: str = Field(..., description="Azure OpenAI API key")
    AZURE_ENDPOINT: str = Field(..., description="Azure OpenAI endpoint URL")


class FPTConfig(BaseConfig):
    FPT_CLOUD_API_KEY: str = Field(..., description="FPT Cloud API key")
    FPT_CLOUD_BASE_URL: str = Field(..., description="FPT Cloud base URL")


class Settings:
    """
    Centralized application settings that aggregates sub-configs.
    """

    @cached_property
    def azure(self) -> AzureConfig:
        return AzureConfig()

    @cached_property
    def fpt(self) -> FPTConfig:
        return FPTConfig()


# Usage
settings = Settings()
