from abc import ABC, abstractmethod

class BaseModelClient(ABC):
    @abstractmethod
    def infer(self, **kwargs):
        pass


class BaseChatModelClient(BaseModelClient):
    @abstractmethod
    def chat(self, prompt: str, **kwargs) -> str:
        pass


class BaseEmbeddingModelClient(BaseModelClient):
    @abstractmethod
    def embed(self, text: str) -> list[float]:
        pass
