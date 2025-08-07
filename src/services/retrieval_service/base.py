from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseRetriever(ABC):
    @abstractmethod
    def search(self, query: Any, top_k: int) -> List[Dict]:
        pass
