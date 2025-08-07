from functools import lru_cache

from src.services.knowledge_base_service import LawDatabase
from src.services.llm_service import AzureChatModel, FPTChatModel
from src.services.retrieval_service import ImageRetriever
from src.services.detection_service import DetectorPipeline


class Service:
    def __init__(self):
        self._initialize_services()

    def _initialize_services(self):
        # LLM
        self.deepseek = AzureChatModel(model_name="DeepSeek-R1-0528")

        # VLM
        self.gemma = FPTChatModel(model_name="gemma-3-27b-it")
        self.qwen = FPTChatModel(model_name="Qwen2.5-VL-7B-Instruct")
        self.llama = FPTChatModel(model_name="Llama-4-Scout-17B-16E")

        # DB
        self.law_db = LawDatabase()

        # Retriever
        # self.image_retriever = ImageRetriever()

        # Detector
        self.detector = DetectorPipeline()

@lru_cache(maxsize=1)
def get_service() -> Service:
    return Service()
