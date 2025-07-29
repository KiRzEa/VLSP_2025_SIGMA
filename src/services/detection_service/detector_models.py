import requests
import base64
from abc import ABC, abstractmethod

# ===============================
# BaseDetector Interface
# ===============================
class BaseDetector(ABC):
    @abstractmethod
    def detect(self, image_path: str) -> list:
        """Trả về danh sách prediction từ ảnh đầu vào"""
        pass


# ===============================
# Roboflow-based Detector
# ===============================
class RoboflowDetector(BaseDetector):
    def __init__(self, api_key: str, model_name: str = "vlsp2025-trafficsign", version: int = 1):
        self.api_key = api_key
        self.model_name = model_name
        self.version = version

    def detect(self, image_path: str) -> list:
        url = f"https://detect.roboflow.com/{self.model_name}/{self.version}?api_key={self.api_key}"
        with open(image_path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode("utf-8")

        response = requests.post(
            url,
            data=img_base64,
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        response.raise_for_status()
        return response.json().get("predictions", [])
