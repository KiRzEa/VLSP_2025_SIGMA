import os
import base64
import requests
from abc import ABC, abstractmethod

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
ROBOFLOW_MODEL_NAME = os.environ.get("ROBOFLOW_MODEL_NAME")
ROBOFLOW_MODEL_VERSION = os.environ.get("ROBOFLOW_MODEL_VERSION")
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
    def __init__(self, version: int = 3):
        self.api_key = ROBOFLOW_API_KEY
        self.model_name = ROBOFLOW_MODEL_NAME
        self.version = ROBOFLOW_MODEL_VERSION

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
