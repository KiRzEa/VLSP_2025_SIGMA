import os
import json
from src.services.detection_service.detector_models import BaseDetector
from src.services.detection_service.crop_utils import crop_item, save_crop

class DetectorPipeline:
    """
    DetectorPipeline là lớp chịu trách nhiệm điều phối toàn bộ quy trình:
    1. Phát hiện đối tượng (biển báo giao thông) từ ảnh đầu vào bằng detector
    2. Crop từng đối tượng từ ảnh gốc
    3. Lưu từng ảnh đã crop
    4. Tạo file metadata JSON chứa thông tin của tất cả ảnh đã crop
    """

    def __init__(self, detector: BaseDetector):
        """
        Khởi tạo pipeline với một detector cụ thể (phải kế thừa BaseDetector)
        """
        self.detector = detector

    def detect(self, image_path: str):
        """
        Chạy bước detect, trả về danh sách prediction từ ảnh

        Args:
            image_path (str): đường dẫn ảnh đầu vào

        Returns:
            List[Dict]: danh sách prediction từ detector
        """
        return self.detector.detect(image_path)

    def crop_and_save(self, image_path: str, predictions: list, output_dir: str, metadata_path: str):
        """
        Từ danh sách prediction, crop và lưu ảnh + metadata

        Args:
            image_path (str): ảnh gốc
            predictions (list): kết quả từ hàm detect()
            output_dir (str): nơi lưu ảnh crop
            metadata_path (str): nơi lưu metadata

        Returns:
            List[Dict]: metadata của từng ảnh đã crop
        """
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        metadata = []

        for idx, pred in enumerate(predictions):
            crop_img, bbox = crop_item(image_path, pred)
            crop_filename = f"{image_id}_{idx}.jpg"
            crop_path = os.path.join(output_dir, crop_filename)

            save_crop(crop_img, crop_path)

            metadata.append({
                "original_image": os.path.basename(image_path),
                "crop_id": crop_filename,
                "bbox": bbox,
                "label": pred.get("class", "unknown"),
                "image_path": crop_path
            })

        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return metadata

    def detect_and_save(self, image_path: str, output_dir: str, metadata_path: str):
        """
        Chạy toàn bộ pipeline detect → crop → save metadata

        Args:
            image_path (str): đường dẫn ảnh đầu vào
            output_dir (str): thư mục lưu ảnh crop
            metadata_path (str): đường dẫn file metadata JSON

        Returns:
            List[Dict]: danh sách metadata của từng ảnh đã crop
        """
        predictions = self.detect(image_path)
        return self.crop_and_save(image_path, predictions, output_dir, metadata_path)