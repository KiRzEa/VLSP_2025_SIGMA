import os
from PIL import Image

def crop_item(image_path: str, prediction: dict):
    """
    Cắt 1 đối tượng từ ảnh theo toạ độ bounding box trong prediction.
    """
    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size

    x, y = prediction["x"], prediction["y"]
    w, h = prediction["width"], prediction["height"]

    x1 = max(int(x - w / 2), 0)
    y1 = max(int(y - h / 2), 0)
    x2 = min(int(x + w / 2), img_w)
    y2 = min(int(y + h / 2), img_h)

    return image.crop((x1, y1, x2, y2)), [x1, y1, x2, y2]

def save_crop(cropped_img: Image.Image, save_path: str):
    """
    Lưu ảnh đã crop vào thư mục chỉ định
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cropped_img.save(save_path)
