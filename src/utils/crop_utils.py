import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from typing import Union

def crop_item(image_path: Union[Image.Image, str], prediction: dict):
    """
    Cắt 1 đối tượng từ ảnh theo toạ độ bounding box trong prediction.
    """
    if isinstance(image_path, Image.Image):
        image = image_path
    elif isinstance(image_path, str):
        image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size

    x, y = prediction["x"], prediction["y"]
    w, h = prediction["width"], prediction["height"]

    x1 = max(int(x - w / 2), 0)
    y1 = max(int(y - h / 2), 0)
    x2 = min(int(x + w / 2), img_w)
    y2 = min(int(y + h / 2), img_h)

    return image.crop((x1, y1, x2, y2))

def save_crop(cropped_img: Image.Image, save_path: str):
    """
    Lưu ảnh đã crop vào thư mục chỉ định
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cropped_img.save(save_path)

def show_image(image, title=None):
    """
    Hiển thị 1 ảnh đơn lẻ (PIL hoặc numpy)

    Args:
        image: PIL.Image hoặc np.ndarray
        title (str, optional): tiêu đề ảnh (nếu có)
    """
    if isinstance(image, Image.Image):
        img = np.array(image)
    else:
        img = image

    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()

def plot_prediction_boxes(image: Union[Image.Image, str], predictions: list):
    """
    Hiển thị ảnh gốc kèm các bounding boxes từ predictions.

    Args:
        image_path (str): Đường dẫn ảnh gốc
        predictions (list): Danh sách các dự đoán từ detector
    """
    if isinstance(image, Image.Image):
        image = image.convert("RGB")
    elif isinstance(image, str):
        image = Image.open(image).convert("RGB")
    
    draw = ImageDraw.Draw(image)

    for pred in predictions:
        x, y = pred["x"], pred["y"]
        w, h = pred["width"], pred["height"]
        label = pred.get("class", "")

        x0 = x - w / 2
        y0 = y - h / 2
        x1 = x + w / 2
        y1 = y + h / 2

        draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
        draw.text((x0, y0 - 10), label, fill="red")

    # Show using matplotlib
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.title("Predictions")
    plt.show()
