import os
import json
from dotenv import load_dotenv
from typing import Any, Tuple

import pandas as pd
from PIL import Image

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR")

def load_json(file_path: str) -> Any:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in {file_path}: {e}")
    return None

def load_image(image_path: str) -> Any:
    try:
        image = Image.open(image_path)
        return image
    except Exception as e:
        print(f"[ERROR] Failed to load image at {image_path}: {e}")
        return None

def load_table(table_path: str) -> Any:
    try:
        df = pd.read_csv(table_path)
        return df
    except Exception as e:
        print(f"[ERROR] Failed to load table at {table_path}: {e}")
        return None

def load_law_db() -> Tuple[dict, dict]:
    law_db_path = os.path.join(DATA_DIR, "law_db", "vlsp2025.json")
    law_db_data = load_json(law_db_path)
    traffic_sign_standard = law_db_data[0]
    traffic_order_safety_law = law_db_data[1]
    return traffic_sign_standard, traffic_order_safety_law