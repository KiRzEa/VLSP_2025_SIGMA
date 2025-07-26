import os
import re
import json
import pandas as pd
from PIL import Image
from io import StringIO
from typing import Any, Tuple
from dotenv import load_dotenv

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
    law_db_path = os.path.join(DATA_DIR, "law_db", "vlsp2025_law.json")
    law_db_data = load_json(law_db_path)
    traffic_sign_standard = law_db_data[0]
    traffic_order_safety_law = law_db_data[1]
    return traffic_sign_standard, traffic_order_safety_law

def convert_html_to_dataframe(law_data: dict, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    table_id = 1  

    for article in law_data["articles"]:
        content = article['text']
        table_pattern = r"<<TABLE:\s*(.*?)\s*/TABLE>>"

        def replace_table(match):
            nonlocal table_id
            html_content = match.group(1)
            try:
                tables = pd.read_html(StringIO(html_content), header=0)  # Sử dụng dòng đầu làm header
                if tables:
                    df = tables[0]
                    table_filename = f"table{table_id:03d}.csv"
                    table_path = os.path.join(save_dir, table_filename)
                    df.to_csv(table_path, index=False, encoding="utf-8-sig")
                    table_id += 1
                    return f"<<TABLE: {table_filename} /TABLE>>"
            except Exception as e:
                print(f"Error parsing table {table_id}: {e}")
                return match.group(0)

        updated_content = re.sub(table_pattern, replace_table, content, flags=re.DOTALL)
        article["text"] = updated_content

    return law_data