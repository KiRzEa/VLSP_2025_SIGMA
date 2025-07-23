import os
import json
import re
from io import StringIO
from typing import Any, Tuple

import pandas as pd

def load_json(file_path: str) -> Any:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in {file_path}: {e}")
    return None

def load_law_db(data_dir: str) -> Tuple[dict, dict]:
    law_db_path = os.path.join(data_dir, "law_db", "vlsp2025_law.json")
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

if __name__ == "__main__":
    DATA_DIR = "./data"
    RAW_DIR = os.path.join(DATA_DIR, "raw")
    TABLE_SAVE_DIR = os.path.join(DATA_DIR, "processed", "law_db", "tables")

    traffic_sign_standard, traffic_order_safety_law = load_law_db(RAW_DIR)

    processed_traffic_sign_standard = convert_html_to_dataframe(traffic_sign_standard, TABLE_SAVE_DIR)
    processed_traffic_order_safety_law = convert_html_to_dataframe(traffic_order_safety_law, TABLE_SAVE_DIR)

    content = [processed_traffic_sign_standard, processed_traffic_order_safety_law]
    with open(os.path.join(DATA_DIR, "processed", "law_db", "vlsp2025_law.json"), "w", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False, indent=4)
    
    traffic_sign_standard_articles, traffic_sign_description_articles = [], []
    law_id, law_title = processed_traffic_sign_standard['id'], processed_traffic_sign_standard['title']
    for article in processed_traffic_sign_standard['articles']:
        custom_article = {
            'law_id': law_id,
            'law_title': law_title,
            'article_id': article['id'],
            'article_title': article['title'],
            'text': article['text'],
            'images': [s.strip() for s in re.findall(r"<<IMAGE:(.*?)/IMAGE>>", article['text'], re.DOTALL)],
            'tables': [s.strip() for s in re.findall(r"<<TABLE:(.*?)/TABLE>>", article['text'], re.DOTALL)]
        }
        if re.match(r"^[B-F]\.", article["title"]):
            traffic_sign_description_articles.append(custom_article)
        else:
            traffic_sign_standard_articles.append(custom_article)
    
    traffic_order_safety_law_articles = []
    law_id, law_title = processed_traffic_order_safety_law['id'], processed_traffic_order_safety_law['title']
    for article in processed_traffic_order_safety_law['articles']:
        custom_article = {
            'law_id': law_id,
            'law_title': law_title,
            'article_id': article['id'],
            'article_title': article['title'],
            'text': article['text'],
            'images': [s.strip() for s in re.findall(r"<<IMAGE:(.*?)/IMAGE>>", article['text'], re.DOTALL)],
            'tables': [s.strip() for s in re.findall(r"<<TABLE:(.*?)/TABLE>>", article['text'], re.DOTALL)]
        }
        traffic_order_safety_law_articles.append(custom_article)
    
    with open(os.path.join(DATA_DIR, "processed", "law_db", "articles", "traffic_sign_standard.json"), "w", encoding="utf-8") as f:
        json.dump(traffic_sign_standard_articles, f, ensure_ascii=False, indent=4)
    
    with open(os.path.join(DATA_DIR, "processed", "law_db", "articles", "traffic_sign_description.json"), "w", encoding="utf-8") as f:
        json.dump(traffic_sign_description_articles, f, ensure_ascii=False, indent=4)
    
    with open(os.path.join(DATA_DIR, "processed", "law_db", "articles", "traffic_order_safety_law.json"), "w", encoding="utf-8") as f:
        json.dump(traffic_order_safety_law_articles, f, ensure_ascii=False, indent=4)