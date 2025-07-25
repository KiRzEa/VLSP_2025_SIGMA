import os
import re
import json

from src.utils import (
    load_law_db,
    convert_html_to_dataframe,
    
)
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