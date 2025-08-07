import os
import re
import json
from pathlib import Path
from tqdm.auto import tqdm

from src.services.detection_service.detector_models import RoboflowDetector
from src.services.detection_service.detector_pipeline import DetectorPipeline

pipeline = DetectorPipeline(
    detector=RoboflowDetector()
)

# Mapping prefix to semantic label
prefix_mapper = {
    "R": "regulatory",
    "W": "warning",
    "P": "prohibition",
    "DP": "prohibition",
    "I": "information",
    "IE": "information_expressway",
    "S": "auxiliary",
    "S.G": "auxiliary",
}

# Regex to extract sign IDs
def extract_sign_ids(text):
    pattern = r'\b(?:P|I|DP|W|R\.E|R|S|S\.G)[\.,]?\d+[a-zA-Z]?\b'
    return re.findall(pattern, text)

# Helper to determine sign type from ID
def get_sign_type(sign_id: str):
    for prefix in sorted(prefix_mapper.keys(), key=len, reverse=True):  # check longer first (e.g., DP before D)
        # Normalize prefix with dot
        normalized_prefix = prefix.replace(".", r"\.")
        if re.match(rf"^{normalized_prefix}[\.,]?\d+", sign_id):
            return prefix_mapper[prefix]
    return None
    

if __name__ == '__main__':
    DATA_DIR = Path("./data")
    if not DATA_DIR.exists():
        import subprocess
        subprocess.run("python -m src.prepare_data.process_data")
    os.makedirs(DATA_DIR / "traffic_sign_db", exist_ok=True)

    input_path = DATA_DIR / "processed/law_db/articles/traffic_sign_description.json"
    output_path = DATA_DIR / "traffic_sign_db" / "traffic_sign_descriptions.json"

    print(f"[INFO] Loading input from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        traffic_sign_descriptions = json.load(f)

    for i, article in tqdm(enumerate(traffic_sign_descriptions), desc=f"[INFO] Processing {len(traffic_sign_descriptions)} articles..."):
        combined_text = article.get("text", "") + " " + article.get("article_title", "")
        sign_ids = extract_sign_ids(combined_text)
        sign_types = {get_sign_type(sign_id) for sign_id in sign_ids}
        sign_types.discard(None)  # Remove unmatched ones

        article["reference_sign_type"] = list(sign_types)

    print("\n[INFO] Example output:")
    for article in traffic_sign_descriptions[:3]:
        print(f"  - {article['article_id']} → {article['reference_sign_type']}")

    print(f"\n[INFO] Saving results to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(traffic_sign_descriptions, f, indent=4, ensure_ascii=False)

    print("[INFO] Done.")

