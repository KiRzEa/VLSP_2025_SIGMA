import os
from pathlib import Path
from typing import List, Dict, Optional

from src.utils import load_json

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))  # fallback to "data" if not set


class LawDatabase:
    """
    A class to load and retrieve legal articles from a structured law database.
    """

    def __init__(self, db_path: Path = DATA_DIR / "law_db/vlsp2025_law.json"):
        """
        Initialize the law database from a JSON file.
        Builds an index for fast article retrieval by (law_id, article_id).
        """
        self.article_index: Dict[str, Dict[str, Dict]] = {}
        self._load_database(db_path)

    def _load_database(self, db_path: Path) -> None:
        """
        Load the law database and build internal indices.
        """
        law_list = load_json(db_path)
        for law in law_list:
            law_id = law.get("id")
            articles = law.get("articles", [])
            self.article_index[law_id] = {article["id"]: article for article in articles}

    def get_article(self, law_id: str, article_id: str) -> Optional[Dict]:
        """
        Retrieve a specific article by law ID and article ID.
        """
        return self.article_index.get(law_id, {}).get(article_id)

    def get_articles_from_annotations(self, annotations: List[Dict]) -> List[Dict]:
        """
        Retrieve multiple articles based on a list of annotation dicts,
        where each dict must have 'law_id' and 'article_id'.
        """
        return [
            article
            for ann in annotations
            if (article := self.get_article(ann.get("law_id"), ann.get("article_id")))
        ]
