from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
from typing import List, Dict, Annotated, TypedDict

class Article(TypedDict):
    title: str
    text: str

    def __str__(self):
        """
        Returns a human-readable string representation of the article,
        showing the title and a truncated version of the text.
        """
        preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"Title: {self.title}\nText: {preview}"

    def __repr__(self):
        """
        Returns a human-readable string representation of the article,
        showing the title and a truncated version of the text.
        """
        preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"Article(title={self.title!r}, text={preview!r})"

class ArticleState(TypedDict, total=False):
    # Input
    question: str
    choices: Dict
    image_analysis: str

    # Phase 1 - Article relevance check
    articles: List[Article]
    current_article_index: int = 0
    relevant_articles: List[Article]

    # Phase 2 - Info filtering from relevant articles
    current_relevant_index: int = 0
    extracted_info: List[Dict]

    # Chat logs
    messages: Annotated[List[AnyMessage], add_messages]