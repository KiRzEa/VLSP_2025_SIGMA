from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage
from typing import List, Dict, Annotated

class Article(BaseModel):
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

class ArticleState(BaseModel):
    # Input
    question: str
    choices: Dict
    image_analysis: str

    # Phase 1 - Article relevance check
    articles: List[Article]
    current_article_index: int = 0
    relevant_articles: List[Article] = Field(default_factory=list)

    # Phase 2 - Info filtering from relevant articles
    current_relevant_index: int = 0
    extracted_info: List[Dict] = Field(default_factory=list)

    # Message history (chat logs)
    messages: Annotated[List[BaseMessage], Field(default_factory=lambda: [
        HumanMessage(content="Bắt đầu phân tích các điều luật liên quan")
    ])]