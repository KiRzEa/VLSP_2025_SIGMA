import json

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage

from src.models import ArticleState
from src.core.logger import setup_logger
from src.graphs.base_graph import BaseGraph
from src.prompts import CHECK_ARTICLE_RELEVANCY_PROMPT
from src.utils import (
    format_choices, 
    extract_json_from_deepseek_response,
    get_article_text
)

# --- Logging setup ---
logger = setup_logger("ArticleAgent")

class ArticleAnalysisSubgraph(BaseGraph):
    def __init__(self, llm):
        self.llm = llm
        self.graph: StateGraph = None
    
    def all_articles_checked(self, state: ArticleState) -> bool:
        return state.current_article_index >= len(state.articles)

    def all_relevant_articles_filtered(self, state: ArticleState) -> bool:
        return state.current_relevant_index >= len(state.relevant_articles)
    
    def article_is_relevant(self, state: ArticleState) -> bool:
        try:
            last_msg = state.messages[-1].content
            parsed = json.loads(last_msg)
            return parsed.get("relevant", False) is True
        except Exception as e:
            logger.warning(f"Failed to parse relevance from LLM response: {e}")
            return False
        
    def check_next_article(self, state: ArticleState):
        if self.all_articles_checked(state):
            logger.info("[check_next_article] No more articles. Start filtering information from relevant articles.")
        else:
            logger.info(f"[check_next_article] Moving to article index {state.current_article_index}")
        
        return state
    
    def check_next_relevant_article(self, state: ArticleState):
        if self.all_relevant_articles_filtered(state):
            logger.info("[check_next_relevant_article] No more relevant articles. Ending subgraph")
        else:
            logger.info(f"[check_next_relevant_article] Moving to relevant article index {state.current_relevant_index}")

    def save_article(self, state: ArticleState):
        logger.info(f"[save_article] Saving relevant article at index {state.current_article_index}")

        state.relevant_articles.append(state.articles[state.current_article_index])
        state.current_article_index += 1
        return state

    
    def skip_article(self, state: ArticleState):
        logger.info(f"[skip_article] Skipping article at index {state.current_article_index}")
        state.current_article_index += 1
        return state
    
    def check_article_relevancy(self, state: ArticleState):
        logger.info(f"[check_article_relevancy] Analyzing article at index {state.current_article_index}")

        prompt = CHECK_ARTICLE_RELEVANCY_PROMPT.format(
            question=state.question,
            choices=format_choices(state.choices),
            article=get_article_text(state.articles, state.current_article_index)
        )

        response = self.llm.invoke([HumanMessage(content=prompt)])

        response.content = extract_json_from_deepseek_response(response) 

        state.messages.append(response)
        return state

    def filter_article_information(self, state: ArticleState):
        
        state.current_relevant_index += 1
        return state

    
    def build(self) -> StateGraph:
        builder = StateGraph(ArticleState)
        # Phase 1
        builder.add_node("check_next_article", self.check_next_article)
        builder.add_node("check_article_relevancy", self.check_article_relevancy)
        builder.add_node("save_article", self.save_article)
        builder.add_node("skip_article", self.skip_article)
        # Phase 2
        builder.add_node("check_next_relevant_article", self.check_next_relevant_article)
        builder.add_node("filter_article_information", self.filter_article_information)

        builder.set_entry_point("check_next_article")

        builder.add_edge("save_article", "check_next_article")
        builder.add_edge("skip_article", "check_next_article")

        builder.add_conditional_edges(
            "check_next_article",
            lambda state: "check_next_relevant_article" if self.all_articles_checked(state) else "check_article_relevancy",
            {
                "check_next_relevant_article": "check_next_relevant_article",
                "check_article_relevancy": "check_article_relevancy"
            }
        )

        builder.add_conditional_edges(
            "check_article_relevancy",
            lambda state: "save_article" if self.article_is_relevant(state) else "skip_article",
            {
                "save_article": "save_article",
                "skip_article": "skip_article"
            }
        )

        builder.add_conditional_edges(
            "check_next_relevant_article",
            lambda state: "END" if self.all_relevant_articles_filtered(state) else "filter_article_information",
            {
                "filter_article_information": "filter_article_information",
                "END": END
            }
        )

        return builder.compile()
    
    def run(self, state: ArticleState) -> ArticleState:
        if not self.graph:
            self.graph = self.build()
        
        output_state = self.graph.invoke(state)
        return ArticleState(**output_state)
