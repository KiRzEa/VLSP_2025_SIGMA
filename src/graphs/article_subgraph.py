import json

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph

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
    
    def no_more_articles(self, state: ArticleState) -> bool:
        return state.current_index >= len(state.articles)

    def article_is_relevant(self, state: ArticleState) -> bool:
        try:
            last_msg = state.messages[-1].content
            parsed = json.loads(last_msg)
            return parsed.get("relevant", False) is True
        except Exception as e:
            logger.warning(f"Failed to parse relevance from LLM response: {e}")
            return False
        
    def check_next_article(self, state: ArticleState):
        if self.no_more_articles(state):
            logger.info("[check_next_article] No more articles. Ending subgraph.")
        else:
            logger.info(f"[check_next_article] Moving to article index {state.current_index}")
        
        return state

    def save_article(self, state: ArticleState):
        logger.info(f"[save_article] Saving relevant article at index {state.current_index}")

        state.relevant_articles.append(state.articles[state.current_index])
        state.current_index += 1
        return state

    
    def skip_article(self, state: ArticleState):
        logger.info(f"[skip_article] Skipping article at index {state.current_index}")
        state.current_index += 1
        return state
    
    def analyze_article(self, state: ArticleState):
        logger.info(f"[analyze_article] Analyzing article at index {state.current_index}")

        prompt = CHECK_ARTICLE_RELEVANCY_PROMPT.format(
            question=state.question,
            choices=format_choices(state.choices),
            article=get_article_text(state.articles, state.current_index)
        )

        response = self.llm.invoke([HumanMessage(content=prompt)])

        response.content = extract_json_from_deepseek_response(response) 

        state.messages.append(response)
        return state
    
    def build(self) -> CompiledStateGraph:
        builder = StateGraph(ArticleState)
        builder.add_node("check_next_article", self.check_next_article)
        builder.add_node("analyze_article", self.analyze_article)
        builder.add_node("save_article", self.save_article)
        builder.add_node("skip_article", self.skip_article)

        builder.set_entry_point("check_next_article")

        builder.add_edge("save_article", "check_next_article")
        builder.add_edge("skip_article", "check_next_article")

        builder.add_conditional_edges(
            "check_next_article",
            lambda state: "END" if self.no_more_articles(state) else "analyze_article",
            {
                "END": END,
                "analyze_article": "analyze_article"
            }
        )

        builder.add_conditional_edges(
            "analyze_article",
            lambda state: "save_article" if self.article_is_relevant(state) else "skip_article",
            {
                "save_article": "save_article",
                "skip_article": "skip_article"
            }
        )

        return builder.compile()
    
    def run(self, state: ArticleState) -> ArticleState:
        if not self.graph:
            self.graph = self.build()
        
        output_state = self.graph.invoke(state)
        return ArticleState(**output_state)

