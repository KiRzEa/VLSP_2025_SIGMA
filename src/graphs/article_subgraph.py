import json

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import HumanMessage

from src.models.article_state import (
    ArticleMetadata,
    ProcessedArticle,
    ArticleInputState,
    ArticleOutputState,
    ArticleProcessingState,
)
from src.core.logger import setup_logger
from src.graphs.base_graph import BaseGraph
from src.services import Service, get_service
from src.prompts import CHECK_ARTICLE_RELEVANCY_PROMPT, ARTICLE_FILTER_INFORMATION_PROMPT, ARTICLE_RETRIEVAL_QUERY_TEMPLATE
from src.utils import (
    format_choices, 
    extract_json_from_deepseek_response,
    format_sign_interpretation,    
)

# --- Logging setup ---
logger = setup_logger("ArticleAgent")

class ArticleSubGraph(BaseGraph):
    def __init__(self, service: Service = get_service()):
        self.service = service
        self.graph = self.build()

    # --------------------
    # NODES
    # --------------------
    def retrieve_candidate_articles(self, state: ArticleInputState) -> ArticleProcessingState:
        logger.info("[retrieve_candidate_articles] Fetching articles for given candidate IDs")
        state = ArticleProcessingState(**state.model_dump())
        query_str = ARTICLE_RETRIEVAL_QUERY_TEMPLATE.format(
            question=state.question,
            choices=format_choices(state.choices),
            sign_interpretation=format_sign_interpretation(state.image_analysis.sign_interpretation),
            scene_text=state.image_analysis.scene_text
        )

        logger.debug(f"[retrieve_candidate_articles] Final query:\n{query_str}")

        # Call retriever
        results = self.service.text_retriever.search(
            query_str=query_str,
            candidate_ids=state.article_ids,
            top_k=5
        )

        # Map results to articles in state
        state.articles = [
            ProcessedArticle(
                metadata=ArticleMetadata(
                    law_id=res["metadata"]["law_id"],
                    article_id=res["metadata"]["article_id"],
                    score=res.get("score"),
                    other_meta={k: v for k, v in res.get("metadata", {}).items()
                                if k not in ("law_id", "article_id")}
                ),
                raw_text=res.get("text", ""),
                filtered_info=None
            )
            for res in results
        ]
        return state
    
    def check_relevancy_and_save(self, state: ArticleProcessingState) -> ArticleProcessingState:
        logger.info("[check_relevancy_and_save] Checking all articles for relevancy")

        for idx, article in enumerate(state.articles):
            prompt = CHECK_ARTICLE_RELEVANCY_PROMPT.format(
                question=state.question,
                choices=format_choices(state.choices),
                sign_interpretation=format_sign_interpretation(state.image_analysis.sign_interpretation),
                scene_text=state.image_analysis.scene_text,
                article_text=article.raw_text
            )

            response = self.service.deepseek.get_client().invoke([HumanMessage(content=prompt)])
            response.content = extract_json_from_deepseek_response(response)
            state.messages.append(response)

            try:
                is_relevant = json.loads(response.content).get("relevant", False)
            except Exception as e:
                logger.warning(f"Failed to parse relevance for article {idx}: {e}")
                is_relevant = False

            if is_relevant:
                state.relevant_articles.append(article)
                logger.info(f"Article {idx} marked as relevant")
            else:
                logger.info(f"Article {idx} skipped")

        logger.info(f"Total relevant articles: {len(state.relevant_articles)}")
        return state

    def filter_relevant_info(self, state: ArticleProcessingState) -> ArticleOutputState:
        logger.info("[filter_relevant_info] Filtering information from relevant articles")

        for idx, article in enumerate(state.relevant_articles):
            prompt = ARTICLE_FILTER_INFORMATION_PROMPT.format(
                question=state.question,
                choices=format_choices(state.choices),
                sign_interpretation=format_sign_interpretation(state.image_analysis.sign_interpretation),
                scene_text=state.image_analysis.scene_text,
                article_text=article.raw_text
            )

            response = self.service.deepseek.get_client().invoke([HumanMessage(content=prompt)])
            filtered = extract_json_from_deepseek_response(response)

            article.filtered_info = json.loads(filtered)['text']
        
        state.processed_articles = state.relevant_articles
        return ArticleOutputState(**state.model_dump())
    
    def build(self) -> CompiledStateGraph:
        builder = StateGraph(
            state_schema=ArticleProcessingState,
            input_schema=ArticleInputState,
            output_schema=ArticleOutputState,
        )

        builder.add_node("retrieve_candidate_articles", self.retrieve_candidate_articles)
        builder.add_node("check_relevancy_and_save", self.check_relevancy_and_save)
        builder.add_node("filtered_relevant_info", self.filter_relevant_info)

        builder.set_entry_point("retrieve_candidate_articles")

        builder.add_edge("retrieve_candidate_articles", "check_relevancy_and_save")
        builder.add_edge("check_relevancy_and_save", "filtered_relevant_info")
        builder.add_edge("filtered_relevant_info", END)        

        return builder.compile()
    
    def run(self, state: ArticleInputState) -> ArticleOutputState:
        if not self.graph:
            self.graph = self.build()
        
        output_state = self.graph.invoke(state)
        return ArticleOutputState(**output_state)
