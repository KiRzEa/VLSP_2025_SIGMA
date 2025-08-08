from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import HumanMessage

from src.core.logger import setup_logger
from src.services import Service, get_service
from src.graphs.base_graph import BaseGraph
from src.graphs.image_subgraph import ImageSubGraph
from src.graphs.article_subgraph import ArticleSubGraph
from src.models.image_state import ImageInputState
from src.models.article_state import ArticleInputState
from src.models.global_state import (
    GlobalInputState,
    GlobalOutputState,
    GlobalProcessingState
)
from src.prompts import FINAL_ANSWER_REASONING_PROMPT
from src.utils import (
    extract_json_from_deepseek_response,
    format_choices,
    format_sign_interpretation,
    format_processed_articles,
)

logger = setup_logger("GlobalGraph")

class GlobalGraph(BaseGraph):
    def __init__(self, service: Service = get_service()):
        self.service = service
        self.image_subgraph = ImageSubGraph(service)
        self.article_subgraph = ArticleSubGraph(service)

        self.graph = self.build()

    
    def run_image_subgraph(self, state: GlobalInputState) -> GlobalProcessingState:
        logger.info("[run_image_graph] Executing ImageSubGraph")
        state = GlobalProcessingState(**state.model_dump())
        image_output = self.image_subgraph.run(
            state=ImageInputState(
                input_image_path=state.input_image_path
            )
        )
        state.image_output = image_output
        return state
    
    def run_article_subgraph(self, state: GlobalProcessingState) -> GlobalProcessingState:
        logger.info("[run_article_graph] Executing ArticleSubGraph")
        article_output = self.article_subgraph.run(
            state=ArticleInputState(
                **state.model_dump(),
                image_analysis=state.image_output,
            )
        )
        state.article_output = article_output
        return state

    def answer_question(self, state: GlobalProcessingState) -> GlobalOutputState:
        logger.info("[answer_question] Composing prompt and generating answer")

        prompt = FINAL_ANSWER_REASONING_PROMPT.format(
            question_type=state.question_type,
            question=state.question,
            choices=format_choices(state.choices),
            sign_interpretation=format_sign_interpretation(state.image_output.sign_interpretation),
            scene_text=state.image_output.scene_text,
            articles_text=format_processed_articles(state.article_output.processed_articles)
        )

        response = self.service.deepseek.get_client().invoke([HumanMessage(content=prompt)])
        parsed = extract_json_from_deepseek_response(response, return_json=True)
        answer = parsed["answer"]
        
        state.predicted_answer = answer
        return GlobalOutputState(**state.model_dump())

    def build(self) -> CompiledStateGraph:
        builder = StateGraph(
            state_schema=GlobalProcessingState,
            input_schema=GlobalInputState,
            output_schema=GlobalOutputState
        )

        builder.add_node("run_image_subgraph", self.run_image_subgraph)
        builder.add_node("run_article_subgraph", self.run_article_subgraph)
        builder.add_node("answer_question", self.answer_question)

        builder.set_entry_point("run_image_subgraph")

        builder.add_edge("run_image_subgraph", "run_article_subgraph")
        builder.add_edge("run_article_subgraph", "answer_question")
        builder.add_edge("answer_question", END)

        return builder.compile()

    def run(self, state: GlobalInputState) -> GlobalOutputState:
        return self.graph.invoke(state)