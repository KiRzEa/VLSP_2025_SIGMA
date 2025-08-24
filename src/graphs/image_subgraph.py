import json
from typing import Dict

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from src.services import Service, get_service
from src.prompts import (
    SIGN_ATTRIBUTES_EXTRACTION_PROMPT,
    INTERPRET_SIGN_MEANING_PROMPT,
    SCENE_DESCRIPTION_PROMPT,
)
from src.models.image_state import (
    ImageInputState,
    ImageOutputState,
    ImageProcessingState
)
from src.core.logger import setup_logger
from src.utils.crop_utils import crop_item
from src.utils import extract_json_from_deepseek_response

# --- Logging setup ---
logger = setup_logger("ImageSubgraph")

class ImageSubGraph(StateGraph):
    def __init__(self, service: Service = get_service(), detector_config: Dict = {"top_k": 5, "min_area_ratio": 1e-2}):
        self.service = service
        self.detector_config = detector_config

        self.graph = self.build()

    # --------------------
    # NODES
    # --------------------
    def detect_signs(self, state: ImageInputState) -> ImageProcessingState:
        logger.info(f"[detect_signs] Processing image: {state.input_image_path}")
        predictions = self.service.detector.detect(
            image=state.input_image_path,
            **self.detector_config
        )

        return ImageProcessingState(
            input_image_path=state.input_image_path,
            sign_bboxes=predictions
        )
    
    def crop_signs(self, state: ImageProcessingState) -> ImageProcessingState:
        logger.info(f"[crop_signs] Cropping {len(state.sign_bboxes)} detected signs")
        cropped = [crop_item(state.input_image_path, prediction=p) for p in state.sign_bboxes]
        state.detected_signs = cropped

        return state
    
    def analyze_query(self, state: ImageProcessingState) -> ImageProcessingState:
        logger.info(f"[analyze_query] Analyzing {len(state.detected_signs)} detected signs")
        sign_descs = []

        for sign, bbox in zip(state.detected_signs, state.sign_bboxes):
            response = self.service.gemma.generate(
                user_prompt=SIGN_ATTRIBUTES_EXTRACTION_PROMPT,
                image_paths=[sign]
            )

            desc = extract_json_from_deepseek_response(response, return_json=True)
            desc[0]['sign_type'] = bbox['class']
            sign_descs.append(desc[0])
        
        state.sign_descriptions = sign_descs
        return state
    
    def interpret_sign(self, state: ImageProcessingState) -> ImageProcessingState:
        logger.info(f"[interpret_sign] Interpreting {len(state.sign_descriptions)} signs")

        prompt = INTERPRET_SIGN_MEANING_PROMPT.format(
            query_descriptions=json.dumps(
                state.sign_descriptions,
                ensure_ascii=False,
                indent=4
            )
        )

        response = self.service.deepseek.generate(
            user_prompt=prompt
        )

        parsed = extract_json_from_deepseek_response(response, return_json=True)

        signs = parsed["individual_signs"]
        for sign, bbox in zip(signs, state.sign_bboxes):
            sign["detector_predicted_type"] = bbox["class"]
        
        parsed["individual_signs"] = signs
        state.sign_interpretation = parsed

        return state

    def generate_scene_text(self, state: ImageProcessingState) -> ImageOutputState:
        logger.info(f"[generate_scene_text] Generate scene text")
        prompt = SCENE_DESCRIPTION_PROMPT.format(
            sign_interpretion=json.dumps(
                state.sign_interpretation,
                ensure_ascii=False,
                indent=4
            )
        )

        response = self.service.gemma.generate(
            system_prompt=prompt,
            image_paths=[state.input_image_path]
        )

        parsed = extract_json_from_deepseek_response(response, return_json=True)

        state.scene_text = parsed['scene_text']

        return state
    
    # --------------------
    # BUILD GRAPH
    # --------------------
    def build(self) -> CompiledStateGraph:
        builder = StateGraph(
            state_schema=ImageProcessingState,
            input_schema=ImageInputState,
            output_schema=ImageOutputState
        )

        builder.add_node("detect_signs", self.detect_signs)
        builder.add_node("crop_signs", self.crop_signs)
        builder.add_node("analyze_query", self.analyze_query)
        builder.add_node("interpret_sign", self.interpret_sign)
        builder.add_node("generate_scene_text", self.generate_scene_text)

        builder.set_entry_point("detect_signs")
        builder.add_edge("detect_signs", "crop_signs")
        builder.add_edge("crop_signs", "analyze_query")
        builder.add_edge("analyze_query", "interpret_sign")
        builder.add_edge("interpret_sign", "generate_scene_text")
        builder.add_edge("generate_scene_text", END)

        return builder.compile()

    def run(self, state: ImageInputState) -> ImageOutputState:
        output_state = self.graph.invoke(state)
        return ImageOutputState(**output_state)