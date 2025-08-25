import json
from time import time
from pathlib import Path
from tqdm.auto import tqdm
from argparse import ArgumentParser

from src.core.logger import setup_logger
from src.graphs.global_graph import GlobalGraph
from src.utils import (
    load_json,
    transform_raw_sample_to_input_state,
    save_json
)

logger = setup_logger("Task-2 Runner")


def load_existing_results(predictions_file: Path):
    """Load existing predictions if available, else return empty results."""
    if predictions_file.exists():
        results = load_json(predictions_file)
        total_time = float(results.get("total_time", 0.0))
        predictions = results.get("predictions", [])
        prediction_ids = {pred["id"] for pred in predictions}
    else:
        total_time = 0.0
        predictions = []
        prediction_ids = set()
    return total_time, predictions, prediction_ids


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--mode", 
        choices=["train", "public_test", "private_test"], 
        default="public_test", 
        help="Choose whether to run in train, public_test, or private_test mode"
    )
    args = parser.parse_args()

    # File paths
    if args.mode == "train":
        data_file = Path("./data/train_data/vlsp_2025_train.json")
        image_root = Path("./data/train_data/train_images/train_images")
    elif args.mode == "public_test":
        data_file = Path("./data/public_test/vlsp_2025_public_test_task2.json")
        image_root = Path("./data/public_test/public_test_images")
    elif args.mode == "private_test":
        data_file = Path("./data/private_test/vlsp2025_submission_task2.json")
        image_root = Path("./data/private_test/private_test_images_jpeg")

    predictions_file = Path(f"./task2_{args.mode}_predictions.json")
    submission_file = Path(f"./{args.mode}_submission_task2.json")

    logger.info(f"Initialized Task-2 Runner in {args.mode.upper()} mode")
    logger.info(f"Data file: {data_file}")
    logger.info(f"Image root: {image_root}")
    logger.info(f"Predictions file: {predictions_file}")

    # Load data and results
    samples = load_json(data_file)
    total_time, predictions, prediction_ids = load_existing_results(predictions_file)

    remaining_samples = [sample for sample in samples if sample["id"] not in prediction_ids]
    logger.info(f"Need to process {len(remaining_samples)} samples out of {len(samples)} total")
    # Initialize graph
    graph = GlobalGraph()

    # Inference loop
    for sample in tqdm(remaining_samples, desc="[INFO] Inference..."):
        try:
            logger.info(f"[INFO] Processing Sample {sample["id"]}")
            start = time()

            input_state = transform_raw_sample_to_input_state(sample, image_root)
            output_state = graph.run(input_state)

            predictions.append({
                **sample,
                "predicted_answer": output_state["predicted_answer"]
            })

            total_time += time() - start

            save_json(predictions_file, {
                "total_time": total_time,
                "predictions": predictions
            })

            save_json(submission_file, [
                {**{k: v for k, v in pred.items() if k != "predicted_answer"},
                 "answer": pred["predicted_answer"]}
                for pred in predictions
            ])

        except Exception as e:
            logger.error(f"Error while processing Sample {sample['id']}: {e}")



if __name__ == "__main__":
    main()
