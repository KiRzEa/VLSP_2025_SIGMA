import json
import os
from pathlib import Path
from tqdm.auto import tqdm
from time import time
from typing import Dict, List, Set, Tuple

from src.core.logger import setup_logger
from src.utils import transform_raw_sample_to_input_state
from src.utils.submission_utils import create_submission_from_predictions
from src.graphs.global_graph import GlobalGraph

logger = setup_logger("Task2Runner")


class Task2Runner:
    """Task2 inference runner with robust sample processing and error handling."""
    
    def __init__(self, mode: str = "test"):
        self.mode = mode
        self.graph = GlobalGraph()
        
        # Configure paths based on mode
        if mode == "train":
            self.data_file = "./data/train_data/vlsp_2025_train.json"
            self.image_root = Path("./data/train_data/train_images/train_images")
            self.predictions_file = "./task2_train_predictions.json"
        elif mode == "test":
            self.data_file = "./data/public_test/vlsp_2025_public_test_task2.json"
            self.image_root = Path("./data/public_test/public_test_images")
            self.predictions_file = "./task2_test_predictions.json"
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'test'")
        
        logger.info(f"Initialized Task2Runner in {mode.upper()} mode")
        logger.info(f"Data file: {self.data_file}")
        logger.info(f"Image root: {self.image_root}")
        logger.info(f"Predictions file: {self.predictions_file}")
    
    def load_existing_predictions(self) -> Set[str]:
        """Load existing predictions from file if it exists."""
        try:
            if os.path.exists(self.predictions_file):
                with open(self.predictions_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return {result["id"] for result in data.get("results", [])}
            return set()
        except (json.JSONDecodeError, KeyError, Exception) as e:
            logger.warning(f"Error loading existing predictions: {e}. Starting fresh.")
            return set()
    
    def save_prediction(self, prediction_result: Dict) -> None:
        """Save a single prediction to file, appending to existing results."""
        try:
            # Load existing data
            existing_data = {"total_time": "0s", "results": []}
            if os.path.exists(self.predictions_file):
                try:
                    with open(self.predictions_file, "r", encoding="utf-8") as f:
                        existing_data = json.load(f)
                except (json.JSONDecodeError, Exception):
                    pass  # Use default empty structure
            
            # Append to existing results
            existing_data["results"].append(prediction_result)
            
            # Save back to file
            with open(self.predictions_file, "w", encoding="utf-8") as f:
                json.dump(existing_data, f, indent=4, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to save prediction for {prediction_result.get('id', 'unknown')}: {e}")
    
    def process_sample(self, sample: Dict) -> Tuple[bool, Dict]:
        """
        Process a single sample and return (success, prediction).
        
        Returns:
            Tuple[bool, Dict]: (success_flag, prediction_dict)
        """
        try:
            prediction = sample.copy()
            input_state = transform_raw_sample_to_input_state(sample, image_root=self.image_root)
            output_state = self.graph.run(input_state)
            
            prediction["predicted_answer"] = output_state["predicted_answer"]
            return True, prediction
            
        except Exception as e:
            logger.error(f"Failed to process sample {sample['id']}: {e}")
            return False, {}
    
    def process_samples_batch(self, samples: List[Dict], desc: str = "Processing samples") -> Tuple[List[Dict], List[Dict]]:
        """
        Process a batch of samples.
        
        Returns:
            Tuple[List[Dict], List[Dict]]: (successful_results, failed_samples)
        """
        successful_results = []
        failed_samples = []
        
        for sample in tqdm(samples, desc=desc):
            success, prediction = self.process_sample(sample)
            
            if success:
                self.save_prediction(prediction)
                successful_results.append(prediction)
                logger.debug(f"✅ Processed {sample['id']}: {prediction['predicted_answer']}")
            else:
                failed_samples.append(sample)
        
        return successful_results, failed_samples
    
    def get_samples_to_process(self) -> Tuple[List[Dict], Set[str]]:
        """
        Load data and determine which samples need processing.
        
        Returns:
            Tuple[List[Dict], Set[str]]: (samples_to_process, completed_sample_ids)
        """
        # Load test data
        with open(self.data_file, "r", encoding="utf-8") as f:
            all_samples = json.load(f)
        
        # Load existing predictions
        completed_samples = self.load_existing_predictions()
        logger.info(f"Found {len(completed_samples)} already completed samples")
        
        # Filter out completed samples
        remaining_samples = [sample for sample in all_samples if sample["id"] not in completed_samples]
        logger.info(f"Need to process {len(remaining_samples)} samples out of {len(all_samples)} total")
        
        return remaining_samples, completed_samples
    
    def finalize_predictions(self, total_time: float) -> None:
        """Update total time and create submission file."""
        # Update total time
        try:
            with open(self.predictions_file, "r", encoding="utf-8") as f:
                final_data = json.load(f)
            
            final_data["total_time"] = f"{total_time:.4f}s"
            
            with open(self.predictions_file, "w", encoding="utf-8") as f:
                json.dump(final_data, f, indent=4, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to update total time: {e}")
        
        # Create submission file
        task_name = f"task2_{self.mode}"
        create_submission_from_predictions(self.predictions_file, task_name)
        logger.info(f"Created submission file: {task_name}_submission.json")
    
    def run(self) -> None:
        """Main execution method."""
        start_time = time()
        all_results = []
        
        # Get samples to process
        samples_to_process, completed_samples = self.get_samples_to_process()
        
        if not samples_to_process:
            logger.info("No samples to process. All samples already completed!")
            self.finalize_predictions(0)
            return
        
        # Process main batch
        logger.info("Starting main processing...")
        results, failed_samples = self.process_samples_batch(
            samples_to_process, 
            "[INFO] Main inference"
        )
        all_results.extend(results)
        
        # Retry failed samples
        if failed_samples:
            logger.info(f"Retrying {len(failed_samples)} failed samples...")
            retry_results, still_failed = self.process_samples_batch(
                failed_samples,
                "[INFO] Retrying failed samples"
            )
            all_results.extend(retry_results)
            
            if still_failed:
                failed_ids = [s["id"] for s in still_failed]
                logger.warning(f"Still failed after retry: {failed_ids}")
        
        # Final verification
        current_completed = self.load_existing_predictions()
        with open(self.data_file, "r", encoding="utf-8") as f:
            all_sample_ids = {sample["id"] for sample in json.load(f)}
        
        missing_ids = all_sample_ids - current_completed
        if missing_ids:
            logger.warning(f"Missing samples detected: {sorted(missing_ids)}")
        else:
            logger.info("✅ All samples processed successfully!")
        
        # Finalize
        total_time = time() - start_time
        logger.info(f"Total processing time: {total_time:.4f}s")
        logger.info(f"Processed {len(all_results)} samples in this run")
        
        self.finalize_predictions(total_time)


def main():
    """Main entry point."""
    # Configuration
    mode = "test"  # Change to "train" for training data
    
    # Run inference
    runner = Task2Runner(mode=mode)
    runner.run()


if __name__ == "__main__":
    main()
