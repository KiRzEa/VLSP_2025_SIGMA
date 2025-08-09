import json
import os
from pathlib import Path
from typing import Optional, List, Dict, Any

from src.core.logger import setup_logger

logger = setup_logger("SubmissionUtils")


def detect_task_type(sample: Dict[str, Any]) -> str:
    """
    Detect task type based on sample structure.
    
    Returns:
        "task1" for open-ended questions
        "task2" for multiple choice questions
    """
    if "choices" in sample and "question_type" in sample:
        return "task2" 
    else:
        return "task1"


def transform_to_submission_format(
    predictions_file: str, 
    output_file: Optional[str] = None,
    task_type: Optional[str] = None
):
    """
    Transform predictions to submission format (only id and predicted_answer).
    
    Args:
        predictions_file: Path to the full predictions file
        output_file: Output path for submission format. If None, auto-generate based on input filename
        task_type: Force specific task type ("task1" or "task2"). If None, auto-detect from data
    """
    if output_file is None:
        # Auto-generate output filename
        base_path = Path(predictions_file)
        output_file = str(base_path.parent / f"{base_path.stem}_submission{base_path.suffix}")
    
    try:
        with open(predictions_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        results = data.get("results", [])
        if not results:
            logger.warning("No results found in predictions file")
            return None
            
        # Auto-detect task type if not specified
        if task_type is None:
            task_type = detect_task_type(results[0])
            logger.info(f"Auto-detected task type: {task_type}")
        
        submission_results = []
        for result in results:
            submission_entry = {
                "id": result["id"],
                "predicted_answer": result["predicted_answer"]
            }
            
            # For task1, we might want to include additional fields in the future
            # For now, both tasks have the same submission format
            submission_results.append(submission_entry)
        
        submission_data = {
            "results": submission_results
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(submission_data, f, indent=4, ensure_ascii=False)
            
        logger.info(f"Submission format saved to {output_file} (detected: {task_type})")
        return output_file
        
    except Exception as e:
        logger.error(f"Failed to create submission format: {e}")
        return None


def transform_to_submission_task2_full_info(predictions_file: str, output_file: str):
    """
    Transform predictions to submission format for task2, keeping only required fields.
    Output is a list of dicts with: id, image_id, question, relevant_articles, question_type (if exists), and answer.
    """
    try:
        with open(predictions_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        results = data.get("results", [])
        submission_results = []
        for result in results:
            # Keep only required fields
            entry = {
                "id": result["id"],
                "image_id": result["image_id"],
                "question": result["question"],
                "relevant_articles": result["relevant_articles"],
                "answer": result["predicted_answer"]  # Rename predicted_answer to answer
            }
            
            # Add question_type if it exists (for multiple choice questions)
            if "question_type" in result:
                entry["question_type"] = result["question_type"]
            
            submission_results.append(entry)
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(submission_results, f, indent=4, ensure_ascii=False)
            
        logger.info(f"Task2 submission format saved to {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"Failed to create task2 submission format: {e}")
        return None


def create_submission_from_predictions(predictions_file: str, task_name: str = "task2"):
    """
    Create submission file from predictions with standard naming.
    Auto-detects task type from data structure.
    
    Args:
        predictions_file: Path to the full predictions file
        task_name: Task name for output filename (e.g., "task1", "task2") 
    """
    output_file = f"./{task_name}_submission.json"
    return transform_to_submission_format(predictions_file, output_file)


def batch_create_submissions(predictions_files: List[str], output_dir: str = "./"):
    """
    Create submission files for multiple prediction files.
    
    Args:
        predictions_files: List of prediction file paths
        output_dir: Directory to save submission files
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []
    
    for pred_file in predictions_files:
        try:
            # Extract task name from filename (e.g., "task1_predictions.json" -> "task1")
            filename = Path(pred_file).stem
            if "_predictions" in filename:
                task_name = filename.replace("_predictions", "")
            else:
                task_name = "unknown"
            
            output_file = Path(output_dir) / f"{task_name}_submission.json"
            result = transform_to_submission_format(pred_file, str(output_file))
            results.append(result)
            
        except Exception as e:
            logger.error(f"Failed to process {pred_file}: {e}")
            results.append(None)
    
    return results


if __name__ == "__main__":
    # Example usage:
    # python -m src.utils.submission_utils <predictions_file> [task_type]
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.utils.submission_utils <predictions_file> [task_type]")
        print("  task_type: 'task1' or 'task2' (optional, will auto-detect if not provided)")
        sys.exit(1)
    
    predictions_file = sys.argv[1]
    task_type = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(predictions_file):
        logger.error(f"File not found: {predictions_file}")
        sys.exit(1)

    result = transform_to_submission_task2_full_info(predictions_file, output_file="./submission_task2.json")
    if result:
        logger.info(f"Successfully created submission file: {result}")
    else:
        logger.error("Failed to create submission file")
        sys.exit(1)
