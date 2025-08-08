import json
from tqdm.auto import tqdm
from time import time

from src.utils import transform_raw_sample_to_input_state
from src.graphs.global_graph import GlobalGraph


def main():
    graph = GlobalGraph()

    with open("./data/train_data/vlsp_2025_train.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    start = time()
    for sample in tqdm(data[:1], desc="[INFO] Inference..."):
        prediction = sample.copy()
        input_state = transform_raw_sample_to_input_state(sample)
        output_state = graph.run(input_state)

        prediction["predicted_answer"] = output_state["predicted_answer"]

        results.append(prediction)
    end = time()

    with open("./results.json", "w", encoding="utf-8") as f:
        content = {
            "total_time": f"{end - start:.4f}s",
            "results": results
        }
        json.dump(content, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()