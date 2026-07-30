# SIGMA: SIGn Multimodal Agents

[![Paper](https://img.shields.io/badge/ACL%20Anthology-2025.vlsp--1.50-b31b1b)](https://aclanthology.org/2025.vlsp-1.50/)
[![VLSP 2025](https://img.shields.io/badge/VLSP%202025-Top%205%20Finalist-blue)](https://aclanthology.org/2025.vlsp-1.50/)

We present **SIGMA (SIGn Multimodal Agents)**, a system for **multimodal legal question answering** on Vietnamese traffic sign rules, developed for the **VLSP 2025 MLQA-TSR shared task**. The task requires combining **traffic sign images** and **legal documents** to answer **multiple-choice** or **yes/no questions**, demanding both accurate **visual interpretation** and **legal grounding**.  

SIGMA adopts a **multi-agent architecture**:
- **Traffic Sign Understanding Agent** – interprets traffic signs and generates scene descriptions.  
- **Legal Relevance Agent** – verifies candidate legal articles.  
- **Legal Filtering Agent** – extracts essential legal provisions.  
- **Reasoning Agent** – integrates all information to produce the final answer with supporting citations.  

On the official evaluation, SIGMA achieved **72% accuracy in Subtask 2**, ranking among the **top-5 finalists**, demonstrating the effectiveness of agent-based multimodal systems for **legal QA in safety-critical domains** such as **traffic law compliance**.

---

## 🔧 Setup

1. Clone this repository:
    ```bash
    git clone https://github.com/KiRzEa/VLSP_2025_SIGMA.git
    cd VLSP_2025_SIGMA
    ```
2. Create a Python environment (Python 3.12.11 recommended):

    Option 1: Using venv (recommended)
    ```bash
    python3.12 -m venv .venv
    source .venv/bin/activate   # On Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    ```

    Option 2: Using ```conda```
    ```bash
    conda create -n sigma python=3.12.11
    conda activate sigma
    pip install -r requirements.txt
    ```

## 📊 Data

The full VLSP 2025 MLQA-TSR dataset (traffic sign images, legal document DB, train/test splits) is not committed to this repo due to size — see the shared task organizers for the official release. A small illustrative subset (sample traffic sign images + a trimmed `law_db`) is included under [`sample/`](sample/) so you can see the expected data format without downloading the full corpus.

## 📂 Project Structure
    ```
    src/
    │── core/         # Core system (graphs, states, orchestration)
    │── prepare_data/ # Data preprocessing and vector DB builders
    │── prompts/      # Prompt templates for different agents
    │── services/     # External/internal services (LLM, retrieval, detection, etc.)
    │── utils/        # Utility functions and helpers
    ```

## 🚀 Usage

The project provides a runner script for **Task 2** (`runtask2`) that performs inference on different dataset splits.  
The `--mode` argument specifies **which dataset to run inference on**:

- `train` → Run inference on the **training set**  
- `public_test` → Run inference on the **public test set** (default)  
- `private_test` → Run inference on the **private test set**  

### Example commands

Run inference on the **training set**:
```bash
python run_task2.py --mode train
```
Run inference on the **public test set**:
```bash
python run_task2.py --mode public_test
```
Run inference on the **private test set**:
```bash
python run_task2.py --mode private_test
```
### Output files
For each mode, two files will be generated:

- Predictions file → **task2_[mode]_predictions.json**
(stores intermediate predictions and total runtime)

- Submission file → **[mode]_submission_task2.json**
(stores the final submission format expected by the shared task)

## 📄 Citation

If you use this work, please cite:

```bibtex
@inproceedings{kiet-etal-2025-metamorphic,
    title = "Metamorphic at {VLSP} 2025: {SIGMA} {--} A Multimodal Agent System for Legal {QA} on {V}ietnamese Traffic Signs",
    author = "Kiet, Nguyen Tuan  and
      Anh, Nguyen Khanh Tuan  and
      Nguyen, Long Hoang Huu  and
      Tai, Dam Vu Trong  and
      Thin, Dang Van",
    editor = "Mai, Luong Chi  and
      Huyen, Nguyen Thi Minh  and
      Trang, Nguyen Thi Thu",
    booktitle = "Proceedings of the 11th International Workshop on Vietnamese Language and Speech Processing",
    month = oct,
    year = "2025",
    address = "Hanoi, Vietnam",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.vlsp-1.50/",
    pages = "418--430"
}
```

Paper: [aclanthology.org/2025.vlsp-1.50](https://aclanthology.org/2025.vlsp-1.50/)