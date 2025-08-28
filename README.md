# SIGMA: SIGn Multimodal Agents

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
    git clone <repo-url>
    cd <repo-folder>
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
python runtask2 --mode train
```
Run inference on the **public test set**:
```bash
python runtask2 --mode public_test
```
Run inference on the **private test set**:
```bash
python runtask2 --mode private_test
```
### Output files
For each mode, two files will be generated:

- Predictions file → **task2_[mode]_predictions.json**
(stores intermediate predictions and total runtime)

- Submission file → **[mode]_submission_task2.json**
(stores the final submission format expected by the shared task)