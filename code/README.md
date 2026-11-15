# ClaimFast — Code Guide

## Requirements
- Python 3
- Ollama
- Ollama model: llama3.2:3b
- Python packages listed in ../requirements.txt

## Setup
From the project root:

```bash
pip install -r requirements.txt
ollama pull llama3.2:3b

## Main experiment scripts
Run scripts from the project root:

python code/zero_shot.py
python code/few_shot_3.py
python code/few_shot_5.py
python code/few_shot_8.py

The Chain-of-Thought prompt is stored in prompts/cot.txt.

## Held-out evaluation
The final held-out experiment runner executes the five prompting strategies across three runs:

python code/run_test_experiments.py

This makes 900 inference requests. Do not rerun it unless you intend to repeat the full experiment.

## Evaluation and analysis

python code/evaluate_final_test.py
python code/advanced_evaluation.py
python code/categorize_errors.py
python code/compare_experiments.py
python code/visualize_results.py

## Data
- data/dev.jsonl: 20 development examples
- data/test.jsonl: 60 held-out test examples
- data/labelling_guide.md: labelling guidance

Few-shot examples are selected from development data. The held-out test set is used for final evaluation.

## Outputs
- outputs/final_test/: raw predictions for each strategy and run
- outputs/evaluation/: evaluation metrics and error analysis
- outputs/results.csv: consolidated strategy comparison
- outputs/comparison_report.csv: comparison results

## Reproducibility
The experiment used llama3.2:3b with temperature 0. The same 60 held-out records were evaluated for each strategy across three runs.

## Limitations
The dataset is synthetic/anonymised and does not establish real-world insurance performance. Automated error categories are suggestions and require human verification. The independent reviewer template must be completed by a second person before claiming independent human agreement.
