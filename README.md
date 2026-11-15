# ClaimFast – LLM-Based Insurance Claim Classification

## 1. Project Overview

ClaimFast is an experimental insurance claim classification and information extraction system using Large Language Models (LLMs).

It processes vehicle insurance messages and generates structured JSON containing claim categories, urgency, policy information, incident details, and other relevant fields.

The project compares five prompting configurations:
- Zero-Shot
- Few-Shot (k=3)
- Few-Shot (k=5)
- Few-Shot (k=8)
- Chain-of-Thought

## 2. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application and evaluation scripts |
| Ollama | Local LLM execution |
| llama3.2:3b | Language model |
| JSON / JSONL | Dataset and structured responses |
| Pandas | Evaluation and CSV processing |
| Matplotlib | Visualizations |

## 3. Dataset

The synthetic and anonymised dataset contains 80 labelled messages:

| Dataset | Records | Purpose |
|---|---:|---|
| data/dev.jsonl | 20 | Prompt development |
| data/test.jsonl | 60 | Held-out final evaluation |

The six claim categories are:
- own_damage
- third_party_liability
- theft
- total_loss
- glass_only
- not_a_claim

See `data/labelling_guide.md` for labelling and extraction rules.

## 4. Experimental Methodology

All five strategies used `llama3.2:3b` through Ollama with temperature 0.

Each strategy was evaluated three times on the same 60 held-out records.

Total experiment size:
- 5 strategies
- 3 runs per strategy
- 60 records per run
- 900 inference requests

The held-out test set was not used for prompt development.

## 5. Final Experimental Results

| Strategy | Accuracy | Macro F1 | Extraction Exact Match |
|---|---:|---:|---:|
| Zero-Shot | 93.33% | 0.940 | 48.33% |
| Few-Shot 3 | 83.33% | 0.836 | 58.33% |
| Few-Shot 5 | 91.67% | 0.933 | 63.33% |
| Few-Shot 8 | 95.00% | 0.954 | 68.33% |
| Chain-of-Thought | 96.67% | 0.970 | 51.67% |

Chain-of-Thought achieved the highest classification accuracy and Macro F1-score. Few-Shot 8 achieved the highest extraction exact-match rate.

Zero-Shot had the lowest average latency at approximately 1.517 seconds per message.

## 6. Evaluation and Supporting Evidence

Evaluation files are stored in `outputs/evaluation/`.

Important files include:
- `strategy_comparison.csv`
- `run_metrics.csv`
- `per_field_metrics.csv`
- `operational_metrics.csv`
- `run_agreement.csv`
- Confusion matrices for all strategies and runs
- `categorized_error_analysis.csv`

The consolidated results table is `outputs/results.csv`.

The error categories are automatically suggested and require human verification.

## 7. Project Structure

```text
ClaimFast/
├── code/
│   ├── zero_shot.py
│   ├── few_shot_3.py
│   ├── few_shot_5.py
│   ├── few_shot_8.py
│   ├── run_test_experiments.py
│   ├── evaluate_final_test.py
│   ├── advanced_evaluation.py
│   ├── categorize_errors.py
│   ├── compare_experiments.py
│   ├── validate_dataset.py
│   └── README.md
├── data/
│   ├── dev.jsonl
│   ├── test.jsonl
│   └── labelling_guide.md
├── prompts/
│   ├── zero_shot.txt
│   ├── few_shot_3.txt
│   ├── few_shot_5.txt
│   ├── few_shot_8.txt
│   └── cot.txt
├── outputs/
│   ├── evaluation/
│   ├── final_test/
│   ├── results.csv
│   └── independent_label_review.csv
├── report/
│   └── claimfast_experiment_report.md
├── requirements.txt
└── README.md
