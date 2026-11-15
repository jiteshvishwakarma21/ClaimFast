import json
import csv
from pathlib import Path
from collections import defaultdict
from statistics import mean

BASE_DIR = Path(__file__).resolve().parent.parent
TEST_FILE = BASE_DIR / "data" / "test.jsonl"
RESULTS_DIR = BASE_DIR / "outputs" / "final_test"
OUTPUT_DIR = BASE_DIR / "outputs" / "evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = [
    "own_damage",
    "third_party_liability",
    "theft",
    "glass_only",
    "not_a_claim",
    "total_loss",
]

FIELDS = [
    "urgency",
    "policy_number",
    "incident_date",
    "location",
    "vehicle_reg_no",
    "estimated_amount_inr",
    "injury_reported",
    "police_report_filed",
]


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def safe_value(value):
    if isinstance(value, str):
        return value.strip().lower()
    return value


def classification_metrics(rows):
    matrix = {g: {p: 0 for p in CLASSES + ["INVALID"]} for g in CLASSES}
    correct = 0
    valid = 0

    for row in rows:
        gold = row["gold"]["claim_type"]
        prediction = row.get("prediction") or {}
        predicted = prediction.get("claim_type")

        if predicted in CLASSES:
            valid += 1
        else:
            predicted = "INVALID"

        matrix[gold][predicted] += 1

        if gold == predicted:
            correct += 1

    accuracy = correct / len(rows) if rows else 0

    f1_scores = []
    for label in CLASSES:
        tp = matrix[label][label]
        fp = sum(matrix[g][label] for g in CLASSES if g != label)
        fn = sum(matrix[label][p] for p in CLASSES + ["INVALID"] if p != label)

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0
        )
        f1_scores.append(f1)

    return accuracy, mean(f1_scores), valid, matrix


def extraction_metrics(rows):
    exact = 0
    total = 0
    field_correct = defaultdict(int)
    field_total = defaultdict(int)
    false_positive = defaultdict(int)
    false_negative = defaultdict(int)

    for row in rows:
        gold = row["gold"]
        prediction = row.get("prediction") or {}

        record_exact = True

        for field in FIELDS:
            actual = safe_value(gold.get(field))
            predicted = safe_value(prediction.get(field))

            field_total[field] += 1
            total += 1

            if actual == predicted:
                field_correct[field] += 1
            else:
                record_exact = False

            if actual is None and predicted is not None:
                false_positive[field] += 1
            elif actual is not None and predicted is None:
                false_negative[field] += 1

        if record_exact:
            exact += 1

    field_accuracy = {
        field: field_correct[field] / field_total[field]
        if field_total[field] else 0
        for field in FIELDS
    }

    return exact / len(rows), field_accuracy, false_positive, false_negative


def main():
    test_data = load_jsonl(TEST_FILE)
    summaries = []
    matrices = {}
    all_results = defaultdict(list)

    files = sorted(RESULTS_DIR.glob("*/run_*.jsonl"))

    print("Files found:", len(files))
    print("Test records:", len(test_data))

    for path in files:
        rows = load_jsonl(path)

        if len(rows) != len(test_data):
            print("WARNING: Incorrect record count:", path)
            continue

        strategy = path.parent.name
        run_name = path.stem

        accuracy, macro_f1, valid, matrix = classification_metrics(rows)
        extraction_exact, field_accuracy, fp, fn = extraction_metrics(rows)

        json_valid = sum(bool(r.get("json_valid")) for r in rows)
        avg_latency = mean(r.get("latency_seconds", 0) for r in rows)
        avg_tokens = mean(
            r.get("prompt_tokens", 0) + r.get("completion_tokens", 0)
            for r in rows
        )

        summary = {
            "strategy": strategy,
            "run": run_name,
            "records": len(rows),
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "valid_classifications": valid,
            "json_valid_count": json_valid,
            "extraction_record_exact_match": extraction_exact,
            "average_latency_seconds": avg_latency,
            "average_total_tokens": avg_tokens,
        }

        summaries.append(summary)
        all_results[strategy].append(summary)
        matrices[f"{strategy}_{run_name}"] = matrix

        print(
            f"{strategy}/{run_name}: "
            f"Accuracy={accuracy:.3f}, "
            f"Macro-F1={macro_f1:.3f}, "
            f"Extraction exact={extraction_exact:.3f}"
        )

    # Save individual run metrics
    with open(OUTPUT_DIR / "run_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)

    # Save average metrics by strategy
    aggregate = []
    for strategy, runs in sorted(all_results.items()):
        aggregate.append({
            "strategy": strategy,
            "runs": len(runs),
            "mean_accuracy": mean(r["accuracy"] for r in runs),
            "mean_macro_f1": mean(r["macro_f1"] for r in runs),
            "mean_extraction_exact_match": mean(
                r["extraction_record_exact_match"] for r in runs
            ),
            "mean_latency_seconds": mean(
                r["average_latency_seconds"] for r in runs
            ),
            "mean_total_tokens": mean(
                r["average_total_tokens"] for r in runs
            ),
        })

    with open(OUTPUT_DIR / "strategy_comparison.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=aggregate[0].keys())
        writer.writeheader()
        writer.writerows(aggregate)

    # Save confusion matrices
    for name, matrix in matrices.items():
        with open(OUTPUT_DIR / f"confusion_{name}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["gold/predicted"] + CLASSES + ["INVALID"])
            for label in CLASSES:
                writer.writerow(
                    [label] + [matrix[label][p] for p in CLASSES + ["INVALID"]]
                )

    # Save extraction field accuracy
    extraction_summary = {}
    for strategy in all_results:
        extraction_summary[strategy] = {}

    with open(OUTPUT_DIR / "evaluation_summary.json", "w") as f:
        json.dump({
            "classification": aggregate,
            "run_metrics": summaries,
            "confusion_matrices": matrices,
        }, f, indent=2)

    print("\nEVALUATION COMPLETE")
    print("Output folder:", OUTPUT_DIR)
    print("Files created:")
    print("- run_metrics.csv")
    print("- strategy_comparison.csv")
    print("- evaluation_summary.json")
    print("- confusion matrix CSV files")


if __name__ == "__main__":
    main()