
import csv
import json
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "outputs" / "final_test"
OUT = BASE / "outputs" / "evaluation"
OUT.mkdir(parents=True, exist_ok=True)

FIELDS = [
    "urgency", "policy_number", "incident_date", "location",
    "vehicle_reg_no", "estimated_amount_inr",
    "injury_reported", "police_report_filed"
]

CLASSES = [
    "own_damage", "third_party_liability", "theft",
    "glass_only", "not_a_claim", "total_loss"
]


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def norm(value):
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ("", "null", "none", "unknown", "n/a"):
            return None
        if value in ("true", "yes"):
            return True
        if value in ("false", "no"):
            return False
    return value


def write_csv(path, rows):
    if not rows:
        print("No rows to write:", path.name)
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    files = sorted(RESULTS.glob("*/run_*.jsonl"))
    grouped = defaultdict(dict)

    for path in files:
        strategy = path.parent.name
        run_number = int(path.stem.split("_")[-1])
        grouped[strategy][run_number] = load_jsonl(path)

    field_stats = defaultdict(lambda: {
        "tp": 0, "fp": 0, "fn": 0,
        "exact": 0, "total": 0,
        "hallucinations": 0, "gold_null": 0
    })

    safety = []
    agreement = []
    errors = []

    for strategy, runs in sorted(grouped.items()):
        if not runs:
            continue

        # Use run 1 as the primary run for detailed field and error analysis.
        primary = runs[1]
        correct_injury = 0
        total_injury = 0
        correct_urgency = 0
        total_urgency = 0
        classification_errors = 0
        extraction_errors = 0

        # Compare predictions across all three runs.
        run_maps = {
            run: {r["id"]: r.get("prediction") or {} for r in rows}
            for run, rows in runs.items()
        }

        ids = sorted(run_maps[1].keys())
        identical = 0

        for record_id in ids:
            predictions = [
                run_maps[run].get(record_id)
                for run in sorted(run_maps)
            ]
            if all(p == predictions[0] for p in predictions):
                identical += 1

        agreement.append({
            "strategy": strategy,
            "records_compared": len(ids),
            "identical_across_all_runs": identical,
            "agreement_rate": round(identical / len(ids), 4)
            if ids else 0
        })

        for row in primary:
            gold = row.get("gold") or {}
            pred = row.get("prediction") or {}
            record_id = row.get("id")
            claim_correct = (
                norm(gold.get("claim_type")) ==
                norm(pred.get("claim_type"))
            )

            if not claim_correct:
                classification_errors += 1

            record_field_errors = []

            for field in FIELDS:
                actual = norm(gold.get(field))
                guessed = norm(pred.get(field))
                stats = field_stats[(strategy, field)]
                stats["total"] += 1

                if actual == guessed:
                    stats["exact"] += 1

                if actual is None:
                    if guessed is None:
                        pass
                    else:
                        stats["fp"] += 1
                        stats["hallucinations"] += 1
                        stats["gold_null"] += 1
                        record_field_errors.append(field)
                else:
                    if guessed is None:
                        stats["fn"] += 1
                        record_field_errors.append(field)
                    elif guessed == actual:
                        stats["tp"] += 1
                    else:
                        stats["fp"] += 1
                        stats["fn"] += 1
                        record_field_errors.append(field)

                if actual is None and guessed is None:
                    stats["gold_null"] += 1

            if record_field_errors:
                extraction_errors += 1

            if gold.get("injury_reported") is True:
                total_injury += 1
                if norm(pred.get("injury_reported")) is True:
                    correct_injury += 1

            if norm(gold.get("urgency")) in ("high", "urgent", "critical"):
                total_urgency += 1
                if norm(pred.get("urgency")) in ("high", "urgent", "critical"):
                    correct_urgency += 1

            if not claim_correct or record_field_errors:
                if not claim_correct and record_field_errors:
                    category = "Classification and extraction error"
                elif not claim_correct:
                    category = "Classification error"
                else:
                    category = "Extraction error"

                errors.append({
                    "strategy": strategy,
                    "record_id": record_id,
                    "category": category,
                    "message": row.get("message", ""),
                    "gold_claim_type": gold.get("claim_type"),
                    "predicted_claim_type": pred.get("claim_type"),
                    "incorrect_extraction_fields": ", ".join(record_field_errors),
                    "gold_json": json.dumps(gold, ensure_ascii=False),
                    "predicted_json": json.dumps(pred, ensure_ascii=False)
                })

        safety.append({
            "strategy": strategy,
            "injury_cases": total_injury,
            "injury_true_positives": correct_injury,
            "injury_recall": round(correct_injury / total_injury, 4)
            if total_injury else "N/A",
            "high_urgency_cases": total_urgency,
            "high_urgency_true_positives": correct_urgency,
            "high_urgency_recall": round(correct_urgency / total_urgency, 4)
            if total_urgency else "N/A",
            "classification_errors_run1": classification_errors,
            "records_with_extraction_errors_run1": extraction_errors
        })

    # Per-field precision, recall, F1 and hallucination rate.
    field_rows = []

    for (strategy, field), s in sorted(field_stats.items()):
        precision = s["tp"] / (s["tp"] + s["fp"]) if s["tp"] + s["fp"] else 0
        recall = s["tp"] / (s["tp"] + s["fn"]) if s["tp"] + s["fn"] else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        hallucination_rate = (
            s["hallucinations"] / s["gold_null"]
            if s["gold_null"] else 0
        )

        field_rows.append({
            "strategy": strategy,
            "field": field,
            "exact_match_rate": round(s["exact"] / s["total"], 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "hallucinations": s["hallucinations"],
            "gold_null_opportunities": s["gold_null"],
            "hallucination_rate": round(hallucination_rate, 4)
        })

    write_csv(OUT / "per_field_metrics.csv", field_rows)
    write_csv(OUT / "operational_metrics.csv", safety)
    write_csv(OUT / "run_agreement.csv", agreement)
    write_csv(OUT / "error_analysis.csv", errors)

    print("\nADVANCED EVALUATION COMPLETE")
    print("Strategies:", len(grouped))
    print("Output folder:", OUT)

    print("\nRUN AGREEMENT")
    for item in agreement:
        print(
            item["strategy"],
            f'{item["identical_across_all_runs"]}/'
            f'{item["records_compared"]}',
            f'({item["agreement_rate"]:.1%})'
        )

    print("\nERROR COUNTS")
    for strategy in sorted(grouped):
        count = sum(1 for e in errors if e["strategy"] == strategy)
        print(f"{strategy}: {count} actual error records")

    print("\nFiles created:")
    print("- per_field_metrics.csv")
    print("- operational_metrics.csv")
    print("- run_agreement.csv")
    print("- error_analysis.csv")


if __name__ == "__main__":
    main()
