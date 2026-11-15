
import csv
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs" / "evaluation"

INPUT_FILE = OUTPUT_DIR / "error_analysis.csv"
OUTPUT_FILE = OUTPUT_DIR / "categorized_error_analysis.csv"

STRATEGIES = [
    "zero_shot",
    "few_shot_3",
    "few_shot_5",
    "few_shot_8",
    "chain_of_thought",
]

FIELDS = [
    "claim_type",
    "urgency",
    "policy_number",
    "incident_date",
    "location",
    "vehicle_reg_no",
    "estimated_amount_inr",
    "injury_reported",
    "police_report_filed",
]

def parse_json(value):
    try:
        return json.loads(value) if value else {}
    except (json.JSONDecodeError, TypeError):
        return {}

def normalize(value):
    if isinstance(value, str):
        value = value.strip()

        if value.lower() == "null":
            return None
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False

    return value

def categorize_all(row):
    gold = parse_json(row.get("gold_json"))
    pred = parse_json(row.get("predicted_json"))

    if not gold or not pred:
        return (
            ["Invalid JSON or missing data"],
            ["High"],
            ["Validate JSON parsing and required output fields."]
        )

    categories = []
    severities = []
    fixes = []

    def add(category, severity, fix):
        if category not in categories:
            categories.append(category)
            severities.append(severity)
            fixes.append(fix)

    gold_type = normalize(gold.get("claim_type"))
    pred_type = normalize(pred.get("claim_type"))

    if not pred_type:
        add(
            "Invalid or missing claim classification",
            "High",
            "Enforce a valid claim_type from the six permitted classes."
        )
    elif gold_type != pred_type:
        add(
            "Incorrect claim classification",
            "High",
            "Improve class definitions and add contrastive examples for confused claim types."
        )

    for field in FIELDS:
        expected = normalize(gold.get(field))
        actual = normalize(pred.get(field))

        if expected == actual:
            continue

        if field == "claim_type":
            continue

        if field == "urgency":
            add(
                "Incorrect urgency",
                "High",
                "Apply explicit urgency rules; do not infer urgency from claim type alone."
            )

        elif field == "incident_date":
            if expected is None and actual is not None:
                add(
                    "Date hallucination",
                    "High",
                    "Extract dates only when explicitly present; otherwise return null."
                )
            else:
                add(
                    "Date extraction error",
                    "Medium",
                    "Improve date recognition and standardize date formats."
                )

        elif field == "location":
            if expected is None and actual is not None:
                add(
                    "Location over-extraction",
                    "Medium",
                    "Extract a specific incident location, not a generic setting or surrounding phrase."
                )
            else:
                add(
                    "Location extraction error",
                    "Medium",
                    "Improve identification of the actual incident location."
                )

        elif field == "vehicle_reg_no":
            add(
                "Vehicle registration extraction or formatting",
                "Medium",
                "Extract only actual registration numbers and normalize spacing before comparison."
            )

        elif field == "policy_number":
            add(
                "Policy number extraction error",
                "Medium",
                "Extract policy identifiers separately from vehicle registration numbers."
            )

        elif field == "injury_reported":
            add(
                "Injury information extraction",
                "High",
                "Distinguish explicitly reported injury, explicitly no injury, and unknown."
            )

        elif field == "police_report_filed":
            if expected is None and actual is True:
                add(
                    "Unsupported police report inference",
                    "High",
                    "Do not assume a police report was filed unless the message explicitly confirms it."
                )
            else:
                add(
                    "Police report status extraction",
                    "Medium",
                    "Recognize both positive and negative statements about police reports."
                )

        elif field == "estimated_amount_inr":
            add(
                "Estimated amount extraction",
                "Medium",
                "Extract monetary amounts accurately and avoid unsupported estimates."
            )

    if not categories:
        add(
            "Other or requires manual review",
            "Low",
            "Inspect the original message and prediction to identify the underlying issue."
        )

    return categories, severities, fixes

def main():
    if not INPUT_FILE.exists():
        print(f"ERROR: Missing input file: {INPUT_FILE}")
        return

    with INPUT_FILE.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    categorized = []

    for strategy in STRATEGIES:
        strategy_rows = [
            row for row in rows
            if row.get("strategy") == strategy
        ]

        for row in strategy_rows[:15]:
            item = dict(row)

            categories, severities, fixes = categorize_all(item)

            item["root_cause_categories"] = " | ".join(categories)
            item["severity_levels"] = " | ".join(severities)
            item["proposed_fixes"] = " | ".join(fixes)
            item["review_status"] = "Suggested - needs human verification"

            categorized.append(item)

    if not categorized:
        print("No matching error records found.")
        return

    fieldnames = list(categorized[0].keys())

    with OUTPUT_FILE.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(categorized)

    print("\nMULTI-CATEGORY ERROR ANALYSIS COMPLETE")
    print(f"Total records: {len(categorized)}")

    category_counts = {}

    for row in categorized:
        for category in row["root_cause_categories"].split(" | "):
            category_counts[category] = category_counts.get(category, 0) + 1

    print("\nERROR CATEGORY COUNTS:")
    for category, count in sorted(category_counts.items()):
        print(f"{category}: {count}")

    print(f"\nSaved to: {OUTPUT_FILE}")
    print("All suggested categories require human verification.")

if __name__ == "__main__":
    main()
