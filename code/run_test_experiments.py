
import json
import time
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime, timezone

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "test.jsonl"
PROMPT_DIR = BASE_DIR / "prompts"
OUTPUT_DIR = BASE_DIR / "outputs" / "final_test"

MODEL = "llama3.2:3b"
API_URL = "http://localhost:11434/api/chat"

RUNS = 3

EXPERIMENTS = {
    "zero_shot": "zero_shot.txt",
    "few_shot_3": "few_shot_3.txt",
    "few_shot_5": "few_shot_5.txt",
    "few_shot_8": "few_shot_8.txt",
    "chain_of_thought": "cot.txt",
}


def load_jsonl(path):
    with open(path, encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def call_ollama(message, prompt_template):
    prompt = prompt_template.replace("{message}", message)

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0
        }
    }

    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    start = time.perf_counter()

    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.loads(response.read().decode("utf-8"))

    latency = time.perf_counter() - start

    raw = result["message"]["content"]

    try:
        prediction = json.loads(raw)
        parse_error = None
    except json.JSONDecodeError as error:
        prediction = None
        parse_error = str(error)

    return {
        "prediction": prediction,
        "raw_response": raw,
        "parse_error": parse_error,
        "json_valid": prediction is not None,
        "latency_seconds": round(latency, 3),
        "prompt_tokens": result.get("prompt_eval_count"),
        "completion_tokens": result.get("eval_count")
    }


def main():
    records = load_jsonl(DATA_FILE)

    if len(records) != 60:
        raise ValueError(
            f"Expected 60 test records, found {len(records)}"
        )

    print("ClaimFast Final Test Evaluation")
    print("Model:", MODEL)
    print("Temperature: 0")
    print("Test records:", len(records))
    print("Runs per strategy:", RUNS)

    for strategy, prompt_file in EXPERIMENTS.items():

        prompt_path = PROMPT_DIR / prompt_file

        if not prompt_path.exists():
            raise FileNotFoundError(prompt_path)

        prompt_template = prompt_path.read_text(
            encoding="utf-8"
        )

        strategy_dir = OUTPUT_DIR / strategy
        strategy_dir.mkdir(parents=True, exist_ok=True)

        for run_number in range(1, RUNS + 1):

            output_file = strategy_dir / f"run_{run_number}.jsonl"

            # Never overwrite an existing experiment run.
            if output_file.exists():
                raise FileExistsError(
                    f"{output_file} already exists. "
                    "Move or rename it before rerunning."
                )

            print(
                f"\n[{strategy}] Run {run_number}/{RUNS}"
            )

            results = []

            for index, record in enumerate(records, start=1):

                print(
                    f"Processing {index}/{len(records)}: "
                    f"{record['id']}"
                )

                try:
                    response = call_ollama(
                        record["message"],
                        prompt_template
                    )

                    item = {
                        "id": record["id"],
                        "message": record["message"],
                        "gold": record["gold"],
                        **response,
                        "strategy": strategy,
                        "run": run_number,
                        "model": MODEL,
                        "temperature": 0,
                        "evaluated_at": datetime.now(
                            timezone.utc
                        ).isoformat()
                    }

                except Exception as error:
                    item = {
                        "id": record["id"],
                        "message": record["message"],
                        "gold": record["gold"],
                        "prediction": None,
                        "raw_response": None,
                        "parse_error": str(error),
                        "json_valid": False,
                        "latency_seconds": None,
                        "prompt_tokens": None,
                        "completion_tokens": None,
                        "strategy": strategy,
                        "run": run_number,
                        "model": MODEL,
                        "temperature": 0
                    }

                    print("ERROR:", error)

                results.append(item)

                if item["prediction"] is not None:
                    print(
                        "  Prediction:",
                        item["prediction"].get("claim_type")
                    )

            with open(
                output_file, "w", encoding="utf-8"
            ) as file:
                for item in results:
                    file.write(
                        json.dumps(item, ensure_ascii=False) + "\n"
                    )

            successful = sum(
                item["json_valid"] for item in results
            )

            print("Saved:", output_file)
            print("Records:", len(results))
            print("Valid JSON:", successful)

    print("\nALL EXPERIMENTS FINISHED")
    print("Results directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
