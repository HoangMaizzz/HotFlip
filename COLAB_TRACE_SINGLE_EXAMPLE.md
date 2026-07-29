# Colab cell: trace one untargeted example

The cell below selects one baseline row by example ID or supporting-document
title and prints each HotFlip beam-search step while it runs.

```python
import csv
import json
import subprocess
import sys
from pathlib import Path

# Set EXAMPLE_ID when it is known. Otherwise the cell searches by gold title.
EXAMPLE_ID = None
DOCUMENT_TITLE = "Big Stone Gap (film)"

TOP_K = 3
MAX_TOKEN_CHANGES = 10
BEAM_WIDTH = 3
UNTARGETED_ANSWER_WEIGHT = 1.0
CANDIDATE_VOCAB_SIZE = 30000
HOTFLIP_TOP_K = 10
CANDIDATES_PER_STATE = 10
GENERATOR_MODEL = "Qwen/Qwen2.5-14B-Instruct"

PROJECT_DIR = Path("/content/HotFlip")
BASELINE_IN_PROJECT = (
    PROJECT_DIR / "outputs" / "colab_baseline" / "baseline_results.csv"
)
UPLOADED_BASELINE = Path("/content/baseline_results.csv")
OUTPUT_DIR = PROJECT_DIR / "outputs" / "single_example_trace"
SELECTED_CSV = OUTPUT_DIR / "selected_baseline_row.csv"

subprocess.run(["nvidia-smi"], check=False)

import torch

if not torch.cuda.is_available():
    raise RuntimeError("Enable a Colab GPU runtime before running this cell.")

if (PROJECT_DIR / ".git").exists():
    subprocess.run(
        ["git", "-C", str(PROJECT_DIR), "pull", "--ff-only"],
        check=True,
    )
else:
    subprocess.run(
        [
            "git", "clone",
            "https://github.com/HoangMaizzz/HotFlip.git",
            str(PROJECT_DIR),
        ],
        check=True,
    )

subprocess.run(
    ["git", "-C", str(PROJECT_DIR), "log", "-1", "--oneline"],
    check=True,
)
subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
        "-r", str(PROJECT_DIR / "requirements-colab.txt"),
    ],
    check=True,
)

if BASELINE_IN_PROJECT.is_file():
    baseline_csv = BASELINE_IN_PROJECT
elif UPLOADED_BASELINE.is_file():
    baseline_csv = UPLOADED_BASELINE
else:
    raise FileNotFoundError(
        "Run baseline first or upload /content/baseline_results.csv"
    )

with baseline_csv.open(encoding="utf-8-sig", newline="") as handle:
    baseline_rows = list(csv.DictReader(handle))

from datasets import load_dataset

dataset = load_dataset(
    "hotpotqa/hotpot_qa", "distractor", split="validation"
)
dataset_by_id = {
    str(item_id): index for index, item_id in enumerate(dataset["id"])
}

matches = []
for row in baseline_rows:
    row_id = str(row["id"])
    if row_id not in dataset_by_id:
        continue
    item = dataset[dataset_by_id[row_id]]
    supporting_titles = set(item["supporting_facts"]["title"])
    id_matches = EXAMPLE_ID is not None and row_id == str(EXAMPLE_ID)
    title_matches = (
        EXAMPLE_ID is None
        and DOCUMENT_TITLE in supporting_titles
    )
    if id_matches or title_matches:
        matches.append((row, item))

if not matches:
    raise ValueError(
        f"No baseline row matched EXAMPLE_ID={EXAMPLE_ID!r} or "
        f"gold DOCUMENT_TITLE={DOCUMENT_TITLE!r}."
    )

selected_row, selected_item = matches[0]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
with SELECTED_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=selected_row.keys())
    writer.writeheader()
    writer.writerow(selected_row)

print("\nSELECTED EXAMPLE")
print("ID              :", selected_row["id"])
print("Question        :", selected_item["question"])
print("Gold answer     :", selected_item["answer"])
print("Baseline answer :", selected_row["llm_answer"])
print("Baseline correct:", selected_row["llm_judge_correct"])
print("Gold titles     :", list(selected_item["supporting_facts"]["title"]))
if len(matches) > 1:
    print(
        f"Note: {len(matches)} rows matched; the first was selected. "
        "Set EXAMPLE_ID to choose another one."
    )

command = [
    sys.executable, "-u", "-m", "hotflip_rag.compare_attacks",
    "--baseline-results", str(SELECTED_CSV),
    "--modes", "untargeted",
    "--num-examples", "1",
    "--top-k", str(TOP_K),
    "--max-token-changes", str(MAX_TOKEN_CHANGES),
    "--search-strategy", "beam",
    "--beam-width", str(BEAM_WIDTH),
    "--untargeted-answer-weight", str(UNTARGETED_ANSWER_WEIGHT),
    "--candidate-vocab-size", str(CANDIDATE_VOCAB_SIZE),
    "--hotflip-top-k", str(HOTFLIP_TOP_K),
    "--candidates-per-state", str(CANDIDATES_PER_STATE),
    "--no-disallow-numeric-replacement",
    "--generator-model", GENERATOR_MODEL,
    "--load-in-4bit",
    "--trace-hotflip",
    "--fail-fast",
    "--output-dir", str(OUTPUT_DIR),
]

print("\nRUNNING")
print(" ".join(command))
print()

process = subprocess.Popen(
    command,
    cwd=str(PROJECT_DIR),
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
for line in process.stdout:
    print(line, end="", flush=True)
return_code = process.wait()
if return_code != 0:
    raise RuntimeError(f"Single-example trace failed: exit code {return_code}")

with (OUTPUT_DIR / "comparison_results.jsonl").open(
    encoding="utf-8"
) as handle:
    result = json.loads(handle.readline())

attack = result["attacks"]["untargeted"]
print("\n" + "=" * 100)
print("BEST HOTFLIP PATH")
for change in attack["hotflip"]["changes"]:
    print(
        f"step={change['step']:2d} | "
        f"position={change['context_position']:3d} | "
        f"{change['original_token']!r} -> {change['replacement_token']!r} | "
        f"objective={change['objective_before']:.6f} -> "
        f"{change['objective_after']:.6f}"
    )

print("\nFINAL RESULT")
print("Baseline answer               :", result["baseline"]["answer"])
print("Attacked answer               :", attack["answer"])
print("Attacked correct              :", attack["gold_judge"]["correct"])
print(
    "Equivalent to baseline        :",
    attack["attacked_vs_baseline_judge"]["correct"],
)
print("Modified document retrieved   :", attack["modified_document_retrieved"])
print("Strict success                :", attack["strict_attack_success"])
print("Relaxed success               :", attack["relaxed_attack_success"])
print("Original document:\n", result["selected_document"]["text"])
print("Modified document:\n", attack["modified_document"])
```
