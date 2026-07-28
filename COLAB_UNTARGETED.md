# Colab cell: retrieval-preserving untargeted HotFlip

Cell này chỉ chạy untargeted; không cần `wrong_targets.json`.

```python
import csv
import json
import subprocess
import sys
from pathlib import Path

NUM_EXAMPLES = 5
TOP_K = 3
MAX_TOKEN_CHANGES = 10
BEAM_WIDTH = 5
UNTARGETED_ANSWER_WEIGHT = 1.0
CANDIDATE_VOCAB_SIZE = 30000
HOTFLIP_TOP_K = 20
CANDIDATES_PER_STATE = 20

PROJECT_DIR = Path("/content/HotFlip")
BASELINE_IN_PROJECT = (
    PROJECT_DIR / "outputs" / "colab_baseline" / "baseline_results.csv"
)
UPLOADED_BASELINE = Path("/content/baseline_results.csv")
OUTPUT_DIR = PROJECT_DIR / "outputs" / "untargeted_comparison"

subprocess.run(["nvidia-smi"], check=False)

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
        sys.executable, "-m", "pip", "install", "-q",
        "-r", str(PROJECT_DIR / "requirements-colab.txt"),
    ],
    check=True,
)

if BASELINE_IN_PROJECT.exists():
    baseline_csv = BASELINE_IN_PROJECT
elif UPLOADED_BASELINE.exists():
    baseline_csv = UPLOADED_BASELINE
else:
    raise FileNotFoundError(
        "Không tìm thấy baseline_results.csv. "
        "Hãy chạy baseline trước hoặc upload vào /content."
    )

with baseline_csv.open(encoding="utf-8-sig", newline="") as file:
    baseline_rows = list(csv.DictReader(file))
if len(baseline_rows) < NUM_EXAMPLES:
    raise ValueError(
        f"Baseline chỉ có {len(baseline_rows)} dòng, "
        f"không đủ {NUM_EXAMPLES} mẫu."
    )

command = [
    sys.executable,
    "-m", "hotflip_rag.compare_attacks",
    "--baseline-results", str(baseline_csv),
    "--modes", "untargeted",
    "--num-examples", str(NUM_EXAMPLES),
    "--top-k", str(TOP_K),
    "--max-token-changes", str(MAX_TOKEN_CHANGES),
    "--search-strategy", "beam",
    "--beam-width", str(BEAM_WIDTH),
    "--untargeted-answer-weight", str(UNTARGETED_ANSWER_WEIGHT),
    "--candidate-vocab-size", str(CANDIDATE_VOCAB_SIZE),
    "--hotflip-top-k", str(HOTFLIP_TOP_K),
    "--candidates-per-state", str(CANDIDATES_PER_STATE),
    "--no-disallow-numeric-replacement",
    "--generator-model", "Qwen/Qwen2.5-14B-Instruct",
    "--load-in-4bit",
    "--output-dir", str(OUTPUT_DIR),
]

print("Command:")
print(" ".join(command))

process = subprocess.Popen(
    command,
    cwd=str(PROJECT_DIR),
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
for line in process.stdout:
    print(line, end="")
return_code = process.wait()
if return_code != 0:
    raise RuntimeError(
        f"Untargeted comparison failed with exit code {return_code}"
    )

with (OUTPUT_DIR / "comparison_aggregate.json").open(
    encoding="utf-8"
) as file:
    aggregate = json.load(file)

metrics = aggregate["untargeted"]
print("\n" + "=" * 100)
print(f"Baseline accuracy             : {aggregate['baseline_accuracy'] * 100:.2f}%")
print(f"Accuracy after untargeted     : {metrics['accuracy_after_attack'] * 100:.2f}%")
print(f"Accuracy drop                 : {metrics['accuracy_drop'] * 100:.2f} points")
print(
    f"ASR on baseline-correct       : "
    f"{metrics['asr_on_baseline_correct'] * 100:.2f}% "
    f"({metrics['asr_on_baseline_correct_successes']}/"
    f"{metrics['asr_on_baseline_correct_examples']})"
)
print(
    f"Modified-document retrieval   : "
    f"{metrics['modified_document_retrieval_rate'] * 100:.2f}%"
)

import pandas as pd
from IPython.display import display

comparison = pd.read_csv(OUTPUT_DIR / "comparison_summary.csv")
display(
    comparison[
        [
            "id",
            "question",
            "gold_answer",
            "baseline_answer",
            "baseline_correct",
            "attacked_answer",
            "attacked_correct",
            "attack_success",
            "attacked_document_title",
            "modified_document_retrieved",
            "original_document",
            "modified_document",
        ]
    ]
)
```
