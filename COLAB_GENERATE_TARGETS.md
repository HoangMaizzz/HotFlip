# Colab cell: sinh target answer sai riêng

Chạy cell này sau baseline và trước attack comparison.

```python
import json
import subprocess
import sys
from pathlib import Path

NUM_TARGETS = 300
CANDIDATE_VOCAB_SIZE = 30000
MAX_ATTEMPTS = 5

PROJECT_DIR = Path("/content/HotFlip")
BASELINE_CSV = (
    PROJECT_DIR / "outputs" / "colab_baseline" / "baseline_results.csv"
)
UPLOADED_BASELINE = Path("/content/baseline_results.csv")
OUTPUT_DIR = PROJECT_DIR / "outputs" / "generated_targets"

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

if not BASELINE_CSV.exists():
    if UPLOADED_BASELINE.exists():
        BASELINE_CSV = UPLOADED_BASELINE
    else:
        raise FileNotFoundError(
            "Không tìm thấy baseline_results.csv. "
            "Hãy chạy baseline trước hoặc upload file vào /content."
        )

command = [
    sys.executable,
    "-m", "hotflip_rag.generate_targets",
    "--baseline-results", str(BASELINE_CSV),
    "--num-examples", str(NUM_TARGETS),
    "--candidate-vocab-size", str(CANDIDATE_VOCAB_SIZE),
    "--max-attempts", str(MAX_ATTEMPTS),
    "--generator-model", "Qwen/Qwen2.5-14B-Instruct",
    "--load-in-4bit",
    "--require-all",
    "--output-dir", str(OUTPUT_DIR),
]

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
        f"Target generation failed with exit code {return_code}. "
        "Các target hợp lệ và failure details vẫn được lưu trong output directory."
    )

with (OUTPUT_DIR / "wrong_target_aggregate.json").open(
    encoding="utf-8"
) as file:
    aggregate = json.load(file)

print(json.dumps(aggregate, ensure_ascii=False, indent=2))

import pandas as pd
from IPython.display import display

targets = pd.read_csv(OUTPUT_DIR / "wrong_target_details.csv")
display(
    targets[
        [
            "question",
            "gold_answer",
            "baseline_answer",
            "wrong_target_answer",
            "retriever_token_ids",
            "retriever_tokens",
            "attempt",
        ]
    ]
)
```
