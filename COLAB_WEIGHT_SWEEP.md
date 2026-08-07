# Colab: tune beta and lambda in one cell

Upload these files to `/content` before running the cell:

- `baseline_results.csv`
- `wrong_targets.json` (or the equivalent generated target-answer JSON/PKL)

The cell selects one fixed set of 30 non-yes/no examples, balanced between
baseline-correct and baseline-incorrect answers. It runs beta values
`0.5, 1, 2` for untargeted attack and lambda values `1.5, 3, 6` for targeted
attack. Contriever, Qwen, and HotpotQA are loaded only once.

```python
import json
import shutil
import subprocess
import sys
from pathlib import Path

NUM_EXAMPLES = 30
PROJECT_DIR = Path("/content/HotFlip")
OUTPUT_DIR = PROJECT_DIR / "outputs" / "weight_sweep_30"

subprocess.run(["nvidia-smi"], check=False)

import torch

if not torch.cuda.is_available():
    raise RuntimeError("Chưa bật GPU trong Runtime > Change runtime type.")
print("GPU:", torch.cuda.get_device_name(0))

if (PROJECT_DIR / ".git").exists():
    subprocess.run(
        ["git", "-C", str(PROJECT_DIR), "pull", "--ff-only"], check=True
    )
else:
    subprocess.run(
        [
            "git", "clone", "https://github.com/HoangMaizzz/HotFlip.git",
            str(PROJECT_DIR),
        ],
        check=True,
    )

subprocess.run(
    ["git", "-C", str(PROJECT_DIR), "log", "-1", "--oneline"], check=True
)
subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "-q", "-r",
        str(PROJECT_DIR / "requirements-colab.txt"),
    ],
    check=True,
)

baseline_candidates = [
    Path("/content/baseline_results.csv"),
    PROJECT_DIR / "outputs" / "colab_baseline" / "baseline_results.csv",
]
target_candidates = [
    Path("/content/wrong_targets.json"),
    Path("/content/hotpotqa_answers_300.pkl"),
    PROJECT_DIR / "outputs" / "generated_targets" / "wrong_targets.json",
]

baseline_csv = next((path for path in baseline_candidates if path.exists()), None)
target_file = next((path for path in target_candidates if path.exists()), None)
if baseline_csv is None:
    raise FileNotFoundError("Hãy upload baseline_results.csv vào /content.")
if target_file is None:
    raise FileNotFoundError("Hãy upload wrong_targets.json vào /content.")

command = [
    sys.executable, "-u", "-m", "hotflip_rag.weight_sweep",
    "--baseline-results", str(baseline_csv),
    "--target-answer-file", str(target_file),
    "--num-examples", str(NUM_EXAMPLES),
    "--untargeted-weights", "0.5", "1", "2",
    "--targeted-weights", "1.5", "3", "6",
    "--top-k", "3",
    "--max-token-changes", "10",
    "--beam-width", "5",
    "--hotflip-top-k", "10",
    "--candidates-per-state", "10",
    "--candidate-vocab-size", "30000",
    "--generator-model", "Qwen/Qwen2.5-14B-Instruct",
    "--load-in-4bit",
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
    raise RuntimeError(f"Weight sweep failed with exit code {return_code}")

import pandas as pd
from IPython.display import display

summary = pd.read_csv(OUTPUT_DIR / "weight_sweep_summary.csv")
display(summary)

with (OUTPUT_DIR / "recommended_weights.json").open(encoding="utf-8") as file:
    recommendations = json.load(file)
print("\nRECOMMENDED PILOT WEIGHTS")
print(json.dumps(recommendations, ensure_ascii=False, indent=2))

archive = shutil.make_archive(str(OUTPUT_DIR), "zip", root_dir=OUTPUT_DIR)
print("\nZIP:", archive)

from google.colab import files
files.download(archive)
```
