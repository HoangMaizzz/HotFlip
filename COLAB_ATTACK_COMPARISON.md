# Colab test: baseline vs untargeted vs targeted

Cell dưới đây dùng lại `baseline_results.csv`, không chạy lại baseline. Nếu CSV
không còn ở `outputs/colab_baseline`, hãy upload nó vào `/content`.

```python
import json
import subprocess
import sys
from pathlib import Path

NUM_EXAMPLES = 5          # thử nhỏ trước; tăng sau khi cell chạy ổn
TOP_K = 3
MAX_TOKEN_CHANGES = 3

PROJECT_DIR = Path("/content/HotFlip")
BASELINE_IN_PROJECT = (
    PROJECT_DIR / "outputs" / "colab_baseline" / "baseline_results.csv"
)
UPLOADED_BASELINE = Path("/content/baseline_results.csv")
OUTPUT_DIR = PROJECT_DIR / "outputs" / "attack_comparison"

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
        "Không tìm thấy baseline_results.csv. Hãy upload file vào /content "
        "hoặc chạy baseline trước trong cùng runtime."
    )

target_file = PROJECT_DIR / "hotpotqa_answers_300.pkl"
if not target_file.exists():
    raise FileNotFoundError(f"Không tìm thấy target file: {target_file}")

command = [
    sys.executable,
    "-m", "hotflip_rag.compare_attacks",
    "--baseline-results", str(baseline_csv),
    "--target-answer-file", str(target_file),
    "--modes", "untargeted", "targeted",
    "--num-examples", str(NUM_EXAMPLES),
    "--top-k", str(TOP_K),
    "--max-token-changes", str(MAX_TOKEN_CHANGES),
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
    raise RuntimeError(f"Comparison failed with exit code {return_code}")

with (OUTPUT_DIR / "comparison_aggregate.json").open(encoding="utf-8") as file:
    aggregate = json.load(file)

print("\nAGGREGATE")
print(json.dumps(aggregate, ensure_ascii=False, indent=2))

import pandas as pd
from IPython.display import display

comparison = pd.read_csv(OUTPUT_DIR / "comparison_summary.csv")
display(comparison)
```

ASR được tính như sau:

- Untargeted: mẫu đủ điều kiện khi baseline được judge là đúng; thành công nếu
  đáp án sau attack được judge là sai so với gold.
- Targeted: mẫu đủ điều kiện khi baseline chưa khớp target; thành công nếu đáp
  án sau attack được judge là đúng so với target.
- Exact Match được nhận tự động; mọi đáp án không exact đều do Qwen judge.
