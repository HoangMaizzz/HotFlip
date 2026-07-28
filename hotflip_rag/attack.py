from __future__ import annotations

import argparse

import torch

from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HotFlip Gold Context with Contriever before RAG retrieval"
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument("--num-examples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retriever-model", default="facebook/contriever")
    parser.add_argument("--generator-model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--generator-dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--max-context-tokens", type=int, default=512)
    parser.add_argument("--max-generator-input-tokens", type=int, default=3072)
    parser.add_argument("--attack-mode", choices=["untargeted", "targeted"], default="untargeted")
    parser.add_argument("--target-answer")
    parser.add_argument("--target-answer-file")
    parser.add_argument("--target-weight", type=float, default=1.0)
    parser.add_argument("--search-strategy", choices=["greedy", "beam"], default="greedy")
    parser.add_argument("--max-token-changes", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--hotflip-top-k", type=int, default=20)
    parser.add_argument("--candidates-per-state", type=int, default=20)
    parser.add_argument("--candidate-policy", choices=["full_vocab", "tokenizer_safe"], default="tokenizer_safe")
    parser.add_argument("--candidate-vocab-size", type=int, default=5000)
    parser.add_argument("--score-chunk-size", type=int, default=2048)
    parser.add_argument("--min-objective-improvement", type=float, default=0.0)
    parser.add_argument("--exact-rerank", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preserve-token-class", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preserve-leading-space", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disallow-punctuation-replacement", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disallow-numeric-replacement", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--only-clean-correct", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clean-f1-threshold", type=float, default=0.5)
    parser.add_argument("--success-f1-threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", default="outputs/hotflip_untargeted")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.attack_mode == "targeted" and not (args.target_answer or args.target_answer_file):
        raise SystemExit("Targeted mode requires --target-answer or --target-answer-file")
    run_pipeline(args)


if __name__ == "__main__":
    main()
