"""Compare two policies on identical held-out Countdown tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer

from rewards import (
    answer_reward,
    direct_answer_expression,
    direct_answer_reward,
    final_answer_expression,
    trace_reward,
)
from train import DIRECT_PROMPT, MODEL_ID, prepare_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--adapter", type=Path, default=Path("runs/qwen35-countdown/final"))
    parser.add_argument(
        "--reference-adapter",
        type=Path,
        help="Compare against this earlier adapter instead of the base model.",
    )
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--output", type=Path, default=Path("runs/qwen35-countdown/eval.jsonl"))
    parser.add_argument("--tasks", type=int, default=16)
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Test a concise answer-only prompt with thinking disabled.",
    )
    return parser.parse_args()


@torch.inference_mode()
def evaluate_policy(model, tokenizer, rows: list[dict], cli: argparse.Namespace, label: str) -> list[dict]:
    examples = [row for row in rows for _ in range(cli.rollouts)]
    results: list[dict] = []
    torch.manual_seed(cli.seed)
    torch.cuda.manual_seed_all(cli.seed)

    for start in range(0, len(examples), cli.batch_size):
        batch = examples[start : start + cli.batch_size]
        prompts = (
            [
                [{"role": "user", "content": DIRECT_PROMPT.format(numbers=row["nums"], target=row["target"])}]
                for row in batch
            ]
            if cli.direct
            else [row["prompt"] for row in batch]
        )
        encoded = tokenizer.apply_chat_template(
            prompts,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=not cli.direct,
            padding=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)
        generated = model.generate(
            **encoded,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            max_new_tokens=cli.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        completion_ids = generated[:, encoded["input_ids"].shape[1] :]
        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=False)
        numbers = [row["nums"] for row in batch]
        targets = [row["target"] for row in batch]
        reward_fn = direct_answer_reward if cli.direct else answer_reward
        correct = reward_fn(completions, nums=numbers, target=targets)
        trace = trace_reward(completions, nums=numbers, target=targets)

        for row, text, is_correct, trace_score, token_ids in zip(
            batch, completions, correct, trace, completion_ids, strict=True
        ):
            eos_positions = (token_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            token_count = int(eos_positions[0].item() + 1) if len(eos_positions) else len(token_ids)
            results.append(
                {
                    "policy": label,
                    "target": row["target"],
                    "nums": row["nums"],
                    "completion": text,
                    "correct": is_correct,
                    "trace_correct": trace_score > 0,
                    "formatted": (
                        direct_answer_expression(text) if cli.direct else final_answer_expression(text)
                    ) is not None,
                    "completion_tokens": token_count,
                    "truncated": token_count >= cli.max_new_tokens,
                }
            )
        completed = min(start + cli.batch_size, len(examples))
        if completed % (8 * cli.batch_size) == 0 or completed == len(examples):
            print(f"{label}: generated {completed}/{len(examples)} completions", flush=True)
    return results


def summarize(results: list[dict], label: str) -> None:
    rows = [row for row in results if row["policy"] == label]
    count = len(rows)
    correct = sum(row["correct"] for row in rows) / count
    formatted = sum(row["formatted"] for row in rows) / count
    trace = sum(row["trace_correct"] for row in rows) / count
    truncated = sum(row["truncated"] for row in rows) / count
    mean_tokens = sum(row["completion_tokens"] for row in rows) / count
    print(
        f"{label}: final_correct={correct:.3f} trace_correct={trace:.3f} "
        f"formatted={formatted:.3f} "
        f"truncated={truncated:.3f} mean_tokens={mean_tokens:.1f} n={count}"
    )
    for arity in sorted({len(row["nums"]) for row in rows}):
        subset = [row for row in rows if len(row["nums"]) == arity]
        solved = int(sum(row["correct"] for row in subset))
        print(f"  {arity} numbers: {solved}/{len(subset)} valid final answers")


def main() -> None:
    cli = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for practical evaluation")
    if cli.tasks < 1 or cli.rollouts < 1:
        raise SystemExit("--tasks and --rollouts must be positive")

    _, eval_dataset = prepare_dataset(seed=42, train_size=50_000, eval_size=512)
    rows = [eval_dataset[index] for index in range(min(cli.tasks, len(eval_dataset)))]
    tokenizer = AutoTokenizer.from_pretrained(cli.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForImageTextToText.from_pretrained(
        cli.model, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda")
    model.eval()

    if cli.reference_adapter:
        model = PeftModel.from_pretrained(
            model, cli.reference_adapter, adapter_name="reference"
        ).eval()
        model.load_adapter(cli.adapter, adapter_name="candidate")
        model.set_adapter("reference")
        reference_label = "reference"
    else:
        reference_label = "base"

    results = evaluate_policy(model, tokenizer, rows, cli, reference_label)
    if cli.reference_adapter:
        model.set_adapter("candidate")
    else:
        model = PeftModel.from_pretrained(model, cli.adapter).eval()
    results.extend(evaluate_policy(model, tokenizer, rows, cli, cli.candidate_label))

    cli.output.parent.mkdir(parents=True, exist_ok=True)
    with cli.output.open("w") as handle:
        for row in results:
            handle.write(json.dumps(row) + "\n")
    summarize(results, reference_label)
    summarize(results, cli.candidate_label)
    print(f"wrote {cli.output}")


if __name__ == "__main__":
    main()
