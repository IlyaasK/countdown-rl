"""Sample a trained Countdown adapter until it produces a verified final answer."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer

from oracle import find_solution
from rewards import direct_answer_expression, evaluate_expression, final_answer_expression
from train import DIRECT_PROMPT, MODEL_ID, PROMPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("numbers", nargs="+", type=int, help="Three or four input numbers")
    parser.add_argument("--target", required=True, type=int)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--adapter", type=Path, help="Saved LoRA adapter; defaults depend on --direct")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--show-thinking", action="store_true")
    parser.add_argument("--direct", action="store_true", help="Use the direct-answer prompt and parser.")
    parser.add_argument(
        "--fallback-oracle",
        action="store_true",
        help="If sampling fails, solve exhaustively and clearly label the non-model answer.",
    )
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    cli = parse_args()
    if len(cli.numbers) not in (3, 4):
        raise SystemExit("Provide exactly three or four numbers")
    if cli.samples < 1 or cli.batch_size < 1 or cli.max_new_tokens < 1:
        raise SystemExit("--samples, --batch-size, and --max-new-tokens must be positive")
    if not torch.cuda.is_available():
        if cli.fallback_oracle:
            expression = find_solution(cli.numbers, cli.target)
            if expression is not None and evaluate_expression(expression, cli.numbers) == cli.target:
                print(f"Oracle only (CUDA unavailable): {expression} = {cli.target}")
                return
        raise SystemExit("CUDA is required for model inference")

    adapter = cli.adapter or Path(
        "runs/qwen35-countdown-direct-four/final"
        if cli.direct
        else "runs/qwen35-countdown-phase3/final"
    )

    torch.manual_seed(cli.seed)
    torch.cuda.manual_seed_all(cli.seed)
    tokenizer = AutoTokenizer.from_pretrained(cli.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForImageTextToText.from_pretrained(
        cli.model, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda")
    model = PeftModel.from_pretrained(base, adapter).eval()
    prompt_text = DIRECT_PROMPT if cli.direct else PROMPT
    prompt = [{"role": "user", "content": prompt_text.format(numbers=cli.numbers, target=cli.target)}]

    for start in range(0, cli.samples, cli.batch_size):
        count = min(cli.batch_size, cli.samples - start)
        encoded = tokenizer.apply_chat_template(
            [prompt] * count,
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
        completions = tokenizer.batch_decode(
            generated[:, encoded["input_ids"].shape[1] :], skip_special_tokens=False
        )
        for index, completion in enumerate(completions, start=start + 1):
            expression = (
                direct_answer_expression(completion)
                if cli.direct
                else final_answer_expression(completion)
            )
            if expression is None:
                continue
            try:
                correct = evaluate_expression(expression, cli.numbers) == cli.target
            except (SyntaxError, TypeError, ValueError, ZeroDivisionError):
                continue
            if correct:
                print(f"Verified on sample {index}/{cli.samples}: {expression} = {cli.target}")
                if cli.show_thinking:
                    print(completion)
                return
        print(f"Checked {start + count}/{cli.samples} samples", flush=True)

    if cli.fallback_oracle:
        expression = find_solution(cli.numbers, cli.target)
        if expression is not None and evaluate_expression(expression, cli.numbers) == cli.target:
            print(f"Oracle fallback (model failed {cli.samples} samples): {expression} = {cli.target}")
            return
    raise SystemExit(f"No verified model answer in {cli.samples} samples")


if __name__ == "__main__":
    main()
