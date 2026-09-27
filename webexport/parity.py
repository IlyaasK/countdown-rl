"""Differential test: webexport/reference.py versus transformers.

Run from the countdown_grpo directory:

    .venv/bin/python webexport/parity.py --tasks 4
    .venv/bin/python webexport/parity.py --tasks 4 --params webexport/out/merged_text_fp32.safetensors

Three levels of evidence, because "the numbers are close" is not enough:

1. weight check -- the exported text weights equal what PeftModel.merge_and_unload
   produces inside transformers (bit-level, since both sides are fp32);
2. per-step logits -- feeding one token at a time through the HF cache path
   (which uses the same recurrent delta-rule branch the browser will use);
3. batched prefill -- a single forward over the whole prompt, whose last-position
   logits must match the stepwise reference (this is what proves the chunked and
   recurrent formulations agree), plus greedy token equality.

Any failure is printed with the offending position and magnitudes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from webexport.reference import TextConfig, forward_step, greedy, load_params, prefill, rope_tables  # noqa: E402

BASE_MODEL_ID = "Qwen3.5-0.8B-Base"
TEXT_PREFIX = "model.language_model."
EVAL_TASKS = Path("runs/qwen35-countdown-100k-optimized/eval-vs-direct-four.jsonl")
ADAPTER = Path("runs/qwen35-countdown-100k-optimized/final")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--adapter", type=Path, default=ADAPTER)
    parser.add_argument("--params", type=Path, help="exported merged text weights (optional)")
    parser.add_argument("--config", type=Path, default=Path("webexport/out/config.json"))
    parser.add_argument("--tasks", type=int, default=4, help="number of held-out tasks to compare")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--logit-tolerance", type=float, default=2e-3)
    return parser.parse_args()


def build_hf_model(cli: argparse.Namespace):
    model = AutoModelForImageTextToText.from_pretrained(cli.base, dtype=torch.float32)
    if cli.adapter is not None:
        model = PeftModel.from_pretrained(model, str(cli.adapter)).merge_and_unload()
    return model.eval().to(cli.device)


def hf_text_params(model) -> dict[str, torch.Tensor]:
    return {
        key.split(TEXT_PREFIX, 1)[1]: value.detach().to(torch.float32)
        for key, value in model.state_dict().items()
        if TEXT_PREFIX in key
    }


def check_weights(params: dict[str, torch.Tensor], model) -> float:
    reference = hf_text_params(model)
    assert params.keys() == reference.keys(), "weight name mismatch"
    worst = 0.0
    offender = ""
    for key, tensor in params.items():
        difference = float((tensor.float().cpu() - reference[key].float().cpu()).abs().max())
        if difference > worst:
            worst, offender = difference, key
    print(f"weights: {len(params)} tensors, max abs diff {worst:.3e} ({offender})")
    return worst


def hf_stepwise_logits(model, tokens: list[int]) -> list[torch.Tensor]:
    """Feed one token at a time through the KV/state cache and collect last-position logits."""
    collected: list[torch.Tensor] = []
    with torch.inference_mode():
        output = model(input_ids=torch.tensor([tokens[:1]], device=model.device), use_cache=True)
        collected.append(output.logits[0, -1].float())
        cache = output.past_key_values
        for token in tokens[1:]:
            output = model(
                input_ids=torch.tensor([[token]], device=model.device),
                past_key_values=cache,
                use_cache=True,
            )
            cache = output.past_key_values
            collected.append(output.logits[0, -1].float())
    return collected


def hf_prefill_logits(model, tokens: list[int]) -> torch.Tensor:
    with torch.inference_mode():
        output = model(input_ids=torch.tensor([tokens], device=model.device), use_cache=False)
    return output.logits[0, -1].float()


def hf_greedy(model, tokens: list[int], max_new_tokens: int, stop_ids: set[int]) -> list[int]:
    produced: list[int] = []
    with torch.inference_mode():
        output = model(input_ids=torch.tensor([tokens], device=model.device), use_cache=True)
        cache = output.past_key_values
        for _ in range(max_new_tokens):
            token = int(output.logits[0, -1].argmax())
            produced.append(token)
            if token in stop_ids:
                break
            output = model(input_ids=torch.tensor([[token]], device=model.device), past_key_values=cache, use_cache=True)
            cache = output.past_key_values
    return produced


def main() -> int:
    cli = parse_args()
    print(f"device={cli.device}")

    tokenizer = AutoTokenizer.from_pretrained(str(cli.adapter))
    config = TextConfig.from_json(cli.config)
    tables = rope_tables(config, 512, cli.device)

    model = build_hf_model(cli)
    if cli.params:
        params = load_params(cli.params, device=cli.device)
        check_weights(params, model)
    else:
        params = {key: value.to(cli.device) for key, value in hf_text_params(model).items()}
        print(f"weights: using in-process HF weights ({len(params)} tensors)")

    tasks = [json.loads(line) for line in EVAL_TASKS.read_text().splitlines() if line.strip()]
    seen: set[tuple] = set()
    selected = []
    for task in tasks:
        key = (tuple(task["nums"]), task["target"])
        if key in seen:
            continue
        seen.add(key)
        selected.append(task)
        if len(selected) == cli.tasks:
            break

    stop_ids = {tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.convert_tokens_to_ids("<|endoftext|>")}
    failures = 0
    for index, task in enumerate(selected):
        text = _direct_prompt(tokenizer, task["nums"], task["target"])
        tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
        print(f"\n[task {index}] nums={task['nums']} target={task['target']} prompt_tokens={len(tokens)}")

        hf_steps = hf_stepwise_logits(model, tokens)
        mine_steps, _ = prefill(params, config, tokens, *tables)

        worst_diff, worst_pos, worst_token = 0.0, -1, -1
        argmax_mismatch = 0
        for position, (expected, actual) in enumerate(zip(hf_steps, mine_steps)):
            actual = actual.to(cli.device)
            difference = float((expected - actual).abs().max())
            if difference > worst_diff:
                worst_diff, worst_pos = difference, position
            if int(expected.argmax()) != int(actual.argmax()):
                argmax_mismatch += 1
                if worst_token < 0:
                    worst_token = position
        verdict = "OK " if worst_diff <= cli.logit_tolerance and argmax_mismatch == 0 else "FAIL"
        if verdict == "FAIL":
            failures += 1
        print(
            f"  stepwise logits: max|diff|={worst_diff:.3e} at pos {worst_pos}, "
            f"argmax mismatches={argmax_mismatch}" + (f" (first at {worst_token})" if worst_token >= 0 else "")
        )

        hf_last = hf_prefill_logits(model, tokens)
        mine_last = mine_steps[-1].to(cli.device)
        prefilled = float((hf_last - mine_last).abs().max())
        print(f"  batched prefill vs stepwise: max|diff|={prefilled:.3e} argmax_equal={int(hf_last.argmax()) == int(mine_last.argmax())}")
        if prefilled > cli.logit_tolerance or int(hf_last.argmax()) != int(mine_last.argmax()):
            failures += 1

        hf_tokens = hf_greedy(model, tokens, cli.max_new_tokens, stop_ids)
        my_tokens = greedy(params, config, tokens, cli.max_new_tokens, stop_ids, *tables)
        equal = hf_tokens == my_tokens
        print(f"  greedy tokens equal={equal}")
        print(f"    hf:  {tokenizer.decode(hf_tokens)!r}")
        print(f"    ref: {tokenizer.decode(my_tokens)!r}")
        if not equal:
            failures += 1

    print(f"\n{'PASS' if failures == 0 else 'FAIL'}: {failures} failing check(s) across {len(selected)} tasks")
    return 1 if failures else 0


def _direct_prompt(tokenizer, numbers: list[int], target: int) -> str:
    from train import DIRECT_PROMPT

    message = DIRECT_PROMPT.format(numbers=numbers, target=target)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": message}],
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=False,
    )


if __name__ == "__main__":
    raise SystemExit(main())
