"""Train Qwen3.5-0.8B-Base on Countdown directly with GRPO."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from rewards import (
    answer_reward,
    direct_answer_reward,
    direct_format_reward,
    format_reward,
    proximity_reward,
    trace_reward,
)


MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
DATASET_ID = "Jiayi-Pan/Countdown-Tasks-3to4"

PROMPT = """Use the numbers {numbers} to make exactly {target}.

Rules:
- Use every supplied number exactly once.
- You may only use +, -, *, /, and parentheses.
- Do not concatenate numbers.
- Division may create fractions, but the final value must equal the target exactly.

Reason concisely: try simple addition and subtraction first, do not narrate every
failed combination, and stop as soon as you find a valid expression. Keep the
thinking section under 100 words. After closing it, respond with only:
<answer>one arithmetic expression</answer>"""

DIRECT_PROMPT = """Use each number in {numbers} exactly once to make {target}.
Use only +, -, *, /, and parentheses. Do not concatenate numbers or add new ones.
Respond with only <answer>one arithmetic expression</answer>. No explanation."""


def make_prompt(example: dict) -> dict:
    text = PROMPT.format(numbers=example["nums"], target=example["target"])
    return {"prompt": [{"role": "user", "content": text}]}


def prepare_dataset(seed: int, train_size: int, eval_size: int) -> tuple[Dataset, Dataset]:
    dataset = load_dataset(DATASET_ID, split="train")
    subset_size = min(len(dataset), train_size + eval_size)
    dataset = dataset.shuffle(seed=seed).select(range(subset_size))
    split = dataset.train_test_split(test_size=eval_size, seed=seed)
    split = split.map(make_prompt, desc="Formatting Countdown prompts")
    return split["train"], split["test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument(
        "--adapter",
        type=Path,
        help="Initialize from a saved LoRA adapter with a fresh optimizer/scheduler phase.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/qwen35-countdown"))
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--train-size", type=int, default=50_000)
    parser.add_argument("--eval-size", type=int, default=512)
    parser.add_argument(
        "--arity",
        type=int,
        choices=(3, 4),
        help="Optionally train only on tasks with this many numbers; hold-out split stays unchanged.",
    )
    parser.add_argument(
        "--proximity-reward",
        action="store_true",
        help="Add a small distance-to-target reward for legal all-number expressions.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Train for a concise answer-only completion with thinking disabled.",
    )
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=3,
        help="How many recent checkpoints to retain; 0 keeps every checkpoint.",
    )
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument(
        "--log-completions",
        action="store_true",
        help="Print sampled completions during training (very verbose).",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        help="Resume optimizer, scheduler, RNG, and adapter state from a checkpoint directory.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow an extremely slow CPU run (useful only for a one-step smoke test).",
    )
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    if not torch.cuda.is_available() and not cli.allow_cpu:
        raise SystemExit(
            "CUDA is not available. GRPO generation/training on CPU is impractical. "
            "Fix GPU access, or pass --allow-cpu only for a tiny smoke test."
        )
    if cli.batch_size % cli.generations != 0:
        raise SystemExit("--batch-size must be divisible by --generations")
    if cli.save_total_limit < 0:
        raise SystemExit("--save-total-limit must be nonnegative")

    train_dataset, eval_dataset = prepare_dataset(cli.seed, cli.train_size, cli.eval_size)
    if cli.direct:
        def direct_prompt(row: dict) -> dict:
            return {
                "prompt": [
                    {
                        "role": "user",
                        "content": DIRECT_PROMPT.format(numbers=row["nums"], target=row["target"]),
                    }
                ]
            }

        train_dataset = train_dataset.map(direct_prompt, desc="Formatting direct prompts")
        eval_dataset = eval_dataset.map(direct_prompt, desc="Formatting direct eval prompts")
    if cli.arity is not None:
        train_dataset = train_dataset.filter(
            lambda row: len(row["nums"]) == cli.arity,
            desc=f"Selecting {cli.arity}-number training tasks",
        )
    dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float32"
    tokenizer = AutoTokenizer.from_pretrained(cli.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    training_args = GRPOConfig(
        output_dir=str(cli.output_dir),
        max_steps=cli.steps,
        learning_rate=cli.learning_rate,
        warmup_steps=max(1, cli.steps // 20),
        lr_scheduler_type="cosine",
        per_device_train_batch_size=cli.batch_size,
        gradient_accumulation_steps=cli.gradient_accumulation_steps,
        num_generations=cli.generations,
        max_completion_length=cli.max_completion_length,
        temperature=1.0,
        top_p=0.95,
        chat_template_kwargs={"enable_thinking": not cli.direct},
        beta=0.0,
        loss_type="dapo",
        scale_rewards="group",
        bf16=dtype == "bfloat16",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        model_init_kwargs=(
            {"dtype": dtype, "attn_implementation": "sdpa"}
            if cli.adapter is None
            else None
        ),
        logging_steps=cli.logging_steps,
        logging_first_step=True,
        log_completions=cli.log_completions,
        num_completions_to_print=4,
        report_to="tensorboard",
        save_strategy="steps",
        save_steps=cli.save_steps,
        save_total_limit=cli.save_total_limit or None,
        seed=cli.seed,
        data_seed=cli.seed,
    )
    lora = LoraConfig(
        task_type="CAUSAL_LM",
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "in_proj_qkv",
            "in_proj_z",
            "in_proj_a",
            "in_proj_b",
            "out_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = cli.model
    trainer_peft = lora
    if cli.adapter is not None:
        base = AutoModelForImageTextToText.from_pretrained(
            cli.model, dtype=dtype, attn_implementation="sdpa"
        )
        model = PeftModel.from_pretrained(base, cli.adapter, is_trainable=True)
        trainer_peft = None

    reward_funcs = (
        [direct_answer_reward, direct_format_reward]
        if cli.direct
        else [answer_reward, format_reward, trace_reward]
    )
    if cli.proximity_reward:
        reward_funcs.append(proximity_reward)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=trainer_peft,
    )
    trainer.train(
        resume_from_checkpoint=(
            str(cli.resume_from_checkpoint) if cli.resume_from_checkpoint else None
        )
    )
    trainer.save_model(str(cli.output_dir / "final"))


if __name__ == "__main__":
    main()
