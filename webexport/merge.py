"""Merge the GRPO LoRA adapter into the base model and export text-only weights.

Run from the countdown_grpo directory:

    .venv/bin/python webexport/merge.py

Writes ``webexport/out/merged_text_fp32.safetensors`` (752M params, fp32, text
tower only -- the vision tower and the mtp head are never used for a text-only
prompt) plus a copy of the base ``config.json`` for the export tools.

The merge runs in fp32 on CPU: quantization error to int4 dwarfs the fp32-vs-bf16
merge difference, and no CUDA work is competing with anything else.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForImageTextToText

BASE_MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
TEXT_PREFIX = "model.language_model."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter",
        type=Path,
        default=Path("runs/qwen35-countdown-100k-optimized/final"),
        help="LoRA adapter directory produced by training",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("webexport/out"))
    parser.add_argument("--base", default=BASE_MODEL_ID)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    cli.output_dir.mkdir(parents=True, exist_ok=True)

    adapter_config = json.loads((cli.adapter / "adapter_config.json").read_text())
    print(f"adapter={cli.adapter} r={adapter_config['r']} alpha={adapter_config['lora_alpha']}")
    print(f"targets={len(adapter_config['target_modules'])} modules")

    base = AutoModelForImageTextToText.from_pretrained(cli.base, dtype=torch.float32)
    model = PeftModel.from_pretrained(base, str(cli.adapter))
    model = model.merge_and_unload()
    model.eval()

    state = model.state_dict()
    text = {
        key.split(TEXT_PREFIX, 1)[1]: value.detach().to(torch.float32).contiguous()
        for key, value in state.items()
        if TEXT_PREFIX in key
    }
    params = sum(tensor.numel() for tensor in text.values())
    print(f"exporting {len(text)} text tensors, {params / 1e6:.1f}M params")

    destination = cli.output_dir / "merged_text_fp32.safetensors"
    save_file(
        text,
        str(destination),
        metadata={
            "base_model": cli.base,
            "adapter": str(cli.adapter),
            "lora_r": str(adapter_config["r"]),
            "lora_alpha": str(adapter_config["lora_alpha"]),
            "note": "text tower only; vision and mtp excluded",
        },
    )
    AutoConfig.from_pretrained(cli.base).to_json_file(str(cli.output_dir / "config.json"))
    print(f"wrote {destination} ({destination.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
