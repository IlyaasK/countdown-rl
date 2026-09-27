"""Prepare the merged Countdown text weights for llama.cpp's GGUF converter.

Run from examples/countdown_grpo after ``webexport/merge.py``. This reuses the
already merged text tower, avoiding a second load and merge of the base model.
The resulting directory is an intermediate export artifact, not a new model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("webexport/out"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("runs/qwen35-countdown-100k-optimized/gguf-source"),
    )
    parser.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B-Base")
    cli = parser.parse_args()

    cli.output.mkdir(parents=True, exist_ok=True)
    config = json.loads((cli.source / "config.json").read_text())
    if config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]:
        raise ValueError("Expected a Qwen3.5 conditional-generation config")

    source = cli.source / "merged_text_fp32.safetensors"
    tensors = {}
    with safe_open(source, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            tensors[f"model.language_model.{name}"] = handle.get_tensor(name).to(torch.float16)
    if "model.language_model.embed_tokens.weight" not in tensors:
        raise ValueError("Merged text weights are missing the embedding table")

    save_file(tensors, str(cli.output / "model.safetensors"))
    (cli.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    AutoTokenizer.from_pretrained(cli.tokenizer, local_files_only=True).save_pretrained(cli.output)
    print(f"Prepared {len(tensors)} tensors in {cli.output}")


if __name__ == "__main__":
    main()
