---
license: apache-2.0
base_model: Qwen/Qwen3.5-0.8B-Base
tags:
- gguf
- reinforcement-learning
- grpo
- countdown
---

# Qwen3.5-0.8B-Base Countdown GRPO · browser GGUF

This is a text-only, Q4_K_M GGUF export of a Qwen3.5-0.8B-Base checkpoint after 100,000 GRPO training steps on Jiayi Pan's Countdown-Tasks-3to4 arithmetic puzzles. No supervised fine-tuning was used. The vision tower and MTP weights are excluded from this browser export. The GGUF was converted with llama.cpp `--no-nextn` and quantized to Q4_K_M.

The model was trained to produce a single `<answer>arithmetic expression</answer>` using each supplied number once. It is experimental and often wrong. On the held-out evaluation used for this project, the final checkpoint solved **56/252 four-number puzzles in one attempt** and **0/260 three-number puzzles**. The browser demo defaults to four numbers and verifies every generated expression using exact rational arithmetic. It does not replace a failed answer with a solver-generated one.

Source base model: [Qwen/Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base). Dataset: [Jiayi-Pan/Countdown-Tasks-3to4](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4). Demo: [ilya.as/countdown](https://ilya.as/countdown/).

Single-file download: `countdown-Q4_K_M.gguf` (SHA-256 `fd5bae67451d67f6513b7563aa388561097bf347a774764404b4698b0c474cfe`). The browser demo loads the equivalent five-part `countdown-Q4_K_M-00001-of-00005.gguf` split, which wllama downloads in parallel and reassembles in memory.
