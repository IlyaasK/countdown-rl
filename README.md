# Countdown GRPO with Qwen3.5-0.8B-Base

This experiment skips supervised fine-tuning and applies GRPO directly to
`Qwen/Qwen3.5-0.8B-Base` on
`Jiayi-Pan/Countdown-Tasks-3to4`.

The main reward is deliberately strict. A completion receives `1` only when a
closed thinking block ends with a tagged answer whose expression:

- evaluates exactly to the target (using rational arithmetic),
- uses every supplied number exactly once, and
- contains only integer literals, parentheses, and `+`, `-`, `*`, `/`.

An exact equation anywhere in a reasoning trace earns a smaller `0.1` bootstrap
reward, and a completed answer envelope earns `0.2` even when its math is wrong.
The expression is independently parsed and evaluated. Thinking mode is
explicitly enabled in the official Qwen chat template.
Without the operand checks, a policy can maximize a naive "hit the target"
reward by simply copying the target from the prompt.

## Setup

From this directory:

```bash
uv venv .venv --python 3.12 --system-site-packages
uv pip install --python .venv/bin/python -r requirements.txt
```

The model is about 1.75 GB. The dataset has 490,364 rows and is only a few MB
compressed. Both are downloaded automatically by the training script, or can
be fetched in advance:

```bash
.venv/bin/hf download Qwen/Qwen3.5-0.8B-Base
.venv/bin/hf download Jiayi-Pan/Countdown-Tasks-3to4 --repo-type dataset
```

## Verify the reward

```bash
.venv/bin/pytest -q test_rewards.py
```

## Train

Run from this directory so outputs land under `runs/`:

```bash
.venv/bin/python train.py --steps 300
```

The trainer checkpoints every 25 steps. Resume an interrupted run with:

```bash
.venv/bin/python train.py --steps 300 \
  --resume-from-checkpoint runs/qwen35-countdown/checkpoint-100
```

The default run uses a reproducible 50,000-row training subset, LoRA, four
completions per GRPO group, 512 completion tokens, and no reference-model KL
term. TensorBoard metrics are written beneath `runs/qwen35-countdown/`.
Pass `--log-completions` to print and save sampled completions (very verbose).
Use `--train-size` to change the subset size.

To watch the reward curve in another terminal:

```bash
.venv/bin/tensorboard --logdir runs/qwen35-countdown
```

Compare the base model and trained adapter on identical held-out tasks:

```bash
.venv/bin/python evaluate.py
```

This writes each completion and score to `runs/qwen35-countdown/eval.jsonl` and
prints final-answer correctness, trace correctness, format, truncation, and
mean-length summaries for both policies. Final-answer correctness is the primary
metric.

If held-out final-answer correctness improves, continue from the adapter. The
optimizer and cosine schedule are intentionally reinitialized for the new phase:

```bash
.venv/bin/python train.py --steps 300 \
  --adapter runs/qwen35-countdown/final \
  --output-dir runs/qwen35-countdown-extend
```

Compare the two adapters using the same held-out prompts and sampling seed:

```bash
.venv/bin/python evaluate.py \
  --reference-adapter runs/qwen35-countdown/final \
  --adapter runs/qwen35-countdown-extend/final \
  --candidate-label phase-2 \
  --output runs/qwen35-countdown-extend/eval.jsonl
```

The first 300-step phase improved valid final answers from 1/64 (base) to
4/64 (adapter) on 16 held-out tasks with four samples each. Correct equations
*anywhere in the trace* rose from 11/64 to 22/64. This is evidence of a useful
learning signal, not yet a reliable solver; the follow-up phase tests whether
rewarding only usable final answers closes that gap.

The 100-step follow-up phase used the strict final-answer reward and smaller
two-generation groups because other GPU jobs were active. On 32 held-out tasks
with four samples each and a 384-token cap, phase 1 solved 5/128 and phase 2
solved 13/128. The corresponding truncation rates were 118/128 and 108/128.
These numbers are encouraging but small; use a larger held-out set before
claiming robust mathematical reasoning.

A third 100-step phase used four completions per group and a 512-token cap.
On the same 32 held-out tasks at 512 tokens, phase 2 solved 12/128 and phase 3
solved 23/128. Almost all of that gain was on three-number tasks: 10/64 to
22/64; four-number tasks changed from 2/64 to 1/64. This is useful as an RL
demonstration, not a reliable four-number solver.

For a harder-task experiment, `--arity 4` trains on only four-number tasks
while preserving the original held-out split. `--proximity-reward` optionally
adds at most 0.15 reward for a legal expression using every number, scaled by
its exact arithmetic distance from the target. The 1.0 exact-final-answer
reward remains dominant. Use this only as an explicitly labeled shaped-reward
experiment; compare it with the prior adapter on held-out tasks.

On a larger 64-task held-out comparison with four samples per task and a
512-token cap, phase 3 solved 46/256 final answers and the four-number-focused
phase solved 52/256. The four-number subset changed from 2/116 to 4/116;
the three-number subset changed from 44/140 to 48/140. These small differences
are inconclusive, and four-number reliability remains poor. Further training
should be gated by a larger held-out gain, not the shaped training reward.
With four attempts per task, the focused adapter found at least one valid
answer on 22/35 three-number tasks and 4/29 four-number tasks.

To try the saved phase-3 adapter interactively, sample and independently
verify each proposed answer:

```bash
.venv/bin/python solve.py 82 8 25 --target 99 --samples 16
```

Use `--adapter runs/qwen35-countdown-phase4-four-numbers/final` to try the
focused adapter. The command prints an answer only if it evaluates exactly to
the target and uses each number once; otherwise it exits with a clear failure.
The examples above were already seen in held-out evaluation and are smoke
tests, not representative success-rate measurements.

## Concise-answer experiment

The original thinking format often loops until the completion limit. Switching
to a non-thinking prefix at inference time alone did **not** work: the phase-4
adapter scored 0/256 valid answers with the direct prompt. A separate 150-step
direct-format GRPO phase learned to emit one `<answer>` element in 100% of
held-out samples, with no truncation and a mean of 20 tokens. It solved 27/256
tasks per sample (25/140 three-number, 2/116 four-number). A further 500-step
four-number-focused direct phase reached 42/256 (37/140 and 5/116). This is a
meaningful format and efficiency improvement, but not a reliable standalone
four-number model.

The exact [oracle](oracle.py) found solutions for all 512 tasks in the held-out
split (260 three-number, 252 four-number). The verifier never uses the oracle
to credit model outputs. For a dependable CLI, request an explicitly labeled
fallback after model sampling fails:

```bash
.venv/bin/python solve.py 53 4 44 28 --target 63 \
  --adapter runs/qwen35-countdown-direct-four/final \
  --direct --samples 16 --fallback-oracle
```

The fallback makes the *tool* reliable for solvable three- and four-number
tasks; its answers must not be counted as model successes. Run `evaluate.py
--direct` without fallback to measure the model itself.

Useful first adjustments:

- The defaults fit an otherwise idle RTX 5070 with 12 GB VRAM. Concurrent GPU
  jobs may require `--generations 2 --batch-size 2
  --gradient-accumulation-steps 2 --max-completion-length 384`.
- If memory is tight, reduce `--max-completion-length`; GRPO still needs at
  least two completions per prompt.
- If correct reward remains exactly zero after 50-100 steps, inspect the
  printed completions. A fully sparse reward cannot learn until the base policy
  produces at least occasional successes.

CUDA is required for a practical run. `--allow-cpu --steps 1` exists only for
integration debugging and will be extremely slow.

## 100,000-step checkpoint and browser demo

The final direct-answer run is `runs/qwen35-countdown-100k-optimized/final`.
On a 512-task held-out evaluation (one sample per task), it solved 56/252
four-number tasks and 0/260 three-number tasks. This is a measurable
four-number improvement over the earlier direct-four adapter (19/252), but
is not a reliable general Countdown solver. The public demo uses this exact
checkpoint, without an oracle fallback: [ilya.as/countdown](https://ilya.as/countdown/).

The browser export merges the final LoRA into the base text tower, converts
to GGUF with llama.cpp, and quantizes to Q4_K_M. The [model repository](https://huggingface.co/IlyaasK/countdown-qwen3.5-0.8b-grpo)
contains the 505 MiB GGUF and five parallel-download shards. The [demo source](demo/)
is a static React/Vite app using wllama WebGPU; it verifies generated arithmetic
with exact rational numbers and never substitutes a solver answer. To build it:

```bash
cd demo
npm ci
npm test
npm run build
```

The build output is static and is hosted on GitHub Pages at `/countdown/`.
