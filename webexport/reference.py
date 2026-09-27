"""State-explicit Qwen3.5 text decoder.

This is the numerical contract for the browser export: it reproduces the
``transformers`` ``Qwen3_5`` text path *one token at a time* with explicit
conv / recurrent / KV state, so the identical maths can be ported to WebGPU.

Deliberate design choices, all of which the browser engine mirrors:

* single-token steps only -- the gated delta rule's recurrent form is
  mathematically identical to transformers' chunked form, but it is the only
  path a KV/state-cached browser decode needs, so there is exactly one code
  path to keep correct;
* fp32 activations everywhere;
* state layouts sized for fixed GPU buffers:
    conv state  [conv_dim, 3]                  raw (pre-activation) qkv inputs
    rec state   [num_v_heads, k_dim, v_dim]    fp32
    kv state    [num_kv_heads, T, head_dim]

``parity.py`` proves this matches transformers, per step and per greedy token.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

TEXT_PREFIX = "model.language_model."


@dataclass(frozen=True)
class TextConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rope_theta: float
    partial_rotary_factor: float
    rms_norm_eps: float
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    conv_kernel_dim: int
    layer_types: tuple[str, ...]

    @property
    def key_dim(self) -> int:
        return self.linear_key_head_dim * self.linear_num_key_heads

    @property
    def value_dim(self) -> int:
        return self.linear_value_head_dim * self.linear_num_value_heads

    @property
    def conv_dim(self) -> int:
        return self.key_dim * 2 + self.value_dim

    @property
    def rope_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def linear_layers(self) -> tuple[int, ...]:
        return tuple(i for i, t in enumerate(self.layer_types) if t == "linear_attention")

    @property
    def full_layers(self) -> tuple[int, ...]:
        return tuple(i for i, t in enumerate(self.layer_types) if t == "full_attention")

    @property
    def kv_groups(self) -> int:
        assert self.num_attention_heads % self.num_key_value_heads == 0
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_json(cls, path: str | Path) -> "TextConfig":
        raw = json.loads(Path(path).read_text())["text_config"]
        rope = raw["rope_parameters"]
        return cls(
            vocab_size=raw["vocab_size"],
            hidden_size=raw["hidden_size"],
            intermediate_size=raw["intermediate_size"],
            num_hidden_layers=raw["num_hidden_layers"],
            num_attention_heads=raw["num_attention_heads"],
            num_key_value_heads=raw["num_key_value_heads"],
            head_dim=raw["head_dim"],
            rope_theta=rope["rope_theta"],
            partial_rotary_factor=rope.get("partial_rotary_factor", 1.0),
            rms_norm_eps=raw["rms_norm_eps"],
            linear_key_head_dim=raw["linear_key_head_dim"],
            linear_value_head_dim=raw["linear_value_head_dim"],
            linear_num_key_heads=raw["linear_num_key_heads"],
            linear_num_value_heads=raw["linear_num_value_heads"],
            conv_kernel_dim=raw["linear_conv_kernel_dim"],
            layer_types=tuple(raw["layer_types"]),
        )


def load_params(path: str | Path, device: str = "cpu", dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
    """Load the text half of a Qwen3.5 checkpoint as a flat name->tensor dict.

    Keys are transformers names with the ``model.language_model.`` prefix stripped.
    Vision and mtp tensors are dropped: the browser model is text-only.
    """
    from safetensors import safe_open

    params: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt") as handle:
        for key in handle.keys():
            if TEXT_PREFIX not in key:
                continue
            params[key.split(TEXT_PREFIX, 1)[1]] = handle.get_tensor(key).to(device=device, dtype=dtype)
    return params


def rope_tables(config: TextConfig, max_pos: int, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Text-only rotary tables. Shape [max_pos, rope_dim // 2].

    The checkpoint stores interleaved mrope sections, but a text-only prompt has
    identical position ids on the temporal/height/width grids, so every section
    collapses to plain rope on the leading ``rope_dim`` channels.
    """
    dim = config.rope_dim
    inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    positions = torch.arange(max_pos, dtype=torch.float32, device=device)
    angles = positions[:, None] * inv_freq[None, :]
    return angles.cos(), angles.sin()


@dataclass
class TextState:
    """Per-layer decode state. Entries are None for the other layer type."""

    conv: list[torch.Tensor | None] = field(default_factory=list)
    rec: list[torch.Tensor | None] = field(default_factory=list)
    keys: list[torch.Tensor | None] = field(default_factory=list)
    values: list[torch.Tensor | None] = field(default_factory=list)

    @classmethod
    def empty(cls, config: TextConfig) -> "TextState":
        return cls(
            conv=[None] * config.num_hidden_layers,
            rec=[None] * config.num_hidden_layers,
            keys=[None] * config.num_hidden_layers,
            values=[None] * config.num_hidden_layers,
        )


def _rms_zero_centered(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen3_5RMSNorm: zero-initialised weight, applied as (1 + w)."""
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * (1.0 + weight)


def _rms_plain(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


def _rms_gated(x: torch.Tensor, weight: torch.Tensor, gate: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen3_5RMSNormGated: normalise, scale, then multiply by silu(gate)."""
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight * F.silu(gate)


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Partial rope on a [heads, head_dim] tensor; cos/sin are [rope_dim // 2]."""
    rope_dim = cos.shape[-1] * 2
    rot, passthrough = x[..., :rope_dim], x[..., rope_dim:]
    first, second = rot[..., : rope_dim // 2], rot[..., rope_dim // 2 :]
    rotated = torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)
    return torch.cat((rotated, passthrough), dim=-1)


def _linear_attention(
    params: dict[str, torch.Tensor],
    config: TextConfig,
    index: int,
    x: torch.Tensor,
    state: TextState,
) -> torch.Tensor:
    prefix = f"layers.{index}.linear_attn."

    mixed_qkv = x @ params[prefix + "in_proj_qkv.weight"].T
    window = mixed_qkv.unsqueeze(-1)
    previous = state.conv[index]
    if previous is None:
        window = F.pad(window, (config.conv_kernel_dim - 1, 0))
    else:
        window = torch.cat((previous, window), dim=-1)
    state.conv[index] = window[:, -config.conv_kernel_dim + 1 :].contiguous()

    conv_weight = params[prefix + "conv1d.weight"].squeeze(1)
    mixed = F.silu((window * conv_weight).sum(-1))

    key_dim, value_dim = config.key_dim, config.value_dim
    query = mixed[:key_dim].view(config.linear_num_key_heads, config.linear_key_head_dim)
    key = mixed[key_dim : 2 * key_dim].view(config.linear_num_key_heads, config.linear_key_head_dim)
    value = mixed[2 * key_dim :].view(config.linear_num_value_heads, config.linear_value_head_dim)

    z = x @ params[prefix + "in_proj_z.weight"].T
    beta = (x @ params[prefix + "in_proj_b.weight"].T).sigmoid()
    a = x @ params[prefix + "in_proj_a.weight"].T
    decay = -params[prefix + "A_log"].float().exp() * F.softplus(a.float() + params[prefix + "dt_bias"].float())

    query = _l2norm(query) * (config.linear_key_head_dim**-0.5)
    key = _l2norm(key)

    recurrent = state.rec[index]
    if recurrent is None:
        recurrent = torch.zeros(
            config.linear_num_value_heads,
            config.linear_key_head_dim,
            config.linear_value_head_dim,
            dtype=torch.float32,
            device=x.device,
        )
    recurrent = recurrent * decay.exp().unsqueeze(-1).unsqueeze(-1)
    kv_memory = (recurrent * key.unsqueeze(-1)).sum(-2)
    delta = (value - kv_memory) * beta.unsqueeze(-1)
    recurrent = recurrent + key.unsqueeze(-1) * delta.unsqueeze(-2)
    state.rec[index] = recurrent

    out = (recurrent * query.unsqueeze(-1)).sum(-2).reshape(-1)
    out = _rms_gated(out, params[prefix + "norm.weight"], z, config.rms_norm_eps)
    return out @ params[prefix + "out_proj.weight"].T


def _full_attention(
    params: dict[str, torch.Tensor],
    config: TextConfig,
    index: int,
    x: torch.Tensor,
    state: TextState,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    prefix = f"layers.{index}.self_attn."

    query_gate = x @ params[prefix + "q_proj.weight"].T
    query, gate = query_gate[: config.num_attention_heads * config.head_dim], query_gate[config.num_attention_heads * config.head_dim :]
    query = query.view(config.num_attention_heads, config.head_dim)
    key = (x @ params[prefix + "k_proj.weight"].T).view(config.num_key_value_heads, config.head_dim)
    value = (x @ params[prefix + "v_proj.weight"].T).view(config.num_key_value_heads, config.head_dim)

    query = _apply_rope(_rms_plain(query, params[prefix + "q_norm.weight"], config.rms_norm_eps), cos, sin)
    key = _apply_rope(_rms_plain(key, params[prefix + "k_norm.weight"], config.rms_norm_eps), cos, sin)

    state.keys[index] = key.unsqueeze(1) if state.keys[index] is None else torch.cat((state.keys[index], key.unsqueeze(1)), dim=1)
    state.values[index] = value.unsqueeze(1) if state.values[index] is None else torch.cat((state.values[index], value.unsqueeze(1)), dim=1)
    keys, values = state.keys[index], state.values[index]

    grouped_keys = keys.repeat_interleave(config.kv_groups, dim=0)
    grouped_values = values.repeat_interleave(config.kv_groups, dim=0)
    scores = (query.unsqueeze(1) @ grouped_keys.transpose(1, 2)).squeeze(1) * (config.head_dim**-0.5)
    probs = scores.softmax(-1)
    attended = (probs.unsqueeze(1) @ grouped_values).squeeze(1).reshape(-1)
    return (attended * gate.sigmoid()) @ params[prefix + "o_proj.weight"].T


def _mlp(params: dict[str, torch.Tensor], config: TextConfig, index: int, x: torch.Tensor) -> torch.Tensor:
    prefix = f"layers.{index}.mlp."
    return F.silu(x @ params[prefix + "gate_proj.weight"].T) * (x @ params[prefix + "up_proj.weight"].T) @ params[
        prefix + "down_proj.weight"
    ].T


def forward_step(
    params: dict[str, torch.Tensor],
    config: TextConfig,
    token: int,
    position: int,
    state: TextState,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
) -> torch.Tensor:
    """Advance the state by one token and return logits [vocab_size]."""
    embed = params["embed_tokens.weight"]
    x = embed[token].to(torch.float32)

    for index, layer_type in enumerate(config.layer_types):
        residual = x
        x = _rms_zero_centered(x, params[f"layers.{index}.input_layernorm.weight"], config.rms_norm_eps)
        if layer_type == "linear_attention":
            x = _linear_attention(params, config, index, x, state)
        else:
            x = _full_attention(params, config, index, x, state, cos_table[position], sin_table[position])
        x = residual + x

        residual = x
        x = _rms_zero_centered(x, params[f"layers.{index}.post_attention_layernorm.weight"], config.rms_norm_eps)
        x = residual + _mlp(params, config, index, x)

    x = _rms_zero_centered(x, params["norm.weight"], config.rms_norm_eps)
    return x @ embed.T


def prefill(
    params: dict[str, torch.Tensor],
    config: TextConfig,
    tokens: list[int],
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
) -> tuple[list[torch.Tensor], TextState]:
    """Run a prompt token by token. Returns per-position logits and the final state."""
    state = TextState.empty(config)
    logits = [forward_step(params, config, token, position, state, cos_table, sin_table) for position, token in enumerate(tokens)]
    return logits, state


def greedy(
    params: dict[str, torch.Tensor],
    config: TextConfig,
    tokens: list[int],
    max_new_tokens: int,
    stop_ids: set[int],
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
) -> list[int]:
    """Greedy continuation used as the behavioural parity check."""
    logits, state = prefill(params, config, tokens, cos_table, sin_table)
    produced: list[int] = []
    position = len(tokens)
    last = logits[-1]
    while len(produced) < max_new_tokens:
        token = int(torch.argmax(last))
        produced.append(token)
        if token in stop_ids:
            break
        last = forward_step(params, config, token, position, state, cos_table, sin_table)
        position += 1
    return produced
