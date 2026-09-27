"""int4 group quantization and the chunked weight container read by the browser engine.

Scheme (frozen -- the WGSL kernels in the demo implement exactly this):

* every 2-D matmul weight is quantized independently along its input dimension in
  groups of 64, symmetric, ``scale = max|w| / 7``, ``q = clamp(round(w / scale), -8, 7)``;
* packed nibbles are row-major pairs: ``byte = q[2j] & 0xF | q[2j+1] << 4`` with
  two's-complement 4-bit values (the shader sign-extends with ``(u << 28) >> 28``);
* scales are stored as fp16 (half the bytes) and widened to fp32 in JavaScript at
  load time, so the engine needs no ``shader-f16`` feature;
* 1-D tensors (norms, A_log, dt_bias) and conv1d stay fp32 -- they are ~0.1% of the
  parameters and sit on precision-critical paths.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import torch

GROUP = 64
INT4_KINDS = {"int4", "fp32"}


def quantize_matrix(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize [out, in] fp32 weights to packed int4 plus fp16 group scales."""
    out_features, in_features = weight.shape
    if in_features % GROUP:
        raise ValueError(f"{tuple(weight.shape)} is not a multiple of the group size {GROUP}")
    grouped = weight.detach().float().reshape(out_features, in_features // GROUP, GROUP)
    scales = grouped.abs().amax(-1) / 7.0
    scales = scales.clamp_min(1e-8)
    quantized = torch.round(grouped / scales.unsqueeze(-1)).clamp_(-8, 7).to(torch.int8)

    nibbles = (quantized.to(torch.int32) & 0xF).to(torch.uint8).reshape(out_features, in_features // 2, 2)
    packed = (nibbles[:, :, 0] | (nibbles[:, :, 1] << 4)).contiguous()
    return packed, scales.to(torch.float16).contiguous()


def dequantize_matrix(packed: torch.Tensor, scales: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """Inverse of :func:`quantize_matrix`, used to simulate int4 damage in torch."""
    out_features, in_features = shape
    codes = packed.to(torch.int32)
    low, high = codes & 0xF, (codes >> 4) & 0xF
    quantized = torch.empty((out_features, in_features), dtype=torch.int32)
    quantized[:, 0::2] = low
    quantized[:, 1::2] = high
    quantized = (quantized << 28) >> 28
    expanded = scales.float().unsqueeze(-1).expand(out_features, in_features // GROUP, GROUP)
    return quantized.float() * expanded.reshape(out_features, in_features)


@dataclass
class ChunkWriter:
    """Concatenates tensor blobs into fixed-size chunk files with a manifest."""

    directory: Path
    chunk_size: int = 32 * 1024 * 1024
    _buffers: list[bytearray] = field(default_factory=list, init=False)
    _written: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self._buffers = [bytearray()]

    @property
    def _current(self) -> bytearray:
        return self._buffers[-1]

    def _ensure(self, size: int) -> None:
        if size > self.chunk_size:
            raise ValueError(f"blob of {size} bytes exceeds the chunk size {self.chunk_size}")
        if self._current and len(self._current) + size > self.chunk_size:
            self._buffers.append(bytearray())

    def add(self, blob: bytes) -> tuple[int, int]:
        """Append a blob, returning (chunk index, byte offset)."""
        self._ensure(len(blob))
        offset = len(self._current)
        self._current.extend(blob)
        return len(self._buffers) - 1, offset

    def add_aligned(self, blob: bytes, alignment: int = 64) -> tuple[int, int]:
        """Append a blob aligned so GPU bindings and typeless reads stay sane."""
        self._ensure(len(blob) + alignment)
        padding = (-len(self._current)) % alignment
        return self.add(b"\x00" * padding + blob) if padding else self.add(blob)

    def finish(self) -> list[dict[str, object]]:
        """Write every non-empty chunk and return manifest entries with digests."""
        entries: list[dict[str, object]] = []
        for index, buffer in enumerate(self._buffers):
            if not buffer:
                continue
            payload = bytes(buffer)
            name = f"weights-{index:03d}.bin"
            (self.directory / name).write_bytes(payload)
            entries.append(
                {
                    "file": name,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
            self._written.append(name)
        return entries


def tensor_blob(tensor: torch.Tensor) -> bytes:
    """Serialise a tensor as little-endian raw bytes in its own dtype."""
    return tensor.detach().contiguous().cpu().numpy().tobytes()


def fp16_scales_to_fp32_blob(scales: torch.Tensor) -> bytes:
    """Scales ship as fp16; this is the fp32 blob the engine uploads."""
    return struct.pack("<f", 0.0) * 0 + tensor_blob(scales.float())


def write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=1, sort_keys=False) + "\n")
