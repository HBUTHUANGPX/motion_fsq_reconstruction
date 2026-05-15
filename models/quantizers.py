"""与在线 rsl_rl DualFSQ 对齐的标量量化器。

职责：
    提供 FSQ 和 iFSQ 的 `[B, latent_dim]` 标量量化。
前置条件：
    输入 latent 是二维 tensor。
后置条件：
    返回 dict，至少包含 `z_q`，可直接被 DualFSQ decoder 使用。
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn


class FSQQuantizer(nn.Module):
    """有限标量量化器。

    前置条件：
        `levels >= 2`，输入 shape 为 `[B, D]`。
    后置条件：
        输出 `z_q` 通过直通估计器保留反向梯度。
    """

    def __init__(self, levels: int | Iterable[int] = 32) -> None:
        super().__init__()
        self._levels = _first_level(levels)

    @property
    def levels(self) -> int:
        """返回每个 latent 标量的 level 数。"""

        return self._levels

    def forward(self, z_e: torch.Tensor) -> dict[str, torch.Tensor]:
        """执行 FSQ 量化。"""

        if z_e.ndim != 2:
            raise ValueError(f"FSQ 输入必须是 [B, D]，实际 shape={tuple(z_e.shape)}。")
        z_bound = torch.tanh(z_e)
        return _quantize_bound_latent(z_bound, self._levels)


class IFSQuantizer(FSQQuantizer):
    """iFSQ 量化器。

    职责：
        使用 sigmoid/tanh boundary transform 改善离散 level 使用率。
    前置条件：
        `boundary_fn` 为 `sigmoid` 或 `tanh`。
    后置条件：
        输出格式与 `FSQQuantizer` 一致。
    """

    def __init__(
        self,
        levels: int | Iterable[int] = 32,
        boundary_fn: str = "sigmoid",
        boundary_scale: float = 1.6,
    ) -> None:
        super().__init__(levels=levels)
        normalized = boundary_fn.lower().strip()
        if normalized not in {"sigmoid", "tanh"}:
            raise ValueError(f"不支持的 iFSQ boundary_fn: {boundary_fn}。")
        self._boundary_fn = normalized
        self._boundary_scale = float(boundary_scale)

    @property
    def boundary_fn(self) -> str:
        """返回 boundary 函数名称。"""

        return self._boundary_fn

    @property
    def boundary_scale(self) -> float:
        """返回 boundary 输入缩放系数。"""

        return self._boundary_scale

    def forward(self, z_e: torch.Tensor) -> dict[str, torch.Tensor]:
        """执行 iFSQ 量化。"""

        if z_e.ndim != 2:
            raise ValueError(f"iFSQ 输入必须是 [B, D]，实际 shape={tuple(z_e.shape)}。")
        scaled = z_e * self._boundary_scale
        if self._boundary_fn == "sigmoid":
            z_bound = torch.sigmoid(scaled) * 2.0 - 1.0
        else:
            z_bound = torch.tanh(scaled)
        return _quantize_bound_latent(z_bound, self.levels)


def build_quantizer(
    *,
    quantizer_type: str,
    levels: int,
    ifsq_boundary_fn: str,
    ifsq_boundary_scale: float,
) -> nn.Module:
    """根据配置构建量化器。"""

    normalized = quantizer_type.lower().strip()
    if normalized == "fsq":
        return FSQQuantizer(levels=levels)
    if normalized == "ifsq":
        return IFSQuantizer(
            levels=levels,
            boundary_fn=ifsq_boundary_fn,
            boundary_scale=ifsq_boundary_scale,
        )
    raise ValueError(f"不支持的 quantizer_type: {quantizer_type}。")


def _first_level(levels: int | Iterable[int]) -> int:
    if isinstance(levels, int):
        value = int(levels)
    else:
        values = list(levels)
        if not values:
            raise ValueError("levels 不允许为空。")
        value = int(values[0])
    if value < 2:
        raise ValueError("levels 必须 >= 2。")
    return value


def _quantize_bound_latent(z_bound: torch.Tensor, levels: int) -> dict[str, torch.Tensor]:
    scale = float(levels - 1)
    scaled = (z_bound + 1.0) * 0.5 * scale
    indices = torch.round(scaled).clamp(0, levels - 1).long()
    z_q = (indices.to(z_bound.dtype) / scale) * 2.0 - 1.0
    z_q_st = z_bound + (z_q - z_bound).detach()
    one_hot = F.one_hot(indices, num_classes=levels).to(dtype=z_bound.dtype)
    per_dim_usage = one_hot.mean(dim=0)
    used = per_dim_usage > 1.0e-6
    unique_per_dim = used.sum(dim=1).clamp(min=1).to(dtype=z_bound.dtype)
    entropy_probs = per_dim_usage / per_dim_usage.sum(dim=1, keepdim=True).clamp_min(1.0e-10)
    entropy = -torch.sum(entropy_probs * torch.log2(entropy_probs.clamp_min(1.0e-10)), dim=1)
    return {
        "z_q": z_q_st,
        "indices": indices,
        "level_histogram": per_dim_usage.mean(dim=0),
        "avg_utilization": used.to(dtype=z_bound.dtype).mean() * 100.0,
        "effective_bits": torch.log2(unique_per_dim).mean(),
        "effective_bits_entropy": entropy.mean(),
    }
