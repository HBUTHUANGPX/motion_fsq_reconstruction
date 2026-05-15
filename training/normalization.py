"""窗口 feature 归一化。

职责：
    按逐帧 feature 统计均值方差，并 repeat 到展平 window 维度。
前置条件：
    输入 frame feature shape 为 `[T, D]`。
后置条件：
    调用对象可归一化 `[B, window_size * D]`。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class NormalizerState:
    """可保存的归一化状态。"""

    mean: torch.Tensor
    std: torch.Tensor
    eps: float


class WindowFeatureNormalizer:
    """展平 window feature 归一化器。"""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, *, eps: float) -> None:
        self._mean = mean
        self._std = std
        self._eps = float(eps)

    @property
    def mean(self) -> torch.Tensor:
        """返回均值 tensor。"""

        return self._mean

    @property
    def std(self) -> torch.Tensor:
        """返回标准差 tensor。"""

        return self._std

    def to(self, device: str | torch.device) -> WindowFeatureNormalizer:
        """移动归一化状态到指定 device。"""

        self._mean = self._mean.to(device)
        self._std = self._std.to(device)
        return self

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        """归一化输入 tensor。"""

        return (value - self._mean) / self._std

    def state_dict(self) -> dict[str, torch.Tensor | float]:
        """返回 checkpoint 可保存状态。"""

        return {"mean": self._mean.detach().cpu(), "std": self._std.detach().cpu(), "eps": self._eps}

    @classmethod
    def from_state_dict(cls, state: dict[str, torch.Tensor | float]) -> WindowFeatureNormalizer:
        """从 checkpoint 状态恢复归一化器。"""

        return cls(
            mean=state["mean"],
            std=state["std"],
            eps=float(state.get("eps", 1.0e-2)),
        )

    @classmethod
    def fit(
        cls,
        frame_feature: torch.Tensor,
        *,
        window_size: int,
        eps: float,
    ) -> WindowFeatureNormalizer:
        """按逐帧统计构建 window 归一化器。"""

        mean = frame_feature.mean(dim=0)
        var = frame_feature.var(dim=0, unbiased=False)
        std = torch.sqrt(var + float(eps))
        return cls(
            mean=mean.repeat(window_size),
            std=std.repeat(window_size).clamp_min(float(eps)),
            eps=eps,
        )
