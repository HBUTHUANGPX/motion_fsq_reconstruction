"""GPU 常驻窗口索引和采样缓冲。

职责：
    在 device 上维护合法中心帧池，并通过向量化索引抽取窗口。
前置条件：
    输入 feature 第一维为总帧数，motion_lengths 描述 clip 边界。
后置条件：
    训练采样不会跨 clip，导出可按 clip 内 clamp 生成全帧窗口。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch

from motion_fsq_reconstruction.features.specs import DualFSQFeatureBundle


@dataclass
class MotionWindowBatch:
    """一批已展平的 DualFSQ 窗口特征。"""

    actor_robot: torch.Tensor
    actor_human: torch.Tensor
    critic_robot: torch.Tensor
    critic_human: torch.Tensor
    center_indices: torch.Tensor
    window_indices: torch.Tensor


class MotionWindowIndex:
    """负责构建合法中心帧和窗口索引。

    前置条件：
        `history`、`future` 非负，`motion_lengths` 均大于 0。
    后置条件：
        `valid_center_indices` 不包含会跨 clip 的中心帧。
    """

    def __init__(
        self,
        motion_lengths: torch.Tensor,
        *,
        history: int,
        future: int,
        device: str | torch.device,
    ) -> None:
        if history < 0 or future < 0:
            raise ValueError("history 和 future 必须非负。")
        self._device = torch.device(device)
        self._motion_lengths = motion_lengths.to(self._device, dtype=torch.long)
        self._history = int(history)
        self._future = int(future)
        self._window_offsets = torch.arange(
            -self._history,
            self._future + 1,
            device=self._device,
            dtype=torch.long,
        )
        self._clip_starts = torch.cat(
            (
                torch.zeros(1, device=self._device, dtype=torch.long),
                torch.cumsum(self._motion_lengths[:-1], dim=0),
            )
        )
        self._valid_center_indices = self._build_valid_center_indices()

    @property
    def window_size(self) -> int:
        """返回窗口帧数。"""

        return int(self._window_offsets.numel())

    @property
    def valid_center_indices(self) -> torch.Tensor:
        """返回合法中心帧索引。"""

        return self._valid_center_indices

    @property
    def all_center_indices(self) -> torch.Tensor:
        """返回所有帧作为中心帧的索引。"""

        return torch.arange(int(self._motion_lengths.sum().item()), device=self._device)

    def window_indices_for(self, centers: torch.Tensor, *, clamp_to_clip: bool) -> torch.Tensor:
        """为中心帧生成窗口索引。

        前置条件：
            `centers` 是全局帧索引。
        后置条件：
            当 `clamp_to_clip=True` 时，边界帧窗口会夹到当前 clip 内。
        """

        centers = centers.to(self._device, dtype=torch.long)
        raw = centers[:, None] + self._window_offsets[None, :]
        if not clamp_to_clip:
            return raw
        clip_ids = torch.bucketize(centers, self._clip_starts[1:], right=False)
        starts = self._clip_starts[clip_ids]
        ends = starts + self._motion_lengths[clip_ids] - 1
        return raw.clamp(min=starts[:, None], max=ends[:, None])

    def _build_valid_center_indices(self) -> torch.Tensor:
        centers: list[torch.Tensor] = []
        for start, length in zip(self._clip_starts.tolist(), self._motion_lengths.tolist()):
            first = start + self._history
            last_exclusive = start + length - self._future
            if first < last_exclusive:
                centers.append(torch.arange(first, last_exclusive, device=self._device))
        if not centers:
            raise ValueError("当前 history/future 下没有合法中心帧。")
        return torch.cat(centers, dim=0)


class MotionWindowBuffer:
    """保存 feature tensor 并提供 batch 采样。

    职责：
        将四路 feature 常驻 device，并输出展平窗口 batch。
    前置条件：
        四路 feature 帧数一致。
    后置条件：
        训练 batch 已展平为 `[B, window_size * feature_dim]`。
    """

    def __init__(
        self,
        features: DualFSQFeatureBundle,
        motion_lengths: torch.Tensor,
        *,
        history: int,
        future: int,
        device: str | torch.device,
    ) -> None:
        self._device = torch.device(device)
        self._features = DualFSQFeatureBundle(
            actor_robot=features.actor_robot.to(self._device),
            actor_human=features.actor_human.to(self._device),
            critic_robot=features.critic_robot.to(self._device),
            critic_human=features.critic_human.to(self._device),
            schema=features.schema,
        )
        self._index = MotionWindowIndex(
            motion_lengths=motion_lengths,
            history=history,
            future=future,
            device=self._device,
        )

    @property
    def window_size(self) -> int:
        """返回窗口帧数。"""

        return self._index.window_size

    @property
    def valid_center_indices(self) -> torch.Tensor:
        """返回合法中心帧池。"""

        return self._index.valid_center_indices

    @property
    def index(self) -> MotionWindowIndex:
        """返回底层窗口索引器。"""

        return self._index

    def iter_epoch_batches(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
    ) -> Iterator[MotionWindowBatch]:
        """按 epoch 随机无放回遍历合法中心帧。"""

        if batch_size <= 0:
            raise ValueError("batch_size 必须为正数。")
        order = torch.randperm(
            self.valid_center_indices.numel(),
            device=self._device,
            generator=generator,
        )
        centers = self.valid_center_indices[order]
        for start in range(0, centers.numel(), batch_size):
            batch_centers = centers[start : start + batch_size]
            yield self.batch_from_centers(batch_centers, clamp_to_clip=False)

    def batch_from_centers(self, centers: torch.Tensor, *, clamp_to_clip: bool) -> MotionWindowBatch:
        """按指定中心帧构造展平窗口 batch。"""

        indices = self._index.window_indices_for(centers, clamp_to_clip=clamp_to_clip)
        return MotionWindowBatch(
            actor_robot=self._flatten(self._features.actor_robot[indices]),
            actor_human=self._flatten(self._features.actor_human[indices]),
            critic_robot=self._flatten(self._features.critic_robot[indices]),
            critic_human=self._flatten(self._features.critic_human[indices]),
            center_indices=centers.to(self._device, dtype=torch.long),
            window_indices=indices,
        )

    def num_batches(self, batch_size: int) -> int:
        """返回一个 epoch 的 batch 数。"""

        return max((self.valid_center_indices.numel() + batch_size - 1) // batch_size, 1)

    @staticmethod
    def _flatten(window: torch.Tensor) -> torch.Tensor:
        return window.reshape(window.shape[0], -1)
