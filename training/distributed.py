"""分布式训练运行时与 motion 分片工具。

职责：
    封装 torch.distributed 初始化、rank 信息、frame-balanced motion 分片和
    全局 normalizer 统计同步。
前置条件：
    使用 torchrun 启动时环境变量包含 RANK、WORLD_SIZE 和 LOCAL_RANK。
后置条件：
    调用方可以用统一接口区分单进程和分布式训练路径。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.distributed as dist

from motion_fsq_reconstruction.data import ResolvedMotionSources
from motion_fsq_reconstruction.training.normalization import WindowFeatureNormalizer


@dataclass(frozen=True)
class DistributedRuntime:
    """分布式训练运行时信息。"""

    enabled: bool
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    backend: str = "gloo"

    @property
    def is_main(self) -> bool:
        """当前进程是否为 rank0。"""

        return self.rank == 0

    @classmethod
    def disabled(cls) -> DistributedRuntime:
        """返回单进程运行时。"""

        return cls(enabled=False)

    @classmethod
    def from_environment(cls, requested_device: str) -> DistributedRuntime:
        """从 torchrun 环境变量创建分布式运行时。

        前置条件：
            torchrun 已设置 RANK、WORLD_SIZE 和 LOCAL_RANK。
        后置条件：
            torch.distributed process group 已初始化。
        """

        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        backend = "nccl" if requested_device.startswith("cuda") else "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        return cls(
            enabled=True,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            backend=backend,
        )

    def barrier(self) -> None:
        """等待所有 rank 到达同一点。"""

        if self.enabled and dist.is_initialized():
            dist.barrier()

    def close(self) -> None:
        """销毁 process group。"""

        if self.enabled and dist.is_initialized():
            dist.destroy_process_group()


@dataclass(frozen=True)
class MotionShardInfo:
    """单个 rank 的 motion 分片信息。"""

    rank: int
    world_size: int
    local_valid_frames: int
    global_valid_frames: int
    local_file_count: int

    def to_dict(self) -> dict[str, int]:
        """返回 checkpoint 可保存的普通字典。"""

        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "local_valid_frames": self.local_valid_frames,
            "global_valid_frames": self.global_valid_frames,
            "local_file_count": self.local_file_count,
        }


def resolve_training_device(requested: str, runtime: DistributedRuntime) -> torch.device:
    """根据请求设备和 rank 解析实际 torch device。"""

    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            return torch.device("cpu")
        if runtime.enabled:
            torch.cuda.set_device(runtime.local_rank)
            return torch.device(f"cuda:{runtime.local_rank}")
    return torch.device(requested)


def shard_motion_sources(
    sources: ResolvedMotionSources,
    *,
    runtime: DistributedRuntime,
    history: int,
    future: int,
) -> tuple[ResolvedMotionSources, MotionShardInfo]:
    """按 valid-frame 数对 motion 文件做 frame-balanced 分片。

    前置条件：
        `sources` 是所有 rank 都能解析到的全量文件列表。
    后置条件：
        返回当前 rank 的文件子集和分片元信息。
    """

    if not runtime.enabled:
        total_valid = sum(_valid_frame_count(path, history, future) for path in sources.paths)
        return sources, MotionShardInfo(
            rank=0,
            world_size=1,
            local_valid_frames=total_valid,
            global_valid_frames=total_valid,
            local_file_count=len(sources.paths),
        )

    items = [
        (path, group, _valid_frame_count(path, history, future))
        for path, group in zip(sources.paths, sources.groups)
    ]
    buckets: list[list[tuple[Path, str, int]]] = [[] for _ in range(runtime.world_size)]
    weights = [0 for _ in range(runtime.world_size)]
    for path, group, valid_frames in sorted(items, key=lambda item: item[2], reverse=True):
        target_rank = min(range(runtime.world_size), key=lambda index: weights[index])
        buckets[target_rank].append((path, group, valid_frames))
        weights[target_rank] += valid_frames

    local_items = sorted(buckets[runtime.rank], key=lambda item: str(item[0]))
    if not local_items:
        raise ValueError(f"rank {runtime.rank} 没有分配到 motion 文件。")
    return (
        ResolvedMotionSources(
            paths=[path for path, _, _ in local_items],
            groups=[group for _, group, _ in local_items],
        ),
        MotionShardInfo(
            rank=runtime.rank,
            world_size=runtime.world_size,
            local_valid_frames=weights[runtime.rank],
            global_valid_frames=sum(weights),
            local_file_count=len(local_items),
        ),
    )


def fit_window_normalizer(
    frame_feature: torch.Tensor,
    *,
    window_size: int,
    eps: float,
    runtime: DistributedRuntime,
) -> WindowFeatureNormalizer:
    """按全局 frame feature 统计构建 window normalizer。

    前置条件：
        `frame_feature` shape 为 `[T, D]`，每个 rank 至少有一帧。
    后置条件：
        分布式下所有 rank 返回完全一致的 normalizer。
    """

    count = torch.tensor(
        [frame_feature.shape[0]],
        dtype=frame_feature.dtype,
        device=frame_feature.device,
    )
    total = frame_feature.sum(dim=0)
    total_square = (frame_feature * frame_feature).sum(dim=0)
    if runtime.enabled:
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_square, op=dist.ReduceOp.SUM)
    mean = total / count.clamp_min(1.0)
    var = total_square / count.clamp_min(1.0) - mean * mean
    std = torch.sqrt(var.clamp_min(0.0) + float(eps)).clamp_min(float(eps))
    return WindowFeatureNormalizer(
        mean=mean.repeat(window_size),
        std=std.repeat(window_size),
        eps=eps,
    )


def average_epoch_totals(
    totals: dict[str, float],
    *,
    batch_count: int,
    device: torch.device,
    runtime: DistributedRuntime,
) -> dict[str, float]:
    """聚合所有 rank 的 epoch 平均 loss。"""

    if batch_count <= 0:
        return {name: 0.0 for name in totals}
    names = sorted(totals)
    values = torch.tensor(
        [totals[name] for name in names] + [float(batch_count)],
        dtype=torch.float64,
        device=device,
    )
    if runtime.enabled:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    global_batch_count = max(float(values[-1].item()), 1.0)
    return {
        name: float(values[index].item() / global_batch_count)
        for index, name in enumerate(names)
    }


def max_int(value: int, *, device: torch.device, runtime: DistributedRuntime) -> int:
    """返回所有 rank 上 int 值的最大值。"""

    tensor = torch.tensor([int(value)], dtype=torch.long, device=device)
    if runtime.enabled:
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return int(tensor.item())


def assert_same_object(value: object, *, runtime: DistributedRuntime, label: str) -> None:
    """确认所有 rank 上的 Python 对象完全一致。"""

    if not runtime.enabled:
        return
    gathered: list[object | None] = [None for _ in range(runtime.world_size)]
    dist.all_gather_object(gathered, value)
    reference = gathered[0]
    for rank, item in enumerate(gathered):
        if item != reference:
            raise ValueError(f"{label} 在 rank0 与 rank{rank} 不一致。")


def _valid_frame_count(path: Path, history: int, future: int) -> int:
    with np.load(path, allow_pickle=True) as data:
        length = int(np.asarray(data["robot_joint_pos" if "robot_joint_pos" in data else "joint_pos"]).shape[0])
    return max(length - int(history) - int(future), 0)
