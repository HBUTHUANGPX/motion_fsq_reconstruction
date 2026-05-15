"""checkpoint 保存与加载。

职责：
    保存训练状态、模型参数、normalizer、feature schema 和配置。
前置条件：
    模型与 optimizer 已完成初始化。
后置条件：
    checkpoint 文件可由 trainer 恢复或由 latent exporter 读取。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from motion_fsq_reconstruction.models import DualFSQTrainingModule
from motion_fsq_reconstruction.training.normalization import WindowFeatureNormalizer


def save_checkpoint(
    *,
    path: str | Path,
    model: DualFSQTrainingModule,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    config: dict[str, Any],
    normalizers: dict[str, WindowFeatureNormalizer],
    feature_schema: dict[str, Any],
) -> Path:
    """保存 checkpoint。

    前置条件：
        `path.parent` 可创建。
    后置条件：
        文件写入成功并返回 Path。
    """

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "config": config,
            "normalizers": {
                name: normalizer.state_dict()
                for name, normalizer in normalizers.items()
            },
            "feature_schema": feature_schema,
        },
        checkpoint_path,
    )
    return checkpoint_path


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """加载 checkpoint 字典。"""

    return torch.load(Path(path), map_location=map_location, weights_only=False)
