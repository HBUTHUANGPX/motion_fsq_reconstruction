"""配置加载与 dataclass schema。"""

from __future__ import annotations

from .io import load_config
from .schema import (
    DataConfig,
    FeatureConfig,
    LossConfig,
    ModelConfig,
    MotionFSQReconstructionConfig,
    OutputConfig,
    QuantizerConfig,
    TrainConfig,
)

__all__ = [
    "DataConfig",
    "FeatureConfig",
    "LossConfig",
    "ModelConfig",
    "MotionFSQReconstructionConfig",
    "OutputConfig",
    "QuantizerConfig",
    "TrainConfig",
    "load_config",
]
