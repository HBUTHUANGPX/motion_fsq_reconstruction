"""离线 DualFSQ 重构训练配置。

职责：
    使用 dataclass 描述 YAML 和 Python 代码共享的稳定配置结构。
前置条件：
    配置值必须能被下游数据加载、特征构建和训练模块解释。
后置条件：
    `to_dict()` 返回可写入 checkpoint 的普通 Python 映射。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from motion_fsq_reconstruction.features.specs import (
    DEFAULT_DESIRE_HUMAN_JOINT_NAMES,
    DEFAULT_HUMAN_BODY_NAMES,
    DEFAULT_ROBOT_BODY_NAMES,
)


@dataclass
class DataConfig:
    """motion 数据来源配置。

    职责：
        描述直接文件、目录或旧 YAML motion 列表。
    前置条件：
        至少一种来源在运行前可解析到 npz 文件。
    后置条件：
        解析器得到有序文件列表和同长度 group 列表。
    """

    motion_yaml: str | None = None
    files: list[str] = field(default_factory=list)
    dirs: list[str] = field(default_factory=list)
    exclude_files: list[str] = field(default_factory=list)
    exclude_dirs: list[str] = field(default_factory=list)
    groups: list[str] = field(default_factory=list)


@dataclass
class FeatureConfig:
    """在线 DualFSQ feature 语义配置。

    职责：
        指定 robot/human anchor 和参与 FSQ 的 human body 名称。
    前置条件：
        这些名称必须存在于 npz 的名字字段中。
    后置条件：
        FeatureBuilder 能构建 actor/critic 四类逐帧特征。
    """

    robot_anchor_body: str = "torso_link"
    robot_body_names: list[str] = field(
        default_factory=lambda: list(DEFAULT_ROBOT_BODY_NAMES)
    )
    robot_joint_names: list[str] = field(default_factory=list)
    desire_human_joint_names: list[str] = field(
        default_factory=lambda: list(DEFAULT_DESIRE_HUMAN_JOINT_NAMES)
    )
    human_anchor_body: str = "Hips"
    human_body_names: list[str] = field(
        default_factory=lambda: list(DEFAULT_HUMAN_BODY_NAMES)
    )


@dataclass
class QuantizerConfig:
    """FSQ/iFSQ 量化器配置。

    职责：
        对齐 rsl_rl DualFSQ 中使用的有限标量量化器配置。
    前置条件：
        `type` 必须是 `fsq` 或 `ifsq`。
    后置条件：
        模型工厂创建共享量化器实例。
    """

    type: str = "ifsq"
    levels: int = 32
    ifsq_boundary_fn: str = "sigmoid"
    ifsq_boundary_scale: float = 1.6


@dataclass
class ModelConfig:
    """双 encoder、共享 quantizer、单 decoder 网络配置。"""

    latent_dim: int = 64
    robot_encoder_hidden_dims: list[int] = field(default_factory=lambda: [512, 256])
    human_encoder_hidden_dims: list[int] = field(default_factory=lambda: [512, 256])
    decoder_hidden_dims: list[int] = field(default_factory=lambda: [256, 512])
    activation: str = "elu"
    quantizer: QuantizerConfig = field(default_factory=QuantizerConfig)


@dataclass
class LossConfig:
    """DualFSQ MSE 组合损失权重。"""

    robot_recon: float = 1.0
    human_recon: float = 1.0
    latent_align: float = 5.0
    cycle_latent: float = 0.25


@dataclass
class TrainConfig:
    """训练循环和采样配置。"""

    device: str = "cuda"
    epochs: int = 100
    batch_size: int = 1024
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    history: int = 0
    future: int = 9
    seed: int = 1
    log_every_steps: int = 20
    checkpoint_interval_epochs: int = 10
    normalizer_eps: float = 1.0e-2
    progress: bool = True


@dataclass
class OutputConfig:
    """训练输出目录配置。"""

    root_dir: str = "outputs/motion_fsq_reconstruction"
    run_name: str | None = None


@dataclass
class MotionFSQReconstructionConfig:
    """包级顶层配置。"""

    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict[str, Any]:
        """返回 checkpoint 可序列化配置。

        前置条件：
            配置对象已完成 dataclass 初始化。
        后置条件：
            返回值只包含 dict/list/str/int/float/bool/None。
        """

        return asdict(self)
