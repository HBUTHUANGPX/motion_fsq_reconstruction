"""DualFSQ feature schema 定义。

职责：
    以可序列化 dataclass 固定离线 feature 与在线 `_make_calculate()` 的契约。
前置条件：
    名称列表来自同一批 motion 文件的 schema。
后置条件：
    checkpoint 和 latent 文件可通过 schema 解释每一段 feature 的语义。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch


DEFAULT_HUMAN_BODY_NAMES = [
    "Spine1",
    "Spine2",
    "Chest",
    "Neck1",
    "Neck2",
    "Head",
    "HeadEnd",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftLeg",
    "LeftShin",
    "LeftFoot",
    "LeftToeBase",
    "LeftToeEnd",
    "RightLeg",
    "RightShin",
    "RightFoot",
    "RightToeBase",
    "RightToeEnd",
]


@dataclass
class FeatureBuilderConfig:
    """feature 构建配置。"""

    robot_anchor_body: str = "torso_link"
    human_anchor_body: str = "Hips"
    human_body_names: list[str] = field(default_factory=lambda: list(DEFAULT_HUMAN_BODY_NAMES))


@dataclass
class DualFSQFeatureSchema:
    """四路 DualFSQ feature 的可序列化描述。"""

    robot_anchor_body: str
    human_anchor_body: str
    human_body_names: list[str]
    robot_joint_names: list[str]
    robot_body_names: list[str]
    source_human_body_names: list[str]
    actor_robot_feature_dim: int
    actor_human_feature_dim: int
    critic_robot_feature_dim: int
    critic_human_feature_dim: int

    def to_dict(self) -> dict[str, object]:
        """返回 checkpoint/npz metadata 可保存的普通字典。"""

        return asdict(self)


@dataclass
class DualFSQFeatureBundle:
    """逐帧 actor/critic robot/human feature 容器。"""

    actor_robot: torch.Tensor
    actor_human: torch.Tensor
    critic_robot: torch.Tensor
    critic_human: torch.Tensor
    schema: DualFSQFeatureSchema

    @property
    def num_frames(self) -> int:
        """返回逐帧 feature 的总帧数。"""

        return int(self.actor_robot.shape[0])
