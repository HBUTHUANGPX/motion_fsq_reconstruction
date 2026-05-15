"""在线 DualFSQ 对齐的 feature builder。

职责：
    从 raw motion tensor 构建 actor/critic 四路逐帧 feature。
前置条件：
    raw motion 名称字段包含配置要求的 anchor 和 human body。
后置条件：
    feature 拼接顺序严格对齐 `MotionCommand._make_calculate()` 当前实现。
"""

from __future__ import annotations

import torch

from motion_fsq_reconstruction.data.raw_motion import RawMotionDataset
from motion_fsq_reconstruction.features.rotation import (
    quat_inverse_rotate_wxyz,
    quat_multiply_wxyz,
    quat_to_rot6d_wxyz,
    subtract_frame_transforms_wxyz,
)
from motion_fsq_reconstruction.features.specs import (
    DualFSQFeatureBundle,
    DualFSQFeatureSchema,
    FeatureBuilderConfig,
)


class DualFSQFeatureBuilder:
    """构建 DualFSQ actor/critic 特征的服务类。

    前置条件：
        输入 raw motion 已经统一 quaternion 为 wxyz。
    后置条件：
        输出四路逐帧 feature 和可序列化 schema。
    """

    def __init__(self, config: FeatureBuilderConfig) -> None:
        self._config = config

    @property
    def config(self) -> FeatureBuilderConfig:
        """返回 feature 构建配置。"""

        return self._config

    def build(self, raw: RawMotionDataset) -> DualFSQFeatureBundle:
        """从 raw motion 构建四路逐帧 feature。"""

        robot_anchor_idx = _index(raw.robot_body_names, self._config.robot_anchor_body, "robot body")
        robot_body_indices = [
            _index(raw.robot_body_names, name, "robot body")
            for name in self._config.robot_body_names
        ]
        human_anchor_idx = _index(raw.human_body_names, self._config.human_anchor_body, "human body")
        human_body_indices = [
            _index(raw.human_body_names, name, "human body")
            for name in self._config.human_body_names
        ]

        actor_robot, critic_robot = self._build_robot_features(
            raw,
            robot_anchor_idx,
            robot_body_indices,
        )
        actor_human, critic_human = self._build_human_features(
            raw,
            human_anchor_idx,
            human_body_indices,
        )
        schema = DualFSQFeatureSchema(
            robot_anchor_body=self._config.robot_anchor_body,
            robot_body_names=list(self._config.robot_body_names),
            robot_joint_names=list(raw.robot_joint_names),
            desire_human_joint_names=list(raw.human_body_names),
            human_anchor_body=self._config.human_anchor_body,
            human_body_names=list(self._config.human_body_names),
            source_human_body_names=list(raw.human_body_names),
            actor_robot_feature_dim=int(actor_robot.shape[-1]),
            actor_human_feature_dim=int(actor_human.shape[-1]),
            critic_robot_feature_dim=int(critic_robot.shape[-1]),
            critic_human_feature_dim=int(critic_human.shape[-1]),
        )
        return DualFSQFeatureBundle(
            actor_robot=actor_robot,
            actor_human=actor_human,
            critic_robot=critic_robot,
            critic_human=critic_human,
            schema=schema,
        )

    def _build_robot_features(
        self,
        raw: RawMotionDataset,
        anchor_idx: int,
        body_indices: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        robot_anchor_quat = raw.body_quat_w[:, anchor_idx]
        robot_anchor_rot6d = quat_to_rot6d_wxyz(robot_anchor_quat)
        robot_anchor_pos = raw.body_pos_w[:, anchor_idx]
        selected_body_pos = raw.body_pos_w[:, body_indices]
        selected_body_quat = raw.body_quat_w[:, body_indices]
        anchor_pos_repeat = robot_anchor_pos[:, None, :].expand(-1, len(body_indices), -1)
        anchor_quat_repeat = robot_anchor_quat[:, None, :].expand(-1, len(body_indices), -1)
        body_pos_in_anchor, body_quat_in_anchor = subtract_frame_transforms_wxyz(
            anchor_pos_repeat.reshape(-1, 3),
            anchor_quat_repeat.reshape(-1, 4),
            selected_body_pos.reshape(-1, 3),
            selected_body_quat.reshape(-1, 4),
        )
        body_pos_in_anchor = body_pos_in_anchor.reshape(raw.num_frames, -1)
        body_rot6d_in_anchor = quat_to_rot6d_wxyz(body_quat_in_anchor).reshape(raw.num_frames, -1)
        actor_robot = torch.cat((robot_anchor_rot6d, raw.joint_pos), dim=-1)
        critic_robot = torch.cat(
            (
                robot_anchor_rot6d,
                robot_anchor_pos,
                raw.joint_pos,
                body_pos_in_anchor,
                body_rot6d_in_anchor,
            ),
            dim=-1,
        )
        return actor_robot, critic_robot

    def _build_human_features(
        self,
        raw: RawMotionDataset,
        anchor_idx: int,
        human_body_indices: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        human_anchor_quat = raw.human_body_quat_w[:, anchor_idx]
        human_anchor_rot6d = quat_to_rot6d_wxyz(human_anchor_quat)
        human_anchor_pos = raw.human_body_pos_w[:, anchor_idx]
        selected_body_pos = raw.human_body_pos_w[:, human_body_indices]
        selected_body_quat = raw.human_body_quat_w[:, human_body_indices]
        selected_joint_quat = raw.human_joint_quat[:, human_body_indices]
        rel_world = selected_body_pos - human_anchor_pos[:, None, :]
        anchor_quat_repeat = human_anchor_quat[:, None, :].expand(
            -1,
            len(human_body_indices),
            -1,
        )
        body_pos_in_anchor = quat_inverse_rotate_wxyz(
            anchor_quat_repeat.reshape(-1, 4),
            rel_world.reshape(-1, 3),
        ).reshape(raw.num_frames, -1)
        body_quat_in_anchor = quat_multiply_wxyz(
            torch.cat(
                (
                    anchor_quat_repeat.reshape(-1, 4)[..., :1],
                    -anchor_quat_repeat.reshape(-1, 4)[..., 1:],
                ),
                dim=-1,
            ),
            selected_body_quat.reshape(-1, 4),
        )
        body_rot6d_in_anchor = quat_to_rot6d_wxyz(body_quat_in_anchor).reshape(raw.num_frames, -1)
        joint_rot6d = quat_to_rot6d_wxyz(selected_joint_quat).reshape(raw.num_frames, -1)
        actor_human = torch.cat((human_anchor_rot6d, body_pos_in_anchor), dim=-1)
        critic_human = torch.cat(
            (
                human_anchor_rot6d,
                human_anchor_pos,
                joint_rot6d,
                body_pos_in_anchor,
                body_rot6d_in_anchor,
            ),
            dim=-1,
        )
        return actor_human, critic_human


def _index(names: list[str], name: str, label: str) -> int:
    try:
        return names.index(name)
    except ValueError as exc:
        raise ValueError(f"未知 {label}: {name}，可选值: {names}") from exc
