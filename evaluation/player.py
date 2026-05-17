"""MuJoCo 重构轨迹播放器。

职责：
    将原始机器人和四路重构机器人写入多实例 MuJoCo model 的 qpos。
前置条件：
    scene XML 的机器人实例顺序与 `instance_names` 一致。
后置条件：
    可 headless 跑完整 FK，也可打开 viewer 播放。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np

from motion_fsq_reconstruction.evaluation.human_overlay import (
    compute_global_joint_positions,
    draw_human_skeleton,
)
from motion_fsq_reconstruction.evaluation.motion import PlaybackMotion
from motion_fsq_reconstruction.evaluation.reconstruction import ReconstructionResult


@dataclass(frozen=True)
class MujocoPlaybackConfig:
    """MuJoCo 播放配置。"""

    xml_path: Path
    show_viewer: bool
    loop: bool = False
    speed: float = 1.0
    instance_spacing: float = 2.0


class MujocoReconstructionPlayer:
    """MuJoCo 多机器人重构播放器。"""

    INSTANCE_NAMES = [
        "original",
        "actor_robot",
        "actor_human",
        "critic_robot",
        "critic_human",
    ]

    def __init__(
        self,
        *,
        config: MujocoPlaybackConfig,
        playback_motion: PlaybackMotion,
        reconstruction: ReconstructionResult,
    ) -> None:
        self._config = config
        self._motion = playback_motion
        self._reconstruction = reconstruction

    def play(self) -> None:
        """播放或 headless 推进完整轨迹。"""

        import mujoco

        model = mujoco.MjModel.from_xml_path(str(self._config.xml_path))
        data = mujoco.MjData(model)
        qpos_per_instance = 7 + len(self._reconstruction.robot_joint_names)
        expected_nq = qpos_per_instance * len(self.INSTANCE_NAMES)
        if model.nq != expected_nq:
            raise ValueError(f"scene nq={model.nq}，期望 {expected_nq}。")

        if not self._config.show_viewer:
            for frame_idx in range(self._reconstruction.num_frames):
                self._write_frame(data.qpos, frame_idx)
                mujoco.mj_forward(model, data)
            return

        import mujoco.viewer

        human_positions = None
        if self._motion.human_local_transforms is not None and self._motion.human_parent_indices is not None:
            human_positions = compute_global_joint_positions(
                self._motion.human_local_transforms,
                self._motion.human_parent_indices,
            )
        with mujoco.viewer.launch_passive(model, data) as viewer:
            _set_default_camera(mujoco, viewer)
            frame_idx = 0
            frame_period = 1.0 / max(self._motion.fps * max(self._config.speed, 1.0e-6), 1.0e-6)
            while viewer.is_running():
                start_time = time.time()
                self._write_frame(data.qpos, frame_idx)
                mujoco.mj_forward(model, data)
                if human_positions is not None and self._motion.human_parent_indices is not None:
                    draw_human_skeleton(
                        mujoco,
                        viewer,
                        human_positions[frame_idx],
                        self._motion.human_parent_indices,
                    )
                viewer.sync()
                frame_idx += 1
                if frame_idx >= self._reconstruction.num_frames:
                    if not self._config.loop:
                        break
                    frame_idx = 0
                sleep_time = frame_period - (time.time() - start_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def _write_frame(self, qpos: np.ndarray, frame_idx: int) -> None:
        root_pos = self._motion.robot_root_pos[frame_idx]
        root_quat = self._motion.robot_root_quat_wxyz[frame_idx]
        joint_sets = [
            self._reconstruction.original_robot_joint_pos,
            self._reconstruction.actor_robot_recon_joint_pos,
            self._reconstruction.actor_human_recon_joint_pos,
            self._reconstruction.critic_robot_recon_joint_pos,
            self._reconstruction.critic_human_recon_joint_pos,
        ]
        qpos_per_instance = 7 + len(self._reconstruction.robot_joint_names)
        for index, joint_pos in enumerate(joint_sets):
            start = index * qpos_per_instance
            offset_x = (index - (len(joint_sets) - 1) / 2.0) * self._config.instance_spacing
            qpos[start : start + 3] = root_pos + np.asarray([offset_x, 0.0, 0.0], dtype=np.float32)
            qpos[start + 3 : start + 7] = root_quat
            qpos[start + 7 : start + qpos_per_instance] = joint_pos[frame_idx]


def _set_default_camera(mujoco_module: object, viewer: object) -> None:
    viewer.cam.distance = 5.5
    viewer.cam.azimuth = 135.0
    viewer.cam.elevation = -18.0
    viewer.cam.fixedcamid = -1
    viewer.cam.type = mujoco_module.mjtCamera.mjCAMERA_FREE
