"""DualFSQ 重构推理。

职责：
    从 checkpoint 恢复模型，并将 actor/critic 四路 encoder-decoder 输出转换为逐帧 robot joint position。
前置条件：
    checkpoint、配置、motion 数据的 feature schema 一致。
后置条件：
    返回可保存、可由 MuJoCo 播放的四路重构关节轨迹。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from motion_fsq_reconstruction.config.schema import MotionFSQReconstructionConfig
from motion_fsq_reconstruction.data import ResolvedMotionSources
from motion_fsq_reconstruction.models import DualFSQTrainingModule
from motion_fsq_reconstruction.pipeline import MotionRuntimeBundle, build_motion_runtime, build_training_module
from motion_fsq_reconstruction.training.checkpoint import load_checkpoint
from motion_fsq_reconstruction.training.normalization import WindowFeatureNormalizer


@dataclass(frozen=True)
class ReconstructionFeatureLayout:
    """展平 robot window 中 joint position 的位置信息。"""

    frame_dim: int
    joint_start: int
    joint_count: int
    history: int


@dataclass
class ReconstructionResult:
    """四路重构结果。

    职责：
        保存原始 robot joint position 和四路 reconstructed joint position。
    前置条件：
        所有数组第一维均为同一 motion 的帧数。
    后置条件：
        可直接写入 npz 或交给 MuJoCo player 播放。
    """

    original_robot_joint_pos: np.ndarray
    actor_robot_recon_joint_pos: np.ndarray
    actor_human_recon_joint_pos: np.ndarray
    critic_robot_recon_joint_pos: np.ndarray
    critic_human_recon_joint_pos: np.ndarray
    robot_joint_names: list[str]
    motion_paths: list[str]
    motion_lengths: np.ndarray
    feature_schema: dict[str, object]

    @property
    def num_frames(self) -> int:
        """返回重构总帧数。"""

        return int(self.original_robot_joint_pos.shape[0])


class ReconstructionEvaluator:
    """DualFSQ checkpoint 重构评估器。"""

    def __init__(
        self,
        *,
        model: DualFSQTrainingModule,
        runtime: MotionRuntimeBundle,
        normalizers: dict[str, WindowFeatureNormalizer],
        config: MotionFSQReconstructionConfig,
        device: str | torch.device,
    ) -> None:
        self._model = model
        self._runtime = runtime
        self._normalizers = normalizers
        self._config = config
        self._device = torch.device(device)

    @property
    def runtime(self) -> MotionRuntimeBundle:
        """返回评估 runtime。"""

        return self._runtime

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        config: MotionFSQReconstructionConfig,
        *,
        device: str | torch.device,
        sources: ResolvedMotionSources | None = None,
        progress: bool = False,
    ) -> ReconstructionEvaluator:
        """从 checkpoint 构建评估器。"""

        device_obj = torch.device(device)
        runtime = build_motion_runtime(config, device=device_obj, progress=progress, sources=sources)
        model = build_training_module(config, runtime).to(device_obj)
        checkpoint = load_checkpoint(checkpoint_path, map_location=device_obj)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        normalizers = {
            name: WindowFeatureNormalizer.from_state_dict(state).to(device_obj)
            for name, state in checkpoint["normalizers"].items()
        }
        return cls(
            model=model,
            runtime=runtime,
            normalizers=normalizers,
            config=config,
            device=device_obj,
        )

    def reconstruct(self, *, batch_size: int | None = None, max_frames: int | None = None) -> ReconstructionResult:
        """执行四路重构。

        前置条件：
            模型已加载并处于 eval 模式。
        后置条件：
            返回逐帧 robot joint position 重构结果。
        """

        batch_size = int(batch_size or self._config.train.batch_size)
        centers = self._runtime.buffer.index.all_center_indices
        if max_frames is not None:
            centers = centers[: int(max_frames)]
        actor_robot_layout = ReconstructionFeatureLayout(
            frame_dim=self._runtime.features.schema.actor_robot_feature_dim,
            joint_start=6,
            joint_count=len(self._runtime.raw.robot_joint_names),
            history=self._config.train.history,
        )
        critic_robot_layout = ReconstructionFeatureLayout(
            frame_dim=self._runtime.features.schema.critic_robot_feature_dim,
            joint_start=9,
            joint_count=len(self._runtime.raw.robot_joint_names),
            history=self._config.train.history,
        )
        result_arrays = _empty_result_arrays(
            centers.numel(),
            len(self._runtime.raw.robot_joint_names),
        )
        iterator = range(0, centers.numel(), batch_size)
        if self._config.train.progress:
            iterator = tqdm(iterator, dynamic_ncols=True, desc="重构 MuJoCo 评估轨迹")

        with torch.inference_mode():
            for start in iterator:
                batch_centers = centers[start : start + batch_size]
                end = start + batch_centers.numel()
                batch = self._runtime.buffer.batch_from_centers(batch_centers, clamp_to_clip=True)
                actor_robot = self._normalizers["actor_robot"](batch.actor_robot)
                actor_human = self._normalizers["actor_human"](batch.actor_human)
                critic_robot = self._normalizers["critic_robot"](batch.critic_robot)
                critic_human = self._normalizers["critic_human"](batch.critic_human)
                actor_output = self._model.actor_dual_fsq(actor_robot, actor_human)
                critic_output = self._model.critic_dual_fsq(critic_robot, critic_human)

                actor_robot_recon = self._normalizers["actor_robot"].denormalize(
                    actor_output.robot_recon_from_robot
                )
                actor_human_recon = self._normalizers["actor_robot"].denormalize(
                    actor_output.robot_recon_from_human
                )
                critic_robot_recon = self._normalizers["critic_robot"].denormalize(
                    critic_output.robot_recon_from_robot
                )
                critic_human_recon = self._normalizers["critic_robot"].denormalize(
                    critic_output.robot_recon_from_human
                )
                result_arrays["actor_robot"][start:end] = extract_current_joint_pos(
                    actor_robot_recon,
                    actor_robot_layout,
                ).detach().cpu().numpy()
                result_arrays["actor_human"][start:end] = extract_current_joint_pos(
                    actor_human_recon,
                    actor_robot_layout,
                ).detach().cpu().numpy()
                result_arrays["critic_robot"][start:end] = extract_current_joint_pos(
                    critic_robot_recon,
                    critic_robot_layout,
                ).detach().cpu().numpy()
                result_arrays["critic_human"][start:end] = extract_current_joint_pos(
                    critic_human_recon,
                    critic_robot_layout,
                ).detach().cpu().numpy()

        center_np = centers.detach().cpu().numpy()
        original = self._runtime.raw.joint_pos[centers].detach().cpu().numpy()
        motion_lengths = self._runtime.raw.motion_lengths.detach().cpu().numpy()
        if max_frames is not None:
            motion_lengths = np.asarray([center_np.shape[0]], dtype=np.int64)
        return ReconstructionResult(
            original_robot_joint_pos=original,
            actor_robot_recon_joint_pos=result_arrays["actor_robot"],
            actor_human_recon_joint_pos=result_arrays["actor_human"],
            critic_robot_recon_joint_pos=result_arrays["critic_robot"],
            critic_human_recon_joint_pos=result_arrays["critic_human"],
            robot_joint_names=list(self._runtime.raw.robot_joint_names),
            motion_paths=list(self._runtime.raw.motion_paths),
            motion_lengths=motion_lengths,
            feature_schema=self._runtime.features.schema.to_dict(),
        )


def extract_current_joint_pos(window: torch.Tensor, layout: ReconstructionFeatureLayout) -> torch.Tensor:
    """从展平 robot window 中取 current frame 的 joint position。"""

    reshaped = window.reshape(window.shape[0], -1, layout.frame_dim)
    current = reshaped[:, layout.history]
    return current[:, layout.joint_start : layout.joint_start + layout.joint_count]


def _empty_result_arrays(num_frames: int, joint_count: int) -> dict[str, np.ndarray]:
    return {
        "actor_robot": np.empty((num_frames, joint_count), dtype=np.float32),
        "actor_human": np.empty((num_frames, joint_count), dtype=np.float32),
        "critic_robot": np.empty((num_frames, joint_count), dtype=np.float32),
        "critic_human": np.empty((num_frames, joint_count), dtype=np.float32),
    }
