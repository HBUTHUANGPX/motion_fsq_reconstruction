"""DualFSQ quantized latent 导出器。

职责：
    从离线训练 checkpoint 恢复模型，并为每个 motion frame 导出四路 q latent。
前置条件：
    checkpoint 与 config 的 feature schema 和模型维度一致。
后置条件：
    输出 npz 包含 actor/critic human/robot quantized latent 和 metadata。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from motion_fsq_reconstruction.config.schema import MotionFSQReconstructionConfig
from motion_fsq_reconstruction.models import DualFSQTrainingModule
from motion_fsq_reconstruction.pipeline import MotionRuntimeBundle, build_motion_runtime, build_training_module
from motion_fsq_reconstruction.training.checkpoint import load_checkpoint
from motion_fsq_reconstruction.training.normalization import WindowFeatureNormalizer


class LatentExporter:
    """checkpoint latent 导出服务。"""

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

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        config: MotionFSQReconstructionConfig,
        *,
        device: str | torch.device = "cpu",
    ) -> LatentExporter:
        """从 checkpoint 构建导出器。"""

        device_obj = torch.device(device)
        runtime = build_motion_runtime(config, device=device_obj, progress=False)
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

    def export(self, output_path: str | Path, *, batch_size: int | None = None) -> Path:
        """导出全帧 quantized latent。

        前置条件：
            checkpoint 已加载。
        后置条件：
            写入包含四路 latent 的 npz 文件。
        """

        batch_size = batch_size or self._config.train.batch_size
        centers = self._runtime.buffer.index.all_center_indices
        actor_q_human: torch.Tensor | None = None
        actor_q_robot: torch.Tensor | None = None
        critic_q_human: torch.Tensor | None = None
        critic_q_robot: torch.Tensor | None = None
        iterator = range(0, centers.numel(), batch_size)
        if self._config.train.progress:
            iterator = tqdm(iterator, dynamic_ncols=True, desc="导出 latent")
        with torch.inference_mode():
            for start in iterator:
                batch_centers = centers[start : start + batch_size]
                end = start + batch_centers.numel()
                batch = self._runtime.buffer.batch_from_centers(batch_centers, clamp_to_clip=True)
                actor_robot = self._normalizers["actor_robot"](batch.actor_robot)
                actor_human = self._normalizers["actor_human"](batch.actor_human)
                critic_robot = self._normalizers["critic_robot"](batch.critic_robot)
                critic_human = self._normalizers["critic_human"](batch.critic_human)
                q_ar = self._model.actor_dual_fsq.encode_robot(actor_robot)
                q_ah = self._model.actor_dual_fsq.encode_human(actor_human)
                q_cr = self._model.critic_dual_fsq.encode_robot(critic_robot)
                q_ch = self._model.critic_dual_fsq.encode_human(critic_human)
                if actor_q_human is None:
                    actor_q_human = torch.empty((centers.numel(), q_ah.shape[-1]), dtype=q_ah.dtype)
                    actor_q_robot = torch.empty((centers.numel(), q_ar.shape[-1]), dtype=q_ar.dtype)
                    critic_q_human = torch.empty((centers.numel(), q_ch.shape[-1]), dtype=q_ch.dtype)
                    critic_q_robot = torch.empty((centers.numel(), q_cr.shape[-1]), dtype=q_cr.dtype)
                actor_q_robot[start:end].copy_(q_ar.detach().cpu())
                actor_q_human[start:end].copy_(q_ah.detach().cpu())
                critic_q_robot[start:end].copy_(q_cr.detach().cpu())
                critic_q_human[start:end].copy_(q_ch.detach().cpu())

        if (
            actor_q_human is None
            or actor_q_robot is None
            or critic_q_human is None
            or critic_q_robot is None
        ):
            raise RuntimeError("没有可导出的 latent。")

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output,
            actor_q_human=actor_q_human.numpy(),
            actor_q_robot=actor_q_robot.numpy(),
            critic_q_human=critic_q_human.numpy(),
            critic_q_robot=critic_q_robot.numpy(),
            motion_lengths=self._runtime.raw.motion_lengths.detach().cpu().numpy(),
            motion_start_indices=self._runtime.raw.motion_start_indices.detach().cpu().numpy(),
            motion_paths=np.asarray(self._runtime.raw.motion_paths, dtype=object),
            feature_schema=np.asarray(self._runtime.features.schema.to_dict(), dtype=object),
            config=np.asarray(self._config.to_dict(), dtype=object),
        )
        return output
