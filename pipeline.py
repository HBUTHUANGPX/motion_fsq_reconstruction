"""训练和导出共享的构建流程。

职责：
    从配置解析 motion、构建 raw dataset、feature bundle、window buffer 和模型。
前置条件：
    配置中的路径、anchor 名称和模型维度有效。
后置条件：
    返回对象可被 trainer 或 latent exporter 直接使用。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from motion_fsq_reconstruction.config.schema import MotionFSQReconstructionConfig
from motion_fsq_reconstruction.data import (
    MotionSourceResolver,
    MotionWindowBuffer,
    RawMotionDataset,
    RawMotionLoader,
    ResolvedMotionSources,
)
from motion_fsq_reconstruction.features import (
    DualFSQFeatureBuilder,
    DualFSQFeatureBundle,
    FeatureBuilderConfig,
)
from motion_fsq_reconstruction.models import DualFSQAutoEncoder, DualFSQTrainingModule, build_quantizer


@dataclass
class MotionRuntimeBundle:
    """训练或导出需要的数据运行时对象。"""

    raw: RawMotionDataset
    features: DualFSQFeatureBundle
    buffer: MotionWindowBuffer

    @property
    def window_size(self) -> int:
        """返回窗口帧数。"""

        return self.buffer.window_size

    @property
    def actor_robot_input_dim(self) -> int:
        """返回 actor robot 展平窗口维度。"""

        return self.features.schema.actor_robot_feature_dim * self.window_size

    @property
    def actor_human_input_dim(self) -> int:
        """返回 actor human 展平窗口维度。"""

        return self.features.schema.actor_human_feature_dim * self.window_size

    @property
    def critic_robot_input_dim(self) -> int:
        """返回 critic robot 展平窗口维度。"""

        return self.features.schema.critic_robot_feature_dim * self.window_size

    @property
    def critic_human_input_dim(self) -> int:
        """返回 critic human 展平窗口维度。"""

        return self.features.schema.critic_human_feature_dim * self.window_size


def resolve_motion_sources(config: MotionFSQReconstructionConfig) -> ResolvedMotionSources:
    """根据配置解析 npz 文件来源。"""

    resolver = MotionSourceResolver(
        files=config.data.files,
        dirs=config.data.dirs,
        exclude_files=config.data.exclude_files,
        exclude_dirs=config.data.exclude_dirs,
        motion_yaml=config.data.motion_yaml,
    )
    return resolver.resolve(groups=config.data.groups or None)


def build_motion_runtime(
    config: MotionFSQReconstructionConfig,
    *,
    device: str | torch.device,
    progress: bool,
    sources: ResolvedMotionSources | None = None,
) -> MotionRuntimeBundle:
    """构建 raw、feature 和窗口 buffer。"""

    resolved = sources or resolve_motion_sources(config)
    raw = RawMotionLoader(
        resolved.paths,
        groups=resolved.groups,
        robot_body_names=config.features.robot_body_names,
        robot_joint_names=config.features.robot_joint_names,
        desire_human_joint_names=config.features.desire_human_joint_names,
    ).load(
        device=device,
        progress=progress,
    )
    features = DualFSQFeatureBuilder(
        FeatureBuilderConfig(
            robot_anchor_body=config.features.robot_anchor_body,
            robot_body_names=config.features.robot_body_names,
            robot_joint_names=config.features.robot_joint_names,
            desire_human_joint_names=config.features.desire_human_joint_names,
            human_anchor_body=config.features.human_anchor_body,
            human_body_names=config.features.human_body_names,
        )
    ).build(raw)
    buffer = MotionWindowBuffer(
        features=features,
        motion_lengths=raw.motion_lengths,
        history=config.train.history,
        future=config.train.future,
        device=device,
    )
    return MotionRuntimeBundle(raw=raw, features=features, buffer=buffer)


def build_training_module(
    config: MotionFSQReconstructionConfig,
    runtime: MotionRuntimeBundle,
) -> DualFSQTrainingModule:
    """根据 runtime 维度构建 actor/critic 双 DualFSQ 模型。"""

    return DualFSQTrainingModule(
        actor_dual_fsq=_build_autoencoder(
            config,
            robot_input_dim=runtime.actor_robot_input_dim,
            human_input_dim=runtime.actor_human_input_dim,
        ),
        critic_dual_fsq=_build_autoencoder(
            config,
            robot_input_dim=runtime.critic_robot_input_dim,
            human_input_dim=runtime.critic_human_input_dim,
        ),
    )


def _build_autoencoder(
    config: MotionFSQReconstructionConfig,
    *,
    robot_input_dim: int,
    human_input_dim: int,
) -> DualFSQAutoEncoder:
    quantizer = build_quantizer(
        quantizer_type=config.model.quantizer.type,
        levels=config.model.quantizer.levels,
        ifsq_boundary_fn=config.model.quantizer.ifsq_boundary_fn,
        ifsq_boundary_scale=config.model.quantizer.ifsq_boundary_scale,
    )
    return DualFSQAutoEncoder(
        robot_input_dim=robot_input_dim,
        human_input_dim=human_input_dim,
        latent_dim=config.model.latent_dim,
        robot_encoder_hidden_dims=config.model.robot_encoder_hidden_dims,
        human_encoder_hidden_dims=config.model.human_encoder_hidden_dims,
        decoder_hidden_dims=config.model.decoder_hidden_dims,
        quantizer=quantizer,
        activation=config.model.activation,
    )
