from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from motion_fsq_reconstruction.config.schema import (
    DataConfig,
    FeatureConfig,
    ModelConfig,
    MotionFSQReconstructionConfig,
    OutputConfig,
    QuantizerConfig,
    TrainConfig,
)
from motion_fsq_reconstruction.export.latent_exporter import LatentExporter
from motion_fsq_reconstruction.features.rotation import quat_to_rot6d_wxyz
from motion_fsq_reconstruction.pipeline import build_motion_runtime
from motion_fsq_reconstruction.training.trainer import DualFSQTrainer


def test_rot6d_matches_motion_command_flatten_order() -> None:
    quat = np.asarray([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)

    actual = quat_to_rot6d_wxyz(torch.as_tensor(quat)).cpu().numpy()

    expected = np.asarray([[0.0, 0.0, 1.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    np.testing.assert_allclose(actual, expected, atol=1.0e-6)


def test_feature_shapes_match_online_dual_fsq_schema(tmp_path: Path) -> None:
    motion_path = _write_motion_npz(tmp_path / "sample.npz")
    config = _make_config(tmp_path, motion_path)

    runtime = build_motion_runtime(config, device="cpu", progress=False)

    assert runtime.features.actor_robot.shape == (6, 8)
    assert runtime.features.actor_human.shape == (6, 12)
    assert runtime.features.critic_robot.shape == (6, 29)
    assert runtime.features.critic_human.shape == (6, 39)
    assert runtime.actor_robot_input_dim == 24
    assert runtime.actor_human_input_dim == 36
    assert runtime.critic_robot_input_dim == 87
    assert runtime.critic_human_input_dim == 117

    identity_rot6d = np.asarray([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    expected_actor_robot = np.concatenate(
        (
            identity_rot6d,
            np.linspace(0.0, 1.0, 12, dtype=np.float32).reshape(6, 2)[0],
        )
    )
    np.testing.assert_allclose(
        runtime.features.actor_robot[0].cpu().numpy(),
        expected_actor_robot,
        atol=1.0e-6,
    )
    expected_actor_human = np.concatenate(
        (
            identity_rot6d,
            np.asarray([0.06, 0.06, 0.06, 0.12, 0.12, 0.12], dtype=np.float32),
        )
    )
    np.testing.assert_allclose(
        runtime.features.actor_human[0].cpu().numpy(),
        expected_actor_human,
        atol=1.0e-6,
    )


def test_loader_reorders_names_like_online_motion_loader(tmp_path: Path) -> None:
    motion_path = _write_shuffled_motion_npz(tmp_path / "shuffled.npz")
    config = MotionFSQReconstructionConfig(
        data=DataConfig(files=[str(motion_path)]),
        features=FeatureConfig(
            robot_anchor_body="torso_link",
            robot_body_names=["torso_link", "pelvis"],
            robot_joint_names=["hip", "knee"],
            desire_human_joint_names=["Hips", "Chest", "HeadEnd"],
            human_anchor_body="Hips",
            human_body_names=["Chest", "HeadEnd"],
        ),
        train=TrainConfig(device="cpu", history=0, future=0, progress=False),
    )

    runtime = build_motion_runtime(config, device="cpu", progress=False)

    assert runtime.raw.robot_body_names == ["torso_link", "pelvis"]
    assert runtime.raw.robot_joint_names == ["hip", "knee"]
    assert runtime.raw.human_body_names == ["Hips", "Chest", "HeadEnd"]
    np.testing.assert_allclose(
        runtime.raw.joint_pos[0].cpu().numpy(),
        np.asarray([1.0, 2.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        runtime.raw.body_pos_w[0, :, 0].cpu().numpy(),
        np.asarray([10.0, 20.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        runtime.raw.human_body_pos_w[0, :, 0].cpu().numpy(),
        np.asarray([100.0, 200.0, 300.0], dtype=np.float32),
    )


def test_loader_prefers_human_local_transforms_for_joint_quat(tmp_path: Path) -> None:
    motion_path = _write_shuffled_motion_npz(tmp_path / "with_extra_joint_quat.npz")
    with np.load(motion_path, allow_pickle=True) as data:
        payload = {name: data[name] for name in data.files}
    wrong_joint_quat = np.zeros((2, 4, 4), dtype=np.float32)
    wrong_joint_quat[..., 2] = 1.0
    payload["human_joint_quat"] = wrong_joint_quat
    np.savez(motion_path, **payload)

    config = MotionFSQReconstructionConfig(
        data=DataConfig(files=[str(motion_path)]),
        features=FeatureConfig(
            robot_anchor_body="torso_link",
            robot_body_names=["torso_link", "pelvis"],
            robot_joint_names=["hip", "knee"],
            desire_human_joint_names=["Hips", "Chest", "HeadEnd"],
            human_anchor_body="Hips",
            human_body_names=["Chest", "HeadEnd"],
        ),
        train=TrainConfig(device="cpu", history=0, future=0, progress=False),
    )

    runtime = build_motion_runtime(config, device="cpu", progress=False)

    expected = np.zeros((3, 4), dtype=np.float32)
    expected[:, 0] = 1.0
    np.testing.assert_allclose(runtime.raw.human_joint_quat[0].cpu().numpy(), expected)


def test_training_and_latent_export_smoke(tmp_path: Path) -> None:
    motion_path = _write_motion_npz(tmp_path / "sample.npz")
    config = _make_config(tmp_path, motion_path)

    trainer = DualFSQTrainer(config)
    latest = trainer.train()

    output_path = tmp_path / "latents.npz"
    LatentExporter.from_checkpoint(latest, config, device="cpu").export(output_path)

    with np.load(output_path, allow_pickle=True) as data:
        assert data["actor_q_human"].shape == (6, 4)
        assert data["actor_q_robot"].shape == (6, 4)
        assert data["critic_q_human"].shape == (6, 4)
        assert data["critic_q_robot"].shape == (6, 4)
        assert data["motion_lengths"].tolist() == [6]
        assert len(data["motion_paths"].tolist()) == 1


def _make_config(tmp_path: Path, motion_path: Path) -> MotionFSQReconstructionConfig:
    return MotionFSQReconstructionConfig(
        data=DataConfig(files=[str(motion_path)]),
        features=FeatureConfig(
            robot_anchor_body="torso_link",
            robot_body_names=["torso_link", "left_link"],
            robot_joint_names=["joint0", "joint1"],
            desire_human_joint_names=["Hips", "Spine1", "LeftHand"],
            human_anchor_body="Hips",
            human_body_names=["Spine1", "LeftHand"],
        ),
        model=ModelConfig(
            latent_dim=4,
            robot_encoder_hidden_dims=[16],
            human_encoder_hidden_dims=[16],
            decoder_hidden_dims=[16],
            quantizer=QuantizerConfig(levels=8),
        ),
        train=TrainConfig(
            device="cpu",
            epochs=1,
            batch_size=2,
            history=1,
            future=1,
            progress=False,
            log_every_steps=100,
            checkpoint_interval_epochs=1,
        ),
        output=OutputConfig(root_dir=str(tmp_path / "runs"), run_name="smoke"),
    )


def _write_shuffled_motion_npz(path: Path) -> Path:
    frames = 2
    robot_body_names = np.asarray(["pelvis", "torso_link", "unused_body"], dtype=object)
    robot_joint_names = np.asarray(["knee", "unused_joint", "hip"], dtype=object)
    human_joint_names = np.asarray(["HeadEnd", "Hips", "Chest", "UnusedHuman"], dtype=object)

    robot_joint_pos = np.asarray([[2.0, 9.0, 1.0], [5.0, 9.0, 4.0]], dtype=np.float32)
    robot_joint_vel = np.zeros_like(robot_joint_pos)
    robot_body_pos = np.zeros((frames, 3, 3), dtype=np.float32)
    robot_body_pos[:, 0, 0] = 20.0
    robot_body_pos[:, 1, 0] = 10.0
    robot_body_pos[:, 2, 0] = 90.0
    robot_body_quat = np.zeros((frames, 3, 4), dtype=np.float32)
    robot_body_quat[..., 0] = 1.0

    human_global_pos = np.zeros((frames, 4, 3), dtype=np.float32)
    human_global_pos[:, 0, 0] = 300.0
    human_global_pos[:, 1, 0] = 100.0
    human_global_pos[:, 2, 0] = 200.0
    human_global_pos[:, 3, 0] = 900.0
    human_global_quat = np.zeros((frames, 4, 4), dtype=np.float32)
    human_global_quat[..., 0] = 1.0
    human_local_transforms = np.zeros((frames, 4, 7), dtype=np.float32)
    human_local_transforms[..., 3] = 1.0

    np.savez(
        path,
        fps=np.asarray(30),
        scalar_first=np.asarray(True),
        robot_joint_names=robot_joint_names,
        robot_body_names=robot_body_names,
        human_joint_names=human_joint_names,
        robot_joint_pos=robot_joint_pos,
        robot_joint_vel=robot_joint_vel,
        robot_body_pos=robot_body_pos,
        robot_body_quat=robot_body_quat,
        robot_body_lin_vel=np.zeros_like(robot_body_pos),
        robot_body_ang_vel=np.zeros_like(robot_body_pos),
        human_global_pos=human_global_pos,
        human_global_quat=human_global_quat,
        human_local_transforms=human_local_transforms,
    )
    return path


def _write_motion_npz(path: Path) -> Path:
    frames = 6
    robot_body_names = np.asarray(["torso_link", "left_link", "right_link"], dtype=object)
    robot_joint_names = np.asarray(["joint0", "joint1"], dtype=object)
    human_joint_names = np.asarray(["Hips", "Spine1", "LeftHand"], dtype=object)

    robot_body_pos = np.arange(frames * 3 * 3, dtype=np.float32).reshape(frames, 3, 3) / 100.0
    human_global_pos = np.arange(frames * 3 * 3, dtype=np.float32).reshape(frames, 3, 3) / 50.0
    identity_robot_quat = np.zeros((frames, 3, 4), dtype=np.float32)
    identity_robot_quat[..., 0] = 1.0
    identity_human_quat = np.zeros((frames, 3, 4), dtype=np.float32)
    identity_human_quat[..., 0] = 1.0
    human_local_transforms = np.zeros((frames, 3, 7), dtype=np.float32)
    human_local_transforms[..., 3] = 1.0

    np.savez(
        path,
        fps=np.asarray(30),
        scalar_first=np.asarray(True),
        robot_joint_names=robot_joint_names,
        robot_body_names=robot_body_names,
        human_joint_names=human_joint_names,
        robot_joint_pos=np.linspace(0.0, 1.0, frames * 2, dtype=np.float32).reshape(frames, 2),
        robot_joint_vel=np.zeros((frames, 2), dtype=np.float32),
        robot_body_pos=robot_body_pos,
        robot_body_quat=identity_robot_quat,
        robot_body_lin_vel=np.zeros((frames, 3, 3), dtype=np.float32),
        robot_body_ang_vel=np.zeros((frames, 3, 3), dtype=np.float32),
        human_global_pos=human_global_pos,
        human_global_quat=identity_human_quat,
        human_local_transforms=human_local_transforms,
    )
    return path
