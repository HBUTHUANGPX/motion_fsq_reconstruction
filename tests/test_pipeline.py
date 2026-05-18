from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
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
from motion_fsq_reconstruction.evaluation.reconstruction import (
    ReconstructionEvaluator,
    ReconstructionFeatureLayout,
    extract_current_joint_pos,
)
from motion_fsq_reconstruction.evaluation.scene import MujocoMultiRobotSceneBuilder
from motion_fsq_reconstruction.features.rotation import quat_to_rot6d_wxyz
from motion_fsq_reconstruction.pipeline import build_motion_runtime
from motion_fsq_reconstruction.data import ResolvedMotionSources
from motion_fsq_reconstruction.training.distributed import DistributedRuntime, shard_motion_sources
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

    exporter = LatentExporter.from_checkpoint(latest, config, device="cpu")

    token_paths = exporter.export_next_to_motion_files(batch_size=2)
    assert token_paths == [motion_path.with_name("sample_token.npz")]
    with np.load(token_paths[0], allow_pickle=True) as data:
        assert data["actor_q_human"].shape == (6, 4)
        assert data["actor_q_robot"].shape == (6, 4)
        assert data["critic_q_human"].shape == (6, 4)
        assert data["critic_q_robot"].shape == (6, 4)
        assert int(data["motion_length"]) == 6
        assert data["frame_indices"].tolist() == list(range(6))
        assert str(data["window_policy"].item()) == "clamp_to_clip"

    reconstruction = ReconstructionEvaluator.from_checkpoint(
        latest,
        config,
        device="cpu",
    ).reconstruct(batch_size=2)
    assert reconstruction.actor_robot_recon_joint_pos.shape == (6, 2)
    assert reconstruction.actor_human_recon_joint_pos.shape == (6, 2)
    assert reconstruction.critic_robot_recon_joint_pos.shape == (6, 2)
    assert reconstruction.critic_human_recon_joint_pos.shape == (6, 2)


def test_frame_balanced_sharding_covers_all_files(tmp_path: Path) -> None:
    paths = [
        _write_motion_npz(tmp_path / f"sample_{index}.npz", frames=frames)
        for index, frames in enumerate((3, 5, 8, 11))
    ]
    sources = ResolvedMotionSources(paths=paths, groups=["g"] * len(paths))
    shard0, info0 = shard_motion_sources(
        sources,
        runtime=DistributedRuntime(enabled=True, rank=0, world_size=2, local_rank=0),
        history=1,
        future=1,
    )
    shard1, info1 = shard_motion_sources(
        sources,
        runtime=DistributedRuntime(enabled=True, rank=1, world_size=2, local_rank=1),
        history=1,
        future=1,
    )

    set0 = {str(path) for path in shard0.paths}
    set1 = {str(path) for path in shard1.paths}
    assert set0.isdisjoint(set1)
    assert set0 | set1 == {str(path) for path in paths}
    assert info0.global_valid_frames == info1.global_valid_frames
    assert abs(info0.local_valid_frames - info1.local_valid_frames) <= 5


def test_distributed_cpu_training_smoke(tmp_path: Path) -> None:
    motion_paths = [
        _write_motion_npz(tmp_path / f"ddp_{index}.npz", frames=frames)
        for index, frames in enumerate((5, 6, 7, 8))
    ]
    config_path = tmp_path / "ddp_config.yaml"
    output_root = tmp_path / "runs"
    config_path.write_text(
        _ddp_config_text(motion_paths, output_root),
        encoding="utf-8",
    )

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        "motion_fsq_reconstruction/cli/train.py",
        "--config",
        str(config_path),
        "--device",
        "cpu",
        "--run-name",
        "ddp_cpu_smoke",
        "--distributed",
    ]
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    checkpoint = output_root / "ddp_cpu_smoke" / "checkpoints" / "latest.pt"
    assert checkpoint.is_file()
    token_paths = LatentExporter.from_checkpoint(
        checkpoint,
        _load_config_for_test(config_path),
        device="cpu",
    ).export_next_to_motion_files()
    assert len(token_paths) == 4
    total_frames = 0
    for token_path in token_paths:
        with np.load(token_path, allow_pickle=True) as data:
            total_frames += int(data["actor_q_human"].shape[0])
            assert data["actor_q_human"].shape[1] == 4
    assert total_frames == sum((5, 6, 7, 8))


def test_extract_current_joint_pos_from_actor_and_critic_windows() -> None:
    actor_window = torch.arange(2 * 3 * 8, dtype=torch.float32).reshape(2, 24)
    critic_window = torch.arange(2 * 3 * 11, dtype=torch.float32).reshape(2, 33)

    actor_joint = extract_current_joint_pos(
        actor_window,
        ReconstructionFeatureLayout(frame_dim=8, joint_start=6, joint_count=2, history=1),
    )
    critic_joint = extract_current_joint_pos(
        critic_window,
        ReconstructionFeatureLayout(frame_dim=11, joint_start=9, joint_count=2, history=1),
    )

    np.testing.assert_allclose(actor_joint.numpy(), np.asarray([[14.0, 15.0], [38.0, 39.0]]))
    np.testing.assert_allclose(critic_joint.numpy(), np.asarray([[20.0, 21.0], [53.0, 54.0]]))


def test_mujoco_multi_robot_scene_loads_with_expected_qpos(tmp_path: Path) -> None:
    mujoco = pytest.importorskip("mujoco")
    source_xml = _write_minimal_mujoco_xml(tmp_path / "single.xml")
    scene_xml = tmp_path / "scene.xml"

    MujocoMultiRobotSceneBuilder(
        source_xml,
        robot_joint_names=["joint0", "joint1"],
        instance_names=["original", "actor_robot", "actor_human"],
    ).write(scene_xml)

    model = mujoco.MjModel.from_xml_path(str(scene_xml))

    assert model.nq == 3 * (7 + 2)
    assert model.njnt == 3 * 3


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


def _write_motion_npz(path: Path, *, frames: int = 6) -> Path:
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
        robot_root_pos=np.zeros((frames, 3), dtype=np.float32),
        robot_root_quat=np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (frames, 1)),
        robot_joint_names=robot_joint_names,
        robot_body_names=robot_body_names,
        human_joint_names=human_joint_names,
        human_parent_indices=np.asarray([-1, 0, 1], dtype=np.int32),
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


def _write_minimal_mujoco_xml(path: Path) -> Path:
    path.write_text(
        """
<mujoco model="minimal_g1">
  <compiler angle="radian"/>
  <asset>
    <material name="body_mat" rgba="0.7 0.7 0.7 1"/>
  </asset>
  <worldbody>
    <light name="light" pos="0 0 3"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.2 0.2 0.2 1"/>
    <body name="pelvis" pos="0 0 1">
      <freejoint name="floating_base_joint"/>
      <geom name="pelvis_geom" type="sphere" size="0.1" material="body_mat"/>
      <body name="link0" pos="0 0 0.1">
        <joint name="joint0" type="hinge" axis="0 1 0"/>
        <geom name="link0_geom" type="capsule" fromto="0 0 0 0 0 0.2" size="0.03"/>
        <body name="link1" pos="0 0 0.2">
          <joint name="joint1" type="hinge" axis="1 0 0"/>
          <geom name="link1_geom" type="capsule" fromto="0 0 0 0 0 0.2" size="0.03"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="joint0_motor" joint="joint0"/>
  </actuator>
  <sensor>
    <jointpos name="joint0_pos" joint="joint0"/>
  </sensor>
</mujoco>
""".strip(),
        encoding="utf-8",
    )
    return path


def _ddp_config_text(paths: list[Path], output_root: Path) -> str:
    files = "\n".join(f"    - {path}" for path in paths)
    return f"""
data:
  files:
{files}

features:
  robot_anchor_body: torso_link
  robot_body_names: [torso_link, left_link]
  robot_joint_names: [joint0, joint1]
  desire_human_joint_names: [Hips, Spine1, LeftHand]
  human_anchor_body: Hips
  human_body_names: [Spine1, LeftHand]

model:
  latent_dim: 4
  robot_encoder_hidden_dims: [16]
  human_encoder_hidden_dims: [16]
  decoder_hidden_dims: [16]
  activation: elu
  quantizer:
    type: ifsq
    levels: 8
    ifsq_boundary_fn: sigmoid
    ifsq_boundary_scale: 1.6

loss:
  robot_recon: 1.0
  human_recon: 1.0
  latent_align: 1.0
  cycle_latent: 0.25

train:
  device: cpu
  epochs: 1
  batch_size: 2
  learning_rate: 0.0003
  weight_decay: 0.0001
  history: 1
  future: 1
  seed: 1
  log_every_steps: 100
  checkpoint_interval_epochs: 1
  normalizer_eps: 0.01
  progress: false

output:
  root_dir: {output_root}
  run_name: ddp_cpu_smoke
"""


def _load_config_for_test(path: Path) -> MotionFSQReconstructionConfig:
    from motion_fsq_reconstruction.config import load_config

    return load_config(path)
