"""MuJoCo 重构评估命令。

职责：
    从 DualFSQ checkpoint 生成四路重构轨迹，保存指标，并可打开 MuJoCo viewer 对比播放。
前置条件：
    checkpoint、config、motion npz 与 G1 XML 路径有效。
后置条件：
    输出 reconstructions.npz、metrics.json 和 scene.xml。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from motion_fsq_reconstruction.config import load_config
from motion_fsq_reconstruction.data import ResolvedMotionSources
from motion_fsq_reconstruction.evaluation.io import save_metrics_json, save_reconstruction_npz
from motion_fsq_reconstruction.evaluation.metrics import ReconstructionMetrics
from motion_fsq_reconstruction.evaluation.motion import load_playback_motion
from motion_fsq_reconstruction.evaluation.player import MujocoPlaybackConfig, MujocoReconstructionPlayer
from motion_fsq_reconstruction.evaluation.reconstruction import ReconstructionEvaluator
from motion_fsq_reconstruction.evaluation.scene import MujocoMultiRobotSceneBuilder
from motion_fsq_reconstruction.pipeline import resolve_motion_sources


def main() -> None:
    """MuJoCo 评估 CLI 主入口。"""

    parser = argparse.ArgumentParser(description="MuJoCo 播放和评估 DualFSQ 重构效果")
    parser.add_argument("--checkpoint", required=True, help="训练 checkpoint 路径")
    parser.add_argument("--config", required=True, help="训练使用的 YAML 配置路径")
    parser.add_argument("--xml", required=True, help="单机器人 MuJoCo XML 路径")
    parser.add_argument("--output", required=True, help="评估输出目录")
    parser.add_argument("--motion", default=None, help="单个 motion npz 路径")
    parser.add_argument("--all-motions", action="store_true", help="评估 config 解析到的全部 motion")
    parser.add_argument("--device", default="cpu", help="推理设备")
    parser.add_argument("--batch-size", type=int, default=None, help="重构 batch size")
    parser.add_argument("--max-frames", type=int, default=None, help="最多评估帧数，主要用于 smoke test")
    parser.add_argument("--show-viewer", action="store_true", help="打开 MuJoCo viewer")
    parser.add_argument("--no-viewer", action="store_true", help="不打开 MuJoCo viewer")
    parser.add_argument("--loop", action="store_true", help="viewer 播放结束后循环")
    parser.add_argument("--speed", type=float, default=1.0, help="viewer 播放速度倍率")
    parser.add_argument("--instance-spacing", type=float, default=2.0, help="并排机器人 x 方向间距")
    args = parser.parse_args()

    if args.motion is None and not args.all_motions:
        raise ValueError("必须传入 --motion 或 --all-motions。")
    if args.motion is not None and args.all_motions:
        raise ValueError("--motion 与 --all-motions 不能同时使用。")
    if args.show_viewer and args.no_viewer:
        raise ValueError("--show-viewer 与 --no-viewer 不能同时使用。")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    config.train.device = args.device
    sources = _resolve_sources(config, motion=args.motion, all_motions=args.all_motions)
    playback_motion = load_playback_motion(sources.paths, max_frames=args.max_frames)
    evaluator = ReconstructionEvaluator.from_checkpoint(
        args.checkpoint,
        config,
        device=args.device,
        sources=sources,
        progress=config.train.progress,
    )
    reconstruction = evaluator.reconstruct(batch_size=args.batch_size, max_frames=args.max_frames)
    _validate_joint_names(playback_motion.robot_joint_names, reconstruction.robot_joint_names)

    metrics = ReconstructionMetrics(include_per_joint=True).compute(reconstruction)
    metrics["motion_paths"] = list(reconstruction.motion_paths)
    metrics["root_pose_policy"] = "MuJoCo 播放复用原始 robot_root_pos / robot_root_quat。"
    scene_xml = output_dir / "scene.xml"
    MujocoMultiRobotSceneBuilder(
        args.xml,
        robot_joint_names=reconstruction.robot_joint_names,
        instance_names=MujocoReconstructionPlayer.INSTANCE_NAMES,
        instance_spacing=args.instance_spacing,
    ).write(scene_xml)
    reconstruction_path = save_reconstruction_npz(
        output_dir / "reconstructions.npz",
        result=reconstruction,
        playback_motion=playback_motion,
    )
    metrics_path = save_metrics_json(output_dir / "metrics.json", metrics)
    show_viewer = bool(args.show_viewer and not args.no_viewer)
    MujocoReconstructionPlayer(
        config=MujocoPlaybackConfig(
            xml_path=scene_xml,
            show_viewer=show_viewer,
            loop=args.loop,
            speed=args.speed,
            instance_spacing=args.instance_spacing,
        ),
        playback_motion=playback_motion,
        reconstruction=reconstruction,
    ).play()
    print(f"scene xml: {scene_xml}")
    print(f"reconstructions: {reconstruction_path}")
    print(f"metrics: {metrics_path}")


def _resolve_sources(config, *, motion: str | None, all_motions: bool) -> ResolvedMotionSources:
    if all_motions:
        return resolve_motion_sources(config)
    if motion is None:
        raise ValueError("缺少 motion 路径。")
    return ResolvedMotionSources(paths=[Path(motion)], groups=["eval"])


def _validate_joint_names(playback_names: list[str], reconstruction_names: list[str]) -> None:
    if playback_names == reconstruction_names:
        return
    raise ValueError(
        "播放 npz 的 robot_joint_names 与重构 feature schema 不一致，"
        f"playback={playback_names}, reconstruction={reconstruction_names}。"
    )


if __name__ == "__main__":
    main()
