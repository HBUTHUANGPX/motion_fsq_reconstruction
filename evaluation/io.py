"""评估结果保存。

职责：
    将重构轨迹和指标写入磁盘。
前置条件：
    输出目录可创建。
后置条件：
    生成 reconstructions.npz 与 metrics.json。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from motion_fsq_reconstruction.evaluation.motion import PlaybackMotion
from motion_fsq_reconstruction.evaluation.reconstruction import ReconstructionResult


def save_reconstruction_npz(
    output_path: str | Path,
    *,
    result: ReconstructionResult,
    playback_motion: PlaybackMotion,
) -> Path:
    """保存 MuJoCo 可播放的重构 npz。"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = result.num_frames
    np.savez(
        path,
        original_robot_joint_pos=result.original_robot_joint_pos,
        actor_robot_recon_joint_pos=result.actor_robot_recon_joint_pos,
        actor_human_recon_joint_pos=result.actor_human_recon_joint_pos,
        critic_robot_recon_joint_pos=result.critic_robot_recon_joint_pos,
        critic_human_recon_joint_pos=result.critic_human_recon_joint_pos,
        robot_root_pos=playback_motion.robot_root_pos[:frames],
        robot_root_quat=playback_motion.robot_root_quat_wxyz[:frames],
        robot_joint_names=np.asarray(result.robot_joint_names, dtype=object),
        fps=np.asarray(playback_motion.fps, dtype=np.float32),
        motion_lengths=result.motion_lengths,
        motion_paths=np.asarray(result.motion_paths, dtype=object),
        feature_schema=np.asarray(result.feature_schema, dtype=object),
    )
    return path


def save_metrics_json(output_path: str | Path, metrics: dict[str, Any]) -> Path:
    """保存 JSON 指标文件。"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path
