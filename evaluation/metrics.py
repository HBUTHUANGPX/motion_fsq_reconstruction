"""重构误差指标。

职责：
    计算原始 robot joint position 与四路重构结果之间的误差。
前置条件：
    原始和重构数组 shape 一致。
后置条件：
    返回可写入 JSON 的普通字典。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from motion_fsq_reconstruction.evaluation.reconstruction import ReconstructionResult


@dataclass
class ReconstructionMetrics:
    """重构指标计算器。"""

    include_per_joint: bool = False

    def compute(self, result: ReconstructionResult) -> dict[str, Any]:
        """计算四路重构的 joint error。"""

        target = result.original_robot_joint_pos
        paths = {
            "actor_robot": result.actor_robot_recon_joint_pos,
            "actor_human": result.actor_human_recon_joint_pos,
            "critic_robot": result.critic_robot_recon_joint_pos,
            "critic_human": result.critic_human_recon_joint_pos,
        }
        output: dict[str, Any] = {
            "num_frames": result.num_frames,
            "robot_joint_names": list(result.robot_joint_names),
            "paths": {},
        }
        for name, value in paths.items():
            output["paths"][name] = self._compute_one(target, value, result.robot_joint_names)
        return output

    def _compute_one(self, target: np.ndarray, value: np.ndarray, joint_names: list[str]) -> dict[str, Any]:
        diff = np.asarray(value, dtype=np.float64) - np.asarray(target, dtype=np.float64)
        mse = np.mean(np.square(diff))
        mae = np.mean(np.abs(diff))
        result: dict[str, Any] = {
            "joint_mse": float(mse),
            "joint_rmse": float(np.sqrt(mse)),
            "joint_mae": float(mae),
            "joint_max_abs": float(np.max(np.abs(diff))),
        }
        if self.include_per_joint:
            per_joint_mse = np.mean(np.square(diff), axis=0)
            result["per_joint_mse"] = {
                name: float(value)
                for name, value in zip(joint_names, per_joint_mse, strict=False)
            }
        return result
