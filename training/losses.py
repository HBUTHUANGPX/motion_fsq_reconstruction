"""DualFSQ 训练损失。

职责：
    计算 actor/critic 两套 FSQ-VAE 的 MSE 组合损失并聚合日志项。
前置条件：
    输出来自 `DualFSQTrainingModule`。
后置条件：
    总损失可反向传播，日志项为 tensor 标量。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from motion_fsq_reconstruction.models.dual_fsq import DualFSQOutput
from motion_fsq_reconstruction.models.training_module import DualFSQTrainingOutput


@dataclass
class DualFSQLossOutput:
    """训练损失输出。"""

    total: torch.Tensor
    terms: dict[str, torch.Tensor]


class DualFSQLoss:
    """actor/critic DualFSQ MSE 组合损失。"""

    def __init__(
        self,
        *,
        robot_recon: float,
        human_recon: float,
        latent_align: float,
        cycle_latent: float,
    ) -> None:
        self._weights = {
            "robot_recon": float(robot_recon),
            "human_recon": float(human_recon),
            "latent_align": float(latent_align),
            "cycle_latent": float(cycle_latent),
        }

    @property
    def weights(self) -> dict[str, float]:
        """返回损失权重副本。"""

        return dict(self._weights)

    def __call__(
        self,
        output: DualFSQTrainingOutput,
        *,
        actor_robot_target: torch.Tensor,
        critic_robot_target: torch.Tensor,
    ) -> DualFSQLossOutput:
        """计算聚合损失。"""

        actor_total, actor_terms = self._single_loss(output.actor, actor_robot_target)
        critic_total, critic_terms = self._single_loss(output.critic, critic_robot_target)
        terms: dict[str, torch.Tensor] = {}
        for name in actor_terms:
            terms[name] = 0.5 * (actor_terms[name] + critic_terms[name])
            terms[f"actor_{name}"] = actor_terms[name]
            terms[f"critic_{name}"] = critic_terms[name]
        terms["actor_total"] = actor_total
        terms["critic_total"] = critic_total
        return DualFSQLossOutput(total=0.5 * (actor_total + critic_total), terms=terms)

    def _single_loss(
        self,
        output: DualFSQOutput,
        robot_target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        terms = {
            "robot_recon": F.mse_loss(output.robot_recon_from_robot, robot_target),
            "human_recon": F.mse_loss(output.robot_recon_from_human, robot_target),
            "latent_align": F.mse_loss(output.q_human, output.q_robot),
            "cycle_latent": F.mse_loss(output.q_cycle, output.q_human),
        }
        total = sum(self._weights[name] * value for name, value in terms.items())
        return total, terms
