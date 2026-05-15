"""actor/critic 双 DualFSQ 训练封装。

职责：
    组合两套 DualFSQAutoEncoder，保持与在线 ActorCriticDualFSQ 的结构一致。
前置条件：
    四路输入维度分别与构造时声明一致。
后置条件：
    forward 返回 actor 和 critic 两套输出。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from motion_fsq_reconstruction.models.dual_fsq import DualFSQAutoEncoder, DualFSQOutput


@dataclass
class DualFSQTrainingOutput:
    """actor/critic 双任务 forward 输出。"""

    actor: DualFSQOutput
    critic: DualFSQOutput


class DualFSQTrainingModule(nn.Module):
    """封装 actor_dual_fsq 与 critic_dual_fsq。"""

    def __init__(
        self,
        *,
        actor_dual_fsq: DualFSQAutoEncoder,
        critic_dual_fsq: DualFSQAutoEncoder,
    ) -> None:
        super().__init__()
        self.actor_dual_fsq = actor_dual_fsq
        self.critic_dual_fsq = critic_dual_fsq

    def forward(
        self,
        actor_robot_window: torch.Tensor,
        actor_human_window: torch.Tensor,
        critic_robot_window: torch.Tensor,
        critic_human_window: torch.Tensor,
    ) -> DualFSQTrainingOutput:
        """执行 actor/critic 两套 DualFSQ forward。"""

        return DualFSQTrainingOutput(
            actor=self.actor_dual_fsq(actor_robot_window, actor_human_window),
            critic=self.critic_dual_fsq(critic_robot_window, critic_human_window),
        )
