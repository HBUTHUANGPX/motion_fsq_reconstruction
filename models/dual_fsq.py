"""DualFSQ 自编码器模型。

职责：
    提供 robot encoder、human encoder、共享 quantizer、robot decoder。
前置条件：
    输入已经展平为 `[B, window_size * feature_dim]`。
后置条件：
    输出包含重构结果、量化 latent 和 cycle latent。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn


@dataclass
class DualFSQOutput:
    """单套 actor 或 critic DualFSQ 输出。"""

    q_robot: torch.Tensor
    q_human: torch.Tensor
    q_cycle: torch.Tensor
    robot_recon_from_robot: torch.Tensor
    robot_recon_from_human: torch.Tensor


class DualFSQAutoEncoder(nn.Module):
    """双 encoder、共享量化器、单 robot decoder 的自编码器。"""

    def __init__(
        self,
        *,
        robot_input_dim: int,
        human_input_dim: int,
        latent_dim: int,
        robot_encoder_hidden_dims: Sequence[int],
        human_encoder_hidden_dims: Sequence[int],
        decoder_hidden_dims: Sequence[int],
        quantizer: nn.Module,
        activation: str = "elu",
    ) -> None:
        super().__init__()
        self.robot_input_dim = int(robot_input_dim)
        self.human_input_dim = int(human_input_dim)
        self.latent_dim = int(latent_dim)
        self.embedding_dim = self.latent_dim
        self.robot_encoder = _make_mlp(
            self.robot_input_dim,
            list(robot_encoder_hidden_dims),
            self.latent_dim,
            activation,
        )
        self.human_encoder = _make_mlp(
            self.human_input_dim,
            list(human_encoder_hidden_dims),
            self.latent_dim,
            activation,
        )
        self.quantizer = quantizer
        self.decoder = _make_mlp(
            self.latent_dim,
            list(decoder_hidden_dims),
            self.robot_input_dim,
            activation,
        )

    def encode_robot(self, robot_window: torch.Tensor) -> torch.Tensor:
        """编码并量化 robot window，返回 quantized latent。"""

        return self.quantizer(self.robot_encoder(robot_window))["z_q"]

    def encode_human(self, human_window: torch.Tensor) -> torch.Tensor:
        """编码并量化 human window，返回 quantized latent。"""

        return self.quantizer(self.human_encoder(human_window))["z_q"]

    def decode_robot(self, quantized_latent: torch.Tensor) -> torch.Tensor:
        """将 quantized latent 解码为 robot window。"""

        return self.decoder(quantized_latent)

    def forward(self, robot_window: torch.Tensor, human_window: torch.Tensor) -> DualFSQOutput:
        """执行完整 DualFSQ forward。"""

        q_robot = self.encode_robot(robot_window)
        q_human = self.encode_human(human_window)
        robot_recon_from_robot = self.decode_robot(q_robot)
        robot_recon_from_human = self.decode_robot(q_human)
        z_cycle = self.robot_encoder(robot_recon_from_human)
        q_cycle = self.quantizer(z_cycle)["z_q"]
        return DualFSQOutput(
            q_robot=q_robot,
            q_human=q_human,
            q_cycle=q_cycle,
            robot_recon_from_robot=robot_recon_from_robot,
            robot_recon_from_human=robot_recon_from_human,
        )


def _make_mlp(input_dim: int, hidden_dims: list[int], output_dim: int, activation: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = int(input_dim)
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(last_dim, int(hidden_dim)))
        layers.append(_activation(activation))
        last_dim = int(hidden_dim)
    layers.append(nn.Linear(last_dim, int(output_dim)))
    return nn.Sequential(*layers)


def _activation(name: str) -> nn.Module:
    normalized = name.lower().strip()
    if normalized == "elu":
        return nn.ELU()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "tanh":
        return nn.Tanh()
    raise ValueError(f"不支持的 activation: {name}。")
