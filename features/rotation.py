"""wxyz quaternion 与 6D rotation 工具。

职责：
    提供与在线环境相同语义的 quaternion 归一化、旋转、相对位姿和 6D 表示。
前置条件：
    输入 quaternion 最后一维为 `[w, x, y, z]`。
后置条件：
    输出 tensor 保持输入 batch 维度。
"""

from __future__ import annotations

import torch


def normalize_quat_wxyz(quat: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """归一化 wxyz quaternion。"""

    return quat / quat.norm(dim=-1, keepdim=True).clamp_min(eps)


def quat_to_matrix_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """将 wxyz quaternion 转换为旋转矩阵。

    前置条件：
        `quat.shape[-1] == 4`。
    后置条件：
        返回 shape 为 `quat.shape[:-1] + (3, 3)` 的旋转矩阵。
    """

    quat = normalize_quat_wxyz(quat)
    w, x, y, z = quat.unbind(dim=-1)
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    row0 = torch.stack((ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)), dim=-1)
    row1 = torch.stack((2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)), dim=-1)
    row2 = torch.stack((2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def quat_to_rot6d_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """将 quaternion 转换为在线 `rot6d_from_quat` 相同顺序的 6D 表示。"""

    quat = normalize_quat_wxyz(quat)
    r, i, j, k = torch.unbind(quat, dim=-1)
    two_s = 2.0 / (quat * quat).sum(dim=-1)
    return torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
        ),
        dim=-1,
    )


def quat_conjugate_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """返回 wxyz quaternion 共轭。"""

    return torch.cat((quat[..., :1], -quat[..., 1:]), dim=-1)


def quat_multiply_wxyz(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """计算两个 wxyz quaternion 的乘积。"""

    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def quat_rotate_wxyz(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """将 vector 按 quaternion 旋转。"""

    quat = normalize_quat_wxyz(quat)
    q_vec = quat[..., 1:]
    q_w = quat[..., :1]
    uv = torch.cross(q_vec, vector, dim=-1)
    uuv = torch.cross(q_vec, uv, dim=-1)
    return vector + 2.0 * (q_w * uv + uuv)


def quat_inverse_rotate_wxyz(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """将 world frame 位移旋转到 quaternion 表示的 local frame。"""

    return quat_rotate_wxyz(quat_conjugate_wxyz(normalize_quat_wxyz(quat)), vector)


def subtract_frame_transforms_wxyz(
    parent_pos: torch.Tensor,
    parent_quat: torch.Tensor,
    child_pos: torch.Tensor,
    child_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算 child frame 在 parent frame 下的相对位姿。

    前置条件：
        `parent_*` 与 `child_*` 的 batch 维度可广播或相同。
    后置条件：
        返回 `(child_pos_in_parent, child_quat_in_parent)`。
    """

    parent_inv = quat_conjugate_wxyz(normalize_quat_wxyz(parent_quat))
    rel_pos = quat_rotate_wxyz(parent_inv, child_pos - parent_pos)
    rel_quat = quat_multiply_wxyz(parent_inv, normalize_quat_wxyz(child_quat))
    return rel_pos, normalize_quat_wxyz(rel_quat)
