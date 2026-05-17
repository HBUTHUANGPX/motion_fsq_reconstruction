"""MuJoCo viewer 中的 human skeleton overlay。

职责：
    将 human_local_transforms 转成全局 skeleton，并绘制为 viewer user scene。
前置条件：
    local transform quaternion 使用 xyzw 顺序，符合现有 soma-retargeter npz。
后置条件：
    viewer.user_scn 中写入球和连线几何体。
"""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_global_joint_positions(local_transforms: np.ndarray, parent_indices: np.ndarray) -> np.ndarray:
    """从局部 human transform 批量计算全局 joint position。"""

    num_frames, num_joints = local_transforms.shape[:2]
    positions = np.zeros((num_frames, num_joints, 3), dtype=np.float32)
    rotations = np.zeros((num_frames, num_joints, 4), dtype=np.float32)
    local_pos = local_transforms[..., :3]
    local_quat = local_transforms[..., 3:7]
    for joint_idx in range(num_joints):
        parent_idx = int(parent_indices[joint_idx])
        if parent_idx < 0:
            positions[:, joint_idx] = local_pos[:, joint_idx]
            rotations[:, joint_idx] = local_quat[:, joint_idx]
            continue
        positions[:, joint_idx] = positions[:, parent_idx] + _quat_rotate_batch(
            rotations[:, parent_idx],
            local_pos[:, joint_idx],
        )
        rotations[:, joint_idx] = _quat_mul_batch(rotations[:, parent_idx], local_quat[:, joint_idx])
    return positions


def draw_human_skeleton(
    mujoco_module: Any,
    viewer: Any,
    positions: np.ndarray,
    parent_indices: np.ndarray,
) -> None:
    """在 MuJoCo viewer 中绘制一帧 human skeleton。"""

    scene = viewer.user_scn
    scene.ngeom = 0
    joint_rgba = np.asarray([1.0, 0.82, 0.1, 0.9], dtype=np.float32)
    bone_rgba = np.asarray([0.25, 0.9, 1.0, 0.72], dtype=np.float32)
    for joint_idx, position in enumerate(positions):
        if scene.ngeom >= scene.maxgeom:
            return
        _draw_sphere(mujoco_module, scene, position, 0.025, joint_rgba)
        parent_idx = int(parent_indices[joint_idx])
        if parent_idx >= 0 and scene.ngeom < scene.maxgeom:
            _draw_line(mujoco_module, scene, positions[parent_idx], position, 0.008, bone_rgba)


def _draw_sphere(mujoco_module: Any, scene: Any, position: np.ndarray, radius: float, rgba: np.ndarray) -> None:
    geom = scene.geoms[scene.ngeom]
    mujoco_module.mjv_initGeom(
        geom,
        type=mujoco_module.mjtGeom.mjGEOM_SPHERE,
        size=np.asarray([radius, 0.0, 0.0], dtype=np.float64),
        pos=np.asarray(position, dtype=np.float64),
        mat=np.eye(3, dtype=np.float64).reshape(-1),
        rgba=rgba,
    )
    scene.ngeom += 1


def _draw_line(
    mujoco_module: Any,
    scene: Any,
    start: np.ndarray,
    end: np.ndarray,
    width: float,
    rgba: np.ndarray,
) -> None:
    geom = scene.geoms[scene.ngeom]
    mujoco_module.mjv_initGeom(
        geom,
        type=mujoco_module.mjtGeom.mjGEOM_CAPSULE,
        size=np.zeros(3, dtype=np.float64),
        pos=np.zeros(3, dtype=np.float64),
        mat=np.eye(3, dtype=np.float64).reshape(-1),
        rgba=rgba,
    )
    mujoco_module.mjv_connector(
        geom,
        type=mujoco_module.mjtGeom.mjGEOM_CAPSULE,
        width=width,
        from_=np.asarray(start, dtype=np.float64),
        to=np.asarray(end, dtype=np.float64),
    )
    scene.ngeom += 1


def _quat_mul_batch(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = np.moveaxis(q1, -1, 0)
    x2, y2, z2, w2 = np.moveaxis(q2, -1, 0)
    return np.stack(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ),
        axis=-1,
    ).astype(np.float32, copy=False)


def _quat_rotate_batch(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    q_xyz = quat[..., :3]
    qw = quat[..., 3:4]
    uv = np.cross(q_xyz, vec)
    uuv = np.cross(q_xyz, uv)
    return (vec + 2.0 * (qw * uv + uuv)).astype(np.float32, copy=False)
