"""MuJoCo 播放所需 motion 字段读取。

职责：
    读取 root pose、human skeleton 和 MuJoCo 播放需要的 metadata。
前置条件：
    输入 npz 至少包含 robot_root_pos、robot_root_quat、robot_joint_pos。
后置条件：
    返回 numpy 数组，quaternion 保持 MuJoCo 可用的 wxyz。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass
class PlaybackMotion:
    """MuJoCo 播放 motion 数据。"""

    robot_root_pos: np.ndarray
    robot_root_quat_wxyz: np.ndarray
    robot_joint_pos: np.ndarray
    robot_joint_names: list[str]
    fps: float
    scalar_first: bool
    human_local_transforms: np.ndarray | None
    human_parent_indices: np.ndarray | None
    human_joint_names: list[str]
    motion_paths: list[str]

    @property
    def num_frames(self) -> int:
        """返回帧数。"""

        return int(self.robot_root_pos.shape[0])


def load_playback_motion(paths: Sequence[str | Path], *, max_frames: int | None = None) -> PlaybackMotion:
    """读取并拼接一个或多个 npz motion。"""

    if not paths:
        raise ValueError("至少需要一个 motion 文件用于 MuJoCo 播放。")
    root_pos_list: list[np.ndarray] = []
    root_quat_list: list[np.ndarray] = []
    joint_pos_list: list[np.ndarray] = []
    human_local_list: list[np.ndarray] = []
    robot_joint_names: list[str] | None = None
    human_joint_names: list[str] | None = None
    human_parent_indices: np.ndarray | None = None
    fps: float | None = None
    scalar_first: bool | None = None
    motion_paths: list[str] = []

    for raw_path in paths:
        path = Path(raw_path)
        with np.load(path, allow_pickle=True) as data:
            file_fps = float(np.asarray(data["fps"]).item())
            fps = file_fps if fps is None else fps
            if float(fps) != file_fps:
                raise ValueError(f"{path} 的 fps={file_fps} 与前面文件不一致。")
            file_scalar_first = _read_scalar_first(data)
            scalar_first = file_scalar_first if scalar_first is None else scalar_first
            file_robot_joint_names = [str(value) for value in np.asarray(data["robot_joint_names"]).tolist()]
            robot_joint_names = file_robot_joint_names if robot_joint_names is None else robot_joint_names
            if robot_joint_names != file_robot_joint_names:
                raise ValueError(f"{path} 的 robot_joint_names 与前面文件不一致。")
            root_pos_list.append(np.asarray(data["robot_root_pos"], dtype=np.float32))
            root_quat_list.append(_to_wxyz(np.asarray(data["robot_root_quat"], dtype=np.float32), file_scalar_first))
            joint_pos_list.append(np.asarray(data["robot_joint_pos"], dtype=np.float32))
            if "human_local_transforms" in data and "human_parent_indices" in data:
                file_human_joint_names = [str(value) for value in np.asarray(data["human_joint_names"]).tolist()]
                human_joint_names = file_human_joint_names if human_joint_names is None else human_joint_names
                if human_joint_names != file_human_joint_names:
                    raise ValueError(f"{path} 的 human_joint_names 与前面文件不一致。")
                file_parent_indices = np.asarray(data["human_parent_indices"], dtype=np.int32)
                human_parent_indices = file_parent_indices if human_parent_indices is None else human_parent_indices
                if not np.array_equal(human_parent_indices, file_parent_indices):
                    raise ValueError(f"{path} 的 human_parent_indices 与前面文件不一致。")
                human_local_list.append(np.asarray(data["human_local_transforms"], dtype=np.float32))
            motion_paths.append(str(path))

    motion = PlaybackMotion(
        robot_root_pos=np.concatenate(root_pos_list, axis=0),
        robot_root_quat_wxyz=np.concatenate(root_quat_list, axis=0),
        robot_joint_pos=np.concatenate(joint_pos_list, axis=0),
        robot_joint_names=robot_joint_names or [],
        fps=float(fps or 30.0),
        scalar_first=bool(scalar_first if scalar_first is not None else True),
        human_local_transforms=np.concatenate(human_local_list, axis=0) if human_local_list else None,
        human_parent_indices=human_parent_indices,
        human_joint_names=human_joint_names or [],
        motion_paths=motion_paths,
    )
    if max_frames is None:
        return motion
    frames = int(max_frames)
    motion.robot_root_pos = motion.robot_root_pos[:frames]
    motion.robot_root_quat_wxyz = motion.robot_root_quat_wxyz[:frames]
    motion.robot_joint_pos = motion.robot_joint_pos[:frames]
    if motion.human_local_transforms is not None:
        motion.human_local_transforms = motion.human_local_transforms[:frames]
    return motion


def _read_scalar_first(data: np.lib.npyio.NpzFile) -> bool:
    if "scalar_first" not in data:
        return True
    value = np.asarray(data["scalar_first"])
    return bool(value.item() if value.shape == () else value.reshape(-1)[0])


def _to_wxyz(quat: np.ndarray, scalar_first: bool) -> np.ndarray:
    if quat.shape[-1] != 4:
        raise ValueError(f"robot_root_quat 最后一维必须是 4，实际 shape={quat.shape}。")
    if scalar_first:
        return quat
    return quat[..., [3, 0, 1, 2]]
