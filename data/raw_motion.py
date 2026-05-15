"""严格 schema 的 npz motion 加载器。

职责：
    将多个 npz 文件加载成连续 GPU/CPU tensor，并统一 quaternion 为 wxyz。
前置条件：
    所有文件的 fps、robot/human 名称顺序一致。
后置条件：
    返回保留 clip 边界的 `RawMotionDataset`，供窗口采样避免跨 clip。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm


FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "joint_pos": ("joint_pos", "robot_joint_pos"),
    "joint_vel": ("joint_vel", "robot_joint_vel"),
    "body_pos_w": ("body_pos_w", "robot_body_pos"),
    "body_quat_w": ("body_quat_w", "robot_body_quat"),
    "body_lin_vel_w": ("body_lin_vel_w", "robot_body_lin_vel"),
    "body_ang_vel_w": ("body_ang_vel_w", "robot_body_ang_vel"),
    "human_body_pos_w": ("human_body_pos_w", "human_global_pos"),
    "human_body_quat_w": ("human_body_quat_w", "human_global_quat"),
    "human_joint_quat": ("human_joint_quat", "human_local_quat"),
}


@dataclass
class RawMotionDataset:
    """拼接后的 raw motion tensor。

    职责：
        保存离线 DualFSQ feature 构建所需的原始运动字段。
    前置条件：
        tensor 第一维均为总帧数。
    后置条件：
        `motion_lengths` 和 `motion_start_indices` 可还原 clip 边界。
    """

    fps: int
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    body_pos_w: torch.Tensor
    body_quat_w: torch.Tensor
    body_lin_vel_w: torch.Tensor
    body_ang_vel_w: torch.Tensor
    human_body_pos_w: torch.Tensor
    human_body_quat_w: torch.Tensor
    human_joint_quat: torch.Tensor
    robot_joint_names: list[str]
    robot_body_names: list[str]
    human_body_names: list[str]
    motion_lengths: torch.Tensor
    motion_start_indices: torch.Tensor
    motion_groups: list[str]
    motion_paths: list[str]

    @property
    def num_frames(self) -> int:
        """返回总帧数。"""

        return int(self.joint_pos.shape[0])


class RawMotionLoader:
    """npz motion 文件加载器。

    职责：
        读取字段、校验 schema、转换 quaternion 顺序并拼接 tensor。
    前置条件：
        `files` 非空且每个路径指向 npz 文件。
    后置条件：
        输出字段语义对齐在线 MotionLoader 的 `_motion_data_np_list_to_tensor`。
    """

    def __init__(
        self,
        files: Sequence[str | Path],
        groups: Sequence[str] | None = None,
        *,
        robot_body_names: Sequence[str] | None = None,
        robot_joint_names: Sequence[str] | None = None,
        desire_human_joint_names: Sequence[str] | None = None,
    ) -> None:
        if not files:
            raise ValueError("至少需要一个 motion npz 文件。")
        self._files = [Path(path) for path in files]
        if groups is None:
            self._groups = ["default"] * len(self._files)
        else:
            if len(groups) != len(self._files):
                raise ValueError("groups 长度必须与 files 一致。")
            self._groups = list(groups)
        self._target_robot_body_names = list(robot_body_names or [])
        self._target_robot_joint_names = list(robot_joint_names or [])
        self._target_human_joint_names = list(desire_human_joint_names or [])

    @property
    def files(self) -> list[Path]:
        """返回待加载文件列表副本。"""

        return list(self._files)

    def load(self, device: str | torch.device = "cpu", *, progress: bool = True) -> RawMotionDataset:
        """加载全部 motion 文件。

        前置条件：
            文件存在且包含必须字段。
        后置条件：
            返回 tensor 已位于 `device`。
        """

        arrays: dict[str, list[np.ndarray]] = {name: [] for name in FIELD_ALIASES}
        fps: int | None = None
        file_robot_joint_names: list[str] | None = None
        file_robot_body_names: list[str] | None = None
        file_human_body_names: list[str] | None = None
        robot_joint_names: list[str] | None = None
        robot_body_names: list[str] | None = None
        human_body_names: list[str] | None = None
        robot_joint_indices: list[int] | None = None
        robot_body_indices: list[int] | None = None
        human_joint_indices: list[int] | None = None
        lengths: list[int] = []

        iterator = _progress(self._files, progress=progress, desc="加载 DualFSQ motion", unit="file")
        for path in iterator:
            if not path.is_file():
                raise FileNotFoundError(f"motion 文件不存在: {path}")
            with np.load(path, allow_pickle=True) as data:
                file_fps = int(np.asarray(data["fps"]).item())
                fps = file_fps if fps is None else fps
                if file_fps != fps:
                    raise ValueError(f"所有 motion fps 必须一致，{path} 为 {file_fps}。")

                file_robot_joint_names = _check_names(
                    "robot_joint_names",
                    file_robot_joint_names,
                    _read_names(data, ("robot_joint_names", "joint_names")),
                    path,
                )
                file_robot_body_names = _check_names(
                    "robot_body_names",
                    file_robot_body_names,
                    _read_names(data, ("robot_body_names", "body_names")),
                    path,
                )
                file_human_body_names = _check_names(
                    "human_body_names",
                    file_human_body_names,
                    _read_names(data, ("human_body_names", "human_joint_names")),
                    path,
                )
                if robot_joint_indices is None:
                    robot_joint_names = self._target_robot_joint_names or list(file_robot_joint_names)
                    robot_joint_indices = _indices(file_robot_joint_names, robot_joint_names, "robot_joint_names")
                if robot_body_indices is None:
                    robot_body_names = self._target_robot_body_names or list(file_robot_body_names)
                    robot_body_indices = _indices(file_robot_body_names, robot_body_names, "robot_body_names")
                if human_joint_indices is None:
                    human_body_names = self._target_human_joint_names or list(file_human_body_names)
                    human_joint_indices = _indices(file_human_body_names, human_body_names, "human_joint_names")

                scalar_first = _read_scalar_first(data)
                for canonical, aliases in FIELD_ALIASES.items():
                    array = _read_field(data, aliases, path, canonical)
                    if canonical in ("joint_pos", "joint_vel"):
                        array = array[:, robot_joint_indices]
                    elif canonical in (
                        "body_pos_w",
                        "body_quat_w",
                        "body_lin_vel_w",
                        "body_ang_vel_w",
                    ):
                        array = array[:, robot_body_indices]
                    elif canonical in (
                        "human_body_pos_w",
                        "human_body_quat_w",
                        "human_joint_quat",
                    ):
                        array = array[:, human_joint_indices]
                    if canonical in ("body_quat_w", "human_body_quat_w", "human_joint_quat"):
                        array = _to_wxyz(array.astype(np.float32), scalar_first)
                    arrays[canonical].append(np.asarray(array, dtype=np.float32))
                lengths.append(int(arrays["joint_pos"][-1].shape[0]))

        starts = np.cumsum([0] + lengths[:-1], dtype=np.int64)
        tensor_kwargs = {"device": torch.device(device), "dtype": torch.float32}
        return RawMotionDataset(
            fps=int(fps or 0),
            joint_pos=torch.as_tensor(np.concatenate(arrays["joint_pos"], axis=0), **tensor_kwargs),
            joint_vel=torch.as_tensor(np.concatenate(arrays["joint_vel"], axis=0), **tensor_kwargs),
            body_pos_w=torch.as_tensor(np.concatenate(arrays["body_pos_w"], axis=0), **tensor_kwargs),
            body_quat_w=torch.as_tensor(np.concatenate(arrays["body_quat_w"], axis=0), **tensor_kwargs),
            body_lin_vel_w=torch.as_tensor(np.concatenate(arrays["body_lin_vel_w"], axis=0), **tensor_kwargs),
            body_ang_vel_w=torch.as_tensor(np.concatenate(arrays["body_ang_vel_w"], axis=0), **tensor_kwargs),
            human_body_pos_w=torch.as_tensor(
                np.concatenate(arrays["human_body_pos_w"], axis=0), **tensor_kwargs
            ),
            human_body_quat_w=torch.as_tensor(
                np.concatenate(arrays["human_body_quat_w"], axis=0), **tensor_kwargs
            ),
            human_joint_quat=torch.as_tensor(
                np.concatenate(arrays["human_joint_quat"], axis=0), **tensor_kwargs
            ),
            robot_joint_names=robot_joint_names or [],
            robot_body_names=robot_body_names or [],
            human_body_names=human_body_names or [],
            motion_lengths=torch.as_tensor(lengths, dtype=torch.long, device=device),
            motion_start_indices=torch.as_tensor(starts, dtype=torch.long, device=device),
            motion_groups=list(self._groups),
            motion_paths=[str(path) for path in self._files],
        )


def _read_field(
    data: np.lib.npyio.NpzFile,
    aliases: tuple[str, ...],
    path: Path,
    canonical: str,
) -> np.ndarray:
    for name in aliases:
        if name in data:
            return np.asarray(data[name])
    if canonical == "human_joint_quat" and "human_local_transforms" in data:
        return np.asarray(data["human_local_transforms"])[..., 3:7]
    raise KeyError(f"{path} 缺少字段 {aliases}。")


def _read_names(data: np.lib.npyio.NpzFile, names: tuple[str, ...]) -> list[str]:
    for name in names:
        if name in data:
            return [str(value) for value in np.asarray(data[name]).tolist()]
    raise KeyError(f"缺少名字字段，期望之一: {names}。")


def _check_names(name: str, expected: list[str] | None, current: list[str], path: Path) -> list[str]:
    if expected is None:
        return current
    if expected != current:
        raise ValueError(f"{path} 的 {name} 与第一份文件不一致。")
    return expected


def _indices(source_names: list[str], target_names: list[str], label: str) -> list[int]:
    indices: list[int] = []
    for name in target_names:
        try:
            indices.append(source_names.index(name))
        except ValueError as exc:
            raise ValueError(f"{label} 缺少 {name}，可选值: {source_names}") from exc
    return indices


def _read_scalar_first(data: np.lib.npyio.NpzFile) -> bool:
    if "scalar_first" not in data:
        return False
    value = np.asarray(data["scalar_first"])
    return bool(value.item() if value.shape == () else value.reshape(-1)[0])


def _to_wxyz(quat: np.ndarray, scalar_first: bool) -> np.ndarray:
    if quat.shape[-1] != 4:
        raise ValueError(f"quaternion 最后一维必须是 4，实际 shape={quat.shape}。")
    if scalar_first:
        return quat
    return quat[..., [3, 0, 1, 2]]


def _progress(iterable: Any, *, progress: bool, **kwargs: Any) -> Any:
    if not progress:
        return iterable
    return tqdm(iterable, disable=False, dynamic_ncols=True, **kwargs)
