"""motion 文件来源解析器。

职责：
    将直接文件、目录或 YAML 来源解析为有序 npz 列表。
前置条件：
    输入路径可以不存在于配置解析时，但运行前必须存在。
后置条件：
    返回文件路径与 group 列表长度一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ResolvedMotionSources:
    """解析后的 motion 文件和 group 名称。"""

    paths: list[Path]
    groups: list[str]


class MotionSourceResolver:
    """解析 motion 来源的轻量服务类。"""

    def __init__(
        self,
        *,
        files: list[str] | None = None,
        dirs: list[str] | None = None,
        exclude_files: list[str] | None = None,
        exclude_dirs: list[str] | None = None,
        motion_yaml: str | None = None,
    ) -> None:
        self._files = [Path(path) for path in files or []]
        self._dirs = [Path(path) for path in dirs or []]
        self._exclude_files = {Path(path).resolve() for path in exclude_files or []}
        self._exclude_dirs = {Path(path).resolve() for path in exclude_dirs or []}
        self._motion_yaml = Path(motion_yaml) if motion_yaml else None

    def resolve(self, groups: list[str] | None = None) -> ResolvedMotionSources:
        """解析并返回 npz 文件。

        前置条件：
            至少一种来源能解析到文件。
        后置条件：
            文件按路径字符串排序，便于复现实验。
        """

        pairs: list[tuple[Path, str]] = []
        pairs.extend((path, "default") for path in self._files)
        for directory in self._dirs:
            pairs.extend((path, directory.name or "default") for path in directory.rglob("*.npz"))
        if self._motion_yaml is not None:
            pairs.extend(self._resolve_yaml(self._motion_yaml))

        if groups:
            group_set = set(groups)
            pairs = [(path, group) for path, group in pairs if group in group_set]

        filtered = []
        for path, group in pairs:
            resolved_path = path.resolve()
            if resolved_path in self._exclude_files:
                continue
            if any(resolved_path.is_relative_to(directory) for directory in self._exclude_dirs):
                continue
            if path.suffix == ".npz":
                filtered.append((path, group))
        filtered = sorted(dict.fromkeys(filtered).keys(), key=lambda item: str(item[0]))
        if not filtered:
            raise ValueError("未解析到任何 npz motion 文件。")
        return ResolvedMotionSources(
            paths=[path for path, _ in filtered],
            groups=[group for _, group in filtered],
        )

    def _resolve_yaml(self, path: Path) -> list[tuple[Path, str]]:
        with path.open("r", encoding="utf-8") as file:
            payload = yaml.safe_load(file) or {}
        pairs: list[tuple[Path, str]] = []
        self._collect_yaml_paths(payload, path.parent, "default", pairs)
        return pairs

    def _collect_yaml_paths(
        self,
        value: Any,
        base_dir: Path,
        group: str,
        pairs: list[tuple[Path, str]],
    ) -> None:
        if isinstance(value, str) and value.endswith(".npz"):
            candidate = Path(value)
            pairs.append((candidate if candidate.is_absolute() else base_dir / candidate, group))
            return
        if isinstance(value, list):
            for item in value:
                self._collect_yaml_paths(item, base_dir, group, pairs)
            return
        if isinstance(value, dict):
            for key, item in value.items():
                next_group = str(key) if isinstance(item, (list, tuple, dict)) else group
                self._collect_yaml_paths(item, base_dir, next_group, pairs)
