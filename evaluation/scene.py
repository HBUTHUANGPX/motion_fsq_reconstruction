"""MuJoCo 多机器人场景生成。

职责：
    从单机器人 MJCF 生成播放专用多机器人 MJCF。
前置条件：
    源 XML 包含一个带 freejoint 的机器人 body。
后置条件：
    生成 XML 可由 MuJoCo 加载，并包含每个实例独立 qpos。
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

INSTANCE_RGBA = [
    "0.82 0.82 0.82 1.0",
    "0.20 0.55 1.00 0.78",
    "1.00 0.48 0.20 0.78",
    "0.35 0.90 0.45 0.78",
    "0.82 0.40 1.00 0.78",
]


class MujocoMultiRobotSceneBuilder:
    """多机器人播放场景构建器。"""

    def __init__(
        self,
        source_xml: str | Path,
        *,
        robot_joint_names: list[str],
        instance_names: list[str],
        instance_spacing: float = 2.0,
    ) -> None:
        self._source_xml = Path(source_xml)
        self._robot_joint_names = list(robot_joint_names)
        self._instance_names = list(instance_names)
        self._instance_spacing = float(instance_spacing)
        if not self._instance_names:
            raise ValueError("至少需要一个机器人实例。")

    @property
    def qpos_per_instance(self) -> int:
        """返回每个机器人实例的 qpos 维度。"""

        return 7 + len(self._robot_joint_names)

    def write(self, output_xml: str | Path) -> Path:
        """写入多机器人 MJCF。"""

        output = Path(output_xml)
        output.parent.mkdir(parents=True, exist_ok=True)
        tree = ET.parse(self._source_xml)
        source_root = tree.getroot()
        root = ET.Element("mujoco", {"model": f"{source_root.get('model', 'motion')}_multi_reconstruction"})
        self._copy_compiler(source_root, root)
        self._copy_top_level(source_root, root, "option")
        self._copy_top_level(source_root, root, "default")
        self._copy_top_level(source_root, root, "asset")
        source_worldbody = source_root.find("worldbody")
        if source_worldbody is None:
            raise ValueError(f"{self._source_xml} 缺少 worldbody。")
        robot_body = self._find_robot_body(source_worldbody)
        worldbody = ET.SubElement(root, "worldbody")
        for child in source_worldbody:
            if child.tag != "body":
                worldbody.append(deepcopy(child))
        for index, instance_name in enumerate(self._instance_names):
            body = deepcopy(robot_body)
            offset = (index - (len(self._instance_names) - 1) / 2.0) * self._instance_spacing
            body.set("pos", f"{offset:.6g} 0 0")
            _prefix_tree(body, f"{instance_name}_", prefix_references=False)
            _tint_robot_geoms(body, INSTANCE_RGBA[index % len(INSTANCE_RGBA)])
            worldbody.append(body)
        ET.indent(root)
        ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
        return output

    def _copy_compiler(self, source_root: ET.Element, target_root: ET.Element) -> None:
        for compiler in source_root.findall("compiler"):
            copied = deepcopy(compiler)
            meshdir = copied.get("meshdir")
            if meshdir and not Path(meshdir).is_absolute():
                copied.set("meshdir", str((self._source_xml.parent / meshdir).resolve()))
            target_root.append(copied)

    def _copy_top_level(self, source_root: ET.Element, target_root: ET.Element, tag: str) -> None:
        for node in source_root.findall(tag):
            target_root.append(deepcopy(node))

    def _find_robot_body(self, worldbody: ET.Element) -> ET.Element:
        bodies = [child for child in worldbody if child.tag == "body"]
        if not bodies:
            raise ValueError(f"{self._source_xml} 的 worldbody 中没有机器人 body。")
        for body in bodies:
            if body.find(".//freejoint") is not None or body.find(".//joint[@type='free']") is not None:
                return body
        return bodies[0]


def _prefix_tree(element: ET.Element, prefix: str, *, prefix_references: bool) -> None:
    name = element.get("name")
    if name:
        element.set("name", prefix + name)
    reference_attrs = {"joint", "body", "site"}
    for attr, value in list(element.attrib.items()):
        if attr in reference_attrs and prefix_references:
            element.set(attr, prefix + value)
    for child in element:
        _prefix_tree(child, prefix, prefix_references=prefix_references)


def _tint_robot_geoms(element: ET.Element, rgba: str) -> None:
    if element.tag == "geom":
        element.attrib.pop("material", None)
        element.set("rgba", rgba)
    for child in element:
        _tint_robot_geoms(child, rgba)
