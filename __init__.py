"""离线 DualFSQ motion 重构训练包。

职责：
    提供与在线 mimic 环境 DualFSQ feature 契约一致的离线训练、checkpoint
    保存和 latent 导出能力。
前置条件：
    调用方提供包含 robot/human motion 字段的 npz 文件。
后置条件：
    包内公开入口保持独立，不修改旧的 motion_reconstruction 包。
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
