"""离线 DualFSQ 训练命令。

职责：
    解析 CLI 参数、加载配置、应用覆盖项并启动训练。
前置条件：
    `--config` 指向有效 YAML 文件。
后置条件：
    训练输出目录下写入 checkpoint 和 TensorBoard log。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from motion_fsq_reconstruction.config import load_config
from motion_fsq_reconstruction.training import DualFSQTrainer


def main() -> None:
    """训练 CLI 主入口。"""

    parser = argparse.ArgumentParser(description="训练离线 DualFSQ motion 重构模型")
    parser.add_argument("--config", required=True, help="YAML 配置路径")
    parser.add_argument("--device", default=None, help="覆盖训练设备，例如 cuda 或 cpu")
    parser.add_argument("--run-name", default=None, help="覆盖输出 run name")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.device is not None:
        config.train.device = args.device
    if args.run_name is not None:
        config.output.run_name = args.run_name
    latest = DualFSQTrainer(config).train()
    print(f"latest checkpoint: {latest}")


if __name__ == "__main__":
    main()
