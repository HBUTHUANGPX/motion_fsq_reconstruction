"""离线 token 导出命令。

职责：
    从训练 checkpoint 为每个 motion 导出同目录 token npz。
前置条件：
    checkpoint、config 与 motion 数据一致。
后置条件：
    每个原始 motion 同目录生成 `xxxx_token.npz`。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from motion_fsq_reconstruction.config import load_config
from motion_fsq_reconstruction.export import LatentExporter


def main() -> None:
    """token 导出 CLI 主入口。"""

    parser = argparse.ArgumentParser(description="导出离线 DualFSQ token")
    parser.add_argument("--checkpoint", default="motion_fsq_reconstruction/checkpoints/20260517_175626/epoch_1000.pt", help="训练 checkpoint 路径")
    parser.add_argument("--config", default="motion_fsq_reconstruction/configs/g1_dual_fsq.yaml", help="训练使用的 YAML 配置路径")
    parser.add_argument("--device", default="cuda", help="导出设备，默认 cpu")
    parser.add_argument("--batch-size", type=int, default=None, help="导出 batch size")
    args = parser.parse_args()

    config = load_config(args.config)
    exporter = LatentExporter.from_checkpoint(
        args.checkpoint,
        config,
        device=args.device,
    )
    outputs = exporter.export_next_to_motion_files(batch_size=args.batch_size)
    print(f"token files: {len(outputs)}")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
