"""离线 latent 导出命令。

职责：
    从训练 checkpoint 导出全帧 quantized latent npz。
前置条件：
    checkpoint、config 与 motion 数据一致。
后置条件：
    输出 npz 包含 actor/critic 四路 latent 和 metadata。
"""

from __future__ import annotations

import argparse

from motion_fsq_reconstruction.config import load_config
from motion_fsq_reconstruction.export import LatentExporter


def main() -> None:
    """latent 导出 CLI 主入口。"""

    parser = argparse.ArgumentParser(description="导出离线 DualFSQ latent")
    parser.add_argument("--checkpoint", required=True, help="训练 checkpoint 路径")
    parser.add_argument("--config", required=True, help="训练使用的 YAML 配置路径")
    parser.add_argument("--output", required=True, help="输出 npz 路径")
    parser.add_argument("--device", default="cpu", help="导出设备，默认 cpu")
    parser.add_argument("--batch-size", type=int, default=None, help="导出 batch size")
    args = parser.parse_args()

    config = load_config(args.config)
    output = LatentExporter.from_checkpoint(
        args.checkpoint,
        config,
        device=args.device,
    ).export(args.output, batch_size=args.batch_size)
    print(f"latent file: {output}")


if __name__ == "__main__":
    main()
