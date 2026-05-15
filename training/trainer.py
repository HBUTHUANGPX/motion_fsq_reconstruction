"""离线 DualFSQ 训练器。

职责：
    编排数据加载、normalizer 拟合、模型训练、TensorBoard 和 checkpoint。
前置条件：
    配置可解析到至少一个 npz 文件。
后置条件：
    `train()` 返回 latest checkpoint 路径。
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from motion_fsq_reconstruction.config.schema import MotionFSQReconstructionConfig
from motion_fsq_reconstruction.models import DualFSQTrainingModule
from motion_fsq_reconstruction.pipeline import MotionRuntimeBundle, build_motion_runtime, build_training_module
from motion_fsq_reconstruction.training.checkpoint import save_checkpoint
from motion_fsq_reconstruction.training.losses import DualFSQLoss
from motion_fsq_reconstruction.training.normalization import WindowFeatureNormalizer


class NullSummaryWriter:
    """空 TensorBoard writer。

    职责：
        在 tensorboard 不可用时吞掉日志调用。
    前置条件：
        无。
    后置条件：
        调用不会产生副作用。
    """

    def add_scalar(self, *args: Any, **kwargs: Any) -> None:
        """忽略 scalar 写入。"""

        return None

    def add_histogram(self, *args: Any, **kwargs: Any) -> None:
        """忽略 histogram 写入。"""

        return None

    def close(self) -> None:
        """关闭 writer。"""

        return None


class DualFSQTrainer:
    """离线 DualFSQ trainer。"""

    def __init__(self, config: MotionFSQReconstructionConfig) -> None:
        self._config = config
        self._device = self._resolve_device(config.train.device)
        self._global_step = 0
        torch.manual_seed(config.train.seed)
        if self._device.type == "cuda":
            torch.cuda.manual_seed_all(config.train.seed)

        self._run_dir = self._make_run_dir()
        self._checkpoint_dir = self._run_dir / "checkpoints"
        self._log_dir = self._run_dir / "tb"
        self._writer = self._make_writer(self._log_dir)
        self._runtime = build_motion_runtime(
            config,
            device=self._device,
            progress=config.train.progress,
        )
        self._normalizers = self._fit_normalizers(self._runtime)
        self._model = build_training_module(config, self._runtime).to(self._device)
        self._optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )
        self._loss = DualFSQLoss(**config.loss.__dict__)

    @property
    def runtime(self) -> MotionRuntimeBundle:
        """返回训练 runtime。"""

        return self._runtime

    def train(self) -> Path:
        """运行完整训练并返回 latest checkpoint。

        前置条件：
            trainer 已完成初始化。
        后置条件：
            至少保存 `latest.pt`。
        """

        generator = torch.Generator(device=self._device)
        generator.manual_seed(self._config.train.seed)
        latest_path = self._checkpoint_dir / "latest.pt"
        epoch_iter = range(1, self._config.train.epochs + 1)
        epoch_bar = self._progress(epoch_iter, total=self._config.train.epochs, desc="DualFSQ epoch")
        try:
            for epoch in epoch_bar:
                start_time = time.time()
                totals: dict[str, float] = {"total": 0.0}
                batch_count = 0
                self._model.train()
                batch_bar = self._progress(
                    self._runtime.buffer.iter_epoch_batches(
                        self._config.train.batch_size,
                        generator=generator,
                    ),
                    total=self._runtime.buffer.num_batches(self._config.train.batch_size),
                    desc=f"epoch {epoch}",
                    leave=False,
                )
                for batch in batch_bar:
                    actor_robot = self._normalizers["actor_robot"](batch.actor_robot)
                    actor_human = self._normalizers["actor_human"](batch.actor_human)
                    critic_robot = self._normalizers["critic_robot"](batch.critic_robot)
                    critic_human = self._normalizers["critic_human"](batch.critic_human)
                    output = self._model(actor_robot, actor_human, critic_robot, critic_human)
                    loss_output = self._loss(
                        output,
                        actor_robot_target=actor_robot,
                        critic_robot_target=critic_robot,
                    )
                    self._optimizer.zero_grad(set_to_none=True)
                    loss_output.total.backward()
                    self._optimizer.step()

                    self._global_step += 1
                    batch_count += 1
                    totals["total"] += float(loss_output.total.detach().cpu())
                    for name, value in loss_output.terms.items():
                        totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())
                    if self._global_step % self._config.train.log_every_steps == 0:
                        self._log_step(loss_output.terms, loss_output.total, output)
                self._log_epoch(epoch, totals, max(batch_count, 1), time.time() - start_time)
                latest_path = self._save(epoch, "latest.pt")
                if (
                    self._config.train.checkpoint_interval_epochs > 0
                    and epoch % self._config.train.checkpoint_interval_epochs == 0
                ):
                    self._save(epoch, f"epoch_{epoch:04d}.pt")
                if hasattr(epoch_bar, "set_postfix"):
                    epoch_bar.set_postfix({"loss": f"{totals['total'] / max(batch_count, 1):.6f}"})
        finally:
            self._writer.close()
        return latest_path

    def _fit_normalizers(self, runtime: MotionRuntimeBundle) -> dict[str, WindowFeatureNormalizer]:
        window_size = runtime.window_size
        eps = self._config.train.normalizer_eps
        return {
            "actor_robot": WindowFeatureNormalizer.fit(
                runtime.features.actor_robot,
                window_size=window_size,
                eps=eps,
            ).to(self._device),
            "actor_human": WindowFeatureNormalizer.fit(
                runtime.features.actor_human,
                window_size=window_size,
                eps=eps,
            ).to(self._device),
            "critic_robot": WindowFeatureNormalizer.fit(
                runtime.features.critic_robot,
                window_size=window_size,
                eps=eps,
            ).to(self._device),
            "critic_human": WindowFeatureNormalizer.fit(
                runtime.features.critic_human,
                window_size=window_size,
                eps=eps,
            ).to(self._device),
        }

    def _save(self, epoch: int, filename: str) -> Path:
        return save_checkpoint(
            path=self._checkpoint_dir / filename,
            model=self._model,
            optimizer=self._optimizer,
            epoch=epoch,
            global_step=self._global_step,
            config=self._config.to_dict(),
            normalizers=self._normalizers,
            feature_schema=self._runtime.features.schema.to_dict(),
        )

    def _log_step(self, terms: dict[str, torch.Tensor], total: torch.Tensor, output: Any) -> None:
        self._writer.add_scalar("train/total", float(total.detach().cpu()), self._global_step)
        for name, value in terms.items():
            self._writer.add_scalar(f"train/{name}", float(value.detach().cpu()), self._global_step)
        self._writer.add_histogram("latent/actor_q_human", output.actor.q_human.detach().cpu(), self._global_step)
        self._writer.add_histogram("latent/critic_q_robot", output.critic.q_robot.detach().cpu(), self._global_step)

    def _log_epoch(self, epoch: int, totals: dict[str, float], batch_count: int, elapsed: float) -> None:
        for name, value in totals.items():
            self._writer.add_scalar(f"epoch/{name}", value / batch_count, epoch)
        self._writer.add_scalar("epoch/time_sec", elapsed, epoch)

    def _make_run_dir(self) -> Path:
        run_name = self._config.output.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(self._config.output.root_dir) / run_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _progress(self, iterable: Any, **kwargs: Any) -> Any:
        if not self._config.train.progress:
            return iterable
        return tqdm(iterable, dynamic_ncols=True, **kwargs)

    def _resolve_device(self, requested: str) -> torch.device:
        if requested.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(requested)

    @staticmethod
    def _make_writer(log_dir: Path) -> Any:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            return NullSummaryWriter()
        log_dir.mkdir(parents=True, exist_ok=True)
        return SummaryWriter(log_dir=str(log_dir), flush_secs=10)
