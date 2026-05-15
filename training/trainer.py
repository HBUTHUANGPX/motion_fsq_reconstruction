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
from torch.nn.parallel import DistributedDataParallel
from tqdm.auto import tqdm

from motion_fsq_reconstruction.config.schema import MotionFSQReconstructionConfig
from motion_fsq_reconstruction.models import DualFSQTrainingModule
from motion_fsq_reconstruction.pipeline import (
    MotionRuntimeBundle,
    build_motion_runtime,
    build_training_module,
    resolve_motion_sources,
)
from motion_fsq_reconstruction.training.checkpoint import save_checkpoint
from motion_fsq_reconstruction.training.distributed import (
    DistributedRuntime,
    average_epoch_totals,
    assert_same_object,
    fit_window_normalizer,
    max_int,
    resolve_training_device,
    shard_motion_sources,
)
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

    def __init__(
        self,
        config: MotionFSQReconstructionConfig,
        *,
        distributed: bool = False,
    ) -> None:
        self._config = config
        self._distributed = (
            DistributedRuntime.from_environment(config.train.device)
            if distributed
            else DistributedRuntime.disabled()
        )
        self._device = resolve_training_device(config.train.device, self._distributed)
        self._global_step = 0
        torch.manual_seed(config.train.seed + self._distributed.rank)
        if self._device.type == "cuda":
            torch.cuda.manual_seed_all(config.train.seed + self._distributed.rank)

        self._run_dir = self._make_run_dir()
        self._checkpoint_dir = self._run_dir / "checkpoints"
        self._log_dir = self._run_dir / "tb"
        self._writer = self._make_writer(self._log_dir)
        resolved_sources = resolve_motion_sources(config)
        self._sources, self._shard_info = shard_motion_sources(
            resolved_sources,
            runtime=self._distributed,
            history=config.train.history,
            future=config.train.future,
        )
        self._print(
            "[MotionFSQShard] "
            f"rank {self._distributed.rank}/{self._distributed.world_size} "
            f"loads {self._shard_info.local_file_count} files "
            f"valid_frames={self._shard_info.local_valid_frames}/"
            f"{self._shard_info.global_valid_frames}."
        )
        self._runtime = build_motion_runtime(
            config,
            device=self._device,
            progress=config.train.progress and self._distributed.is_main,
            sources=self._sources,
        )
        assert_same_object(
            self._runtime.features.schema.to_dict(),
            runtime=self._distributed,
            label="feature_schema",
        )
        self._normalizers = self._fit_normalizers(self._runtime)
        model = build_training_module(config, self._runtime).to(self._device)
        self._model: DualFSQTrainingModule | DistributedDataParallel
        if self._distributed.enabled:
            ddp_kwargs: dict[str, Any] = {}
            if self._device.type == "cuda":
                ddp_kwargs["device_ids"] = [self._distributed.local_rank]
                ddp_kwargs["output_device"] = self._distributed.local_rank
            self._model = DistributedDataParallel(model, **ddp_kwargs)
        else:
            self._model = model
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
        generator.manual_seed(self._config.train.seed + self._distributed.rank)
        latest_path = self._checkpoint_dir / "latest.pt"
        epoch_iter = range(1, self._config.train.epochs + 1)
        epoch_bar = self._progress(epoch_iter, total=self._config.train.epochs, desc="DualFSQ epoch")
        try:
            for epoch in epoch_bar:
                start_time = time.time()
                totals: dict[str, float] = {"total": 0.0}
                batch_count = 0
                self._model.train()
                epoch_num_batches = None
                if self._distributed.enabled:
                    epoch_num_batches = max_int(
                        self._runtime.buffer.num_batches(self._config.train.batch_size),
                        device=self._device,
                        runtime=self._distributed,
                    )
                batch_bar = self._progress(
                    self._runtime.buffer.iter_epoch_batches(
                        self._config.train.batch_size,
                        generator=generator,
                        num_batches=epoch_num_batches,
                    ),
                    total=epoch_num_batches or self._runtime.buffer.num_batches(self._config.train.batch_size),
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
                averaged = average_epoch_totals(
                    totals,
                    batch_count=max(batch_count, 1),
                    device=self._device,
                    runtime=self._distributed,
                )
                self._log_epoch(epoch, averaged, time.time() - start_time)
                if self._distributed.is_main:
                    latest_path = self._save(epoch, "latest.pt")
                if (
                    self._distributed.is_main
                    and
                    self._config.train.checkpoint_interval_epochs > 0
                    and epoch % self._config.train.checkpoint_interval_epochs == 0
                ):
                    self._save(epoch, f"epoch_{epoch:04d}.pt")
                if hasattr(epoch_bar, "set_postfix"):
                    epoch_bar.set_postfix({"loss": f"{averaged['total']:.6f}"})
                self._distributed.barrier()
        finally:
            self._writer.close()
            self._distributed.barrier()
            self._distributed.close()
        return latest_path

    def _fit_normalizers(self, runtime: MotionRuntimeBundle) -> dict[str, WindowFeatureNormalizer]:
        window_size = runtime.window_size
        eps = self._config.train.normalizer_eps
        return {
            "actor_robot": fit_window_normalizer(
                runtime.features.actor_robot,
                window_size=window_size,
                eps=eps,
                runtime=self._distributed,
            ).to(self._device),
            "actor_human": fit_window_normalizer(
                runtime.features.actor_human,
                window_size=window_size,
                eps=eps,
                runtime=self._distributed,
            ).to(self._device),
            "critic_robot": fit_window_normalizer(
                runtime.features.critic_robot,
                window_size=window_size,
                eps=eps,
                runtime=self._distributed,
            ).to(self._device),
            "critic_human": fit_window_normalizer(
                runtime.features.critic_human,
                window_size=window_size,
                eps=eps,
                runtime=self._distributed,
            ).to(self._device),
        }

    def _save(self, epoch: int, filename: str) -> Path:
        model = self._unwrap_model()
        return save_checkpoint(
            path=self._checkpoint_dir / filename,
            model=model,
            optimizer=self._optimizer,
            epoch=epoch,
            global_step=self._global_step,
            config=self._config.to_dict(),
            normalizers=self._normalizers,
            feature_schema=self._runtime.features.schema.to_dict(),
            metadata={
                "distributed_world_size": self._distributed.world_size,
                "distributed_shard_info": self._shard_info.to_dict(),
            },
        )

    def _log_step(self, terms: dict[str, torch.Tensor], total: torch.Tensor, output: Any) -> None:
        if not self._distributed.is_main:
            return
        self._writer.add_scalar("train/total", float(total.detach().cpu()), self._global_step)
        for name, value in terms.items():
            self._writer.add_scalar(f"train/{name}", float(value.detach().cpu()), self._global_step)
        self._writer.add_histogram("latent/actor_q_human", output.actor.q_human.detach().cpu(), self._global_step)
        self._writer.add_histogram("latent/critic_q_robot", output.critic.q_robot.detach().cpu(), self._global_step)

    def _log_epoch(self, epoch: int, averages: dict[str, float], elapsed: float) -> None:
        if not self._distributed.is_main:
            return
        for name, value in averages.items():
            self._writer.add_scalar(f"epoch/{name}", value, epoch)
        self._writer.add_scalar("epoch/time_sec", elapsed, epoch)

    def _make_run_dir(self) -> Path:
        run_name = self._config.output.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(self._config.output.root_dir) / run_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _progress(self, iterable: Any, **kwargs: Any) -> Any:
        if not self._config.train.progress or not self._distributed.is_main:
            return iterable
        return tqdm(iterable, dynamic_ncols=True, **kwargs)

    def _unwrap_model(self) -> DualFSQTrainingModule:
        if isinstance(self._model, DistributedDataParallel):
            return self._model.module
        return self._model

    def _print(self, message: str) -> None:
        if self._distributed.is_main:
            print(message)

    def _make_writer(self, log_dir: Path) -> Any:
        if not self._distributed.is_main:
            return NullSummaryWriter()
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            return NullSummaryWriter()
        log_dir.mkdir(parents=True, exist_ok=True)
        return SummaryWriter(log_dir=str(log_dir), flush_secs=10)
