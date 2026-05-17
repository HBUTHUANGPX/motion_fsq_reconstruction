# motion_fsq_reconstruction

`motion_fsq_reconstruction` 是一个独立的离线 DualFSQ/iFSQ 重构训练包，用于从 `.npz` motion 数据中构建与在线 mimic RL 环境一致的 FSQ feature，训练 actor/critic 两套 DualFSQ autoencoder，并导出逐帧 quantized latent，供后续 RL 训练直接读取。

这个包不依赖 Isaac 环境运行，主要目标是把 FSQ encoder 的重计算从在线 RL 中剥离出来。

## 功能概览

- 加载大量 `.npz` motion 文件，兼容 `scripts/rsl_rl/motion_file*.yaml` 风格的数据列表。
- 按 `MotionCommand._make_calculate()` 的当前语义构建四路 feature：
  - `actor_robot`
  - `actor_human`
  - `critic_robot`
  - `critic_human`
- 训练 actor/critic 两套 DualFSQ autoencoder：
  - robot encoder
  - human encoder
  - 共享 FSQ/iFSQ quantizer
  - robot decoder
- 支持单卡训练和 `torch.distributed.run` 多卡 DDP 训练。
- 保存 checkpoint、normalizer、feature schema 和 TensorBoard 日志。
- 导出离线 latent：
  - `actor_q_human`
  - `actor_q_robot`
  - `critic_q_human`
  - `critic_q_robot`

## 目录结构

```text
motion_fsq_reconstruction/
  cli/                 训练和 latent 导出入口
  configs/             示例配置
  data/                motion 来源解析、raw loader、窗口采样
  features/            feature schema、rotation 工具、feature builder
  models/              DualFSQ autoencoder 和 FSQ/iFSQ quantizer
  training/            trainer、loss、normalizer、DDP、checkpoint
  export/              latent exporter
  tests/               smoke 和一致性测试
  docs/                代码导读文档
```

更细的代码阅读顺序见：

```text
motion_fsq_reconstruction/docs/code_walkthrough.md
```

## 环境要求

建议使用项目已有 conda 环境：

```bash
conda activate mimic_baseline
```

常用依赖：

```text
torch
numpy
pyyaml
tqdm
tensorboard
pytest
```

如果 TensorBoard 不存在：

```bash
/home/hpx/miniconda3/envs/mimic_baseline/bin/pip install tensorboard
```

## 配置文件

默认配置：

```text
motion_fsq_reconstruction/configs/g1_dual_fsq.yaml
```

主要配置块：

```yaml
data:
  motion_yaml: scripts/rsl_rl/motion_file_mr.yaml
  files: []
  dirs: []

features:
  robot_anchor_body: torso_link
  robot_body_names: [...]
  robot_joint_names: [...]
  desire_human_joint_names: [...]
  human_anchor_body: Hips
  human_body_names: [...]

model:
  latent_dim: 64
  robot_encoder_hidden_dims: [...]
  human_encoder_hidden_dims: [...]
  decoder_hidden_dims: [...]
  quantizer:
    type: ifsq
    levels: 32

loss:
  robot_recon: 1.0
  human_recon: 1.0
  latent_align: 2.5
  cycle_latent: 0.25

train:
  device: cuda
  epochs: 100
  batch_size: 1024
  history: 0
  future: 9

output:
  root_dir: outputs/motion_fsq_reconstruction
  run_name:
```

`history + 1 + future` 决定 FSQ window 大小。默认 `history=0, future=9`，即每个样本包含当前帧和未来 9 帧。

## Feature 语义

当前实现对齐在线 `MotionCommand._make_calculate()`。

单帧 feature 拼接为：

```text
actor_robot:
  robot_anchor_rot6d
  robot_joint_pos

actor_human:
  human_anchor_rot6d
  human_body_pos_in_anchor

critic_robot:
  robot_anchor_rot6d
  robot_anchor_pos
  robot_joint_pos
  robot_body_pos_in_anchor
  robot_body_rot6d_in_anchor

critic_human:
  human_anchor_rot6d
  human_anchor_pos
  human_joint_rot6d
  human_body_pos_in_anchor
  human_body_rot6d_in_anchor
```

注意：

- quaternion 在 loader 中统一为 `wxyz`。
- `rot6d` 展平顺序与 `commands.py` 中的 `rot6d_from_quat()` 一致。
- `human_joint_quat` 优先来自 `human_local_transforms[..., 3:7]`，与在线 `MotionLoader_human` 一致。
- `robot_joint_names`、`robot_body_names`、`desire_human_joint_names` 会按配置顺序重排，避免依赖 npz 原始字段顺序。

## 单卡训练

从仓库根目录运行：

```bash
python motion_fsq_reconstruction/cli/train.py \
  --config motion_fsq_reconstruction/configs/g1_dual_fsq.yaml \
  --device cuda \
  --run-name test_run
```

使用指定 Python：

```bash
/home/hpx/miniconda3/envs/mimic_baseline/bin/python \
  motion_fsq_reconstruction/cli/train.py \
  --config motion_fsq_reconstruction/configs/g1_dual_fsq.yaml \
  --device cuda \
  --run-name test_run
```

输出目录：

```text
outputs/motion_fsq_reconstruction/test_run/
  checkpoints/
    latest.pt
    epoch_0010.pt
  tb/
```

## 多卡 DDP 训练

使用 `torch.distributed.run` 启动：

```bash
python -m torch.distributed.run \
  --nnodes=1 \
  --nproc_per_node=8 \
  motion_fsq_reconstruction/cli/train.py \
  --config motion_fsq_reconstruction/configs/g1_dual_fsq.yaml \
  --device cuda \
  --run-name ddp_run \
  --distributed
```

多节点示例：

```bash
python -m torch.distributed.run \
  --nnodes=2 \
  --node_rank=0 \
  --nproc_per_node=8 \
  --master_addr=<master_ip> \
  --master_port=29500 \
  motion_fsq_reconstruction/cli/train.py \
  --config motion_fsq_reconstruction/configs/g1_dual_fsq.yaml \
  --device cuda \
  --run-name ddp_run \
  --distributed
```

分布式训练策略：

- 每个 rank 只加载自己的 motion 文件分片。
- 分片按可训练帧数做 frame-balanced 分配。
- `train.batch_size` 表示每卡 batch size。
- normalizer 使用所有 rank 的全局统计。
- TensorBoard 和 checkpoint 只由 rank0 写入。
- 每个 rank 每个 epoch 使用相同 step 数，避免 DDP backward 等待不一致。

启动后会看到类似：

```text
[MotionFSQShard] rank 0/8 loads 17778 files valid_frames=6327956/50623599.
```

含义：

```text
rank 0/8                  当前 rank 是 0，总 rank 数是 8
loads 17778 files         当前 rank 加载 17778 个 npz 文件
valid_frames=6327956/...  当前 rank 可采样帧数 / 全局可采样帧数
```

## TensorBoard

启动：

```bash
/home/hpx/miniconda3/envs/mimic_baseline/bin/tensorboard \
  --logdir outputs/motion_fsq_reconstruction
```

常用指标：

```text
train/total
train/robot_recon
train/human_recon
train/latent_align
train/cycle_latent
epoch/total
epoch/time_sec
latent/actor_q_human
latent/critic_q_robot
```

如果 `tensorboard` 命令不存在，请先安装：

```bash
/home/hpx/miniconda3/envs/mimic_baseline/bin/pip install tensorboard
```

## 导出 latent

训练完成后导出：

```bash
python motion_fsq_reconstruction/cli/export_latents.py \
  --checkpoint outputs/motion_fsq_reconstruction/test_run/checkpoints/latest.pt \
  --config motion_fsq_reconstruction/configs/g1_dual_fsq.yaml \
  --output outputs/motion_fsq_reconstruction/test_run/latents.npz \
  --device cuda
```

导出的 `.npz` 包含：

```text
actor_q_human
actor_q_robot
critic_q_human
critic_q_robot
motion_lengths
motion_start_indices
motion_paths
feature_schema
config
```

latent shape 为：

```text
[num_frames, latent_dim]
```

## MuJoCo 重构评估

训练完成后，可以用 MuJoCo 同时查看原始机器人轨迹和四路 DualFSQ 重构轨迹。

可视化单个 motion：

```bash
python motion_fsq_reconstruction/cli/evaluate_mujoco.py \
  --checkpoint outputs/motion_fsq_reconstruction/test_run/checkpoints/latest.pt \
  --config motion_fsq_reconstruction/configs/g1_dual_fsq.yaml \
  --motion soma-retargeter/assets/motions/soma_uniform_bvh_export/240918/body_check_001__A548.npz \
  --xml assets/unitree_g1/g1_29dof_rev_1_0.xml \
  --output outputs/motion_fsq_reconstruction/test_run/mujoco_eval \
  --device cuda \
  --show-viewer
```

不打开 viewer，直接对配置解析到的全部 motion 做评估：

```bash
python motion_fsq_reconstruction/cli/evaluate_mujoco.py \
  --checkpoint outputs/motion_fsq_reconstruction/test_run/checkpoints/latest.pt \
  --config motion_fsq_reconstruction/configs/g1_dual_fsq.yaml \
  --xml assets/unitree_g1/g1_29dof_rev_1_0.xml \
  --output outputs/motion_fsq_reconstruction/test_run/mujoco_eval_all \
  --device cuda \
  --all-motions \
  --no-viewer
```

使用指定 Python：

```bash
/home/hpx/miniconda3/envs/mimic_baseline/bin/python \
  motion_fsq_reconstruction/cli/evaluate_mujoco.py \
  --checkpoint outputs/motion_fsq_reconstruction/test_run/checkpoints/latest.pt \
  --config motion_fsq_reconstruction/configs/g1_dual_fsq.yaml \
  --motion <motion.npz> \
  --xml assets/unitree_g1/g1_29dof_rev_1_0.xml \
  --output outputs/motion_fsq_reconstruction/test_run/mujoco_eval \
  --device cuda \
  --show-viewer
```

MuJoCo viewer 中会看到五套完整机器人，沿 x 方向并排显示：

```text
original      原始 robot motion
actor_robot   actor robot encoder -> actor decoder
actor_human   actor human encoder -> actor decoder
critic_robot  critic robot encoder -> critic decoder
critic_human  critic human encoder -> critic decoder
```

如果 npz 中包含 `human_local_transforms` 和 `human_parent_indices`，viewer 会额外绘制原始 human skeleton overlay。

评估输出目录包含：

```text
scene.xml
reconstructions.npz
metrics.json
```

`reconstructions.npz` 包含：

```text
original_robot_joint_pos
actor_robot_recon_joint_pos
actor_human_recon_joint_pos
critic_robot_recon_joint_pos
critic_human_recon_joint_pos
robot_root_pos
robot_root_quat
robot_joint_names
fps
motion_lengths
motion_paths
feature_schema
```

`metrics.json` 包含四路重构的 joint 误差：

```text
joint_mse
joint_rmse
joint_mae
joint_max_abs
per_joint_mse
```

注意：当前 decoder 的目标是 robot feature window，不是完整 MuJoCo root qpos。评估播放时 root pose 使用原始 npz 的 `robot_root_pos` 和 `robot_root_quat`，重构结果只替换 robot joint position。这样可以稳定观察关节重构质量，并避免从 anchor body pose 反解 root pose 带来的额外误差。

## Loss

每套 actor/critic DualFSQ 都包含四项 loss：

```text
robot_recon:
  decoder(robot_encoder(robot_window)) -> robot_window

human_recon:
  decoder(human_encoder(human_window)) -> robot_window

latent_align:
  mse(q_human, q_robot.detach())

cycle_latent:
  mse(q_cycle, q_human.detach())
```

总 loss：

```text
0.5 * (actor_total + critic_total)
```

## Checkpoint

checkpoint 保存：

```text
model
optimizer
epoch
global_step
config
normalizers
feature_schema
metadata
```

DDP 训练时，checkpoint 保存的是解包后的普通 model state dict，因此 latent exporter 不需要特殊处理。

## 测试

编译检查：

```bash
find motion_fsq_reconstruction -name '*.py' -print0 \
  | xargs -0 /home/hpx/miniconda3/envs/mimic_baseline/bin/python -m py_compile
```

单元和 smoke 测试：

```bash
/home/hpx/miniconda3/envs/mimic_baseline/bin/python \
  -m pytest motion_fsq_reconstruction/tests/test_pipeline.py -q
```

测试覆盖：

- rot6d 顺序与在线 `MotionCommand` 一致。
- loader 按配置重排 robot/human 名称。
- `human_local_transforms` 优先级与在线 loader 一致。
- 单卡训练和 latent 导出。
- MuJoCo 多机器人 scene 生成。
- DualFSQ 重构 joint 轨迹导出。
- frame-balanced sharding。
- 2 进程 CPU DDP smoke。

## 常见问题

### 未解析到任何 npz motion 文件

检查配置：

```yaml
data:
  motion_yaml: scripts/rsl_rl/motion_file_mr.yaml
  files: []
  dirs: []
```

如果使用 `motion_yaml`，确认里面的 `folder_name` 或 `file_name` 指向真实存在的 `.npz`。

### TensorBoard 未找到

安装：

```bash
/home/hpx/miniconda3/envs/mimic_baseline/bin/pip install tensorboard
```

### 多卡训练卡住

优先检查：

- 是否使用 `python -m torch.distributed.run`
- 是否传入 `--distributed`
- 每个节点是否能访问同一份 motion 数据路径
- `MASTER_ADDR`、`MASTER_PORT`、`node_rank` 是否正确

### CUDA rank 与显卡不匹配

DDP 模式下 `--device cuda` 会自动映射为：

```text
cuda:${LOCAL_RANK}
```

不要手动给每个 rank 写不同配置。

## 与在线 RL 的关系

这个包负责离线训练 FSQ-VAE 和导出 latent。后续 mimic RL 可以直接读取导出的 latent 文件，把对应帧的 latent 注入 actor/critic，避免在线反复运行 FSQ encoder。

需要注意：

- 离线训练、导出 latent、在线读取 latent 必须使用同一份 motion 数据顺序和 feature schema。
- 如果后续 `MotionCommand._make_calculate()` 的 feature 拼接规则变化，本包的 `DualFSQFeatureBuilder` 也必须同步更新。
