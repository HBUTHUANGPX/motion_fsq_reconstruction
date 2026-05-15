# motion_fsq_reconstruction 代码导读

这份文档按一次真实使用流程来读代码：先看配置和入口，再看数据加载、feature 构建、窗口采样、模型、训练、checkpoint，最后看 latent 导出。读完后，你应该能清楚知道每个模块负责什么，以及后续要接入 mimic RL 时应该从哪里拿数据。

## 1. 包的目标

`motion_fsq_reconstruction` 是一个独立的离线 DualFSQ 训练包。它不修改旧的 `motion_reconstruction` 包，也不依赖 Isaac 环境运行。

它做三件事：

1. 从大量 `.npz` motion 文件加载 robot/human 参考数据。
2. 按在线 `MotionCommand._make_calculate()` 当前使用的规则构建 DualFSQ feature。
3. 训练 actor/critic 两套 DualFSQ autoencoder，并导出每一帧的 quantized latent。

最终导出的 latent 以后可以被 mimic RL 直接读取：

```text
actor_q_human[time_step]  -> actor 使用
critic_q_robot[time_step] -> critic 使用
```

## 2. 推荐阅读顺序

建议按这个顺序读：

```text
configs/g1_dual_fsq.yaml
cli/train.py
config/schema.py
pipeline.py
data/raw_motion.py
features/builder.py
data/window_buffer.py
models/training_module.py
models/dual_fsq.py
training/losses.py
training/trainer.py
export/latent_exporter.py
```

这个顺序基本对应一次训练和导出的真实调用链。

## 3. 配置文件

入口配置在：

```text
motion_fsq_reconstruction/configs/g1_dual_fsq.yaml
```

这里分为六块：

```text
data      motion 文件来源
features  anchor body 和 human body 选择
model     DualFSQ 网络结构和量化器
loss      四项 MSE loss 权重
train     epoch、batch、history/future、device
output    输出目录
```

其中最关键的是 `features` 和 `train.history/future`。

`features` 决定离线 feature 的语义必须和在线环境一致：

```yaml
features:
  robot_anchor_body: torso_link
  human_anchor_body: Hips
```

`train.history` 和 `train.future` 决定窗口大小：

```text
window_size = history + 1 + future
```

当前默认：

```yaml
history: 0
future: 9
```

也就是每个训练样本使用当前帧加未来 9 帧，共 10 帧。

## 4. 配置 dataclass

配置 schema 在：

```text
motion_fsq_reconstruction/config/schema.py
```

主要类是：

```python
MotionFSQReconstructionConfig
```

它包含：

```python
data: DataConfig
features: FeatureConfig
model: ModelConfig
loss: LossConfig
train: TrainConfig
output: OutputConfig
```

这个文件只描述配置结构，不包含训练逻辑。这样做的好处是：CLI、trainer、exporter 都依赖同一份配置对象，避免参数在多个地方散开。

YAML 加载逻辑在：

```text
motion_fsq_reconstruction/config/io.py
```

入口函数是：

```python
load_config(path)
```

它会把 YAML 递归填入 dataclass，没写的字段使用默认值。

## 5. 训练入口

训练入口在：

```text
motion_fsq_reconstruction/cli/train.py
```

命令：

```bash
python motion_fsq_reconstruction/cli/train.py \
  --config motion_fsq_reconstruction/configs/g1_dual_fsq.yaml \
  --device cuda \
  --run-name test_run
```

多卡训练使用 torchrun：

```bash
python -m torch.distributed.run \
  --nnodes=N \
  --nproc_per_node=M \
  motion_fsq_reconstruction/cli/train.py \
  --config motion_fsq_reconstruction/configs/g1_dual_fsq.yaml \
  --device cuda \
  --run-name test_run \
  --distributed
```

这个文件很薄，只做三件事：

1. 解析命令行参数。
2. 调用 `load_config()` 加载 YAML。
3. 创建 `DualFSQTrainer(config, distributed=...)` 并调用 `train()`。

也就是说，真正的训练逻辑不在 CLI 里，而在：

```text
motion_fsq_reconstruction/training/trainer.py
```

## 6. pipeline 总装配

训练器不会自己知道怎么加载数据和建模型，它通过：

```text
motion_fsq_reconstruction/pipeline.py
```

来装配运行时对象。

核心函数有两个：

```python
build_motion_runtime(config, device, progress)
build_training_module(config, runtime)
```

`build_motion_runtime()` 做：

```text
解析 motion 文件
加载 raw motion
构建四路 feature
创建窗口 buffer
```

返回：

```python
MotionRuntimeBundle
```

里面包含：

```python
raw       RawMotionDataset
features  DualFSQFeatureBundle
buffer    MotionWindowBuffer
```

`build_training_module()` 根据 runtime 里的输入维度创建 actor/critic 两套 DualFSQ 模型。

## 7. 数据加载

数据加载在：

```text
motion_fsq_reconstruction/data/raw_motion.py
```

核心类：

```python
RawMotionLoader
RawMotionDataset
```

`RawMotionLoader` 负责读取多个 npz，并拼成一个连续 tensor buffer。

它会读取这些语义字段：

```text
robot_joint_pos
robot_joint_vel
robot_body_pos
robot_body_quat
robot_body_lin_vel
robot_body_ang_vel
human_global_pos
human_global_quat
human_local_transforms[..., 3:7] -> human_joint_quat
```

内部统一把 quaternion 转成 `wxyz`。

注意：如果 npz 里有：

```text
scalar_first = True
```

说明原始 quaternion 已经是 `wxyz`。否则按 `xyzw` 转成 `wxyz`。

`RawMotionDataset` 还保存：

```python
motion_lengths
motion_start_indices
motion_paths
motion_groups
```

这些信息后面用于保证窗口采样不会跨 motion clip。

## 8. 四路 FeatureBuilder

最关键的文件是：

```text
motion_fsq_reconstruction/features/builder.py
```

核心类：

```python
DualFSQFeatureBuilder
```

它输出：

```python
DualFSQFeatureBundle(
    actor_robot,
    actor_human,
    critic_robot,
    critic_human,
    schema,
)
```

四路 feature 严格对齐在线 `commands.py` 当前实际拼接。

actor robot：

```text
robot_anchor_rot6d
robot_joint_pos
```

actor human：

```text
human_anchor_rot6d
human_body_pos_in_human_anchor
```

critic robot：

```text
robot_anchor_rot6d
robot_anchor_pos
robot_joint_pos
robot_body_pos_in_robot_anchor
robot_body_rot6d_in_robot_anchor
```

critic human：

```text
human_anchor_rot6d
human_anchor_pos
human_joint_rot6d
human_body_pos_in_human_anchor
human_body_rot6d_in_human_anchor
```

这些 feature 都是逐帧的，shape 是：

```text
[num_frames, feature_dim]
```

还没有展开成 window。

## 9. 旋转工具

旋转工具在：

```text
motion_fsq_reconstruction/features/rotation.py
```

这里统一使用 `wxyz` quaternion。

常用函数：

```python
quat_to_rot6d_wxyz()
quat_inverse_rotate_wxyz()
quat_multiply_wxyz()
subtract_frame_transforms_wxyz()
```

其中 `subtract_frame_transforms_wxyz()` 对应在线环境里的相对 frame 计算：

```text
child pose in parent frame
```

robot body in anchor、human body in anchor 都依赖这些工具。

## 10. 窗口采样

窗口采样在：

```text
motion_fsq_reconstruction/data/window_buffer.py
```

核心类：

```python
MotionWindowIndex
MotionWindowBuffer
```

`MotionWindowIndex` 只负责索引：

```text
valid_center_indices
window_offsets
window_indices_for()
```

它会保证训练采样时不跨 clip。

例如：

```text
history = 1
future = 1
center = 10
window = [9, 10, 11]
```

`MotionWindowBuffer` 保存四路 feature，并把窗口展开成网络输入：

```text
[B, window_size, feature_dim]
       ↓ reshape
[B, window_size * feature_dim]
```

训练时调用：

```python
iter_epoch_batches(batch_size)
```

导出 latent 时调用：

```python
batch_from_centers(centers, clamp_to_clip=True)
```

导出需要覆盖所有帧，所以 clip 边界处会 clamp 到当前 clip 内。

## 11. 模型封装

模型封装分两层。

第一层是单套 DualFSQ：

```text
motion_fsq_reconstruction/models/dual_fsq.py
```

核心类：

```python
DualFSQAutoEncoder
```

结构是：

```text
robot_window -> robot_encoder -> quantizer -> q_robot -> decoder -> robot_recon_from_robot
human_window -> human_encoder -> quantizer -> q_human -> decoder -> robot_recon_from_human
robot_recon_from_human -> robot_encoder -> quantizer -> q_cycle
```

第二层是 actor/critic 双模型：

```text
motion_fsq_reconstruction/models/training_module.py
```

核心类：

```python
DualFSQTrainingModule
```

它内部有：

```python
actor_dual_fsq
critic_dual_fsq
```

这对应在线：

```python
ActorCriticDualFSQ.actor_dual_fsq
ActorCriticDualFSQ.critic_dual_fsq
```

## 12. 量化器

量化器在：

```text
motion_fsq_reconstruction/models/quantizers.py
```

目前有：

```python
FSQQuantizer
IFSQuantizer
```

输出是 dict，包含训练和导出实际需要的：

```python
z_q
```

也就是 quantized latent。

## 13. Loss

损失在：

```text
motion_fsq_reconstruction/training/losses.py
```

核心类：

```python
DualFSQLoss
```

每套 actor/critic 都计算四项：

```text
robot_recon:
  decoder(robot_encoder(robot_window)) 重构 robot_window

human_recon:
  decoder(human_encoder(human_window)) 重构 robot_window

latent_align:
  q_human 和 q_robot.detach() 对齐

cycle_latent:
  human -> decoder -> robot_encoder 得到的 q_cycle
  与 q_human.detach() 对齐
```

最后 actor loss 和 critic loss 做平均。

## 14. Normalizer

归一化在：

```text
motion_fsq_reconstruction/training/normalization.py
```

核心类：

```python
WindowFeatureNormalizer
```

它按逐帧 feature 统计均值方差，然后 repeat 到 window 维度。

例如：

```text
frame feature dim = 8
window size = 10
normalizer dim = 80
```

训练和 latent 导出都会使用 checkpoint 里保存的同一份 normalizer。

这只影响离线 FSQ-VAE 训练和离线 latent 预计算。后续 RL 如果直接读取 `q`，就不需要再处理 raw FSQ feature normalizer。

## 15. Trainer

训练器在：

```text
motion_fsq_reconstruction/training/trainer.py
```

核心类：

```python
DualFSQTrainer
```

初始化时做：

```text
解析 device
分布式下解析 rank/local_rank/world_size
创建 run_dir
rank0 创建 TensorBoard writer
build_motion_runtime()
fit normalizers，分布式下 all-reduce 全局统计
build_training_module()
分布式下用 DistributedDataParallel 包装模型
创建 AdamW optimizer
创建 DualFSQLoss
```

训练循环里每个 batch：

```python
actor_robot = normalizer(batch.actor_robot)
actor_human = normalizer(batch.actor_human)
critic_robot = normalizer(batch.critic_robot)
critic_human = normalizer(batch.critic_human)

output = model(actor_robot, actor_human, critic_robot, critic_human)
loss = loss_fn(output, actor_robot_target=actor_robot, critic_robot_target=critic_robot)
```

保存 checkpoint 时包含：

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

## 16. Checkpoint

checkpoint 工具在：

```text
motion_fsq_reconstruction/training/checkpoint.py
```

主要函数：

```python
save_checkpoint()
load_checkpoint()
```

保存路径：

```text
outputs/motion_fsq_reconstruction/<run_name>/checkpoints/latest.pt
```

## 17. Latent 导出

导出入口在：

```text
motion_fsq_reconstruction/cli/export_latents.py
```

核心实现：

```text
motion_fsq_reconstruction/export/latent_exporter.py
```

用法：

```bash
python -m motion_fsq_reconstruction.cli.export_latents \
  --checkpoint outputs/motion_fsq_reconstruction/test_run/checkpoints/latest.pt \
  --config motion_fsq_reconstruction/configs/g1_dual_fsq.yaml \
  --output outputs/motion_fsq_reconstruction/test_run/latents.npz \
  --device cuda
```

`LatentExporter` 会：

```text
加载 config
重新 build_motion_runtime
重新 build_training_module
加载 checkpoint model state
加载 checkpoint normalizers
遍历所有 center frame
导出四路 quantized latent
```

输出 `.npz` 包含：

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

## 18. RL 侧后续怎么接

后续 mimic RL 不应该再把 raw FSQ feature 传入 actor/critic，也不应该再运行 FSQ encoder。

推荐新增 RL obs group：

```text
actor_fsq_latent
critic_fsq_latent
```

在 `MotionCommand` 或类似数据源中按当前 `time_steps` 读取：

```python
actor_fsq_latent = actor_q_human[time_steps]
critic_fsq_latent = critic_q_robot[time_steps]
```

然后 policy 里：

```python
actor_input = cat(actor_obs_normalized, actor_fsq_latent)
critic_input = cat(critic_obs_normalized, critic_fsq_latent)
```

注意不要把 latent 混入普通 `policy` obs group 里，否则会被 `actor_obs_normalizer` 一起归一化。

## 19. 测试

包内测试在：

```text
motion_fsq_reconstruction/tests/test_pipeline.py
```

它做了两件事：

1. 生成一个小型 npz，检查四路 feature shape 和部分数值。
2. 跑 1 epoch 训练，再导出 latent，检查四路 latent shape。

运行：

```bash
/home/hpx/miniconda3/envs/mimic_baseline/bin/python \
  -m pytest motion_fsq_reconstruction/tests/test_pipeline.py -q
```

## 20. 当前边界

第一版没有做：

```text
Mujoco 可视化
RL 侧 latent 注入
分布式训练
多 schema 自动重排
```

这些都可以后续加，但当前包已经把最核心的离线 DualFSQ 训练和 latent 导出链路打通了。
