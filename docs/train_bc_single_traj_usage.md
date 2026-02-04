# train_bc_single_traj.py 使用指南

## 功能说明

`train_bc_single_traj.py` 支持三种运行模式：

1. **纯训练模式**：只训练，不评估
2. **训练+评估模式**：训练完成后自动评估
3. **纯评估模式**：只评估已训练的模型（支持对比参考轨迹）

## 使用方法

### 模式1：纯训练模式

只训练BC策略，不进行评估：

```bash
.venv/bin/python examples/train_bc_single_traj.py \
    --exp_name=gear_assembly \
    --demo_path=./demo_data/gear_assembly_5_demos_2026-02-04_09-01-56.pkl \
    --traj_index=0 \
    --bc_checkpoint_path=./checkpoints/bc_single_traj_test \
    --train_steps=20000 \
    --eval_n_trajs=0 \
    --filter_max_consecutive=5 \
    --enable_filtering=True \
    --seed=42
```

### 模式2：训练+评估模式

训练完成后自动进行评估：

```bash
.venv/bin/python examples/train_bc_single_traj.py \
    --exp_name=gear_assembly \
    --demo_path=./demo_data/gear_assembly_5_demos_2026-02-04_09-01-56.pkl \
    --traj_index=0 \
    --bc_checkpoint_path=./checkpoints/bc_single_traj_test \
    --train_steps=20000 \
    --eval_n_trajs=3 \
    --use_sim \
    --isaac_server_url=http://192.168.31.198:5001/ \
    --filter_max_consecutive=5 \
    --enable_filtering=True \
    --seed=42
```

### 模式3：纯评估模式（新增功能）⭐

**只评估已训练的模型，并对比参考轨迹的动作误差**：

```bash
.venv/bin/python examples/train_bc_single_traj.py \
    --exp_name=gear_assembly \
    --demo_path=./demo_data/gear_assembly_5_demos_2026-02-04_09-01-56.pkl \
    --traj_index=0 \
    --bc_checkpoint_path=./checkpoints/bc_single_traj_test \
    --train_steps=0 \
    --eval_n_trajs=3 \
    --use_sim \
    --isaac_server_url=http://192.168.31.198:5001/ \
    --seed=42
```

**关键参数**：
- `--train_steps=0`：设置为0表示只评估，不训练
- `--eval_n_trajs=3`：运行3次测试
- `--use_sim` 或 `--isaac_server_url`：指定使用仿真环境

## 评估模式功能

### 自动加载参考轨迹

在评估模式下，脚本会：
1. 自动从 `{bc_checkpoint_path}/reference_trajectory.pkl` 加载参考轨迹
2. 如果参考轨迹存在，会在第一次测试时对比BC策略输出和参考轨迹的动作误差
3. 显示详细的动作误差统计（平均、最大、最小、标准差）

### 评估输出示例

```
[BC EVAL] Loading checkpoint from ./checkpoints/bc_single_traj_test/checkpoint_20000
[BC EVAL] Loading reference trajectory from ./checkpoints/bc_single_traj_test/reference_trajectory.pkl
[BC EVAL] Reference trajectory loaded: 418 transitions
[BC EVAL] Starting evaluation with 3 trajectories...

[BC EVAL] Starting episode 1/3
[BC EVAL] 注意: 使用参考轨迹的初始状态进行测试
[BC EVAL] Step 50, Action error: 0.023456
[BC EVAL] Step 100, Action error: 0.019823
...
[BC EVAL] Episode 1 SUCCESS (steps: 418, time: 45.23s)
[BC EVAL] 参考轨迹长度: 418, BC策略轨迹长度: 418
[BC EVAL] 平均动作误差: 0.021234
[BC EVAL] 最大动作误差: 0.045678
[BC EVAL] 最小动作误差: 0.001234
[BC EVAL] 标准差: 0.012345

[BC EVAL] Starting episode 2/3
[BC EVAL] Episode 2 SUCCESS (steps: 425, time: 46.12s)

[BC EVAL] Starting episode 3/3
[BC EVAL] Episode 3 SUCCESS (steps: 412, time: 44.89s)

[BC EVAL] 最终结果: 3/3 成功
[BC EVAL] 动作误差统计:
  平均: 0.021234
  最大: 0.045678
  最小: 0.001234
  标准差: 0.012345
```

## 验证标准

### 成功标准

- **成功率 ≥ 2/3**：BC能复现轨迹，数据质量合格 ✅
- **动作误差 < 0.1**：BC输出与参考轨迹动作接近 ✅
- **轨迹长度相近**：BC策略轨迹长度与参考轨迹相近（±10%）✅

### 需要关注

- **成功率 1/3-2/3**：部分成功，可能需要更多数据或改进数据质量 ⚠️
- **动作误差 0.1-0.3**：动作差异较大，但可能仍能完成任务 ⚠️

### 失败标准

- **成功率 < 1/3**：BC无法学到有效策略，需要检查数据质量或重新采集数据 ❌
- **动作误差 > 0.3**：动作差异过大，可能无法完成任务 ❌

## 注意事项

1. **参考轨迹文件**：只有在训练时设置了 `--save_reference_traj=True`（默认True），才会保存参考轨迹
2. **环境连接**：评估模式需要实际连接环境（仿真或真实），确保环境可用
3. **随机种子**：使用相同的 `--seed` 可以确保可复现性
4. **Checkpoint路径**：确保 `--bc_checkpoint_path` 指向正确的checkpoint目录

## 常见问题

### Q: 评估时提示找不到参考轨迹文件？

**A**: 可能原因：
1. 训练时没有保存参考轨迹（检查 `--save_reference_traj` 参数）
2. Checkpoint路径不正确
3. 参考轨迹文件被删除

**解决方案**：重新训练并确保 `--save_reference_traj=True`

### Q: 评估时环境连接失败？

**A**: 检查：
1. Isaac Sim服务器是否运行
2. `--isaac_server_url` 是否正确
3. 网络连接是否正常

### Q: 动作误差很大但成功率很高？

**A**: 这是正常的，说明：
- BC学到了策略，能完成任务
- 但动作细节与演示数据有差异
- 如果任务完成，可以接受

## 完整验证流程示例

```bash
# 1. 训练BC策略（单条轨迹）
.venv/bin/python examples/train_bc_single_traj.py \
    --exp_name=gear_assembly \
    --demo_path=./demo_data/gear_assembly_5_demos_2026-02-04_09-01-56.pkl \
    --traj_index=0 \
    --bc_checkpoint_path=./checkpoints/bc_single_traj_test \
    --train_steps=20000 \
    --eval_n_trajs=0 \
    --filter_max_consecutive=5 \
    --enable_filtering=True \
    --seed=42

# 2. 单独评估（对比参考轨迹）
.venv/bin/python examples/train_bc_single_traj.py \
    --exp_name=gear_assembly \
    --demo_path=./demo_data/gear_assembly_5_demos_2026-02-04_09-01-56.pkl \
    --traj_index=0 \
    --bc_checkpoint_path=./checkpoints/bc_single_traj_test \
    --train_steps=0 \
    --eval_n_trajs=3 \
    --use_sim \
    --isaac_server_url=http://192.168.31.198:5001/ \
    --seed=42
```
