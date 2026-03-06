# RLPD 训练指南 - Isaac Sim 仿真环境

## 部署架构

本指南支持两种部署方式：

1. **单机部署**：Isaac Sim、Learner、Actor 在同一台机器
2. **分布式部署**：Isaac Sim 和 RL 训练在不同机器（推荐，GPU 资源更充足）

---

## 一、训练前准备

### 1.1 已完成的工作 ✅

- ✅ **演示数据采集完成**
  - 文件：`./demo_data/gear_assembly_25_demos_2026-01-30_12-56-10.pkl`
  - 数据量：25 条轨迹，20,752 个 transitions
  - 文件大小：3.9 GB

- ✅ **Isaac Sim 环境配置完成**
  - 环境类：`IsaacSimGearAssemblyEnvEnhanced`
  - 配置：`IsaacSimEnvConfig`
  - 服务器：`isaac_sim_server.py`

### 1.2 训练代码检查

训练代码 `examples/train_rlpd.py` 已支持 Isaac Sim 仿真环境：

- ✅ **环境选择逻辑**（第 374-378 行）：
  ```python
  use_fake_env = FLAGS.use_sim if FLAGS.actor else FLAGS.learner
  env = config.get_environment(
      fake_env=use_fake_env,
      save_video=FLAGS.save_video,
      classifier=not use_fake_env,  # 仿真环境使用逻辑奖励
  )
  ```

- ✅ **配置支持**（`config.py`）：
  - `setup_mode = "single-arm-continuous-gripper"` ✅
  - `image_keys = ["wrist_1", "wrist_2"]` ✅
  - `encoder_type = "resnet-pretrained"` ✅

- ✅ **演示数据加载**（第 466-473 行）：
  - 支持加载 `.pkl` 格式的演示数据
  - 自动处理 `grasp_penalty` 字段

**结论：训练代码已就绪，可以开始训练！** ✅

---

## 二、训练架构

RLPD 训练采用 **Actor-Learner 分离架构**，需要同时运行两个进程：

```
┌─────────────────┐         ┌─────────────────┐
│   Actor 进程    │         │  Learner 进程   │
│                 │         │                 │
│  - 环境交互      │◄───────►│  - 模型训练     │
│  - 数据收集      │  网络   │  - 参数更新     │
│  - 干预记录      │  通信   │  - 检查点保存   │
└─────────────────┘         └─────────────────┘
```

### 2.1 进程职责

**Learner 进程**：
- 加载演示数据到 `demo_buffer`
- 接收 Actor 收集的在线数据到 `replay_buffer`
- 执行模型训练（50/50 混合采样）
- 定期保存检查点
- 发布更新后的网络参数给 Actor

**Actor 进程**：
- 与环境交互（Isaac Sim）
- 使用策略采样动作
- 收集训练数据
- 支持手柄干预（可选）
- 接收 Learner 更新的参数

---

## 三、训练命令

### 3.0 部署架构选择

**场景 A：单机部署**
- Isaac Sim、Learner、Actor 在同一台机器
- GPU 资源需要共享
- 内存设置：Learner 30%，Actor 10%，Isaac Sim 60%

**场景 B：分布式部署**（推荐）
- 机器 1：Isaac Sim（不使用 GPU 进行 RL 训练）
- 机器 2：Learner + Actor（可以使用全部 GPU）
- 内存设置：Learner 80%，Actor 15-20%，系统预留 5%

---

### 3.1 启动 Isaac Sim 服务器（必需）

**机器 1 - Isaac Sim 服务器**：

**终端 1 - Isaac Sim 服务器**：
```bash
./run_isaac.sh serl_robot_infra/robot_servers/isaac_sim_server.py \
    --flask_url=0.0.0.0 \
    --flask_port=5001 \
    --headless=False \
    --sim_width=1280 \
    --sim_height=720 \
    --sim_hz=60.0 \
    --usd_path=examples/experiments/gear_assembly/HIL_franka_gear.usda \
    --robot_prim_path=/World/franka \
    --camera_prim_paths=/World/franka/panda_hand/wrist_1,/World/franka/panda_hand/wrist_2 \
    --config_module=examples.experiments.gear_assembly.config
```

**注意**：服务器必须保持运行，直到训练完成。

---

### 3.2 启动 Learner 进程

**场景 A：单机部署（Isaac Sim + Learner + Actor 在同一台机器）**

**终端 2 - Learner**：
```bash
# 1. 激活虚拟环境
source .venv/bin/activate

# 2. 设置 CUDA 环境变量（必需！详见 docs/gpu_setup_guide.md）
source setup_cuda_env.sh

# 3. XLA 内存设置（详见 docs/xla_memory_settings.md）
# PREALLOCATE=false: 按需分配内存，避免多进程冲突
# MEM_FRACTION=.8: 使用 80% GPU 内存（分布式部署，Isaac Sim 在另一台机器）
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.8

# 4. 启动训练
python examples/train_rlpd.py \
    --exp_name=gear_assembly \
    --learner \
    --demo_path=./demo_data/gear_assembly_25_demos_2026-01-30_12-56-10.pkl \
    --checkpoint_path=./checkpoints/gear_assembly_rlpd \
    --seed=42
```

**场景 B：分布式部署（Isaac Sim 和 RL 训练在不同机器）**

**机器 2 - Learner**：
```bash
# XLA 内存设置（Isaac Sim 在另一台机器，GPU 资源全部用于 RL 训练）
# PREALLOCATE=false: 按需分配内存
# MEM_FRACTION=.8: 使用 80% GPU 内存（为系统和其他进程预留 20%）
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.8

python examples/train_rlpd_pretrain.py \
    --exp_name=gear_assembly_pretrain \
    --learner \
    --demo_path=./demo_data/gear_assembly_25_demos_2026-03-06_10-44-01_filtered.pkl \
    --checkpoint_path=./checkpoints/gear_pretrain \
    --seed=42
```

**参数说明**：
- `--exp_name=gear_assembly`：实验名称（对应配置）
- `--learner`：启动 Learner 进程
- `--demo_path`：演示数据路径（支持多个文件，用 `--demo_path` 重复指定）
- `--checkpoint_path`：检查点保存路径
- `--seed`：随机种子

**可选参数**：
- `--save_video`：保存训练视频
- `--debug`：调试模式

---

### 3.3 启动 Actor 进程

**场景 A：单机部署（Isaac Sim + Learner + Actor 在同一台机器）**

**终端 3 - Actor**：
```bash
# Actor 只需要推理，内存需求较小
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1

python examples/train_rlpd.py \
    --exp_name=gear_assembly \
    --actor \
    --use_sim \
    --ip=localhost \
    --isaac_server_url=http://localhost:5001/ \
    --checkpoint_path=./checkpoints/gear_assembly_rlpd \
    --seed=42
```

**注意**：单机部署时，`--isaac_server_url` 可以省略（使用 config 默认值），但显式指定更清晰。

**场景 B：分布式部署（Isaac Sim 和 RL 训练在不同机器）**

**机器 2 - Actor**：
```bash
# 1. 激活虚拟环境
source .venv/bin/activate

# 2. 设置 CUDA 环境变量（必需！）
source setup_cuda_env.sh

# 3. XLA 内存设置
# Actor 只需要推理，但 GPU 资源充足，可以适当增加
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.15 

python examples/train_rlpd_pretrain.py \
    --exp_name=gear_assembly_pretrain \
    --actor \
    --use_sim \
    --ip=localhost \
    --isaac_server_url=http://192.168.31.198:5001/ \
    --checkpoint_path=./checkpoints/gear_pretrain \
    --seed=42
```

**重要参数说明**：
- `--ip=localhost`：**Learner 的 IP 地址**（Actor 和 Learner 在同一台机器，使用 localhost）
- `--isaac_server_url`：**Isaac Sim 服务器的 URL**（机器 1 的 IP 地址和端口）
  - 格式：`http://<IsaacSim机器IP>:5001/`
  - 例如：`http://192.168.1.100:5001/`
  - 如果不提供，使用 `config.py` 中 `IsaacSimEnvConfig.SERVER_URL` 的默认值

**注意**：
- Actor 需要同时连接两个服务：
  1. **Learner**：通过 `--ip` 参数（同一台机器，使用 localhost）
  2. **Isaac Sim 服务器**：通过 `--isaac_server_url` 参数（远程机器）
- 确保两台机器网络互通，防火墙允许通信
- 确保 Isaac Sim 服务器已启动并监听 `0.0.0.0:5001`

**参数说明**：
- `--actor`：启动 Actor 进程
- `--use_sim`：**重要**：使用 Isaac Sim 仿真环境
- `--ip`：**Learner 的 IP 地址**
  - 同一台机器：`localhost` 或 `127.0.0.1`
  - 不同机器：Learner 机器的 IP 地址
- `--isaac_server_url`：**Isaac Sim 服务器的 URL**（可选）
  - 格式：`http://<IP>:<端口>/`
  - 如果不提供，使用 `config.py` 中的默认值
  - 分布式部署时必须提供（Isaac Sim 在另一台机器）
- `--checkpoint_path`：检查点路径（与 Learner 相同）

**注意**：
- Actor 需要同时连接两个服务：
  1. **Learner**：通过 `--ip` 参数
  2. **Isaac Sim 服务器**：通过 `--isaac_server_url` 参数（或使用 config 默认值）
- Actor 会等待 Learner 启动并连接
- 如果 Learner 未启动，Actor 会一直等待
- 确保 Isaac Sim 服务器已启动并可以访问

---

## 四、训练流程

### 4.1 启动顺序

1. **第一步**：启动 Isaac Sim 服务器（终端 1）
2. **第二步**：启动 Learner 进程（终端 2）
3. **第三步**：启动 Actor 进程（终端 3）

### 4.2 训练过程

1. **初始化阶段**：
   - Learner 加载演示数据到 `demo_buffer`
   - Learner 启动 `TrainerServer`，等待 Actor 连接
   - Actor 连接 Learner，接收初始策略参数

2. **数据收集阶段**：
   - Actor 开始与环境交互
   - 前 `random_steps` 步使用随机动作探索（默认 0）
   - 之后使用策略采样动作
   - 数据发送到 Learner 的 `replay_buffer`

3. **训练阶段**：
   - Learner 等待 `replay_buffer` 达到 `training_starts` 大小（默认 100）
   - 开始训练循环：
     - 从 `replay_buffer` 和 `demo_buffer` 各采样 50%
     - 更新 Critic（多次）和 Actor（一次）
     - 定期发布参数给 Actor
     - 定期保存检查点

4. **持续迭代**：
   - Actor 持续收集新数据
   - Learner 持续训练并更新参数
   - 策略性能逐步提升

---

## 五、训练监控

### 5.1 日志输出

**Learner 日志**：
- `demo buffer size: X`：演示数据量
- `online buffer size: X`：在线数据量
- `Filling up replay buffer`：等待缓冲区填满
- `sent initial network to actor`：参数已发送给 Actor
- 训练损失和指标（如果配置了 wandb）

**Actor 日志**：
- `starting actor loop`：Actor 已启动
- 环境交互信息
- 干预记录（如果使用手柄）

### 5.2 检查点保存

检查点保存在 `--checkpoint_path` 指定的目录：
```
./checkpoints/gear_assembly_rlpd/
├── checkpoint_0
├── checkpoint_5000
├── checkpoint_10000
├── ...
├── buffer/          # 在线数据缓冲区
└── demo_buffer/     # 演示数据缓冲区
```

**保存频率**：由 `config.checkpoint_period` 控制（默认 5000 步）

### 5.3 WandB 可视化配置

WandB（Weights & Biases）用于实时监控训练指标、可视化训练曲线和对比不同实验。

#### 5.3.1 注册 WandB 账号

**方法一：命令行注册（推荐）**

首次运行训练命令时，WandB 会提示：
```
wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results
wandb: Enter your choice:
```

选择 `1`，然后：
1. 按提示在浏览器中打开注册链接
2. 使用 GitHub/Google 账号登录，或创建新账号
3. 复制显示的 API key 并粘贴到终端

**方法二：网页注册**

1. 访问 https://wandb.ai/signup
2. 使用 GitHub/Google 账号登录，或创建新账号
3. 登录后访问 https://wandb.ai/authorize 获取 API key
4. 在终端运行：
   ```bash
   wandb login
   ```
5. 粘贴 API key

#### 5.3.2 查看训练可视化

注册成功后，训练日志会自动上传到 WandB。访问：
- **项目页面**：https://wandb.ai/your-username/hil-serl
- **实验页面**：训练启动时会显示 URL，类似：
  ```
  View run at: https://wandb.ai/your-username/hil-serl/runs/gear_assembly_20260130_154119
  ```

**记录的指标**：
- `critic_loss`：Critic 网络损失
- `actor_loss`：Actor 网络损失
- `rewards`：平均奖励
- `temperature`：SAC 温度参数
- `entropy`：策略熵
- `environment/episode/*`：Episode 统计（奖励、长度、干预次数等）
- `timer/*`：各阶段耗时

#### 5.3.3 禁用 WandB

如果不需要可视化，可以在命令中添加 `--debug` 参数：
```bash
python examples/train_rlpd.py \
    --exp_name=gear_assembly \
    --learner \
    --demo_path=./demo_data/gear_assembly_25_demos_2026-01-30_12-56-10.pkl \
    --checkpoint_path=./checkpoints/gear_assembly_rlpd \
    --seed=42 \
    --debug  # 禁用 WandB
```

---

## 六、训练配置

### 6.1 关键配置参数

在 `examples/experiments/gear_assembly/config.py` 中：

```python
class TrainConfig(DefaultTrainingConfig):
    batch_size = 256              # 批次大小
    cta_ratio = 2                 # Critic/Actor 更新比例
    discount = 0.97               # 折扣因子
    max_steps = 1000000           # 最大训练步数
    replay_buffer_capacity = 200000  # 缓冲区容量
    random_steps = 0              # 随机探索步数
    training_starts = 100          # 开始训练的缓冲区大小
    steps_per_update = 50         # 每次更新的步数
    checkpoint_period = 5000      # 检查点保存周期
    buffer_period = 1000          # 缓冲区保存周期
```

### 6.2 50/50 混合采样

RLPD 的核心特性：
- 每次训练迭代从两个缓冲区各采样 `batch_size // 2` 的数据
- 确保策略始终"看到"成功的演示数据
- 在稀疏奖励任务中特别有效

---

## 七、常见问题

### 7.1 Actor 无法连接 Learner

**症状**：Actor 启动后一直等待连接

**解决方案**：
1. 确认 Learner 已启动
2. 检查 `--ip` 参数是否正确
3. 检查防火墙设置

### 7.2 内存不足

**症状**：OOM (Out of Memory) 错误

**解决方案**：

**单机部署**：
```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3  # 降低到 0.2 或 0.1
```

**分布式部署**（Isaac Sim 在另一台机器）：
```bash
# 如果仍然 OOM，可以适当降低
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.7  # 从 .8 降到 .7
# 或进一步降低
export XLA_PYTHON_CLIENT_MEM_FRACTION=.6  # 降到 .6
```

### 7.3 训练速度慢

**可能原因**：
- Isaac Sim 服务器性能
- GPU 利用率低
- 网络通信延迟（分布式部署）

**优化建议**：
- **单机部署**：
  - 使用 `--headless=True` 运行 Isaac Sim（无 GUI）
  - 降低 `sim_hz`（如果允许）
  - 检查 GPU 使用情况

- **分布式部署**：
  - 确保网络带宽充足（建议千兆以太网或更高）
  - 检查网络延迟（`ping` 测试）
  - 如果 GPU 利用率低，可以增加 `MEM_FRACTION` 到 `.9`（如果内存充足）
  - 考虑使用更快的网络连接（InfiniBand、10GbE 等）

### 7.4 恢复训练

如果训练中断，可以从检查点恢复：

```bash
# Learner 和 Actor 使用相同的 --checkpoint_path
# 程序会自动检测并加载最新的检查点
python examples/train_rlpd.py \
    --exp_name=gear_assembly \
    --learner \
    --checkpoint_path=./checkpoints/gear_assembly_rlpd \
    --demo_path=./demo_data/gear_assembly_25_demos_2026-01-30_12-56-10.pkl
```

---

## 八、训练完成后的评估

### 8.1 评估检查点

```bash
python examples/train_rlpd.py \
    --exp_name=gear_assembly \
    --actor \
    --use_sim \
    --checkpoint_path=./checkpoints/gear_assembly_rlpd \
    --eval_checkpoint_step=50000 \
    --eval_n_trajs=10 \
    --save_video
```

**参数说明**：
- `--eval_checkpoint_step`：要评估的检查点步数
- `--eval_n_trajs`：评估轨迹数量
- `--save_video`：保存评估视频

---

## 九、总结

### 9.1 训练检查清单

- [x] 演示数据已收集（25 条轨迹）
- [x] Isaac Sim 环境配置完成
- [x] 训练代码支持仿真环境
- [ ] Isaac Sim 服务器已启动
- [ ] Learner 进程已启动
- [ ] Actor 进程已启动
- [ ] 训练正常进行

### 9.2 下一步

1. **开始训练**：按照上述命令启动三个进程
2. **监控训练**：观察日志和检查点保存
3. **评估策略**：训练一段时间后评估性能
4. **持续改进**：根据结果调整超参数或收集更多演示数据

---

## 十、参考文档

- `docs/RLPD_Algorithm_Detail.md`：RLPD 算法详解
- `docs/三种训练策略核心分析.md`：训练策略对比
- `docs/原项目分析/train_rlpd_analysis.md`：训练代码分析

---

**祝训练顺利！** 🚀

