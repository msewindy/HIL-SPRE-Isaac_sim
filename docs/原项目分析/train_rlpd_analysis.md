# train_rlpd.py 代码逻辑分析

## 概述

`train_rlpd.py` 是 HIL-SERL 框架中用于 RLPD (Reinforcement Learning from Preferential Demonstrations) 训练的主脚本。该脚本实现了 **Actor-Learner 分离架构**，支持在线强化学习与演示数据混合训练，并集成了人类干预机制。

---

## 核心架构

### 1. Actor-Learner 分离架构

```
┌─────────────────┐         ┌─────────────────┐
│   Actor 进程    │         │  Learner 进程   │
│                 │         │                 │
│  - 数据收集      │◄───────►│  - 模型训练     │
│  - 环境交互      │  网络   │  - 参数更新     │
│  - 干预记录      │  通信   │  - 检查点保存   │
└─────────────────┘         └─────────────────┘
```

**关键组件**：
- **Actor**: 负责与环境交互，收集训练数据，支持人类干预
- **Learner**: 负责模型训练，参数更新，数据管理
- **通信**: 使用 `agentlace` 框架的 `TrainerClient` 和 `TrainerServer` 进行异步通信

---

## 主要函数分析

### 1. `main()` - 主入口函数

**功能**：初始化环境和智能体，根据命令行参数启动 Actor 或 Learner 进程。

**关键步骤**：

1. **配置加载**
   ```python
   config = CONFIG_MAPPING[FLAGS.exp_name]()
   ```
   - 从 `experiments.mappings` 加载实验配置
   - 支持多种任务：`ram_insertion`, `usb_pickup_insertion`, `object_handover`, `egg_flip`

2. **环境创建**
   ```python
   env = config.get_environment(
       fake_env=FLAGS.learner,  # Learner 使用 fake_env=True (仿真)
       save_video=FLAGS.save_video,
       classifier=True,
   )
   ```
   - `fake_env=True`: Learner 使用仿真环境（不连接真实机器人）
   - `fake_env=False`: Actor 使用真实环境（连接真实机器人）

3. **智能体创建**（根据设置模式）
   - **单臂固定夹爪** (`single-arm-fixed-gripper`): `SACAgent`
   - **单臂学习夹爪** (`single-arm-learned-gripper`): `SACAgentHybridSingleArm`
   - **双臂学习夹爪** (`dual-arm-learned-gripper`): `SACAgentHybridDualArm`

4. **检查点恢复**
   - 如果 `checkpoint_path` 存在，自动加载最新检查点
   - 恢复智能体状态和训练缓冲区

5. **进程启动**
   - `FLAGS.learner=True`: 启动 Learner 进程
   - `FLAGS.actor=True`: 启动 Actor 进程

---

### 2. `actor()` - Actor 循环函数

**功能**：Actor 进程的主循环，负责数据收集、环境交互和干预记录。

#### 2.1 评估模式

如果设置了 `eval_checkpoint_step`，Actor 进入评估模式：

```python
if FLAGS.eval_checkpoint_step:
    # 加载检查点
    ckpt = checkpoints.restore_checkpoint(...)
    agent = agent.replace(state=ckpt)
    
    # 运行评估轨迹
    for episode in range(FLAGS.eval_n_trajs):
        # 执行策略，收集成功率统计
```

**输出**：
- 成功率 (`success rate`)
- 平均完成时间 (`average time`)

#### 2.2 训练模式

**初始化**：

1. **数据存储**
   ```python
   data_store = QueuedDataStore(50000)  # 在线经验缓冲区
   intvn_data_store = QueuedDataStore(50000)  # 干预数据缓冲区
   ```

2. **TrainerClient 连接**
   ```python
   client = TrainerClient(
       "actor_env",
       FLAGS.ip,  # Learner 的 IP 地址
       make_trainer_config(),
       data_stores=datastore_dict,
       wait_for_server=True,
   )
   ```

3. **参数更新回调**
   ```python
   def update_params(params):
       nonlocal agent
       agent = agent.replace(state=agent.state.replace(params=params))
   
   client.recv_network_callback(update_params)
   ```
   - 当 Learner 发布新参数时，自动更新 Actor 的策略网络

**主循环**：

```python
for step in range(start_step, config.max_steps):
    # 1. 动作采样
    if step < config.random_steps:
        actions = env.action_space.sample()  # 随机探索
    else:
        actions = agent.sample_actions(obs, seed=key)  # 策略采样
    
    # 2. 环境步进
    next_obs, reward, done, truncated, info = env.step(actions)
    
    # 3. 干预处理
    if "intervene_action" in info:
        actions = info.pop("intervene_action")  # 使用干预动作
        intervention_steps += 1
        if not already_intervened:
            intervention_count += 1
        already_intervened = True
    
    # 4. 数据存储
    transition = {
        "observations": obs,
        "actions": actions,  # 注意：如果发生干预，这里是干预动作
        "next_observations": next_obs,
        "rewards": reward,
        "masks": 1.0 - done,
        "dones": done,
    }
    data_store.insert(transition)  # 所有数据存入在线缓冲区
    
    if already_intervened:
        intvn_data_store.insert(transition)  # 干预数据单独存储
```

**关键特性**：

1. **干预机制**
   - 通过 `SpacemouseIntervention` wrapper 检测人类干预
   - 当 SpaceMouse 有输入时，`info["intervene_action"]` 包含干预动作
   - 干预动作会**覆盖**策略动作，但记录时使用干预动作
   - 干预数据同时存入 `data_store` 和 `intvn_data_store`

2. **数据持久化**
   ```python
   if step % config.buffer_period == 0:
       # 保存到 pickle 文件
       pkl.dump(transitions, f"buffer/transitions_{step}.pkl")
       pkl.dump(demo_transitions, f"demo_buffer/transitions_{step}.pkl")
   ```
   - 定期将缓冲区数据保存到磁盘
   - 支持训练中断后恢复

3. **统计信息上报**
   ```python
   if done or truncated:
       info["episode"]["intervention_count"] = intervention_count
       info["episode"]["intervention_steps"] = intervention_steps
       client.request("send-stats", {"environment": info})
   ```
   - 每个回合结束后，将统计信息发送给 Learner
   - Learner 通过 wandb 记录这些信息

---

### 3. `learner()` - Learner 循环函数

**功能**：Learner 进程的主循环，负责模型训练、参数更新和数据管理。

#### 3.1 初始化

1. **TrainerServer 创建**
   ```python
   server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
   server.register_data_store("actor_env", replay_buffer)
   server.register_data_store("actor_env_intvn", demo_buffer)
   server.start(threaded=True)
   ```
   - 注册两个数据存储：在线经验缓冲区 (`actor_env`) 和演示缓冲区 (`actor_env_intvn`)

2. **演示数据加载**
   ```python
   for path in FLAGS.demo_path:
       with open(path, "rb") as f:
           transitions = pkl.load(f)
           for transition in transitions:
               demo_buffer.insert(transition)
   ```
   - 从 pickle 文件加载预收集的演示数据
   - 支持多个演示文件

3. **缓冲区恢复**
   ```python
   # 恢复在线缓冲区
   for file in glob.glob("buffer/*.pkl"):
       transitions = pkl.load(f)
       replay_buffer.insert(transition)
   
   # 恢复演示缓冲区
   for file in glob.glob("demo_buffer/*.pkl"):
       transitions = pkl.load(f)
       demo_buffer.insert(transition)
   ```

4. **等待缓冲区填充**
   ```python
   while len(replay_buffer) < config.training_starts:
       time.sleep(1)  # 等待 Actor 收集足够数据
   ```

5. **发布初始网络**
   ```python
   server.publish_network(agent.state.params)
   ```
   - 将初始策略参数发送给 Actor

#### 3.2 训练循环

**关键特性：50/50 采样策略（RLPD 核心）**

```python
# 创建两个迭代器，各采样 batch_size // 2
replay_iterator = replay_buffer.get_iterator(
    batch_size=config.batch_size // 2,  # 一半来自在线经验
)
demo_iterator = demo_buffer.get_iterator(
    batch_size=config.batch_size // 2,  # 一半来自演示数据
)

# 训练时合并两个批次
batch = concat_batches(batch, demo_batch, axis=0)
```

**训练步骤**：

1. **Critic 更新**（多次）
   ```python
   for critic_step in range(config.cta_ratio - 1):
       batch = concat_batches(next(replay_iterator), next(demo_iterator))
       agent, critics_info = agent.update(
           batch,
           networks_to_update={"critic"},  # 或 {"critic", "grasp_critic"}
       )
   ```
   - 执行 `cta_ratio - 1` 次 Critic 更新
   - 减少 CPU-GPU 数据传输次数，提高训练效率

2. **Critic + Actor 更新**（一次）
   ```python
   batch = concat_batches(next(replay_iterator), next(demo_iterator))
   agent, update_info = agent.update(
       batch,
       networks_to_update={"critic", "actor", "temperature"},
   )
   ```

3. **参数发布**
   ```python
   if step % config.steps_per_update == 0:
       server.publish_network(agent.state.params)
   ```
   - 定期将更新后的参数发送给 Actor

4. **检查点保存**
   ```python
   if step % config.checkpoint_period == 0:
       checkpoints.save_checkpoint(checkpoint_path, agent.state, step=step)
   ```

---

## 关键机制详解

### 1. 干预机制（Intervention）

**目的**：允许人类专家在训练过程中实时纠正策略行为。

**实现流程**：

1. **检测干预**
   - `SpacemouseIntervention` wrapper 检测 SpaceMouse 输入
   - 如果有输入，在 `info["intervene_action"]` 中设置干预动作

2. **动作覆盖**
   ```python
   if "intervene_action" in info:
       actions = info.pop("intervene_action")  # 使用干预动作
   ```
   - 策略动作被干预动作覆盖
   - 环境执行的是干预动作

3. **数据记录**
   - 所有 transition 存入 `data_store`（在线经验）
   - 干预的 transition 额外存入 `intvn_data_store`（演示数据）
   - Learner 从 `intvn_data_store` 采样演示数据

**统计信息**：
- `intervention_count`: 每个回合的干预次数
- `intervention_steps`: 每个回合的干预步数

### 2. RLPD 训练策略

**核心思想**：混合在线强化学习和演示学习。

**数据来源**：
1. **在线经验** (`replay_buffer`): Actor 与环境交互收集的数据
2. **演示数据** (`demo_buffer`): 
   - 预收集的演示数据（`--demo_path`）
   - 训练过程中的干预数据（`intvn_data_store`）

**采样策略**：
- **50/50 混合**：每个训练批次中，50% 来自在线经验，50% 来自演示数据
- 这确保了策略既能从演示中学习，又能通过在线探索改进

**重要说明**：虽然采样比例是固定的 50/50，但**通过奖励信号，模型会隐式地更多地学习演示数据**：
- 演示数据通常有奖励（`rewards > 0`），产生更大的梯度
- 在线经验在训练初期可能没有奖励，梯度较小
- 因此，虽然显式权重相同，但隐式权重不同

**关键机制**：在人类干预情况下能够正常完成几次任务，那么这几次奖励就会极大地影响 Critic 的分布：
- **稀疏奖励放大效应**：在稀疏奖励任务中，演示数据中的正奖励样本密度远高于在线经验（80-90% vs 1-2%）
- **梯度贡献不成比例**：虽然采样比例是 50/50，但演示数据贡献了 96% 以上的有效梯度
- **Critic 分布偏移**：Critic 学习到演示数据的高 Q 值（0.8-1.0），在线经验的低 Q 值（0.0-0.2）
- **价值传播效应**：通过 Bellman 方程，奖励信号会向前传播，整个演示轨迹的 Q 值都被提升

**详细分析**：
- 参见 `docs/critic_learning_mechanism.md` 了解为什么人类干预的奖励会极大地影响 Critic 分布的详细数学分析

### 3. 数据同步机制

**Actor → Learner**：
- `data_store.insert(transition)`: 异步插入数据
- `TrainerClient` 自动将数据发送到 `TrainerServer`

**Learner → Actor**：
- `server.publish_network(params)`: 定期发布参数更新
- `client.recv_network_callback(update_params)`: Actor 自动接收并更新

**优势**：
- 异步通信，不阻塞训练
- 支持多个 Actor 同时收集数据
- 自动处理网络延迟和重连

### 4. 多种智能体类型

**SACAgent**（固定夹爪）：
- 标准 SAC 算法
- 夹爪动作由环境控制（如 `GripperCloseEnv` wrapper）

**SACAgentHybridSingleArm**（学习夹爪）：
- 混合 SAC 算法
- 包含 `grasp_critic` 网络
- 支持 `grasp_penalty` 奖励项

**SACAgentHybridDualArm**（双臂学习夹爪）：
- 双臂版本
- 两个夹爪独立学习

---

## 命令行参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `--exp_name` | string | 实验名称（对应配置文件夹） |
| `--seed` | int | 随机种子（默认 42） |
| `--learner` | bool | 是否作为 Learner 运行 |
| `--actor` | bool | 是否作为 Actor 运行 |
| `--ip` | string | Learner 的 IP 地址（默认 localhost） |
| `--demo_path` | multi_string | 演示数据路径（可多个） |
| `--checkpoint_path` | string | 检查点保存路径 |
| `--eval_checkpoint_step` | int | 评估模式：检查点步数 |
| `--eval_n_trajs` | int | 评估模式：轨迹数量 |
| `--save_video` | bool | 是否保存视频 |
| `--debug` | bool | 调试模式（禁用 wandb） |

---

## 数据流图

```
┌─────────────────────────────────────────────────────────────┐
│                      Actor 进程                             │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  env.step() → transition → data_store ──────┐               │
│         │                                    │               │
│         └─ intervention? → intvn_data_store ─┘               │
│                                                               │
│         ┌──────────────────────────────────┐                │
│         │  TrainerClient                   │                │
│         │  - 发送数据到 Learner            │                │
│         │  - 接收参数更新                  │                │
│         └──────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────┘
                          │
                          │ 网络通信
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                     Learner 进程                             │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐      ┌──────────────┐                    │
│  │ replay_buffer │      │ demo_buffer  │                    │
│  │ (在线经验)    │      │ (演示数据)   │                    │
│  └──────┬───────┘      └──────┬───────┘                    │
│         │                      │                             │
│         └──────────┬───────────┘                             │
│                    │ 50/50 采样                              │
│                    ▼                                         │
│            agent.update(batch)                               │
│                    │                                         │
│                    ▼                                         │
│         server.publish_network()                             │
│                                                               │
│         ┌──────────────────────────────────┐                │
│         │  TrainerServer                   │                │
│         │  - 接收 Actor 数据               │                │
│         │  - 发布参数更新                  │                │
│         └──────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

---

## 训练流程示例

### 启动训练

**终端 1 - Learner**:
```bash
python examples/train_rlpd.py \
    --exp_name ram_insertion \
    --learner \
    --demo_path /path/to/demos.pkl \
    --checkpoint_path ./checkpoints/ram_insertion
```

**终端 2 - Actor**:
```bash
python examples/train_rlpd.py \
    --exp_name ram_insertion \
    --actor \
    --ip localhost \
    --checkpoint_path ./checkpoints/ram_insertion
```

### 训练过程

1. **初始化阶段**
   - Learner 加载演示数据到 `demo_buffer`
   - Learner 启动 `TrainerServer`，等待 Actor 连接
   - Actor 连接 Learner，接收初始策略参数

2. **数据收集阶段**
   - Actor 开始与环境交互
   - 前 `random_steps` 步使用随机动作探索
   - 之后使用策略采样动作
   - 人类可以通过 SpaceMouse 进行干预

3. **训练阶段**
   - Learner 等待 `replay_buffer` 达到 `training_starts` 大小
   - 开始训练循环：
     - 从 `replay_buffer` 和 `demo_buffer` 各采样 50%
     - 更新 Critic（多次）和 Actor（一次）
     - 定期发布参数给 Actor
     - 定期保存检查点

4. **持续迭代**
   - Actor 持续收集新数据
   - Learner 持续训练并更新参数
   - 策略性能逐步提升

---

## 关键设计决策

### 1. 为什么使用 Actor-Learner 分离？

- **实时性**：Actor 需要实时控制机器人，不能等待训练完成
- **资源利用**：训练在 GPU 上进行，数据收集在 CPU/机器人上进行
- **可扩展性**：支持多个 Actor 同时收集数据

### 2. 为什么使用 50/50 采样？

- **平衡探索和利用**：在线经验提供探索，演示数据提供利用
- **稳定训练**：演示数据确保策略不会偏离太远
- **加速学习**：演示数据提供高质量样本

### 3. 为什么干预数据单独存储？

- **数据质量**：干预数据是专家演示，质量更高
- **灵活采样**：可以独立控制演示数据和在线经验的采样比例
- **分析方便**：可以单独分析干预数据的效果

---

## 与相关脚本的对比

### `train_rlpd.py` vs `train_hgdagger.py`

| 特性 | train_rlpd.py | train_hgdagger.py |
|------|---------------|-------------------|
| 算法 | RLPD (SAC + 演示) | HG-DAgger (BC + 在线) |
| 数据存储 | 所有数据 + 干预数据 | 仅干预数据 |
| 采样策略 | 50/50 混合 | 仅干预数据 |
| 智能体类型 | SACAgent | BCAgent |

### `train_rlpd.py` vs `record_demos.py`

- `record_demos.py`: 仅用于收集演示数据，不进行训练
- `train_rlpd.py`: 使用演示数据进行训练，同时支持在线收集

---

## 总结

`train_rlpd.py` 是 HIL-SERL 框架的核心训练脚本，实现了：

1. **Actor-Learner 分离架构**：支持分布式训练
2. **RLPD 训练策略**：混合在线强化学习和演示学习
3. **人类干预机制**：允许实时纠正策略行为
4. **灵活的数据管理**：支持检查点恢复、数据持久化
5. **多种智能体支持**：适应不同的任务设置

该脚本为 HIL-SERL 框架提供了完整的训练基础设施，支持从演示数据中学习，同时通过在线探索持续改进策略。
