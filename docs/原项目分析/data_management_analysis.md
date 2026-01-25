# 原项目数据管理机制分析

## 概述

原项目的数据管理采用**分布式架构**，数据在 Actor（数据收集）和 Learner（模型训练）之间通过网络传输。Flask 服务器**仅负责机械臂控制和状态查询**，不参与数据管理。

## 数据架构

```
┌─────────────────────────────────────────────────────────────┐
│                     数据管理架构                              │
└─────────────────────────────────────────────────────────────┘

┌──────────────┐         ┌──────────────┐         ┌──────────────┐
│   Actor 进程  │         │  Learner 进程 │         │  磁盘存储     │
│              │         │              │         │              │
│ ┌──────────┐ │         │ ┌──────────┐ │         │ ┌──────────┐ │
│ │ 环境交互  │ │         │ │ 模型训练  │ │         │ │ Pickle   │ │
│ │ 数据收集  │ │         │ │ 数据采样  │ │         │ │ 文件     │ │
│ └────┬─────┘ │         │ └────┬─────┘ │         │ └────┬─────┘ │
│      │       │         │      │       │         │      │       │
│ ┌────▼─────┐ │         │ ┌────▼─────┐ │         │      │       │
│ │QueuedData│ │────────►│ │ReplayBuffer│ │◄────────┘      │       │
│ │  Store   │ │ 网络    │ │DataStore  │ │   恢复数据      │       │
│ └──────────┘ │         │ └──────────┘ │                  │       │
│              │         │              │                  │       │
│ ┌──────────┐ │         │ ┌──────────┐ │                  │       │
│ │QueuedData│ │────────►│ │ReplayBuffer│ │                  │       │
│ │  Store   │ │ 网络    │ │DataStore  │ │                  │       │
│ │(干预数据) │ │         │ │(演示数据) │ │                  │       │
│ └──────────┘ │         │ └──────────┘ │                  │       │
│              │         │              │                  │       │
│ TrainerClient│◄────────►│ TrainerServer│                  │       │
└──────────────┘         └──────────────┘                  │       │
      │                        │                             │       │
      └────────────────────────┴─────────────────────────────┘       │
                   定期保存到磁盘                                      │
```

## 数据类型

### 1. 示范数据（Demonstration Data）

**收集方式**：
- 使用 `examples/record_demos.py` 脚本收集
- 通过 SpaceMouse 人工控制机器人完成任务
- 只保存成功完成的轨迹（`info["succeed"] == True`）

**存储格式**：
```python
# 每个 transition 的格式
transition = {
    "observations": {
        "images": {"wrist_1": np.ndarray, "wrist_2": np.ndarray},
        "state": {
            "tcp_pose": np.ndarray,      # 7维：xyz + quat
            "tcp_vel": np.ndarray,       # 6维
            "gripper_pose": np.ndarray,  # 1维
            "tcp_force": np.ndarray,     # 3维
            "tcp_torque": np.ndarray,    # 3维
        }
    },
    "actions": np.ndarray,              # 7维动作
    "next_observations": {...},         # 同 observations
    "rewards": float,                   # 奖励信号
    "masks": float,                     # 1.0 - done
    "dones": bool,                      # 是否结束
    "infos": dict,                      # 额外信息
}
```

**存储位置**：
- 文件：`./demo_data/{exp_name}_{n}_demos_{timestamp}.pkl`
- 格式：pickle 文件，包含所有成功轨迹的 transitions 列表

**使用方式**：
```python
# 在 Learner 中加载
for path in FLAGS.demo_path:
    with open(path, "rb") as f:
        transitions = pkl.load(f)
        for transition in transitions:
            demo_buffer.insert(transition)
```

### 2. 在线数据（Online Experience Data）

**收集方式**：
- Actor 在训练过程中与环境交互收集
- 包括策略动作和随机探索动作
- 所有 transition 都存入在线缓冲区

**数据流**：
```python
# Actor 端：examples/train_rlpd.py
for step in range(config.max_steps):
    # 1. 采样动作
    actions = agent.sample_actions(obs)  # 或随机动作
    
    # 2. 环境步进
    next_obs, reward, done, truncated, info = env.step(actions)
    
    # 3. 创建 transition
    transition = {
        "observations": obs,
        "actions": actions,
        "next_observations": next_obs,
        "rewards": reward,
        "masks": 1.0 - done,
        "dones": done,
    }
    
    # 4. 存储到本地队列
    data_store.insert(transition)  # QueuedDataStore
    
    # 5. 自动通过网络发送到 Learner
    # (TrainerClient 自动处理)
```

**存储位置**：
- **内存**：Actor 端使用 `QueuedDataStore(capacity=50000)` 作为本地队列
- **网络传输**：通过 `TrainerClient` 自动发送到 Learner
- **Learner 内存**：使用 `MemoryEfficientReplayBufferDataStore(capacity=200000)` 存储
- **磁盘备份**：定期保存到 `{checkpoint_path}/buffer/transitions_{step}.pkl`

**持久化**：
```python
# 定期保存到磁盘
if step % config.buffer_period == 0:
    buffer_path = os.path.join(FLAGS.checkpoint_path, "buffer")
    with open(os.path.join(buffer_path, f"transitions_{step}.pkl"), "wb") as f:
        pkl.dump(transitions, f)
        transitions = []  # 清空列表
```

### 3. 人工干预数据（Intervention Data）

**收集方式**：
- 通过 `SpacemouseIntervention` wrapper 检测人类干预
- 当 SpaceMouse 有输入时，`info["intervene_action"]` 包含干预动作
- 干预动作会**覆盖**策略动作，但记录时使用干预动作

**数据流**：
```python
# Actor 端：examples/train_rlpd.py
next_obs, reward, done, truncated, info = env.step(actions)

# 检测干预
if "intervene_action" in info:
    actions = info.pop("intervene_action")  # 使用干预动作
    intervention_steps += 1
    already_intervened = True
else:
    already_intervened = False

# 创建 transition（使用干预动作）
transition = {
    "observations": obs,
    "actions": actions,  # 如果是干预，这里是干预动作
    "next_observations": next_obs,
    "rewards": reward,
    "masks": 1.0 - done,
    "dones": done,
}

# 所有数据存入在线缓冲区
data_store.insert(transition)

# 干预数据额外存入演示缓冲区
if already_intervened:
    intvn_data_store.insert(transition)  # 单独存储
```

**存储位置**：
- **内存**：Actor 端使用 `QueuedDataStore(capacity=50000)` 作为本地队列
- **网络传输**：通过 `TrainerClient` 自动发送到 Learner
- **Learner 内存**：使用 `MemoryEfficientReplayBufferDataStore` 作为 `demo_buffer`
- **磁盘备份**：定期保存到 `{checkpoint_path}/demo_buffer/transitions_{step}.pkl`

**特点**：
- 干预数据**同时**存入在线缓冲区和演示缓冲区
- 在 RLPD 训练中，干预数据被视为演示数据的一部分
- 统计信息：`intervention_count`（每个回合的干预次数）和 `intervention_steps`（每个回合的干预步数）

## 数据存储组件

### 1. QueuedDataStore（Actor 端）

**位置**：`agentlace.data.data_store.QueuedDataStore`

**功能**：
- 本地队列缓冲区（容量：50000）
- 自动通过网络将数据发送到 Learner
- 线程安全的数据插入

**使用**：
```python
from agentlace.data.data_store import QueuedDataStore

# Actor 端创建
data_store = QueuedDataStore(50000)  # 在线经验
intvn_data_store = QueuedDataStore(50000)  # 干预数据

# 连接到 Learner
client = TrainerClient(
    "actor_env",
    FLAGS.ip,  # Learner 的 IP 地址
    make_trainer_config(),
    data_stores={
        "actor_env": data_store,
        "actor_env_intvn": intvn_data_store,
    },
    wait_for_server=True,
)

# 插入数据（自动发送到 Learner）
data_store.insert(transition)
```

### 2. ReplayBufferDataStore（Learner 端）

**位置**：`serl_launcher.serl_launcher.data.data_store.ReplayBufferDataStore`

**功能**：
- 本地 ReplayBuffer 存储（容量：200000）
- 从 Actor 接收数据
- 支持数据采样用于训练

**使用**：
```python
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

# Learner 端创建
replay_buffer = MemoryEfficientReplayBufferDataStore(
    env.observation_space,
    env.action_space,
    capacity=config.replay_buffer_capacity,  # 200000
    image_keys=config.image_keys,
)

demo_buffer = MemoryEfficientReplayBufferDataStore(
    env.observation_space,
    env.action_space,
    capacity=config.replay_buffer_capacity,
    image_keys=config.image_keys,
)

# 注册到 TrainerServer
server = TrainerServer(make_trainer_config())
server.register_data_store("actor_env", replay_buffer)
server.register_data_store("actor_env_intvn", demo_buffer)
server.start(threaded=True)

# 从缓冲区采样
replay_iterator = replay_buffer.get_iterator(
    sample_args={
        "batch_size": config.batch_size // 2,
        "pack_obs_and_next_obs": True,
    },
    device=sharding.replicate(),
)
```

### 3. TrainerClient / TrainerServer（网络通信）

**位置**：`agentlace.trainer`

**功能**：
- Actor 和 Learner 之间的网络通信
- 数据自动同步
- 策略参数更新推送

**通信流程**：
```
Actor (TrainerClient)              Learner (TrainerServer)
     │                                    │
     │  connect()                         │
     ├───────────────────────────────────►│
     │                                    │
     │  insert(transition)                │
     ├───────────────────────────────────►│
     │                                    │  replay_buffer.insert()
     │                                    │
     │  recv_network_callback()           │
     │◄───────────────────────────────────┤  publish_network(params)
     │  update agent params               │
```

## 数据持久化

### 1. 定期保存

**Actor 端**：
```python
# examples/train_rlpd.py
if step % config.buffer_period == 0:
    # 保存在线数据
    buffer_path = os.path.join(FLAGS.checkpoint_path, "buffer")
    with open(os.path.join(buffer_path, f"transitions_{step}.pkl"), "wb") as f:
        pkl.dump(transitions, f)
        transitions = []
    
    # 保存干预数据
    demo_buffer_path = os.path.join(FLAGS.checkpoint_path, "demo_buffer")
    with open(os.path.join(demo_buffer_path, f"transitions_{step}.pkl"), "wb") as f:
        pkl.dump(demo_transitions, f)
        demo_transitions = []
```

**文件结构**：
```
{checkpoint_path}/
├── buffer/
│   ├── transitions_1000.pkl
│   ├── transitions_2000.pkl
│   └── ...
└── demo_buffer/
    ├── transitions_1000.pkl
    ├── transitions_2000.pkl
    └── ...
```

### 2. 数据恢复

**Learner 端**：
```python
# examples/train_rlpd.py
# 恢复在线缓冲区
if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path):
    buffer_path = os.path.join(FLAGS.checkpoint_path, "buffer")
    for file in glob.glob(os.path.join(buffer_path, "*.pkl")):
        with open(file, "rb") as f:
            transitions = pkl.load(f)
            for transition in transitions:
                replay_buffer.insert(transition)

# 恢复演示缓冲区
demo_buffer_path = os.path.join(FLAGS.checkpoint_path, "demo_buffer")
for file in glob.glob(os.path.join(demo_buffer_path, "*.pkl")):
    with open(file, "rb") as f:
        transitions = pkl.load(f)
        for transition in transitions:
            demo_buffer.insert(transition)
```

## 数据采样策略（RLPD）

### 1. 50/50 混合采样

**核心思想**：每个训练批次中，50% 来自在线经验，50% 来自演示数据。

**实现**：
```python
# Learner 端：examples/train_rlpd.py
replay_iterator = replay_buffer.get_iterator(
    sample_args={
        "batch_size": config.batch_size // 2,  # 一半来自在线经验
        "pack_obs_and_next_obs": True,
    },
    device=sharding.replicate(),
)

demo_iterator = demo_buffer.get_iterator(
    sample_args={
        "batch_size": config.batch_size // 2,  # 一半来自演示数据
        "pack_obs_and_next_obs": True,
    },
    device=sharding.replicate(),
)

# 训练循环
for step in range(config.max_steps):
    # 采样批次
    batch = next(replay_iterator)  # 在线经验
    demo_batch = next(demo_iterator)  # 演示数据
    
    # 合并批次
    batch = concat_batches(batch, demo_batch, axis=0)
    
    # 更新策略
    agent, update_info = agent.update(batch, ...)
```

### 2. 隐式权重差异

虽然采样比例是固定的 50/50，但**通过奖励信号，模型会隐式地更多地学习演示数据**：

- **演示数据**：通常有奖励（`rewards > 0`），产生更大的梯度
- **在线经验**：在训练初期可能没有奖励，梯度较小
- **结果**：虽然显式权重相同，但隐式权重不同

## 数据流总结

### 完整数据流

```
1. 示范数据收集
   record_demos.py
   └─► SpaceMouse 控制
       └─► 成功轨迹
           └─► ./demo_data/*.pkl

2. 训练时数据收集（Actor）
   env.step(actions)
   └─► transition
       ├─► data_store.insert()  ──► QueuedDataStore ──► TrainerClient ──► Learner
       └─► (if intervention) intvn_data_store.insert() ──► QueuedDataStore ──► Learner

3. 数据接收（Learner）
   TrainerServer
   └─► replay_buffer.insert()  (在线经验)
   └─► demo_buffer.insert()    (干预数据)

4. 数据采样（Learner）
   replay_buffer.sample()  (50%)
   demo_buffer.sample()    (50%)
   └─► 合并批次
       └─► agent.update()

5. 数据持久化
   Actor: 定期保存到 buffer/*.pkl 和 demo_buffer/*.pkl
   Learner: 从磁盘恢复数据
```

## 关键要点

1. **Flask 服务器不参与数据管理**
   - Flask 仅负责机械臂控制和状态查询
   - 数据管理完全由 Actor/Learner 架构处理

2. **分布式数据存储**
   - Actor 端：本地队列（QueuedDataStore）
   - Learner 端：ReplayBuffer（MemoryEfficientReplayBufferDataStore）
   - 网络传输：TrainerClient/Server

3. **数据分类**
   - **示范数据**：预收集的成功轨迹（pickle 文件）
   - **在线数据**：训练过程中的所有经验
   - **干预数据**：训练过程中的人类干预（同时存入在线和演示缓冲区）

4. **数据持久化**
   - 定期保存到磁盘（pickle 格式）
   - 支持训练中断后恢复
   - 文件路径：`{checkpoint_path}/buffer/` 和 `{checkpoint_path}/demo_buffer/`

5. **采样策略**
   - RLPD：50/50 混合采样（在线经验 + 演示数据）
   - 通过奖励信号实现隐式权重差异

## 与 Isaac Sim 方案的关联

在 Isaac Sim 方案中，数据管理机制**完全保持不变**：

- Flask 服务器仍然只负责仿真环境控制和状态查询
- Actor/Learner 架构和数据管理机制不变
- 数据存储、传输、采样策略都不变
- 唯一变化：环境从真机（Franka Robot）变为仿真（Isaac Sim）

这意味着：
- ✅ 数据管理代码可以完全复用
- ✅ 训练流程不需要修改
- ✅ 只需要替换环境接口（从 `FrankaEnv` 到 `IsaacSimFrankaEnv`）
