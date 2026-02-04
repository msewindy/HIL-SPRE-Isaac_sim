# RLPD 步数关系分析：Learner 和 Actor

## 一、核心问题

1. **文档中提到的"100步左右"和 learner 日志里的步数是什么关系？**
2. **Learner 迭代频率和 Actor 参数更新频率分别是多少？**
3. **100步是 Actor 从 Learner 更新了 100 次吗？**

## 二、两种不同的"步数"

### 2.1 Learner 步数（训练迭代次数）

**定义**：Learner 的训练迭代次数

**代码位置**：
```python
# examples/train_rlpd.py, line 334-336
for step in tqdm.tqdm(
    range(start_step, config.max_steps), dynamic_ncols=True, desc="learner"
):
```

**含义**：
- 这是 **训练迭代次数**，不是环境交互次数
- 每次迭代包括：
  - `(cta_ratio - 1)` 次 Critic 更新（默认 1 次）
  - 1 次 Critic + Actor 更新
  - 数据采样和 GPU 计算

**日志示例**：
```
learner:   1%|▊| 12804/1000000 [09:53<12:21:42, 22.18it/s]
```
- `12804`：Learner 的训练迭代次数
- `1000000`：最大训练迭代次数（`config.max_steps`）

### 2.2 Actor 步数（环境交互次数）

**定义**：Actor 与环境交互的次数

**代码位置**：
```python
# examples/train_rlpd.py, line 161-162
pbar = tqdm.tqdm(range(start_step, config.max_steps), dynamic_ncols=True)
for step in pbar:
```

**含义**：
- 这是 **环境交互次数**，每次迭代：
  - 采样动作
  - 执行 `env.step(actions)`
  - 收集 transition 数据

**关键点**：
- Actor 和 Learner 的步数是 **异步的**，没有直接对应关系
- Actor 可能比 Learner 快很多（只做推理，不做训练）

## 三、参数更新频率

### 3.1 Learner 发布参数的频率

**代码位置**：
```python
# examples/train_rlpd.py, line 360-362
if step > 0 and step % (config.steps_per_update) == 0:
    agent = jax.block_until_ready(agent)
    server.publish_network(agent.state.params)
```

**配置**：
```python
# examples/experiments/config.py
steps_per_update: int = 50  # 默认值
```

**含义**：
- Learner **每 50 个训练迭代**发布一次参数给 Actor
- 即：Learner 步数 50, 100, 150, 200, ... 时发布参数

**计算公式**：
```
参数发布次数 = Learner步数 / steps_per_update
```

**示例**：
- Learner 步数 100 → Actor 收到 2 次更新（步数 50 和 100）
- Learner 步数 200 → Actor 收到 4 次更新（步数 50, 100, 150, 200）

### 3.2 Actor 接收参数的频率

**代码位置**：
```python
# examples/train_rlpd.py, line 141-146
def update_params(params):
    nonlocal agent
    agent = agent.replace(state=agent.state.replace(params=params))

client.recv_network_callback(update_params)
```

**含义**：
- Actor **异步接收** Learner 发布的参数
- 每当 Learner 发布参数时，Actor 自动更新
- 更新是**立即生效**的，不需要等待

## 四、步数关系详解

### 4.1 时间线示例

假设 `steps_per_update = 50`：

| 时间 | Learner 步数 | Actor 步数 | 参数更新 | 说明 |
|------|-------------|-----------|---------|------|
| T0 | 0 | 0 | 初始参数 | Learner 发布初始参数 |
| T1 | 50 | ~500-1000 | 第1次更新 | Learner 训练50步后发布 |
| T2 | 100 | ~1000-2000 | 第2次更新 | Learner 训练100步后发布 |
| T3 | 150 | ~1500-3000 | 第3次更新 | Learner 训练150步后发布 |
| T4 | 200 | ~2000-4000 | 第4次更新 | Learner 训练200步后发布 |

**关键发现**：
- ✅ **Learner 步数 ≠ Actor 步数**：两者是异步的
- ✅ **Actor 通常比 Learner 快**：Actor 只做推理，Learner 需要训练
- ✅ **参数更新频率**：每 50 个 Learner 步更新一次

### 4.2 文档中"100步左右"的含义

**文档原文**：
> "约 100-200 steps 后就有明显改进"

**含义澄清**：
- 这里的 **"100步"** 指的是 **Learner 的训练步数**，不是 Actor 的步数
- 对应关系：
  - Learner 步数 100 → Actor 收到 **2 次参数更新**（步数 50 和 100）
  - 此时 Actor 可能已经执行了 **1000-2000 次环境交互**

**为什么是"100步左右"？**
- Learner 训练 100 步 = 2 次参数更新
- 每次更新后，Actor 的策略质量会提升
- 2 次更新后，策略已经有明显改进

## 五、频率计算

### 5.1 Learner 迭代频率

**配置**：
- `cta_ratio = 2`（默认）
- 每次迭代包括：
  - 1 次 Critic 更新
  - 1 次 Critic + Actor 更新

**实际频率**：
- 从日志看：`22.18 it/s`（每秒 22.18 次迭代）
- 这意味着每秒完成约 22 次训练迭代

**影响因素**：
- GPU 性能
- 批次大小（`batch_size = 256`）
- 数据采样速度

### 5.2 Actor 参数更新频率

**计算公式**：
```
Actor参数更新频率 = Learner迭代频率 / steps_per_update
```

**示例计算**：
- Learner 迭代频率：22.18 it/s
- `steps_per_update = 50`
- Actor 参数更新频率：22.18 / 50 ≈ **0.44 次/秒**

**含义**：
- Actor 大约每 **2.3 秒**收到一次参数更新
- 或者每 **50 个 Learner 步**更新一次

### 5.3 Actor 环境交互频率

**典型值**：
- Actor 通常比 Learner 快很多
- 如果环境步进频率是 10 Hz，Actor 每秒执行 10 次环境交互
- 如果环境步进频率是 20 Hz，Actor 每秒执行 20 次环境交互

**与参数更新的关系**：
- 如果 Actor 每秒执行 10 次环境交互
- 参数更新频率是 0.44 次/秒
- 那么 Actor 每执行约 **23 次环境交互**，就会收到一次参数更新

## 六、完整示例

### 6.1 训练开始阶段

**T0（0 秒）**：
- Learner 步数：0
- Actor 步数：0
- 事件：Learner 发布初始参数（随机初始化）

**T1（约 2.3 秒后）**：
- Learner 步数：50
- Actor 步数：~23-46（假设 10-20 Hz）
- 事件：Learner 发布第1次更新后的参数

**T2（约 4.6 秒后）**：
- Learner 步数：100
- Actor 步数：~46-92
- 事件：Learner 发布第2次更新后的参数
- **此时策略开始有明显改进**

**T3（约 6.9 秒后）**：
- Learner 步数：150
- Actor 步数：~69-138
- 事件：Learner 发布第3次更新后的参数

### 6.2 文档中"100步"的完整含义

**文档原文**：
> "约 100-200 steps 后就有明显改进"

**完整解释**：
- **Learner 步数**：100-200 步
- **参数更新次数**：2-4 次（100/50 到 200/50）
- **Actor 步数**：可能已经执行了 1000-4000 次环境交互
- **时间**：约 4.5-9 秒（基于 22.18 it/s 的 Learner 速度）

**为什么"100步左右"？**
- 2 次参数更新后，策略质量有明显提升
- 此时 Actor 已经收集了足够的数据
- 策略开始从演示数据中学习到有效模式

## 七、关键配置参数

### 7.1 相关配置

```python
# examples/experiments/config.py
class DefaultTrainingConfig:
    max_steps: int = 1000000           # 最大训练步数（Learner 和 Actor 共用）
    steps_per_update: int = 50          # Learner 每多少步发布一次参数
    cta_ratio: int = 2                  # Critic/Actor 更新比例
    batch_size: int = 256               # 批次大小
    training_starts: int = 100          # 开始训练前需要的在线样本数
```

### 7.2 参数更新频率表

| steps_per_update | Learner 步数 | 参数更新次数 | 更新频率（基于 22 it/s） |
|------------------|-------------|-------------|------------------------|
| 50 | 100 | 2 | 每 2.3 秒 |
| 50 | 200 | 4 | 每 2.3 秒 |
| 50 | 500 | 10 | 每 2.3 秒 |
| 100 | 100 | 1 | 每 4.5 秒 |
| 100 | 200 | 2 | 每 4.5 秒 |

## 八、常见误解澄清

### 误解 1：100 步 = Actor 更新 100 次

**错误**：认为 Learner 步数 100 意味着 Actor 更新了 100 次

**正确**：
- Learner 步数 100 → Actor 更新 **2 次**（100 / 50 = 2）
- 参数更新频率 = Learner 步数 / steps_per_update

### 误解 2：Learner 步数 = Actor 步数

**错误**：认为 Learner 和 Actor 的步数是对应的

**正确**：
- Learner 步数：训练迭代次数
- Actor 步数：环境交互次数
- 两者是**异步的**，没有直接对应关系
- Actor 通常比 Learner 快很多

### 误解 3：文档中的"100步"是 Actor 步数

**错误**：认为文档中的"100步"指的是 Actor 的环境交互次数

**正确**：
- 文档中的"100步"指的是 **Learner 的训练步数**
- 此时 Actor 可能已经执行了 1000-2000 次环境交互
- 但 Actor 只收到了 2 次参数更新

## 九、实际训练中的观察

### 9.1 日志解读

**Learner 日志**：
```
learner:   1%|▊| 12804/1000000 [09:53<12:21:42, 22.18it/s]
```
- Learner 已完成 12804 次训练迭代
- 参数更新次数：12804 / 50 ≈ **256 次**

**Actor 日志**（如果有）：
```
actor:  45%|████████████████████████████████▌| 45000/100000 [05:23<06:32, 140.23it/s]
```
- Actor 已完成 45000 次环境交互
- 参数更新次数：取决于 Learner 的步数，不是 Actor 的步数

### 9.2 策略改进时间线

**基于实际配置**（`steps_per_update = 50`，Learner 速度 22 it/s）：

| Learner 步数 | 参数更新次数 | 时间（秒） | Actor 步数（估算） | 策略质量 |
|-------------|-------------|-----------|-------------------|---------|
| 0 | 0（初始） | 0 | 0 | 随机 |
| 50 | 1 | 2.3 | ~23-46 | 开始改进 |
| 100 | 2 | 4.5 | ~46-92 | **明显改进** |
| 200 | 4 | 9.0 | ~92-184 | 快速改进 |
| 500 | 10 | 22.5 | ~230-460 | 接近演示水平 |

## 十、总结

### 10.1 关键关系

1. **Learner 步数**：训练迭代次数（日志中的 12804）
2. **Actor 步数**：环境交互次数（异步，通常更快）
3. **参数更新频率**：每 `steps_per_update`（默认 50）个 Learner 步更新一次
4. **文档中的"100步"**：指的是 Learner 步数，对应 2 次参数更新

### 10.2 计算公式

```
参数更新次数 = Learner步数 / steps_per_update
Actor参数更新频率 = Learner迭代频率 / steps_per_update
```

### 10.3 实际示例

**当前训练状态**（Learner 步数 12804）：
- 参数更新次数：12804 / 50 ≈ **256 次**
- 如果 Learner 速度是 22.18 it/s：
  - 已训练时间：12804 / 22.18 ≈ **577 秒**（约 9.6 分钟）
  - 参数更新频率：22.18 / 50 ≈ **0.44 次/秒**
  - Actor 大约每 2.3 秒收到一次更新

### 10.4 文档中"100步"的含义

**完整解释**：
- **Learner 步数**：100 步
- **参数更新次数**：2 次（100 / 50）
- **时间**：约 4.5 秒（100 / 22.18）
- **Actor 步数**：可能已经执行了 1000-2000 次环境交互
- **策略质量**：开始有明显改进

**为什么是"100步左右"？**
- 2 次参数更新后，策略质量有明显提升
- 此时策略开始从演示数据中学习到有效模式
- 人工干预的价值开始提升（从"替代随机动作"变为"纠正错误"）

