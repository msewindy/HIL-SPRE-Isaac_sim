# 演示采集 Y 键重置与轨迹清理 — 需求分析与修改方案

## 一、需求回顾

1. **演示采集阶段**：采集中若出现不可恢复状态，需用手柄 Y 键重置场景；重置后**当前轨迹无效**，应丢弃并清空，再继续采集，以保证演示数据连续、干净，利于策略初学。
2. **RL 训练阶段**：手柄 Y 键重置时，应**记录为一条失败轨迹**（与超时/失败一致），**不**做“清理数据”，正常存入 replay。
3. **两种重置触发**：
   - **手柄 Y 键**：用户主动重置场景；
   - **任务结束**：任务超时未完成或任务成功完成（`done/truncated`）。

---

## 二、与文档/算法的一致性分析

### 2.1 RLPD 与演示数据质量

- **RLPD_Algorithm_Detail.md**：强调演示数据应“高质量”“成功轨迹”“密集正奖励”；50/50 采样下，**有效梯度主要来自演示数据**。若演示中混入“重置前的不连续片段”，会引入错误的状态-动作对应，不利于 Critic 学价值、Actor 学策略。
- **RL_Strategy_Selection_Guide.md**：演示数据要“尽可能干净高效”，以解决稀疏奖励下的探索问题。

**结论**：演示阶段在 Y 键重置时丢弃当前轨迹、只保留重置后的新轨迹，与“演示数据干净、连续”的目标一致，**合理**。

### 2.2 RL 训练阶段对“失败轨迹”的利用

- 训练时 replay_buffer 会混合成功与失败数据；失败轨迹（`dones=1`）用于学习“哪些状态/动作导致结束”，对 TD 目标和探索也有用。
- Y 键重置表示“用户认为本局应结束”，与超时/失败在语义上一致，应**作为一条失败轨迹写入 buffer**，而不是丢弃。

**结论**：RL 阶段 Y 键重置 → 记录失败轨迹、不清理数据，**合理**。

### 2.3 当前实现缺口

- **GamepadIntervention**（`wrappers.py`）：Y 键仅调用 `reset_scene()`，**不**设置 `done`，**不**通知上层“发生了用户重置”。
- **record_demos.py**：只根据 `done` 分支处理（成功则保存轨迹，失败/超时则丢弃）；**无法区分“因 Y 键重置”**，因此无法实现“Y 键重置时丢弃当前轨迹并重置”。
- **train_rlpd.py**：只在 `done or truncated` 时做 episode 结束与 `env.reset()`；Y 键仅触发场景重置，**当前轨迹不会以失败轨迹形式结束并写入**。

因此需要：  
- 在 wrapper 中**显式标记“本步发生了 Y 键触发的场景重置”**；  
- 在 **record_demos** 和 **train_rlpd** 中**分别**根据该标记实现“丢弃轨迹+重置”与“记录失败轨迹+重置”。

---

## 三、修改方案概览

| 层次 | 修改内容 |
|------|----------|
| **1. GamepadIntervention** | Y 键触发 `reset_scene()` 时，在本步 `info` 中写入 `user_reset_scene=True`，供上层区分。 |
| **2. record_demos.py** | 检测 `user_reset_scene`：丢弃当前轨迹、不写入 transitions、调用 `env.reset()`，然后继续采集。 |
| **3. train_rlpd.py** | 将“Y 键重置”视为 episode 结束：本步 transition 的 `dones=1`，并执行与 `done/truncated` 相同的结束逻辑（发统计、reset）。 |
| **4. isaac_sim_gear_env_enhanced.py** | 无需为“区分阶段”改逻辑；阶段区分完全由 record_demos / train_rlpd 的调用方式决定。 |

---

## 四、详细设计

### 4.1 GamepadIntervention（`serl_robot_infra/franka_env/envs/wrappers.py`）

- **行为**：在 Y 键边缘触发时，除调用 `reset_scene()` 外，设置一个**本步有效**的标记（例如 `self._y_triggered_reset = True`）。
- **在 `step()` 中**：在调用 `env.step(new_action)` 得到 `obs, rew, done, truncated, info` 之后，若 `_y_triggered_reset` 为 True，则：
  - `info["user_reset_scene"] = True`
  - 将 `_y_triggered_reset` 置为 False
- **不**在 wrapper 层强制 `done=True`，由上层根据 `info["user_reset_scene"]` 自行决定是否当作 episode 结束处理（record_demos 要清轨迹+reset，train_rlpd 要记失败+reset）。

这样：
- 演示/训练共用同一套 wrapper，无需传“是否演示”参数；
- 两种重置触发（Y 键 vs 任务成功/超时）在上层语义清晰：Y 键 = `user_reset_scene`，任务结束 = `done/truncated`。

### 4.2 record_demos.py

- 在 `env.step(actions)` 之后、**在把本步 transition  append 到 trajectory 之前**：
  - 若 `info.get("user_reset_scene")`：
    - 丢弃当前 `trajectory`（不加入 `transitions`）；
    - 打印提示（例如“用户 Y 键重置，已丢弃当前轨迹”）；
    - `obs, info = env.reset()`；
    - `trajectory = []`，`returns = 0`；
    - `continue`（不执行本步的 append 和后面的 `if done` 分支）。
- 否则按原逻辑：先 append 本步 transition，再根据 `done` 处理成功/失败（成功则整条 trajectory 写入 `transitions`，失败则丢弃 trajectory），最后在 `done` 时 `env.reset()`。

这样保证：**Y 键重置后，当前这条不完整轨迹完全不会进入 `transitions`**，只保留重置之后新开的轨迹，满足“演示数据连续、干净”的要求。

### 4.3 train_rlpd.py

- 定义 **effective_done**：`effective_done = done or truncated or info.get("user_reset_scene", False)`。
- 构造 transition 时使用 **effective_done**：
  - `masks = 1.0 - effective_done`
  - `dones = effective_done`
- 插入 `data_store` / `intvn_data_store` 以及 `transitions` / `demo_transitions` 的逻辑不变，仍按“是否干预”等原有条件插入。
- 原 `if done or truncated:` 的 episode 结束分支改为 **`if effective_done:`**：
  - 发送 stats、更新 pbar、清零 running_return / intervention 计数、`already_intervened = False`、`client.update()`、`obs, _ = env.reset()`。

这样：Y 键重置时，当前步会作为“失败结束”的 transition 写入 buffer，并触发一次 env.reset()，与“RL 阶段 Y 键 = 记录失败轨迹”一致。

### 4.4 isaac_sim_gear_env_enhanced.py

- **无需修改**。重置的两个触发点已明确：
  - **Y 键**：在 wrapper 中检测并设 `user_reset_scene`，由 record_demos / train_rlpd 分别处理；
  - **任务超时/成功**：由现有 `done`/`truncated` 与 `info["succeed"]` 表达，逻辑保持不变。

---

## 五、数据流小结

| 场景 | 触发 | 行为 |
|------|------|------|
| **演示采集** | 任务成功 | 整条 trajectory 写入 transitions，success_count += 1，然后 env.reset()。 |
| **演示采集** | 任务失败/超时 | 丢弃当前 trajectory，env.reset()。 |
| **演示采集** | 手柄 Y 键 | 丢弃当前 trajectory（不写入 transitions），env.reset()，继续采集。 |
| **RL 训练** | 任务成功/失败/超时 | 按现有逻辑写 transition（dones=1），发 stats，env.reset()。 |
| **RL 训练** | 手柄 Y 键 | 本步 transition 记为 dones=1（失败轨迹），发 stats，env.reset()。 |

---

## 六、实现清单（供你确认后改代码）

1. **wrappers.py — GamepadIntervention**
   - 在 `__init__` 中增加：`self._y_triggered_reset = False`。
   - 在 Y 键边缘触发并调用 `reset_scene()` 的分支中，增加：`self._y_triggered_reset = True`。
   - 在 `step()` 中，`env.step(new_action)` 返回后，若 `self._y_triggered_reset`：`info["user_reset_scene"] = True`，并设 `self._y_triggered_reset = False`。

2. **record_demos.py**
   - 在 `next_obs, rew, done, truncated, info = env.step(actions)` 之后、构建并 append transition 之前，增加对 `info.get("user_reset_scene")` 的分支：清空 trajectory、打印提示、`env.reset()`、重置 trajectory/returns、`continue`。
   - 其余逻辑（成功保存、失败丢弃、done 时 reset）保持不变。

3. **train_rlpd.py**
   - 引入 `effective_done = done or truncated or info.get("user_reset_scene", False)`。
   - 构造 transition 时用 `effective_done` 替代 `done`（masks、dones）。
   - 将原 `if done or truncated:` 改为 `if effective_done:`，分支内逻辑不变。

4. **isaac_sim_gear_env_enhanced.py**
   - 无需改动。

若你确认该方案，我将按此清单在仓库中直接修改代码。
