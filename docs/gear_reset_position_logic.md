# Gear / Gear Base 位置设定与场景重置逻辑梳理

## 一、配置文件 (examples/experiments/gear_assembly/config.py)

| 配置项 | 含义 | 被谁使用 |
|--------|------|----------|
| `RESET_POSE` | 机械臂重置位姿 [x,y,z, rx,ry,rz] | server 读作 `default_reset_pose`，用于机械臂复位后自动移动到的位姿 |
| `TARGET_POSE` | 组装目标位姿（大齿轮/底座位置 y=-0.6, z~0.41） | 任务/客户端用，**未**用于 server 内 gear/base 位置设定 |
| `GRASP_POSE` | 抓取位姿（注释写“中齿轮初始位置” y=-0.75, z~0.40） | 任务/客户端用，**未**用于 server 内 gear 位置设定 |
| `RESET_RANDOMIZE_GEAR_AND_BASE` | 是否在场景重置时随机化 gear 与 gear base | server 读作 `reset_randomize_gear_and_base` |

**现状**：配置里**没有** gear / gear base 的“重置位置”。  
需求：应有一个“reset position”：关闭随机化时，每次重置 gear 和 gear base 都回到该位置；打开随机化时，在该位置基础上加随机偏移。

---

## 二、场景初始化 (isaac_sim_server.py __init__)

1. 加载 USD 场景。
2. `world.reset()` 一次。
3. 机械臂设到 `initial_q`，再根据 config 设 `RESET_POSE` 为 `default_reset_pose` 并 `set_pose`。
4. 从 config 读 `RESET_RANDOMIZE_GEAR_AND_BASE`、命令行 `--reset_randomize_gear` 覆盖。

**与 gear/base 相关**：  
- **没有**对 gear / gear base 做任何显式位置设置。  
- 二者完全由 **USD 场景的初始状态** 决定（即 USD 里写死的默认位姿）。

---

## 三、场景重置 (_execute_reset_scene)

流程概要：

1. 解绑夹爪与齿轮、夹爪打开。
2. **（此前错误实现）** 若 `reset_randomize_gear_and_base==False`：保存当前 gear/base 位姿，准备“恢复”。
3. `world.reset()`：物理世界重置，**所有刚体（含 gear、gear base）会被重置回 USD 里的默认状态**，所以会“乱跳”。
4. 机械臂回 `initial_q`，夹爪打开。
5. 齿轮/底座位姿：
   - **若 randomize=True**：  
     - Gear：在 **当前 USD 的 translate** 上改 x,y 为随机数：`x ∈ [-0.10, 0.10]`，`y ∈ [-0.52, -0.45]`，z 保持 `current_val[2]`。  
     - Base：在 **当前 USD 的 rotate** 上 Z 轴加 `rand_deg ∈ [-10, 10]` 度，并同步更新 FixedJoint `localRot1`。  
     - 即：随机化是相对于 **world.reset() 后的 USD 默认状态** 做的，**没有**统一的“config reset position”。
   - **若 randomize=False（当前错误实现）**：  
     - 用 `_get_gear_and_base_world_poses()` 保存的“重置前当前位姿”再写回 → 相当于“保持重置前位姿”，**不是**“重置到 config 的固定位置”。
6. Warmup、状态同步、机械臂移动到 `default_reset_pose`。

**需求对应**：  
- 关闭随机化：应把 gear 和 gear base **设为 config 里定义的 reset position**（固定位置），而不是“恢复重置前位姿”。  
- 打开随机化：应在 **同一 reset position** 基础上加随机偏移（当前逻辑可保留为“在该位置上的随机范围”）。

---

## 四、与 gear/base 位置相关的代码位置

| 位置 | 作用 |
|------|------|
| `__init__` 约 245 行 | `world.reset()`，未设 gear/base |
| `__init__` 约 529–668 行 | 从 config 读 `RESET_POSE`、`RESET_RANDOMIZE_GEAR_AND_BASE`，未读 gear/base reset |
| `_execute_reset_scene` 约 2282–2432 行 | 重置流程；当前 randomize 分支：硬编码 x∈[-0.10,0.10]、y∈[-0.52,-0.45]、base Z +[-10,10]；非 randomize 分支：恢复“保存的当前位姿”（错误） |
| `_get_gear_and_base_world_poses` 约 2173–2209 行 | 读取当前 gear/base 世界位姿（用于之前的“恢复”逻辑） |
| `_restore_gear_and_base_poses` 约 2210–2280 行 | 把保存的位姿写回 gear/base（用于之前的“恢复”逻辑） |

---

## 五、需求与实现要点（已实现，无冗余 config）

1. **不增加 config**  
   - 不在 config 里增加 GEAR_RESET_POSITION / GEAR_BASE_RESET_ANGLE_DEG。  
   - “reset position”即**场景默认固定位置**，由 USD 场景决定。

2. **场景初始化**  
   - 不改：gear/base 仍由 USD 初始状态决定。  
   - 在 `__init__` 末尾**从场景 USD 读取一次** gear/base 的世界位姿，存入 `self._default_gear_base_poses`，作为“默认固定位置”。

3. **场景重置（_execute_reset_scene）**  
   - **关闭随机化**：在 `world.reset()` 和机械臂复位后，调用 `_apply_gear_and_base_reset_position(do_random=False)`，将 gear 和 gear base **恢复到 `_default_gear_base_poses`**（即 __init__ 时保存的默认位姿）。  
   - **打开随机化**：调用 `_apply_gear_and_base_reset_position(do_random=True)`，在 **默认位姿** 上加随机偏移（gear xy：±0.10 / ±0.035，base Z：±10°）。

4. **Server 实现（不读 config 的 reset position）**  
   - `_get_gear_and_base_world_poses()`：从当前 stage 读取 gear/base 世界位姿。  
   - `_set_gear_and_base_poses(saved)`：将 gear/base 设为 saved 中的位姿。  
   - `_apply_gear_and_base_reset_position(do_random)`：  
     - `do_random=False`：调用 `_set_gear_and_base_poses(self._default_gear_base_poses)` 恢复到默认位姿。  
     - `do_random=True`：在 `_default_gear_base_poses` 基础上加随机偏移并写入 stage，并更新 base 的 FixedJoint。

满足：“关闭随机化时，每次场景重置 gear 和 gear base 都重置到场景默认固定位置（__init__ 时从 USD 保存）；打开随机化时，在该默认位置基础上在范围内随机；不增加冗余 config”。
