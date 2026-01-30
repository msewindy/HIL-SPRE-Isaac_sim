# 抓取齿轮后按 A 键夹爪仍继续夹紧导致穿透 — 原因与修复

## 一、现象

- 夹爪抓住齿轮后，夹爪与齿轮之间已建立 **FixedJoint**，设计上此时夹爪不应再响应“继续夹紧”的命令。
- 实际测试：抓取齿轮后按下手柄 **A 键**，夹爪仍会继续夹紧，导致夹爪穿过齿轮。

## 二、代码中已有的逻辑

在 `isaac_sim_server.py` 的 `_execute_set_gripper()` 中，确实有一段“已抓取则禁止夹紧”的逻辑：

```python
# [GRASP FIX] Dynamic Constraint
if self.grasp_joint is not None:
     current_pos = gripper_joint.get_joint_position()
     if target_position < (current_pos - 0.001):
         return  # Block closing command when grasped
```

即：当存在 `grasp_joint` 时，若目标位置比当前位置小 **超过 1mm**（`current_pos - 0.001`），则直接 return，不执行夹爪目标设置。

## 三、根本原因分析

### 原因 1：主路径判断条件过严，在“几乎闭合”时失效

- 条件为：`target_position < (current_pos - 0.001)` 才拦截。
- 抓住齿轮后，夹爪被齿轮撑开，但若关节读数 `current_pos` 已经很小（例如 **≤ 0.001 m**），则：
  - `current_pos - 0.001 ≤ 0`
  - 用户按 A 发 `gripper_pos=0` → `target_position = 0`
  - `0 < (current_pos - 0.001)` 可能为 **false**（例如 current_pos=0.0008 时，0 < -0.0002 为 false）
- 结果：**不会 return**，会继续执行 `set_joint_position_target(0)`，夹爪继续夹紧并穿透齿轮。

**结论**：应用“夹紧方向”拦截，而不是“比当前小 1mm 才拦截”。即：已抓取时，只要 **target_position < current_pos** 就应拦截（或 `target_position <= current_pos`），不应再减 0.001。

### 原因 2：AttributeError 分支完全没有 grasp 检查

- 夹爪控制主路径使用：`self.franka.gripper.get_joints()` 和 `gripper_joint.get_joint_position()`。
- 若其中任一步抛出 **AttributeError**（例如 API 差异、运行时无该方法），会进入 `except AttributeError` 分支，使用：
  - `self.franka.gripper.set_joint_positions([target_position])`，或
  - `self.franka.gripper.close() / open()`
- 该分支内**没有任何 `self.grasp_joint` 判断**，因此一旦走 fallback，就会**无条件执行夹紧/张开**，已抓取时按 A 仍会夹紧并穿透。

**结论**：在 fallback 分支中，若 `self.grasp_joint is not None` 且当前命令为“夹紧”（例如 `gripper_pos` 较小），应直接 return，不执行夹爪闭合。

### 原因 3：run 循环顺序（仅说明为何“有逻辑仍会夹紧”）

- 主循环顺序：`_process_actions()` → `world.step()` → `_update_grasping_logic_v2()`。
- 即：先处理 set_gripper，再物理步进，再更新 grasp_joint。因此当 grasp_joint 已存在时，下一帧的 set_gripper 会先执行，此时会进入“已抓取则禁止夹紧”的判断；问题不在顺序，而在**上述两个判断漏洞**。

## 四、修复建议

| 位置 | 修改内容 |
|------|----------|
| **主路径（方法1）** | 将条件由 `target_position < (current_pos - 0.001)` 改为 **`target_position < current_pos`**（或 `<= current_pos`），使“已抓取时只要目标比当前更闭合就拦截”。 |
| **except AttributeError 分支** | 在分支内执行 set_joint_positions / close 之前，增加：若 `self.grasp_joint is not None` 且 `gripper_pos < 0.5`（表示夹紧意图），则 **return**，不执行夹紧。 |

按上述修改后，抓取齿轮后按 A 键将不再驱动夹爪继续夹紧，避免穿透。
