# 演示采集重置与 Gear 消失问题说明

## 问题 1：按 Y 键后场景重置两次

**原因**：  
- 手柄 Y 键按下时，`GamepadIntervention` 在 `action()` 里调用了 `env.reset_scene()`（第一次重置）。  
- 随后 `step()` 返回 `info["user_reset_scene"]=True`，`record_demos` 里再调用 `env.reset()`。  
- `isaac_sim_gear_env_enhanced.reset()` 内部又会调用 `self.reset_scene()`（第二次重置）。  

因此同一次 Y 键操作触发了两次场景重置。

**修改**：  
- 在 `GamepadIntervention` 中，Y 键按下时**只设置** `_y_triggered_reset = True`，**不再**调用 `reset_scene()`。  
- 由上层在检测到 `user_reset_scene` 后调用 `env.reset()`，在 `env.reset()` 里统一执行一次 `reset_scene()`。

---

## 问题 2：成功装配后卡死约 10 秒、重置后 gear_medium 消失

### 2.1 卡死 10 秒

**原因**：  
- `record_demos.py` 里在“成功”分支中有 `time.sleep(10)`，用于留时间查看结果。  

**修改**：  
- 增加启动参数 `--success_sleep_sec`（默认改为 2.0 秒）。  
- 设为 0 可完全跳过等待，例如：  
  `python examples/record_demos.py --exp_name=gear_assembly --success_sleep_sec=0 ...`

### 2.2 重置后 gear_medium 消失（根本原因与修复）

**根本原因**：  
- 机械臂和齿轮**同时**复位时，齿轮与夹爪仍通过 grasp FixedJoint **绑在一起**。  
- 机械臂突然回到初始位置，把绑在一起的齿轮一起带走，齿轮被甩飞、掉到地上，看起来像“消失”。

**修复逻辑**（在 `_execute_reset_scene()` 中）：  
1. **先解绑**：若存在 `self.grasp_joint`（齿轮与夹爪的 FixedJoint），从 stage 中移除该 prim，并设 `self.grasp_joint = None`。  
2. **夹爪打开**：`_execute_set_gripper(1.0)`，确保齿轮不再被夹持。  
3. **world.reset()**：重置物理世界。  
4. **机械臂先回到初始位置**：`set_joint_positions(initial_q)`，夹爪再次打开。  
5. **齿轮再进入位置随机摆放**：在域随机化块中设置 gear_medium 的 Xform 位置（机械臂已复位后再设，避免甩飞）。  
6. 其余：base 旋转、warmup、set_pose(default_reset_pose) 等不变。

**其他说明**：  
- 若 `/World/factory_gear_medium` 无效，会尝试子节点并对其父 Xform 设位置；若仍无效则打 `[WARNING] Reset: gear_medium prim not found at ...`。
