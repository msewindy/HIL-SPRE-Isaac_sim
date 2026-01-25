# 游戏手柄代替 SpaceMouse 控制机械臂 - 新映射方案

## 测试发现的关键信息

### 手柄输入特性

1. **左右摇杆**：
   - 初始值：`0.0`（中心位置）
   - 极限位置：`-1.0` 到 `1.0`
   - 类型：**连续模拟量**

2. **LT (Left Trigger) - 左扳机键**：
   - 轴索引：**轴 2**（不是轴 4）
   - 初始值：`-1.0`（未按下）
   - 极限位置：`1.0`（完全按下）
   - 类型：**连续模拟量**

3. **RT (Right Trigger) - 右扳机键**：
   - 轴索引：**轴 5**（正确）
   - 初始值：`-1.0`（未按下）
   - 极限位置：`1.0`（完全按下）
   - 类型：**连续模拟量**

4. **LB (Left Bumper) - 左肩键**：
   - 按钮索引：按钮 4
   - 类型：**开关量**（0 或 1）

5. **RB (Right Bumper) - 右肩键**：
   - 按钮索引：按钮 5
   - 类型：**开关量**（0 或 1）

### 设计原则

1. **不操作时输出为 0.0**：所有控制量在不操作手柄时应该输出 `0.0`
2. **连续控制**：所有 6DOF 控制都应该是连续模拟量，与 SpaceMouse 一致
3. **直观映射**：左手控制位置，右手控制旋转

---

## 新映射方案设计

### 方案：左右手分工 + 扳机键组合控制

```
6DOF 动作映射：
├─ 位置控制 (x, y, z) - 左手控制
│  ├─ 左摇杆 X 轴（轴 0） → x 平移（左右移动）
│  │  └─ 初始值 0.0，范围 [-1.0, 1.0]
│  ├─ 左摇杆 Y 轴（轴 1） → y 平移（前后移动，注意取反）
│  │  └─ 初始值 0.0，范围 [-1.0, 1.0]，取反后使用
│  └─ z 平移：LT（轴 2）+ LB（按钮 4）组合控制
│     ├─ LB 未按下：LT 的 [-1, 1] 映射为 z 的 [0, -1]（向下）
│     └─ LB 按下：LT 的 [-1, 1] 映射为 z 的 [0, 1]（向上）
│
└─ 旋转控制 (roll, pitch, yaw) - 右手控制
   ├─ 右摇杆 X 轴（轴 3） → yaw 旋转（左右旋转）
   │  └─ 初始值 0.0，范围 [-1.0, 1.0]
   ├─ 右摇杆 Y 轴（轴 4） → pitch 旋转（上下旋转，注意取反）
   │  └─ 初始值 0.0，范围 [-1.0, 1.0]，取反后使用
   └─ roll 旋转：RT（轴 5）+ RB（按钮 5）组合控制
      ├─ RB 未按下：RT 的 [-1, 1] 映射为 roll 的 [0, -1]（向左）
      └─ RB 按下：RT 的 [-1, 1] 映射为 roll 的 [0, 1]（向右）

夹爪控制：
├─ A 键（按钮 0） → 关闭夹爪
└─ B 键（按钮 1） → 打开夹爪
```

### 详细映射逻辑

#### 1. 位置控制（左手）

**x 平移**：
```python
# 左摇杆 X 轴（轴 0）
# 初始值：0.0，范围：[-1.0, 1.0]
# 直接使用，无需处理
x = apply_deadzone(joystick.get_axis(0), deadzone) * sensitivity
```

**y 平移**：
```python
# 左摇杆 Y 轴（轴 1）
# 初始值：0.0，范围：[-1.0, 1.0]
# 注意：pygame 中摇杆向上为负值，需要取反
y = -apply_deadzone(joystick.get_axis(1), deadzone) * sensitivity
```

**z 平移**（LT + LB 组合）：
```python
# LT（轴 2）：初始值 -1.0，按下后到 1.0
# LB（按钮 4）：0（未按下）或 1（按下）
left_trigger = joystick.get_axis(2)  # 范围 [-1.0, 1.0]
left_bumper = joystick.get_button(4)  # 0 或 1

# 将 LT 的 [-1, 1] 映射到 [0, 1]
# 未按下时：-1 → 0，完全按下时：1 → 1
lt_normalized = (left_trigger + 1.0) / 2.0  # 范围 [0.0, 1.0]

# 根据 LB 状态决定方向
if left_bumper == 0:
    # LB 未按下：z 向下（负值）
    # lt_normalized [0, 1] → z [-1, 0]
    z = -lt_normalized
else:
    # LB 按下：z 向上（正值）
    # lt_normalized [0, 1] → z [0, 1]
    z = lt_normalized

# 应用死区和灵敏度
z = apply_deadzone(z, deadzone) * sensitivity
```

**z 平移逻辑总结**：
- LT 未按下（-1.0）→ `lt_normalized = 0.0` → `z = 0.0`（不操作）
- LT 按下 + LB 未按下 → `z` 从 `0.0` 到 `-1.0`（向下）
- LT 按下 + LB 按下 → `z` 从 `0.0` 到 `1.0`（向上）

#### 2. 旋转控制（右手）

**yaw 旋转**：
```python
# 右摇杆 X 轴（轴 3）
# 初始值：0.0，范围：[-1.0, 1.0]
# 直接使用
yaw = apply_deadzone(joystick.get_axis(3), deadzone) * sensitivity
```

**pitch 旋转**：
```python
# 右摇杆 Y 轴（轴 4）
# 初始值：0.0，范围：[-1.0, 1.0]
# 注意：pygame 中摇杆向上为负值，需要取反
pitch = -apply_deadzone(joystick.get_axis(4), deadzone) * sensitivity
```

**roll 旋转**（RT + RB 组合）：
```python
# RT（轴 5）：初始值 -1.0，按下后到 1.0
# RB（按钮 5）：0（未按下）或 1（按下）
right_trigger = joystick.get_axis(5)  # 范围 [-1.0, 1.0]
right_bumper = joystick.get_button(5)  # 0 或 1

# 将 RT 的 [-1, 1] 映射到 [0, 1]
# 未按下时：-1 → 0，完全按下时：1 → 1
rt_normalized = (right_trigger + 1.0) / 2.0  # 范围 [0.0, 1.0]

# 根据 RB 状态决定方向
if right_bumper == 0:
    # RB 未按下：roll 向左（负值）
    # rt_normalized [0, 1] → roll [-1, 0]
    roll = -rt_normalized
else:
    # RB 按下：roll 向右（正值）
    # rt_normalized [0, 1] → roll [0, 1]
    roll = rt_normalized

# 应用死区和灵敏度
roll = apply_deadzone(roll, deadzone) * sensitivity
```

**roll 旋转逻辑总结**：
- RT 未按下（-1.0）→ `rt_normalized = 0.0` → `roll = 0.0`（不操作）
- RT 按下 + RB 未按下 → `roll` 从 `0.0` 到 `-1.0`（向左）
- RT 按下 + RB 按下 → `roll` 从 `0.0` 到 `1.0`（向右）

#### 3. 夹爪控制

```python
# A 键（按钮 0）：关闭夹爪
# B 键（按钮 1）：打开夹爪
buttons = [0, 0, 0, 0]
buttons[0] = joystick.get_button(0)  # A 键
buttons[1] = joystick.get_button(1)  # B 键
```

---

## 轴索引映射表

| 手柄输入 | 轴/按钮索引 | 初始值 | 范围 | 类型 |
|---------|-----------|--------|------|------|
| 左摇杆 X | 轴 0 | 0.0 | [-1.0, 1.0] | 连续量 |
| 左摇杆 Y | 轴 1 | 0.0 | [-1.0, 1.0] | 连续量 |
| **LT** | **轴 2** | -1.0 | [-1.0, 1.0] | 连续量 |
| 右摇杆 X | 轴 3 | 0.0 | [-1.0, 1.0] | 连续量 |
| 右摇杆 Y | 轴 4 | 0.0 | [-1.0, 1.0] | 连续量 |
| RT | 轴 5 | -1.0 | [-1.0, 1.0] | 连续量 |
| A 键 | 按钮 0 | 0 | [0, 1] | 开关量 |
| B 键 | 按钮 1 | 0 | [0, 1] | 开关量 |
| LB | 按钮 4 | 0 | [0, 1] | 开关量 |
| RB | 按钮 5 | 0 | [0, 1] | 开关量 |

---

## 输出范围验证

### 不操作时（初始状态）

| 控制量 | 输入值 | 处理后输出 | 状态 |
|--------|--------|-----------|------|
| x | 轴 0 = 0.0 | 0.0 | ✅ |
| y | 轴 1 = 0.0 | 0.0 | ✅ |
| z | LT = -1.0, LB = 0 | `lt_normalized = 0.0` → `z = 0.0` | ✅ |
| yaw | 轴 3 = 0.0 | 0.0 | ✅ |
| pitch | 轴 4 = 0.0 | 0.0 | ✅ |
| roll | RT = -1.0, RB = 0 | `rt_normalized = 0.0` → `roll = 0.0` | ✅ |

**结论**：✅ 所有控制量在不操作时输出为 `0.0`，符合要求。

### 操作时（极限状态）

| 控制量 | 操作 | 输入值 | 处理后输出 | 状态 |
|--------|------|--------|-----------|------|
| x | 左摇杆右推到底 | 轴 0 = 1.0 | 1.0 | ✅ |
| x | 左摇杆左推到底 | 轴 0 = -1.0 | -1.0 | ✅ |
| y | 左摇杆前推到底 | 轴 1 = -1.0 | 1.0（取反） | ✅ |
| y | 左摇杆后拉到底 | 轴 1 = 1.0 | -1.0（取反） | ✅ |
| z | LT 按下到底 + LB 未按下 | LT = 1.0, LB = 0 | `z = -1.0` | ✅ |
| z | LT 按下到底 + LB 按下 | LT = 1.0, LB = 1 | `z = 1.0` | ✅ |
| yaw | 右摇杆右推到底 | 轴 3 = 1.0 | 1.0 | ✅ |
| yaw | 右摇杆左推到底 | 轴 3 = -1.0 | -1.0 | ✅ |
| pitch | 右摇杆上推到底 | 轴 4 = -1.0 | 1.0（取反） | ✅ |
| pitch | 右摇杆下拉到底 | 轴 4 = 1.0 | -1.0（取反） | ✅ |
| roll | RT 按下到底 + RB 未按下 | RT = 1.0, RB = 0 | `roll = -1.0` | ✅ |
| roll | RT 按下到底 + RB 按下 | RT = 1.0, RB = 1 | `roll = 1.0` | ✅ |

**结论**：✅ 所有控制量在操作时都能达到 `[-1.0, 1.0]` 的完整范围，与 SpaceMouse 一致。

---

## 操作说明

### z 平移控制

1. **z 向下移动**：
   - 保持 LB 未按下
   - 按下 LT 扳机键
   - LT 按下越多，z 向下移动越快（0.0 → -1.0）

2. **z 向上移动**：
   - 按下 LB 肩键
   - 同时按下 LT 扳机键
   - LT 按下越多，z 向上移动越快（0.0 → 1.0）

### roll 旋转控制

1. **roll 向左旋转**：
   - 保持 RB 未按下
   - 按下 RT 扳机键
   - RT 按下越多，roll 向左旋转越快（0.0 → -1.0）

2. **roll 向右旋转**：
   - 按下 RB 肩键
   - 同时按下 RT 扳机键
   - RT 按下越多，roll 向右旋转越快（0.0 → 1.0）

---

## 优势

1. ✅ **不操作时输出为 0.0**：所有控制量在初始状态都输出 `0.0`，符合操控要求
2. ✅ **连续控制**：所有 6DOF 控制都是连续模拟量，与 SpaceMouse 一致
3. ✅ **直观操作**：左手控制位置，右手控制旋转，符合人体工学
4. ✅ **双向控制**：LT/RT 与 LB/RB 组合，实现 z 和 roll 的双向连续控制
5. ✅ **充分利用硬件**：使用所有可用的连续输入通道

---

## 代码实现要点

### 关键修正

1. **LT 轴索引修正**：从轴 4 改为**轴 2**
2. **RT 轴索引确认**：保持为**轴 5**
3. **LT/RT 归一化**：将 `[-1, 1]` 映射到 `[0, 1]`，确保未按下时为 0.0
4. **组合控制逻辑**：根据 LB/RB 状态决定 z/roll 的方向

### 实现代码示例

```python
# z 平移：LT（轴 2）+ LB（按钮 4）组合
left_trigger = joystick.get_axis(2)  # 修正：使用轴 2
left_bumper = joystick.get_button(4)

# 归一化：[-1, 1] → [0, 1]
lt_normalized = (left_trigger + 1.0) / 2.0

# 根据 LB 状态决定方向
if left_bumper == 0:
    z = -lt_normalized  # 向下（负值）
else:
    z = lt_normalized    # 向上（正值）

# roll 旋转：RT（轴 5）+ RB（按钮 5）组合
right_trigger = joystick.get_axis(5)
right_bumper = joystick.get_button(5)

# 归一化：[-1, 1] → [0, 1]
rt_normalized = (right_trigger + 1.0) / 2.0

# 根据 RB 状态决定方向
if right_bumper == 0:
    roll = -rt_normalized  # 向左（负值）
else:
    roll = rt_normalized   # 向右（正值）
```

---

## 总结

新映射方案解决了以下问题：

1. ✅ **修正轴索引**：LT 使用正确的轴 2（不是轴 4）
2. ✅ **确保初始值为 0.0**：所有控制量在不操作时输出 `0.0`
3. ✅ **实现连续控制**：LT/RT 与 LB/RB 组合，实现 z 和 roll 的双向连续控制
4. ✅ **与 SpaceMouse 一致**：输出范围 `[-1.0, 1.0]`，所有控制都是连续模拟量

该方案完全符合 SpaceMouse 的输出特性，实现了真正的连续控制，同时保持了直观的操作方式。

---

# 场景重置接口和手柄映射方案

## 一、场景重置接口实现

### 1.1 功能说明

**场景重置接口** (`/reset_scene`)：
- **功能**：重置整个 USD 场景到初始状态
- **作用**：
  1. 重置物理世界（`world.reset()`）
  2. 重置所有对象到初始位置
  3. 清除所有约束
  4. 重置机器人到初始状态
  5. 重置夹爪到打开状态

**使用场景**：
- 快速重置仿真环境
- 任务失败后快速恢复
- 测试和调试时使用

### 1.2 实现位置

**Server 端**：`serl_robot_infra/robot_servers/isaac_sim_server.py`
- 路由：`POST /reset_scene`
- 功能：调用 `world.reset()` 重置物理世界

**环境端**：`serl_robot_infra/franka_env/envs/isaac_sim_env.py`
- 方法：`reset_scene()`
- 功能：通过 HTTP 调用服务器端的场景重置接口

---

## 二、手柄映射方案

### 2.1 手柄按钮分配

**当前按钮映射**：
- **A 键（按钮 0）**：关闭夹爪
- **B 键（按钮 1）**：打开夹爪
- **X 键（按钮 2）**：预留（未使用）
- **Y 键（按钮 3）**：**场景重置**（新增）

### 2.2 映射规则设计

#### 设计原则
1. **直观性**：按钮功能应该直观易懂
2. **安全性**：重要操作（如场景重置）应该不容易误触发
3. **一致性**：与现有映射保持一致

#### 映射方案

```
手柄按钮映射：
├─ A 键（按钮 0）→ 关闭夹爪
├─ B 键（按钮 1）→ 打开夹爪
├─ X 键（按钮 2）→ 预留（可用于其他功能）
└─ Y 键（按钮 3）→ 场景重置（边缘触发，避免重复触发）
```

#### Y 键场景重置实现

**边缘触发机制**：
- 检测 Y 键从"未按下"到"按下"的跳变
- 只在跳变时触发一次场景重置
- 避免按住 Y 键时重复触发

**实现逻辑**：
```python
# 在 GamepadIntervention.action() 中
if self.y_button and not hasattr(self, '_y_button_pressed'):
    # Y 键刚按下（边缘触发）
    self._y_button_pressed = True
    # 触发场景重置
    self.env.reset_scene()
elif not self.y_button:
    # Y 键释放，重置标志
    self._y_button_pressed = False
```

### 2.3 使用说明

**场景重置操作**：
1. 按下手柄的 **Y 键**
2. 系统检测到 Y 键按下（边缘触发）
3. 自动调用场景重置接口
4. 场景恢复到初始状态

**注意事项**：
- Y 键是边缘触发，按住不会重复触发
- 场景重置会重置所有对象，包括机器人位置
- 重置后需要等待场景稳定（约 1-2 秒）

---

## 三、实现细节

### 3.1 Server 端实现

```python
@webapp.route("/reset_scene", methods=["POST"])
def reset_scene():
    """
    重置整个 USD 场景
    
    功能：
    1. 重置物理世界（world.reset()）
    2. 重置所有对象到初始位置
    3. 清除所有约束
    4. 重置机器人到初始状态
    """
    try:
        # 重置物理世界（这会重置所有物理对象到初始状态）
        isaac_sim_server.world.reset()
        
        # 重置机器人到初始关节位置
        if hasattr(isaac_sim_server.franka, 'set_joint_positions'):
            initial_joint_positions = np.zeros(7)
            isaac_sim_server.franka.set_joint_positions(initial_joint_positions)
        
        # 重置夹爪到打开状态
        isaac_sim_server.set_gripper(1.0)
        
        return jsonify({"status": "success", "message": "Scene reset completed"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
```

### 3.2 环境端实现

```python
def reset_scene(self):
    """
    重置整个 USD 场景
    
    通过 HTTP 接口调用服务器端的场景重置功能
    """
    try:
        response = self.session.post(self.url + "reset_scene", timeout=2.0)
        if response.status_code == 200:
            print("[INFO] Scene reset successful")
    except Exception as e:
        print(f"[WARNING] Failed to reset scene: {e}")
```

### 3.3 手柄包装器实现

```python
# 在 GamepadIntervention.action() 中
# 处理场景重置（Y 键）
if self.y_button and not hasattr(self, '_y_button_pressed'):
    # 检测 Y 键按下（边缘触发）
    self._y_button_pressed = True
    try:
        if hasattr(self.env, 'reset_scene'):
            self.env.reset_scene()
    except Exception as e:
        print(f"[WARNING] Failed to reset scene: {e}")
elif not self.y_button:
    # Y 键释放，重置标志
    self._y_button_pressed = False
```

---

## 四、测试方法

### 4.1 测试场景重置接口

```bash
# 测试 HTTP 接口
curl -X POST http://127.0.0.1:5001/reset_scene
```

### 4.2 测试手柄映射

1. **连接手柄**
2. **运行环境**（使用 `GamepadIntervention`）
3. **按下 Y 键**
4. **观察**：场景应该重置到初始状态

### 4.3 验证清单

- [ ] Y 键按下时触发场景重置
- [ ] 场景重置后所有对象回到初始位置
- [ ] 机器人回到初始关节位置
- [ ] 夹爪打开
- [ ] 按住 Y 键不会重复触发（边缘触发）

---

## 五、扩展功能（可选）

### 5.1 其他按钮映射

如果将来需要更多功能，可以考虑：

- **X 键（按钮 2）**：
  - 选项 1：暂停/继续仿真
  - 选项 2：切换相机视角
  - 选项 3：切换控制模式

- **方向键（D-Pad）**：
  - 上：增加控制灵敏度
  - 下：减少控制灵敏度
  - 左/右：切换功能

### 5.2 组合键功能

- **LB + Y**：强制场景重置（即使有错误）
- **RB + Y**：重置并重新加载 USD 场景

---

## 六、参考文档

- 手柄映射方案：`docs/isaac sim虚拟环境改造/usb手柄代替spacemouse_新映射方案.md`
- 手柄集成状态：`docs/原项目分析/gamepad_integration_status.md`
