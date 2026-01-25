# 游戏手柄使用示例

## 快速开始

### 方式 1：直接替换（推荐）

在配置文件中，将 `SpacemouseIntervention` 替换为 `GamepadIntervention`：

```python
# 修改前
from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
    GripperCloseEnv
)

def get_environment(self, fake_env=False, ...):
    # ...
    if not fake_env:
        env = SpacemouseIntervention(env)
    # ...

# 修改后
from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    GamepadIntervention,  # 替换为 GamepadIntervention
    MultiCameraBinaryRewardClassifierWrapper,
    GripperCloseEnv
)

def get_environment(self, fake_env=False, ...):
    # ...
    if not fake_env:
        env = GamepadIntervention(env)  # 使用游戏手柄
    # ...
```

### 方式 2：通过配置选择

可以根据配置选择使用 SpaceMouse 或游戏手柄：

```python
from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    GamepadIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
    GripperCloseEnv
)

def get_environment(self, fake_env=False, use_gamepad=False, ...):
    # ...
    if not fake_env:
        if use_gamepad:
            env = GamepadIntervention(env, deadzone=0.0, sensitivity=1.0)
        else:
            env = SpacemouseIntervention(env)
    # ...
```

---

## 完整示例

### 示例 1：ram_insertion 任务

**文件**: `examples/experiments/ram_insertion/config.py`

```python
from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    GamepadIntervention,  # 使用游戏手柄
    MultiCameraBinaryRewardClassifierWrapper,
    GripperCloseEnv
)

class TrainConfig(DefaultTrainingConfig):
    # ...
    
    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        # ... 环境初始化代码 ...
        
        # 1. 固定夹爪包装器
        env = GripperCloseEnv(env)
        
        # 2. 游戏手柄干预（真实环境必需）
        if not fake_env:
            env = GamepadIntervention(env, deadzone=0.0, sensitivity=1.0)
        
        # 3. 其他包装器
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        # ...
        
        return env
```

### 示例 2：自定义参数

如果需要调整死区或灵敏度：

```python
# 增加死区，过滤摇杆噪声
env = GamepadIntervention(env, deadzone=0.1, sensitivity=1.0)

# 降低灵敏度，实现更精细的控制
env = GamepadIntervention(env, deadzone=0.0, sensitivity=0.5)

# 使用第二个手柄（如果有多个手柄）
env = GamepadIntervention(env, deadzone=0.0, sensitivity=1.0, joystick_id=1)
```

---

## 参数说明

### GamepadIntervention 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `env` | `gym.Env` | - | 要包装的环境（必需） |
| `action_indices` | `List[int]` | `None` | 可选，指定哪些动作可以被覆盖 |
| `deadzone` | `float` | `0.0` | 死区阈值（0.0-1.0），默认 0.0 与 SpaceMouse 一致 |
| `sensitivity` | `float` | `1.0` | 灵敏度缩放（0.0-2.0），默认 1.0 与 SpaceMouse 一致 |
| `joystick_id` | `int` | `0` | 手柄设备 ID（如果有多个手柄） |

### 参数建议

- **默认参数** (`deadzone=0.0, sensitivity=1.0`): 与 SpaceMouse 完全一致，推荐使用
- **增加死区** (`deadzone=0.1`): 如果摇杆有噪声，可以增加死区过滤
- **降低灵敏度** (`sensitivity=0.5`): 如果需要更精细的控制，可以降低灵敏度

---

## 操作说明

### 位置控制（左手）

- **左摇杆左右** → x 平移（左右移动）
- **左摇杆前后** → y 平移（前后移动）
- **LT + LB 组合** → z 平移
  - LT 按下（不按 LB）→ z 向下
  - LT + LB 同时按下 → z 向上

### 旋转控制（右手）

- **右摇杆左右** → yaw 旋转（左右旋转）
- **右摇杆上下** → pitch 旋转（上下旋转）
- **RT + RB 组合** → roll 旋转
  - RT 按下（不按 RB）→ roll 向左
  - RT + RB 同时按下 → roll 向右

### 夹爪控制

- **A 键** → 关闭夹爪
- **B 键** → 打开夹爪

---

## 故障排除

### 问题 1：手柄未检测到

**错误信息**: `RuntimeError: No gamepad detected`

**解决方案**:
1. 确保手柄已连接（USB 或蓝牙）
2. 确保手柄处于 X 模式（XInput）
3. 检查系统是否识别手柄：`lsusb` 或 `jstest /dev/input/js0`

### 问题 2：导入错误

**错误信息**: `ModuleNotFoundError: No module named 'franka_env'`

**解决方案**:
```bash
cd serl_robot_infra
pip install -e .
```

### 问题 3：pygame 未安装

**错误信息**: `ImportError: pygame is required for gamepad support`

**解决方案**:
```bash
pip install pygame>=2.0.0
```

### 问题 4：映射不正确

**症状**: 某些控制量方向错误或没有响应

**解决方案**:
1. 运行测试脚本验证映射：`python serl_robot_infra/franka_env/gamepad/gamepad_test_complete.py`
2. 检查手柄是否处于 X 模式
3. 检查轴索引是否正确（Xbox 360 手柄应使用标准映射）

---

## 与 SpaceMouse 的对比

| 特性 | SpaceMouse | Gamepad |
|------|-----------|---------|
| **硬件成本** | 高（数千元） | 低（数百元） |
| **6DOF 控制** | ✅ 连续模拟量 | ✅ 连续模拟量 |
| **操作方式** | 推/拉/旋转 SpaceMouse 帽 | 摇杆 + 扳机键组合 |
| **学习曲线** | 需要适应 | 更直观（类似游戏操作） |
| **接口兼容性** | - | ✅ 完全兼容 |
| **输出范围** | `[-1.0, 1.0]` | `[-1.0, 1.0]` |

---

## 总结

✅ **手柄已经完全可以代替 SpaceMouse 使用！**

只需要在配置文件中将 `SpacemouseIntervention` 替换为 `GamepadIntervention` 即可，无需修改其他代码。
