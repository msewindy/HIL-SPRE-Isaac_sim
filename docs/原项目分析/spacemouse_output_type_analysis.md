# SpaceMouse 输出类型分析：连续值 vs 开关量

本文档详细分析 SpaceMouse 的所有控制量（xyz 平移、rpy 旋转、按钮）的输出类型，确定哪些是连续模拟量，哪些是离散开关量。

## 分析目标

确定 SpaceMouse 的以下控制量的输出类型：
1. **x, y, z 平移**：连续值还是开关量？
2. **roll, pitch, yaw 旋转**：连续值还是开关量？
3. **按钮（buttons）**：连续值还是开关量？

## 代码分析

### 1. 数据读取方式

**文件位置**：`serl_robot_infra/franka_env/spacemouse/pyspacemouse.py`

#### 1.1 6DOF 轴（x, y, z, roll, pitch, yaw）的处理

**关键代码**（`process` 函数，第 251-258 行）：

```python
for name, (chan, b1, b2, flip) in self.__mappings.items():
    if data[0] == chan:
        dof_changed = True
        if b1 < len(data) and b2 < len(data):
            self.dict_state[name] = (
                flip * to_int16(data[b1], data[b2]) / float(self.axis_scale)
            )
```

**关键函数**（第 29-32 行）：

```python
def to_int16(y1, y2):
    """Convert two 8-bit bytes to a signed 16-bit integer."""
    x = (y1) | (y2 << 8)
    if x >= 32768:
        x = -(65536 - x)
    return x
```

**分析**：
- `to_int16` 将两个 8 位字节转换为**16 位有符号整数**
- 返回值范围：**[-32768, 32767]**
- 除以 `axis_scale` (350.0) 后，理论范围：**[-93.6, 93.6]**
- 实际输出被限制在：**[-1.0, 1.0]**

**结论**：✅ **6DOF 轴（x, y, z, roll, pitch, yaw）都是连续模拟量**

#### 1.2 按钮（buttons）的处理

**关键代码**（`process` 函数，第 260-267 行）：

```python
for button_index, (chan, byte, bit) in enumerate(self.button_mapping):
    if data[0] == chan:
        button_changed = True
        mask = 1 << bit
        self.dict_state["buttons"][button_index] = (
            1 if (data[byte] & mask) != 0 else 0
        )
```

**分析**：
- 按钮通过**位掩码**读取
- 输出值只有两种：**0 或 1**
- 这是典型的**离散开关量**

**结论**：✅ **按钮（buttons）是离散开关量**

### 2. 设备物理特性

**SpaceMouse 设备特性**：
- SpaceMouse 是一个**6DOF 模拟输入设备**
- 用户可以通过**推/拉/旋转** SpaceMouse 的帽来产生连续的位置和旋转输入
- 设备内部使用**压力传感器**或**光学编码器**来检测 6 个自由度的连续位移
- 按钮是物理按键，只有按下/未按下两种状态

### 3. 输出格式验证

**从 `spacemouse_expert.py` 验证**：

```python
def _read_spacemouse(self):
    while True:
        state = pyspacemouse.read_all()
        action = [
            -state[0].y, state[0].x, state[0].z,
            -state[0].roll, -state[0].pitch, -state[0].yaw
        ]
        buttons = state[0].buttons
```

**观察**：
- `action` 数组包含 6 个浮点数，范围 `[-1.0, 1.0]`
- `buttons` 是整数列表，每个元素是 0 或 1

## 结论总结

| 控制量 | 输出类型 | 输出范围 | 物理输入方式 |
|--------|---------|---------|-------------|
| **x, y, z 平移** | ✅ **连续模拟量** | `[-1.0, 1.0]` | 推/拉 SpaceMouse 帽 |
| **roll, pitch, yaw 旋转** | ✅ **连续模拟量** | `[-1.0, 1.0]` | 旋转 SpaceMouse 帽 |
| **按钮（buttons）** | ✅ **离散开关量** | `0` 或 `1` | 按下物理按键 |

## 对手柄映射的启示

### 关键发现

1. **SpaceMouse 的所有 6DOF 控制（xyz 和 rpy）都是连续模拟量**
2. **只有按钮是开关量**

### 手柄映射建议

**应该使用连续模拟输入**：
- ✅ **摇杆**：可以输出连续值，适合映射到 xyz 和 rpy
- ✅ **模拟扳机键（LT/RT）**：如果手柄支持模拟扳机键，可以输出连续值，适合映射到 z 和 roll

**不应该使用开关量输入**：
- ❌ **数字按钮（LB/RB）**：只能输出 0 或 1，不适合直接映射到连续值
- ❌ **数字扳机键**：如果扳机键是数字的（只有按下/未按下），不适合映射到连续值

### 当前手柄映射的问题

**当前设计**：
- LT/LB 控制 z 平移
- RT/RB 控制 roll 旋转

**问题**：
- 如果 LB 和 RB 是**数字按钮**（开关量），它们只能输出 0 或 1
- 当按下 LB 时，z 值会突然跳到 -1.0，这不是连续控制
- 当按下 RB 时，roll 值会突然跳到 -1.0，这不是连续控制

**建议修复**：
1. **只使用模拟输入**：LT/RT 作为模拟扳机键，LB/RB 不使用
2. **或者使用组合控制**：LT 控制 z 向上（0→1），RT 控制 z 向下（0→-1），不使用 LB/RB
3. **或者使用摇杆组合**：左摇杆控制 x, y，右摇杆控制 pitch, yaw，LT/RT 控制 z，LB/RB 控制 roll

## 代码证据

### 证据 1：轴值计算方式

```python
# pyspacemouse.py line 256-258
self.dict_state[name] = (
    flip * to_int16(data[b1], data[b2]) / float(self.axis_scale)
)
```

- `to_int16` 返回 16 位整数，范围 `[-32768, 32767]`
- 除以 `axis_scale` (350.0) 后得到连续浮点数
- **这是连续模拟量的典型处理方式**

### 证据 2：按钮值计算方式

```python
# pyspacemouse.py line 265-267
self.dict_state["buttons"][button_index] = (
    1 if (data[byte] & mask) != 0 else 0
)
```

- 使用位掩码判断，输出只有 0 或 1
- **这是开关量的典型处理方式**

### 证据 3：文档注释

```python
# pyspacemouse.py line 233
# axis [x,y,z,roll,pitch,yaw] in range [-1.0, 1.0]
```

- 明确说明轴值在 `[-1.0, 1.0]` 范围内
- 这是连续值的范围，不是离散值

## 验证方法

可以通过以下方式验证：

1. **运行 SpaceMouse 测试脚本**：
   ```bash
   python serl_robot_infra/franka_env/spacemouse/spacemouse_test.py
   ```
   观察输出值是否连续变化

2. **检查输出值**：
   - 6DOF 轴值应该在 `[-1.0, 1.0]` 范围内连续变化
   - 按钮值应该只有 0 或 1

## 总结

✅ **SpaceMouse 的 xyz 和 rpy 都是连续模拟量，范围 `[-1.0, 1.0]`**

✅ **SpaceMouse 的按钮是离散开关量，只有 0 或 1**

✅ **手柄映射应该使用连续模拟输入（摇杆、模拟扳机键），而不是开关量（数字按钮）**
