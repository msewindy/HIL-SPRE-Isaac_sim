# SpaceMouse 控制 Franka 机械臂机制分析

本文档详细分析了原项目中 SpaceMouse 如何控制 Franka 机械臂的完整流程。SpaceMouse 在项目中用于收集演示数据和训练过程中的人类干预。

## 目录

1. [整体架构](#整体架构)
2. [详细流程分析](#详细流程分析)
3. [关键代码解析](#关键代码解析)
4. [数据流图](#数据流图)
5. [应用场景](#应用场景)

---

## 整体架构

SpaceMouse 控制 Franka 机械臂采用**分层架构**，从硬件输入到机器人执行共经过 5 层处理：

```
SpaceMouse 硬件输入
    ↓
1. SpaceMouseExpert (输入读取层)
    ↓
2. SpacemouseIntervention (干预检测与动作覆盖层)
    ↓
3. FrankaEnv.step (动作转换为机器人命令层)
    ↓
4. HTTP 请求 (命令发送层)
    ↓
5. ROS 控制 (机器人执行层)
    ↓
Franka 机械臂执行
```

---

## 详细流程分析

### 1. SpaceMouse 输入读取层 (`SpaceMouseExpert`)

**文件位置**: `serl_robot_infra/franka_env/spacemouse/spacemouse_expert.py`

**核心功能**:
- 使用独立进程持续读取 SpaceMouse 状态
- 将原始输入映射为 6 维动作向量
- 通过共享内存传递最新状态

**关键代码**:

```python
def _read_spacemouse(self):
    while True:
        state = pyspacemouse.read_all()
        action = [0.0] * 6
        buttons = [0, 0, 0, 0]

        if len(state) == 2:  # 双 SpaceMouse 设备
            action = [
                -state[0].y, state[0].x, state[0].z,
                -state[0].roll, -state[0].pitch, -state[0].yaw,
                -state[1].y, state[1].x, state[1].z,
                -state[1].roll, -state[1].pitch, -state[1].yaw
            ]
            buttons = state[0].buttons + state[1].buttons
        elif len(state) == 1:  # 单 SpaceMouse 设备
            action = [
                -state[0].y, state[0].x, state[0].z,
                -state[0].roll, -state[0].pitch, -state[0].yaw
            ]
            buttons = state[0].buttons

        # 更新共享状态
        self.latest_data["action"] = action
        self.latest_data["buttons"] = buttons
```

**关键特点**:
- **坐标映射**: 将 SpaceMouse 的原始输入映射为机器人坐标系
  - 位置: `[-y, x, z]` (注意 y 轴取反)
  - 旋转: `[-roll, -pitch, -yaw]` (所有旋转轴取反)
- **多设备支持**: 支持单/双 SpaceMouse 设备
- **异步读取**: 使用独立进程 (`multiprocessing.Process`) 持续读取，避免阻塞主线程
- **共享内存**: 使用 `multiprocessing.Manager().dict()` 实现进程间数据共享

---

### 2. 干预检测与动作覆盖层 (`SpacemouseIntervention`)

**文件位置**: `serl_robot_infra/franka_env/envs/wrappers.py`

**核心功能**:
- 检测 SpaceMouse 是否有输入（干预）
- 如果有输入，用 SpaceMouse 动作覆盖策略动作
- 处理夹爪按钮控制
- 标记干预动作供后续记录

**关键代码**:

```python
def action(self, action: np.ndarray) -> np.ndarray:
    """
    Input:
    - action: policy action
    Output:
    - action: spacemouse action if nonezero; else, policy action
    """
    expert_a, buttons = self.expert.get_action()
    self.left, self.right = tuple(buttons)
    intervened = False
    
    # 检测是否有 SpaceMouse 输入
    if np.linalg.norm(expert_a) > 0.001:
        intervened = True

    # 处理夹爪控制
    if self.gripper_enabled:
        if self.left:  # 左键：关闭夹爪
            gripper_action = np.random.uniform(-1, -0.9, size=(1,))
            intervened = True
        elif self.right:  # 右键：打开夹爪
            gripper_action = np.random.uniform(0.9, 1, size=(1,))
            intervened = True
        else:
            gripper_action = np.zeros((1,))
        expert_a = np.concatenate((expert_a, gripper_action), axis=0)

    # 动作过滤（如果指定了 action_indices）
    if self.action_indices is not None:
        filtered_expert_a = np.zeros_like(expert_a)
        filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
        expert_a = filtered_expert_a

    if intervened:
        return expert_a, True  # 返回干预动作

    return action, False  # 返回策略动作

def step(self, action):
    new_action, replaced = self.action(action)
    
    obs, rew, done, truncated, info = self.env.step(new_action)
    if replaced:
        info["intervene_action"] = new_action  # 标记为干预动作
    info["left"] = self.left
    info["right"] = self.right
    return obs, rew, done, truncated, info
```

**关键特点**:
- **干预检测**: 通过 `np.linalg.norm(expert_a) > 0.001` 判断是否有 SpaceMouse 输入
- **动作覆盖**: 检测到输入时，SpaceMouse 动作**完全覆盖**策略动作
- **夹爪控制**:
  - 左键 (`left`): 关闭夹爪，动作值 `-1` 到 `-0.9`
  - 右键 (`right`): 打开夹爪，动作值 `0.9` 到 `1`
- **干预标记**: 在 `info["intervene_action"]` 中记录干预动作，供训练脚本识别和记录

---

### 3. 动作转换为机器人命令层 (`FrankaEnv.step`)

**文件位置**: `serl_robot_infra/franka_env/envs/franka_env.py`

**核心功能**:
- 将归一化的动作增量转换为机器人位姿增量
- 应用安全边界框限制
- 发送位姿命令和夹爪命令

**关键代码**:

```python
def step(self, action: np.ndarray) -> tuple:
    """standard gym step function."""
    start_time = time.time()
    action = np.clip(action, self.action_space.low, self.action_space.high)
    xyz_delta = action[:3]

    # 计算新位置（增量控制）
    self.nextpos = self.currpos.copy()
    self.nextpos[:3] = self.nextpos[:3] + xyz_delta * self.action_scale[0]

    # 计算新姿态（旋转向量增量）
    self.nextpos[3:] = (
        Rotation.from_rotvec(action[3:6] * self.action_scale[1])
        * Rotation.from_quat(self.currpos[3:])
    ).as_quat()

    # 处理夹爪动作
    gripper_action = action[6] * self.action_scale[2]
    self._send_gripper_command(gripper_action)
    
    # 发送位姿命令（应用安全边界框）
    self._send_pos_command(self.clip_safety_box(self.nextpos))

    # 控制执行频率
    self.curr_path_length += 1
    dt = time.time() - start_time
    time.sleep(max(0, (1.0 / self.hz) - dt))

    # 更新状态并返回
    self._update_currpos()
    ob = self._get_obs()
    reward = self.compute_reward(ob)
    done = self.curr_path_length >= self.max_episode_length or reward or self.terminate
    return ob, int(reward), done, False, {"succeed": reward}
```

**关键特点**:
- **增量控制**: SpaceMouse 输入是**相对增量**，不是绝对位姿
  - 位置增量: `nextpos[:3] = currpos[:3] + action[:3] * action_scale[0]`
  - 旋转增量: 使用旋转向量 (rotvec) 与当前姿态复合
- **动作缩放**: 通过 `action_scale` 控制增量大小
  - `action_scale[0]`: 位置增量缩放（单位：米）
  - `action_scale[1]`: 旋转增量缩放（单位：弧度）
  - `action_scale[2]`: 夹爪动作缩放
- **安全边界**: 通过 `clip_safety_box()` 限制位姿在工作空间内
- **频率控制**: 按 `hz`（默认 10Hz）控制执行频率

---

### 4. HTTP 命令发送层 (`FrankaEnv._send_pos_command`)

**文件位置**: `serl_robot_infra/franka_env/envs/franka_env.py`

**核心功能**:
- 通过 HTTP POST 请求发送位姿命令到 Flask 服务器
- 位姿格式：`[x, y, z, qx, qy, qz, qw]` (位置 + 四元数)

**关键代码**:

```python
def _send_pos_command(self, pos: np.ndarray):
    """Internal function to send position command to the robot."""
    self._recover()  # 清除错误状态
    arr = np.array(pos).astype(np.float32)
    data = {"arr": arr.tolist()}
    requests.post(self.url + "pose", json=data)
```

**关键特点**:
- **HTTP 通信**: 使用 `requests.post()` 发送 JSON 格式的位姿数据
- **服务器 URL**: 通过 `self.url` 配置（默认 `http://127.0.0.1:5000/`）
- **错误恢复**: 每次发送命令前调用 `_recover()` 清除可能的错误状态

---

### 5. ROS 控制层 (`franka_server.py`)

**文件位置**: `serl_robot_infra/robot_servers/franka_server.py`

**核心功能**:
- Flask 服务器接收 HTTP 请求
- 转换为 ROS 消息
- 发布到阻抗控制器

**关键代码**:

```python
# Flask 路由
@webapp.route("/pose", methods=["POST"])
def pose():
    pos = np.array(request.json["arr"])
    robot_server.move(pos)
    return "Moved"

# 转换为 ROS 消息并发布
def move(self, pose: list):
    """Moves to a pose: [x, y, z, qx, qy, qz, qw]"""
    assert len(pose) == 7
    msg = geom_msg.PoseStamped()
    msg.header.frame_id = "0"
    msg.header.stamp = rospy.Time.now()
    msg.pose.position = geom_msg.Point(pose[0], pose[1], pose[2])
    msg.pose.orientation = geom_msg.Quaternion(pose[3], pose[4], pose[5], pose[6])
    self.eepub.publish(msg)  # 发布到 ROS topic
```

**关键特点**:
- **Flask 服务器**: 接收 HTTP POST 请求到 `/pose` 端点
- **ROS 消息转换**: 将位姿数组转换为 `geometry_msgs/PoseStamped` 消息
- **Topic 发布**: 发布到 `/cartesian_impedance_controller/equilibrium_pose`
- **阻抗控制**: 阻抗控制器接收目标位姿，执行柔顺跟踪

---

## 关键代码解析

### SpaceMouse 坐标映射

SpaceMouse 的原始输入需要映射到机器人坐标系：

```python
# 原始输入: state[0].x, state[0].y, state[0].z, state[0].roll, state[0].pitch, state[0].yaw
# 映射后: [-y, x, z, -roll, -pitch, -yaw]
action = [
    -state[0].y,      # x 方向：使用 SpaceMouse 的 -y
    state[0].x,       # y 方向：使用 SpaceMouse 的 x
    state[0].z,       # z 方向：使用 SpaceMouse 的 z
    -state[0].roll,   # roll：取反
    -state[0].pitch,  # pitch：取反
    -state[0].yaw      # yaw：取反
]
```

**为什么需要映射？**
- SpaceMouse 的坐标系与机器人坐标系不一致
- 需要将 SpaceMouse 的输入转换为机器人期望的坐标系

### 干预检测阈值

```python
if np.linalg.norm(expert_a) > 0.001:
    intervened = True
```

**阈值选择**: `0.001` 是一个很小的阈值，用于过滤噪声，确保只有真正的 SpaceMouse 输入才会触发干预。

### 夹爪控制逻辑

```python
if self.left:  # 左键：关闭夹爪
    gripper_action = np.random.uniform(-1, -0.9, size=(1,))
elif self.right:  # 右键：打开夹爪
    gripper_action = np.random.uniform(0.9, 1, size=(1,))
```

**为什么使用随机值？**
- 使用 `np.random.uniform()` 而不是固定值，可能是为了增加动作的多样性
- 关闭夹爪: `-1` 到 `-0.9`（负值表示关闭）
- 打开夹爪: `0.9` 到 `1`（正值表示打开）

---

## 数据流图

```
┌─────────────────────────────────────────────────────────────┐
│                    SpaceMouse 硬件                            │
│  [x, y, z, roll, pitch, yaw, buttons]                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  SpaceMouseExpert._read_spacemouse()                         │
│  - 坐标映射: [-y, x, z, -roll, -pitch, -yaw]                │
│  - 共享内存更新                                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  SpacemouseIntervention.action()                             │
│  - 检测干预: norm(expert_a) > 0.001                          │
│  - 动作覆盖: 策略动作 → SpaceMouse 动作                      │
│  - 夹爪控制: 左键关闭，右键打开                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  FrankaEnv.step()                                            │
│  - 增量计算: currpos + action * scale                        │
│  - 安全边界: clip_safety_box()                              │
│  - 频率控制: 10Hz                                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  HTTP POST: /pose                                             │
│  - URL: http://127.0.0.1:5000/pose                          │
│  - 数据: {"arr": [x, y, z, qx, qy, qz, qw]}                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  franka_server.py                                             │
│  - Flask 路由: @webapp.route("/pose")                        │
│  - ROS 消息: PoseStamped                                     │
│  - Topic: /cartesian_impedance_controller/equilibrium_pose   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  阻抗控制器 (serl_franka_controllers)                        │
│  - 接收目标位姿                                              │
│  - 执行柔顺跟踪                                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Franka 机械臂执行运动                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 应用场景

### 1. 演示数据收集 (`record_demos.py`)

在演示数据收集中，策略动作始终为零，完全由 SpaceMouse 控制：

```python
while success_count < success_needed:
    actions = np.zeros(env.action_space.sample().shape)  # 策略动作为零
    next_obs, rew, done, truncated, info = env.step(actions)
    
    if "intervene_action" in info:
        actions = info["intervene_action"]  # 使用 SpaceMouse 动作
    
    # 记录 transition
    transition = dict(
        observations=obs,
        actions=actions,  # 记录的是 SpaceMouse 动作
        ...
    )
```

**特点**:
- 策略不参与控制，完全由人类通过 SpaceMouse 操作
- 所有动作都来自 SpaceMouse，因此所有动作都被标记为 `intervene_action`

### 2. 训练过程中的干预 (`train_rlpd.py`)

在策略训练过程中，SpaceMouse 用于实时干预：

```python
for step in range(start_step, config.max_steps):
    # 1. 策略采样动作
    if step < config.random_steps:
        actions = env.action_space.sample()  # 随机探索
    else:
        actions = agent.sample_actions(obs, seed=key)  # 策略采样
    
    # 2. 环境步进（SpaceMouse 可能覆盖动作）
    next_obs, reward, done, truncated, info = env.step(actions)
    
    # 3. 检测干预
    if "intervene_action" in info:
        actions = info.pop("intervene_action")  # 使用干预动作
        intervention_steps += 1
    
    # 4. 数据存储
    transition = {
        "observations": obs,
        "actions": actions,  # 如果发生干预，这里是干预动作
        ...
    }
    data_store.insert(transition)  # 所有数据存入在线缓冲区
    
    if already_intervened:
        intvn_data_store.insert(transition)  # 干预数据单独存储
```

**特点**:
- 策略正常执行，但当 SpaceMouse 有输入时，动作会被覆盖
- 干预数据会同时存入两个缓冲区：
  - `data_store`: 所有数据（包括干预数据）
  - `intvn_data_store`: 仅干预数据（用于演示学习）

---

## 关键设计特点总结

1. **增量控制**: SpaceMouse 输入是相对增量，不是绝对位姿，更符合人类操作习惯
2. **实时覆盖**: 检测到 SpaceMouse 输入时立即覆盖策略动作，实现无缝切换
3. **安全边界**: 通过 `clip_safety_box()` 限制工作空间，确保机器人安全
4. **异步读取**: 使用独立进程读取 SpaceMouse，避免阻塞主线程
5. **阻抗控制**: 使用阻抗控制器实现柔顺控制，提高操作安全性
6. **干预标记**: 通过 `info["intervene_action"]` 标记干预动作，便于数据记录和分析

---

## 相关文件

- **SpaceMouse 接口**: `serl_robot_infra/franka_env/spacemouse/spacemouse_expert.py`
- **干预包装器**: `serl_robot_infra/franka_env/envs/wrappers.py`
- **环境实现**: `serl_robot_infra/franka_env/envs/franka_env.py`
- **服务器实现**: `serl_robot_infra/robot_servers/franka_server.py`
- **演示收集**: `examples/record_demos.py`
- **训练脚本**: `examples/train_rlpd.py`

---

## 参考文档

- [RAM 插入任务训练指南](./ram_insertion_training_guide.md)
- [RLPD 训练流程分析](./train_rlpd_analysis.md)
- [数据管理机制分析](./data_management_analysis.md)
