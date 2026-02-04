# record_demos.py 详细逻辑分析 - USB Pickup Insertion 示例

## 一、整体流程概览

`record_demos.py` 的核心任务是收集成功的演示轨迹数据，用于后续的行为克隆（BC）或强化学习训练。

### 1.1 主循环流程

```python
# 主循环：收集指定数量的成功演示
while success_count < success_needed:
    # 1. 执行动作（初始为零动作）
    actions = np.zeros(env.action_space.sample().shape)
    
    # 2. 环境步进
    next_obs, rew, done, truncated, info = env.step(actions)
    
    # 3. 检查是否有干预动作（来自 SpaceMouse/Gamepad）
    if "intervene_action" in info:
        actions = info["intervene_action"]
    
    # 4. 构建 transition 并添加到轨迹
    transition = {
        observations=obs,
        actions=actions,  # 可能是零动作或干预动作
        next_observations=next_obs,
        rewards=rew,
        masks=1.0 - done,
        dones=done,
        infos=info,
    }
    trajectory.append(transition)
    
    # 5. 如果 episode 结束且成功，保存轨迹
    if done and info["succeed"]:
        transitions.extend(trajectory)
        success_count += 1
```

---

## 二、演示数据格式

### 2.1 Transition 数据结构

每个 transition 是一个字典，包含以下字段：

```python
transition = {
    "observations": dict,      # 当前时刻的观察
    "actions": np.ndarray,     # 执行的动作 [7] (x, y, z, rx, ry, rz, gripper)
    "next_observations": dict, # 下一时刻的观察
    "rewards": float,          # 奖励值（0 或 1）
    "masks": float,            # 折扣掩码 (1.0 - done)
    "dones": bool,             # 是否结束
    "infos": dict,             # 额外信息（包含 succeed, intervene_action 等）
}
```

### 2.2 Observation 数据结构

经过所有 wrapper 处理后，observation 的最终格式：

```python
observation = {
    "state": np.ndarray,  # 展平后的本体感觉状态
    "side_policy": np.ndarray,  # [H, W, 3] 图像
    "wrist_1": np.ndarray,      # [H, W, 3] 图像
    "wrist_2": np.ndarray,      # [H, W, 3] 图像
}
```

**state 维度详解**（USB Pickup Insertion 配置）：
- `proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]`
- `tcp_pose`: [6] - (x, y, z, roll, pitch, yaw) 在相对坐标系中的位姿
- `tcp_vel`: [6] - (vx, vy, vz, wx, wy, wz) 在末端执行器坐标系中的速度
- `tcp_force`: [3] - (fx, fy, fz) 力
- `tcp_torque`: [3] - (tx, ty, tz) 力矩
- `gripper_pose`: [1] - 夹爪开合度 [0, 1]

**图像维度**：
- 所有图像：`[128, 128, 3]` (uint8)
- `side_policy`: 侧视相机（策略用）
- `wrist_1`, `wrist_2`: 两个腕部相机

### 2.3 Action 数据结构

```python
action = np.ndarray([7])  # [x, y, z, rx, ry, rz, gripper]
# x, y, z: 位置增量（在末端执行器坐标系中）
# rx, ry, rz: 旋转增量（旋转向量，在末端执行器坐标系中）
# gripper: 夹爪动作 [-1, 1]，-1 关闭，1 打开
```

---

## 三、环境包装器堆叠顺序

USB Pickup Insertion 的环境包装器堆叠顺序（从内到外）：

```
1. USBEnv (FrankaEnv 子类)
   ↓
2. SpacemouseIntervention (如果 fake_env=False)
   ↓
3. RelativeFrame
   ↓
4. Quat2EulerWrapper
   ↓
5. SERLObsWrapper
   ↓
6. ChunkingWrapper (obs_horizon=1, act_exec_horizon=None)
   ↓
7. MultiCameraBinaryRewardClassifierWrapper (如果 classifier=True)
   ↓
8. GripperPenaltyWrapper
```

---

## 四、各维度数据的生成过程

### 4.1 Actions（动作）的生成流程

#### 4.1.1 初始动作生成

**时间点**：`record_demos.py` 主循环开始时

```python
# 第 31 行
actions = np.zeros(env.action_space.sample().shape)  # [7] 全零
```

**发出者**：`record_demos.py` 主循环

**流向**：`actions` → `env.step(actions)`

---

#### 4.1.2 SpaceMouse 干预检测与替换

**时间点**：`env.step()` 调用期间，在 `SpacemouseIntervention.step()` 中

**发出者**：`SpaceMouseExpert.get_action()` 从 SpaceMouse 硬件读取

**处理流程**：

```python
# SpacemouseIntervention.action() (第 220-253 行)
expert_a, buttons = self.expert.get_action()  # 从 SpaceMouse 读取
self.left, self.right = tuple(buttons)

intervened = False
if np.linalg.norm(expert_a) > 0.001:  # 检测到 SpaceMouse 输入
    intervened = True

# 处理夹爪动作
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

if intervened:
    return expert_a, True  # 返回干预动作
else:
    return action, False   # 返回原始动作（零动作）
```

**流向**：
1. `SpacemouseIntervention.action()` → 返回 `(new_action, replaced)`
2. `SpacemouseIntervention.step()` → 如果 `replaced=True`，设置 `info["intervene_action"] = new_action`
3. `new_action` → `self.env.step(new_action)` → 传递给下层环境

**写入时间点**：
- 如果发生干预：`info["intervene_action"]` 在 `SpacemouseIntervention.step()` 中设置（第 260 行）
- 在 `record_demos.py` 中：第 34-35 行检查并替换 `actions`

---

#### 4.1.3 动作坐标系转换（RelativeFrame）

**时间点**：`RelativeFrame.step()` 中，在调用下层环境之前

**处理流程**：

```python
# RelativeFrame.step() (第 39-55 行)
def step(self, action: np.ndarray):
    # action 在末端执行器坐标系中
    transformed_action = self.transform_action(action)  # 转换到基坐标系
    obs, reward, done, truncated, info = self.env.step(transformed_action)
    
    # 如果 info 中有 intervene_action，需要转换回末端执行器坐标系
    if "intervene_action" in info:
        info["intervene_action"] = self.transform_action_inv(info["intervene_action"])
```

**transform_action 逻辑**（第 98-105 行）：
```python
def transform_action(self, action: np.ndarray):
    # 使用当前 transform_matrix 将动作从末端执行器坐标系转换到基坐标系
    action[:6] = self.transform_matrix @ action[:6]
    return action
```

**transform_matrix 更新**（第 51 行）：
```python
# 在每次 step 后，基于当前 tcp_pose 更新变换矩阵
self.transform_matrix = construct_transform_matrix(obs["state"]["tcp_pose"])
```

**流向**：
- 输入动作（末端执行器坐标系）→ `transform_action()` → 基坐标系动作 → `USBEnv.step()`
- 如果存在 `intervene_action`（基坐标系）→ `transform_action_inv()` → 末端执行器坐标系 → 返回给上层

---

#### 4.1.4 动作执行（USBEnv / FrankaEnv）

**时间点**：`FrankaEnv.step()` 中

**处理流程**：

```python
# FrankaEnv.step() (第 213-241 行)
def step(self, action: np.ndarray):
    action = np.clip(action, self.action_space.low, self.action_space.high)
    
    # 1. 计算位置增量
    xyz_delta = action[:3]
    self.nextpos[:3] = self.nextpos[:3] + xyz_delta * self.action_scale[0]
    
    # 2. 计算旋转增量（旋转向量）
    self.nextpos[3:] = (
        Rotation.from_rotvec(action[3:6] * self.action_scale[1])
        * Rotation.from_quat(self.currpos[3:])
    ).as_quat()
    
    # 3. 处理夹爪
    gripper_action = action[6] * self.action_scale[2]
    
    # 4. 发送命令到机器人服务器
    self._send_gripper_command(gripper_action)
    self._send_pos_command(self.clip_safety_box(self.nextpos))
    
    # 5. 等待控制频率
    time.sleep(max(0, (1.0 / self.hz) - dt))
    
    # 6. 更新当前状态
    self._update_currpos()
```

**ACTION_SCALE**（USB Pickup Insertion）：
```python
ACTION_SCALE = np.array([0.015, 0.1, 1])
# [0]: 位置缩放 0.015m
# [1]: 旋转缩放 0.1 rad
# [2]: 夹爪缩放 1.0
```

**流向**：
- `_send_pos_command()` → HTTP POST 到机器人服务器 (`http://127.0.0.2:5000/`)
- `_send_gripper_command()` → HTTP POST 到机器人服务器

---

#### 4.1.5 动作记录到 Transition

**时间点**：`record_demos.py` 主循环中，`env.step()` 返回后

**处理流程**：

```python
# 第 34-35 行
if "intervene_action" in info:
    actions = info["intervene_action"]  # 使用干预动作（已在相对坐标系中）

# 第 36-46 行
transition = copy.deepcopy({
    "actions": actions,  # 记录的动作（末端执行器坐标系）
    ...
})
```

**最终记录的动作**：
- 如果发生干预：`info["intervene_action"]`（已在 `RelativeFrame` 中转换回末端执行器坐标系）
- 如果没有干预：零动作 `np.zeros([7])`

---

### 4.2 Observations（观察）的生成流程

#### 4.2.1 原始观察获取（FrankaEnv._get_obs）

**时间点**：`FrankaEnv.step()` 中，执行动作后

**发出者**：`FrankaEnv._get_obs()`（第 501-510 行）

**处理流程**：

```python
def _get_obs(self) -> dict:
    # 1. 获取图像
    images = self.get_im()
    
    # 2. 构建状态观察
    state_observation = {
        "tcp_pose": self.currpos,           # [7] (x, y, z, qx, qy, qz, qw) 基坐标系
        "tcp_vel": self.currvel,            # [6] 基坐标系
        "gripper_pose": self.curr_gripper_pos,  # [1]
        "tcp_force": self.currforce,        # [3]
        "tcp_torque": self.currtorque,     # [3]
    }
    return copy.deepcopy(dict(images=images, state=state_observation))
```

**数据来源**：
- `self.currpos`, `self.currvel`, `self.currforce`, `self.currtorque`：通过 `_update_currpos()` 从机器人服务器获取
- `self.curr_gripper_pos`：从机器人服务器获取
- `images`：通过 `get_im()` 从相机捕获

**流向**：`_get_obs()` → 返回原始观察 → `FrankaEnv.step()` → 返回给上层

---

#### 4.2.2 图像获取（get_im）

**时间点**：`FrankaEnv.get_im()` 中

**发出者**：RealSense 相机硬件

**处理流程**：

```python
# FrankaEnv.get_im() (第 258-288 行)
def get_im(self) -> Dict[str, np.ndarray]:
    images = {}
    for key, cap in self.cap.items():
        rgb = cap.read()  # 从相机读取原始图像
        
        # 裁剪
        cropped_rgb = self.config.IMAGE_CROP[key](rgb) if key in self.config.IMAGE_CROP else rgb
        
        # 调整大小到 128x128
        resized = cv2.resize(
            cropped_rgb, 
            self.observation_space["images"][key].shape[:2][::-1]  # (128, 128)
        )
        
        # BGR → RGB 转换
        images[key] = resized[..., ::-1]
    
    return images
```

**USB Pickup Insertion 相机配置**：
```python
REALSENSE_CAMERAS = {
    "wrist_1": {"serial_number": "127122270350", "dim": (1280, 720), "exposure": 10500},
    "wrist_2": {"serial_number": "127122270146", "dim": (1280, 720), "exposure": 10500},
    "side_policy": {"serial_number": "130322274175", "dim": (1280, 720), "exposure": 13000},
    "side_classifier": {"serial_number": "130322274175", ...},  # 复用 side_policy
}

IMAGE_CROP = {
    "wrist_1": lambda img: img[50:-200, 200:-200],
    "wrist_2": lambda img: img[:-200, 200:-200],
    "side_policy": lambda img: img[250:500, 350:650],
    "side_classifier": lambda img: img[270:398, 500:628],
}
```

**流向**：相机硬件 → `cap.read()` → 裁剪 → 调整大小 → RGB 转换 → `images` 字典

---

#### 4.2.3 相对坐标系转换（RelativeFrame）

**时间点**：`RelativeFrame.step()` 中，从下层环境获取观察后

**处理流程**：

```python
# RelativeFrame.step() (第 39-55 行)
obs, reward, done, truncated, info = self.env.step(transformed_action)

# 更新变换矩阵
self.transform_matrix = construct_transform_matrix(obs["state"]["tcp_pose"])

# 转换观察到相对坐标系
transformed_obs = self.transform_observation(obs)
return transformed_obs, reward, done, truncated, info
```

**transform_observation 逻辑**（第 79-96 行）：

```python
def transform_observation(self, obs):
    transform_inv = np.linalg.inv(self.transform_matrix)
    
    # 1. 转换速度到末端执行器坐标系
    obs["state"]["tcp_vel"] = transform_inv @ obs["state"]["tcp_vel"]
    
    # 2. 转换位姿到相对坐标系（相对于 reset pose）
    if self.include_relative_pose:
        T_b_o = construct_homogeneous_matrix(obs["state"]["tcp_pose"])  # 基坐标系中的当前位姿
        T_b_r = self.T_r_o_inv @ T_b_o  # 转换到相对坐标系
        
        # 提取位置和四元数
        p_b_r = T_b_r[:3, 3]
        theta_b_r = R.from_matrix(T_b_r[:3, :3]).as_quat()
        obs["state"]["tcp_pose"] = np.concatenate((p_b_r, theta_b_r))
    
    return obs
```

**流向**：
- 基坐标系观察 → `transform_observation()` → 相对坐标系观察 → 返回给上层

---

#### 4.2.4 四元数转欧拉角（Quat2EulerWrapper）

**时间点**：`Quat2EulerWrapper.observation()` 中

**处理流程**：

```python
# Quat2EulerWrapper.observation() (第 117-123 行)
def observation(self, observation):
    tcp_pose = observation["state"]["tcp_pose"]  # [7] (x, y, z, qx, qy, qz, qw)
    
    # 转换四元数到欧拉角
    observation["state"]["tcp_pose"] = np.concatenate(
        (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))  # [6] (x, y, z, roll, pitch, yaw)
    )
    return observation
```

**流向**：相对坐标系位姿（四元数）→ 欧拉角转换 → 返回给上层

---

#### 4.2.5 观察展平（SERLObsWrapper）

**时间点**：`SERLObsWrapper.observation()` 中

**处理流程**：

```python
# SERLObsWrapper.observation() (第 28-36 行)
def observation(self, obs):
    obs = {
        "state": flatten(
            self.proprio_space,
            {key: obs["state"][key] for key in self.proprio_keys},
        ),
        **(obs["images"]),
    }
    return obs
```

**proprio_keys**（USB Pickup Insertion）：
```python
proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
```

**展平结果**：
- `tcp_pose`: [6] → 直接拼接
- `tcp_vel`: [6] → 直接拼接
- `tcp_force`: [3] → 直接拼接
- `tcp_torque`: [3] → 直接拼接
- `gripper_pose`: [1] → 直接拼接
- **最终 state**: [6 + 6 + 3 + 3 + 1] = [19]

**流向**：字典格式观察 → `flatten()` → 展平后的 state 向量 → 返回给上层

---

#### 4.2.6 观察历史堆叠（ChunkingWrapper）

**时间点**：`ChunkingWrapper.step()` 中

**处理流程**：

```python
# ChunkingWrapper.step() (第 61-72 行)
def step(self, action, *args):
    obs, reward, done, trunc, info = self.env.step(action[0], *args)
    self.current_obs.append(obs)  # 添加到历史队列
    return (stack_obs(self.current_obs), reward, done, trunc, info)
```

**obs_horizon=1** 时：
- `current_obs` 只包含一个观察
- `stack_obs()` 返回的观察格式不变（只是添加了时间维度 `[1, ...]`）

**流向**：单步观察 → 添加到历史队列 → 堆叠 → 返回给上层

---

#### 4.2.7 观察记录到 Transition

**时间点**：`record_demos.py` 主循环中

**处理流程**：

```python
# 第 36-46 行
transition = copy.deepcopy({
    "observations": obs,              # 当前观察（已处理）
    "next_observations": next_obs,    # 下一观察（已处理）
    ...
})
```

**最终记录的观察格式**：
```python
{
    "state": np.ndarray([19]),  # 展平后的本体感觉状态
    "side_policy": np.ndarray([128, 128, 3]),  # uint8
    "wrist_1": np.ndarray([128, 128, 3]),      # uint8
    "wrist_2": np.ndarray([128, 128, 3]),     # uint8
}
```

---

### 4.3 Rewards（奖励）的生成流程

#### 4.3.1 基础奖励计算（FrankaEnv.compute_reward）

**时间点**：`FrankaEnv.step()` 中，获取观察后

**处理流程**：

```python
# FrankaEnv.compute_reward() (第 243-256 行)
def compute_reward(self, obs) -> bool:
    current_pose = obs["state"]["tcp_pose"]  # 基坐标系中的位姿
    
    # 计算位置和旋转误差
    current_rot = Rotation.from_quat(current_pose[3:]).as_matrix()
    target_rot = Rotation.from_euler("xyz", self._TARGET_POSE[3:]).as_matrix()
    diff_rot = current_rot.T @ target_rot
    diff_euler = Rotation.from_matrix(diff_rot).as_euler("xyz")
    
    delta = np.abs(np.hstack([
        current_pose[:3] - self._TARGET_POSE[:3], 
        diff_euler
    ]))
    
    # 检查是否在阈值内
    if np.all(delta < self._REWARD_THRESHOLD):
        return True  # 成功
    else:
        return False  # 未成功
```

**流向**：`compute_reward()` → 返回 `bool` → `FrankaEnv.step()` → `reward = int(reward)` → 返回给上层

---

#### 4.3.2 分类器奖励（MultiCameraBinaryRewardClassifierWrapper）

**时间点**：`MultiCameraBinaryRewardClassifierWrapper.step()` 中

**处理流程**：

```python
# MultiCameraBinaryRewardClassifierWrapper.step() (第 52-61 行)
def step(self, action):
    obs, rew, done, truncated, info = self.env.step(action)
    rew = self.compute_reward(obs)  # 使用分类器计算奖励
    done = done or rew
    info['succeed'] = bool(rew)
    return obs, rew, done, truncated, info

def compute_reward(self, obs):
    if self.reward_classifier_func is not None:
        return self.reward_classifier_func(obs)
    return 0
```

**reward_func**（USB Pickup Insertion 配置，第 131-133 行）：

```python
def reward_func(obs):
    sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
    logit = classifier(obs)  # 分类器输出
    return int(sigmoid(logit) > 0.7 and obs["state"][0, 0] > 0.4)
    # 条件1: 分类器置信度 > 0.7
    # 条件2: 夹爪开合度 > 0.4
```

**流向**：
- 观察 → `classifier(obs)` → logit → sigmoid → 阈值判断 → 奖励（0 或 1）

---

#### 4.3.3 奖励记录到 Transition

**时间点**：`record_demos.py` 主循环中

**处理流程**：

```python
# 第 32 行
next_obs, rew, done, truncated, info = env.step(actions)
returns += rew  # 累计奖励（用于显示）

# 第 41 行
transition = copy.deepcopy({
    "rewards": rew,  # 记录奖励（0 或 1）
    ...
})
```

**最终记录的奖励**：
- 0：未成功
- 1：成功（分类器判断成功且夹爪打开）

---

### 4.4 Masks 和 Dones 的生成流程

#### 4.4.1 Done 标志生成

**时间点**：`FrankaEnv.step()` 中

**处理流程**：

```python
# FrankaEnv.step() (第 240 行)
done = (self.curr_path_length >= self.max_episode_length) or reward or self.terminate
```

**终止条件**：
1. `curr_path_length >= max_episode_length`（120 步）
2. `reward == True`（任务成功）
3. `self.terminate == True`（用户按 ESC 键）

**流向**：`FrankaEnv.step()` → `done` → 各 wrapper 传递 → `record_demos.py`

---

#### 4.4.2 Mask 计算

**时间点**：`record_demos.py` 主循环中

**处理流程**：

```python
# 第 42 行
"masks": 1.0 - done,  # 如果 done=True，mask=0.0；否则 mask=1.0
```

**用途**：用于折扣因子计算，`mask=0.0` 表示 episode 结束，后续奖励不计算

---

### 4.5 Infos 的生成流程

#### 4.5.1 基础 Info（FrankaEnv）

**时间点**：`FrankaEnv.step()` 中

**处理流程**：

```python
# FrankaEnv.step() (第 241 行)
return ob, int(reward), done, False, {"succeed": reward}
```

**基础 info**：
```python
info = {"succeed": bool}  # 是否成功
```

---

#### 4.5.2 干预信息（SpacemouseIntervention）

**时间点**：`SpacemouseIntervention.step()` 中

**处理流程**：

```python
# SpacemouseIntervention.step() (第 255-264 行)
obs, rew, done, truncated, info = self.env.step(new_action)
if replaced:
    info["intervene_action"] = new_action  # 记录干预动作（基坐标系）
info["left"] = self.left   # 左键状态
info["right"] = self.right # 右键状态
return obs, rew, done, truncated, info
```

**流向**：
- `info["intervene_action"]` → `RelativeFrame.step()` → 转换回末端执行器坐标系
- `info["left"]`, `info["right"]` → 直接传递

---

#### 4.5.3 分类器信息（MultiCameraBinaryRewardClassifierWrapper）

**时间点**：`MultiCameraBinaryRewardClassifierWrapper.step()` 中

**处理流程**：

```python
# MultiCameraBinaryRewardClassifierWrapper.step() (第 57 行)
info['succeed'] = bool(rew)  # 覆盖 succeed 标志
```

---

#### 4.5.4 夹爪惩罚信息（GripperPenaltyWrapper）

**时间点**：`GripperPenaltyWrapper.step()` 中

**处理流程**：

```python
# GripperPenaltyWrapper.step() (第 114-128 行)
if (action[-1] < -0.5 and self.last_gripper_pos > 0.9) or (
    action[-1] > 0.5 and self.last_gripper_pos < 0.9
):
    info["grasp_penalty"] = self.penalty  # -0.02
else:
    info["grasp_penalty"] = 0.0
```

**条件**：检测夹爪状态突变（快速开合）

---

#### 4.5.5 Info 记录到 Transition

**时间点**：`record_demos.py` 主循环中

**处理流程**：

```python
# 第 44 行
"infos": info,  # 完整的 info 字典
```

**最终记录的 info**：
```python
{
    "succeed": bool,              # 是否成功
    "intervene_action": np.ndarray([7]) or None,  # 干预动作（如果存在）
    "left": bool,                 # SpaceMouse 左键
    "right": bool,                # SpaceMouse 右键
    "grasp_penalty": float,       # 夹爪惩罚（如果存在）
    "original_state_obs": dict,   # 原始状态观察（RelativeFrame 添加）
}
```

---

## 五、数据写入时间点总结

### 5.1 每个 Step 的数据流时间线

```
T0: record_demos.py 主循环开始
    ↓
T1: actions = np.zeros([7])  # 生成零动作
    ↓
T2: env.step(actions) 调用
    ↓
T3: SpacemouseIntervention.action()
    - 从 SpaceMouse 读取输入
    - 如果检测到输入，替换为零动作
    - 返回 (new_action, replaced)
    ↓
T4: SpacemouseIntervention.step()
    - 调用 self.env.step(new_action)
    - 如果 replaced=True，设置 info["intervene_action"]
    ↓
T5: RelativeFrame.step()
    - transform_action(): 动作从末端执行器坐标系 → 基坐标系
    - 调用 self.env.step(transformed_action)
    - 如果存在 intervene_action，转换回末端执行器坐标系
    - transform_observation(): 观察从基坐标系 → 相对坐标系
    ↓
T6: Quat2EulerWrapper.observation()
    - tcp_pose: 四元数 → 欧拉角
    ↓
T7: SERLObsWrapper.observation()
    - 展平 state 字典为向量
    ↓
T8: ChunkingWrapper.step()
    - 添加观察到历史队列
    - 堆叠观察
    ↓
T9: MultiCameraBinaryRewardClassifierWrapper.step()
    - 使用分类器计算奖励
    - 设置 info["succeed"]
    ↓
T10: GripperPenaltyWrapper.step()
    - 检测夹爪状态突变
    - 设置 info["grasp_penalty"]
    ↓
T11: USBEnv.step() / FrankaEnv.step()
    - 执行动作（发送到机器人服务器）
    - _update_currpos(): 从服务器获取状态
    - _get_obs(): 获取观察（图像 + 状态）
    - compute_reward(): 计算基础奖励
    ↓
T12: 返回 (next_obs, rew, done, truncated, info) 到 record_demos.py
    ↓
T13: record_demos.py 处理返回
    - 如果 "intervene_action" in info，替换 actions
    - 构建 transition
    - 添加到 trajectory
    ↓
T14: 如果 done and info["succeed"]:
    - 将 trajectory 中的所有 transition 添加到 transitions
    - success_count += 1
```

### 5.2 最终数据保存

**时间点**：收集到 `success_needed` 个成功演示后

**处理流程**：

```python
# 第 62-68 行
if not os.path.exists("./demo_data"):
    os.makedirs("./demo_data")

uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
file_name = f"./demo_data/{FLAGS.exp_name}_{success_needed}_demos_{uuid}.pkl"

with open(file_name, "wb") as f:
    pkl.dump(transitions, f)  # 保存所有成功的 transition
```

**保存格式**：
- 文件格式：`.pkl` (pickle)
- 内容：`transitions` 列表，每个元素是一个 transition 字典
- 文件命名：`{exp_name}_{success_needed}_demos_{timestamp}.pkl`

---

## 六、关键设计要点

### 6.1 动作记录策略

- **零动作作为默认**：`record_demos.py` 始终发送零动作，依赖 SpaceMouse 干预来执行实际动作
- **干预动作优先**：如果检测到干预，记录干预动作而不是零动作
- **坐标系一致性**：所有记录的动作都在末端执行器坐标系中（相对坐标系）

### 6.2 观察处理流程

- **多层转换**：基坐标系 → 相对坐标系 → 四元数 → 欧拉角 → 展平
- **图像处理**：裁剪 → 调整大小 → BGR→RGB
- **状态展平**：字典格式 → 向量格式

### 6.3 奖励计算

- **双重判断**：分类器置信度 + 夹爪状态
- **二进制奖励**：0（失败）或 1（成功）

### 6.4 数据完整性

- **深拷贝**：使用 `copy.deepcopy()` 确保数据不被后续修改影响
- **完整信息**：记录 observations、next_observations、actions、rewards、masks、dones、infos

---

## 七、总结

`record_demos.py` 的核心逻辑是：

1. **循环收集成功演示**：直到收集到指定数量的成功演示
2. **零动作 + 干预检测**：发送零动作，检测 SpaceMouse 干预，记录实际执行的动作
3. **多层包装器处理**：动作和观察经过多个 wrapper 的转换和处理
4. **成功判断**：使用分类器判断任务是否成功
5. **数据保存**：将成功的轨迹保存为 pickle 文件

整个流程确保了演示数据的完整性和一致性，为后续的行为克隆和强化学习训练提供了高质量的数据。
