# Gear 组装任务迁移影响分析

## 一、任务变更概述

### 1.1 原任务：RAM 插入
- **任务对象**：RAM 条、主板插槽、RAM 支架
- **任务目标**：将 RAM 条插入到主板插槽中
- **机器人路径**：`/World/franka`
- **相机路径**：`/World/franka/panda_hand/wrist_1`, `/World/franka/panda_hand/wrist_2`

### 1.2 新任务：Gear 组装
- **任务对象**：`factory_gear_medium`（待安装）、`factory_gear_base`（目标位置，已安装）、`factory_gear_large`（已安装）
- **任务目标**：将 `gear_medium` 安装到 `gear_base` 上
- **机器人路径**：`/World/franka`（已在 USD 文件中正确配置）
- **相机路径**：`/World/franka/panda_hand/wrist_1`, `/World/franka/panda_hand/wrist_2`（已在 USD 文件中正确命名）

---

## 二、USD 文件结构变化

### 2.1 机器人路径
```python
# 配置（USD 文件已正确设置）
robot_prim_path = "/World/franka"
```
✅ **已确认**：USD 文件 `HIL_franka_gear.usda` 中的机器人路径为 `/World/franka`，与原配置一致，**不需要修改**。

### 2.2 相机路径
```python
# 配置（USD 文件已正确设置）
camera_prim_paths = [
    "/World/franka/panda_hand/wrist_1",
    "/World/franka/panda_hand/wrist_2"
]
```
✅ **已确认**：USD 文件中的相机已正确命名为 `wrist_1` 和 `wrist_2`，与原配置一致，**不需要修改**。

### 2.3 相机类型说明
**重要发现**：原项目（真机运行）使用的相机类型：

1. **硬件**：Intel RealSense 深度相机（如 D435）
2. **软件使用**：**仅使用 RGB 彩色流，不使用深度信息**
   - 代码位置：`serl_robot_infra/franka_env/camera/rs_capture.py`
   - `RSCapture` 类默认参数 `depth=False`
   - 只启用 `rs.stream.color`（RGB 彩色流）
   - 返回的是 RGB 图像（BGR 格式，通过 OpenCV 转换为 RGB）

3. **配置命名**：
   - `REALSENSE_CAMERAS` 这个命名是因为硬件是 RealSense 相机
   - 但实际只使用 RGB 图像，不使用深度信息
   - 在 Isaac Sim 仿真环境中，使用虚拟 RGB 相机，功能等效

4. **相机配置**：
```python
# config.py 中的配置
REALSENSE_CAMERAS = {
    "wrist_1": {
        "serial_number": "127122270146",  # 真实环境：RealSense 序列号
        "dim": (1280, 720),                # RGB 图像分辨率
        "exposure": 40000,                 # 曝光参数
    },
    "wrist_2": {
        "serial_number": "127122270350",
        "dim": (1280, 720),
        "exposure": 40000,
    },
}

# Isaac Sim 环境配置（虚拟相机）
# 注意：对于 Isaac Sim 环境，只需要键名，字段值（serial_number、dim、exposure）是多余的
# Isaac Sim 服务器通过 camera_prim_paths 参数加载相机，不依赖这些字段
REALSENSE_CAMERAS = {
    "wrist_1": {},  # 只需要键名，用于定义观察空间和图像键名
    "wrist_2": {},  # 字段值在 Isaac Sim 中不使用
}

# 或者更简洁的方式（如果基类允许）：
# REALSENSE_CAMERAS = {
#     "wrist_1": None,
#     "wrist_2": None,
# }
```

**重要说明**：
- ✅ USD 文件中的相机路径和命名已正确，**不需要修改**
- ✅ `REALSENSE_CAMERAS` 配置**只需要键名**（`wrist_1`、`wrist_2`），用于：
  - 定义观察空间（`isaac_sim_env.py` 第 164 行：`for key in self.config.REALSENSE_CAMERAS.keys()`）
  - 获取图像键名（`isaac_sim_env.py` 第 306 行）
- ⚠️ **字段值（serial_number、dim、exposure）在 Isaac Sim 中不使用**：
  - Isaac Sim 服务器通过 `camera_prim_paths` 参数加载相机
  - 相机参数（分辨率、曝光等）在 USD 文件中定义
  - 这些字段值只是为了保持配置结构一致，实际不使用
- ✅ 仿真环境使用 RGB 虚拟相机，与真实环境（RealSense RGB 流）功能等效

---

## 三、原项目相机类型详细说明

### 3.1 硬件类型
- **硬件设备**：Intel RealSense 深度相机（如 D435、D435i 等）
- **参考**：[HIL-SERL 项目网站](https://hil-serl.github.io/)

### 3.2 软件使用方式
**重要发现**：虽然硬件是深度相机，但**软件层面只使用 RGB 彩色流，不使用深度信息**。

**代码证据**：
1. **`serl_robot_infra/franka_env/camera/rs_capture.py`**：
   ```python
   def __init__(self, name, serial_number, dim=(640, 480), fps=15, depth=False, ...):
       # depth=False 是默认参数
       self.cfg.enable_stream(rs.stream.color, dim[0], dim[1], rs.format.bgr8, fps)
       if self.depth:  # 只有当 depth=True 时才启用深度流
           self.cfg.enable_stream(rs.stream.depth, dim[0], dim[1], rs.format.z16, fps)
   ```

2. **`serl_robot_infra/franka_env/envs/franka_env.py`**：
   ```python
   def init_cameras(self, name_serial_dict=None):
       for cam_name, kwargs in name_serial_dict.items():
           cap = VideoCapture(
               RSCapture(name=cam_name, **kwargs)  # 没有传入 depth=True
           )
   ```

3. **返回数据**：
   ```python
   def get_im(self) -> Dict[str, np.ndarray]:
       rgb = cap.read()  # 只返回 RGB 图像，没有深度信息
   ```

### 3.3 配置命名说明
- **`REALSENSE_CAMERAS`**：这个命名是因为硬件是 RealSense 相机
- **实际使用**：只使用 RGB 彩色图像，不使用深度信息
- **仿真环境**：使用 Isaac Sim 虚拟 RGB 相机，功能等效

### 3.4 对迁移的影响
✅ **无影响**：
- USD 文件中的虚拟相机是 RGB 相机，与真实环境（RealSense RGB 流）功能等效
- 配置保持不变，键名 `wrist_1`、`wrist_2` 已匹配
- 不需要任何修改

---

## 四、代码影响分析

### 4.1 必须修改的文件

#### 4.1.1 `isaac_sim_server.py` 启动参数
**影响程度**：🟢 **不需要修改**（USD 文件路径已正确）

**说明**：
- ✅ USD 文件 `HIL_franka_gear.usda` 中的机器人路径为 `/World/franka`，与默认配置一致
- ✅ 相机路径为 `/World/franka/panda_hand/wrist_1` 和 `/World/franka/panda_hand/wrist_2`，与默认配置一致
- ✅ **可以直接使用默认参数启动服务器**

**启动命令**：
```bash
# 使用默认参数即可（机器人路径和相机路径与 USD 文件一致）
python serl_robot_infra/robot_servers/isaac_sim_server.py \
    --usd_path=/path/to/HIL_franka_gear.usda \
    --flask_url=0.0.0.0 \
    --flask_port=5001 \
    --headless=True
    # robot_prim_path 和 camera_prim_paths 使用默认值即可
```

**代码位置**：
- `serl_robot_infra/robot_servers/isaac_sim_server.py` 第 47-48 行（默认值已匹配，无需修改）

#### 4.1.2 `config.py` - 位姿配置
**影响程度**：🔴 **必须修改**

**需要修改的内容**：
1. **TARGET_POSE**：gear_medium 安装到 gear_base 的目标位姿
2. **GRASP_POSE**：抓取 gear_medium 的位姿
3. **RESET_POSE**：重置位姿
4. **ABS_POSE_LIMIT_LOW/HIGH**：探索边界框（根据新任务空间调整）

**修改位置**：
- `examples/experiments/ram_insertion/config.py`
  - `EnvConfig` 类（第 44-48 行）
  - `IsaacSimEnvConfig` 类（第 133-137 行）

**如何获取新位姿**：
1. 在 Isaac Sim 中手动移动机器人到目标位置
2. 使用 `curl -X POST http://127.0.0.1:5001/getstate` 获取当前位姿
3. 或通过 `isaac_sim_server.py` 的 `/getstate` 接口

**示例修改**：
```python
# 需要根据实际测量值修改
IsaacSimEnvConfig:
    TARGET_POSE = np.array([x, y, z, roll, pitch, yaw])  # gear_medium 安装位置
    GRASP_POSE = np.array([x, y, z, roll, pitch, yaw])   # gear_medium 抓取位置
    # ... 其他位姿配置
```

#### 4.1.3 `config.py` - 相机配置
**影响程度**：🟢 **可以简化**（字段值在 Isaac Sim 中不使用）

**说明**：
- ✅ USD 文件中的相机已正确命名为 `wrist_1` 和 `wrist_2`
- ✅ `REALSENSE_CAMERAS` 配置中的键名保持不变（`wrist_1`、`wrist_2`）
- ⚠️ **字段值（serial_number、dim、exposure）在 Isaac Sim 中不使用**，可以简化

**代码证据**：
- `isaac_sim_env.py` 只使用 `self.config.REALSENSE_CAMERAS.keys()` 获取键名
- Isaac Sim 服务器通过 `camera_prim_paths` 参数加载相机，不依赖配置中的字段值

**推荐配置（简化版）**：
```python
# config.py - IsaacSimEnvConfig
# 对于 Isaac Sim 环境，只需要键名，字段值不使用
REALSENSE_CAMERAS = {
    "wrist_1": {},  # 只需要键名，用于定义观察空间和图像键名
    "wrist_2": {},  # 字段值在 Isaac Sim 中不使用
}
```

**或者保持原样（兼容性）**：
```python
# 如果基类或代码需要字段存在，可以保留（但值不会被使用）
REALSENSE_CAMERAS = {
    "wrist_1": {
        "serial_number": "virtual_camera_1",  # 虚拟标识（不使用）
        "dim": (1280, 720),                   # RGB 图像分辨率（不使用）
        "exposure": 40000,                    # 曝光参数（不使用）
    },
    "wrist_2": {
        "serial_number": "virtual_camera_2",
        "dim": (1280, 720),
        "exposure": 40000,
    },
}
```

**注意**：
- `REALSENSE_CAMERAS` 这个命名是因为原项目使用 RealSense 硬件相机
- 但实际只使用 RGB 彩色流，不使用深度信息
- 在 Isaac Sim 中，使用虚拟 RGB 相机，功能等效
- **字段值只是为了保持配置结构一致，实际不使用**

#### 3.1.4 `wrapper.py` - 任务特定逻辑
**影响程度**：🟡 **可能需要修改**

**需要检查的方法**：
1. `regrasp()` 方法（第 68-103 行）
   - 包含 "Place RAM in holder" 的提示文本（第 83 行）
   - 需要修改为 "Place gear_medium in holder" 或类似文本
   - **注意**：如果新任务不需要重新抓取功能，可以保留或禁用

2. `go_to_reset()` 方法（第 24-66 行）
   - 逻辑通用，不需要修改
   - 但可能需要调整位姿偏移量

**修改建议**：
```python
# wrapper.py 第 83 行
# 原代码
input("Place RAM in holder and press enter to grasp...")

# 修改为
input("Place gear_medium in holder and press enter to grasp...")
```

#### 4.1.5 `isaac_sim_ram_env_enhanced.py` - 环境实现
**影响程度**：🟡 **可能需要修改**

**需要检查的内容**：
1. 类名和注释（第 23-35 行）
   - 类名 `IsaacSimRAMEnvEnhanced` 包含 "RAM"，但可以保留（如果不想重命名）
   - 注释中的 "RAM 插入任务" 可以更新为 "Gear 组装任务"

2. 对象引用（第 46-48 行）
   - `self.ram_stick`、`self.motherboard`、`self.ram_holder` 可以重命名为 gear 相关
   - 或保留变量名，仅更新注释

3. 方法中的 RAM 相关逻辑
   - `_attach_ram_to_gripper()`（第 89-99 行）
   - `_detach_ram_from_gripper()`（第 101-111 行）
   - `_reset_ram_stick_to_holder()`（第 220-235 行）
   - 这些方法名和逻辑可以保留，仅更新内部实现

**修改建议**：
```python
# 选项 1：最小修改（推荐）
# 只更新注释和打印信息，保持方法名不变
class IsaacSimRAMEnvEnhanced(IsaacSimFrankaEnv):
    """
    Gear 组装任务的高保真度仿真环境
    （原为 RAM 插入任务，已迁移到 Gear 组装）
    """
    def __init__(self, ...):
        # 对象引用可以重命名，或保留原名称
        self.gear_medium = None  # 或 self.ram_stick = None（保留兼容性）
        self.gear_base = None    # 或 self.motherboard = None
        # ...

# 选项 2：完整重构
# 重命名所有 RAM 相关的变量和方法
class IsaacSimGearAssemblyEnv(IsaacSimFrankaEnv):
    # ...
```

**建议**：使用选项 1，最小化修改，保持代码结构稳定。

---

### 4.2 可能需要修改的文件

#### 4.2.1 `create_ram_scene_usd.py`
**影响程度**：🟢 **不需要修改**（如果使用新的 USD 文件）

**说明**：
- 此脚本用于创建 RAM 插入任务的 USD 场景
- 如果直接使用 `HIL_franka_gear.usda`，则不需要此脚本
- 如果需要程序化创建 gear 场景，可以创建新脚本 `create_gear_scene_usd.py`

#### 4.2.2 训练脚本
**影响程度**：🟢 **不需要修改**

**说明**：
- `run_actor.sh` 和 `run_learner.sh` 是通用的训练脚本
- 只需要确保 `config.py` 中的配置正确即可

---

### 4.3 不需要修改的文件

#### 4.3.1 基础环境类
- `serl_robot_infra/franka_env/envs/isaac_sim_env.py`
- `serl_robot_infra/franka_env/envs/franka_env.py`
- **原因**：这些是通用基类，不包含任务特定逻辑

#### 4.3.2 服务器核心功能
- `isaac_sim_server.py` 的核心功能（HTTP 接口、WebSocket、仿真循环）
- **原因**：服务器是通用的，通过参数配置适配不同任务

#### 4.3.3 训练框架
- `serl_launcher` 相关代码
- **原因**：训练框架是通用的，不依赖具体任务

---

## 五、文档影响分析

### 5.1 需要更新的文档

#### 5.1.1 `USD_SCENE_SETUP.md`
**影响程度**：🟡 **建议更新**

**需要更新的内容**：
- 任务对象描述（从 RAM 条、主板、支架改为 gear_medium、gear_base、gear_large）
- Prim 路径说明
- 对象位置和尺寸说明

**建议**：
- 创建新文档 `GEAR_ASSEMBLY_SCENE_SETUP.md`
- 或更新现有文档，添加 gear 组装任务的说明

#### 5.1.2 `ram_insertion_sim_setup_plan.md`
**影响程度**：🟡 **建议更新**

**需要更新的内容**：
- 任务描述
- 对象几何特征
- 物理参数（如果 gear 的参数与 RAM 不同）

**建议**：
- 创建新文档 `gear_assembly_sim_setup_plan.md`
- 或更新现有文档，添加 gear 组装任务的章节

#### 4.1.3 `isaac_sim_server_integration_test.md`
**影响程度**：🟡 **建议更新**

**需要更新的内容**：
- 测试步骤中的 prim 路径
- 对象验证清单

**建议**：
- 更新测试文档，添加 gear 组装任务的测试用例

---

## 六、迁移步骤建议

### 6.1 第一步：更新配置和路径
1. ✅ 修改 `config.py` 中的位姿配置（TARGET_POSE、GRASP_POSE 等）
2. ✅ **相机配置不需要修改**（USD 文件已正确命名，键名匹配）
3. ✅ **服务器启动参数不需要修改**（使用默认值即可）

### 6.2 第二步：测试服务器启动
1. ✅ 使用新的 USD 文件启动服务器
2. ✅ 验证机器人、相机、任务对象正确加载
3. ✅ 测试 HTTP 接口（`/getstate`、`/pose` 等）

### 5.3 第三步：更新任务特定代码
1. ✅ 更新 `wrapper.py` 中的提示文本
2. ✅ 更新 `isaac_sim_ram_env_enhanced.py` 中的注释和对象引用
3. ✅ 测试环境重置和重新抓取功能

### 6.4 第四步：验证训练流程
1. ✅ 测试环境初始化
2. ✅ 测试观察空间和动作空间
3. ✅ 测试奖励分类器（如果需要）
4. ✅ 运行一个简短的训练测试

### 5.5 第五步：更新文档
1. ✅ 更新或创建场景设置文档
2. ✅ 更新测试文档
3. ✅ 记录位姿配置和任务参数

---

## 七、关键注意事项

### 7.1 位姿配置
⚠️ **重要**：新任务的位姿配置必须根据实际测量值设置，不能直接使用 RAM 插入任务的位姿。

**获取位姿的方法**：
1. 在 Isaac Sim 中手动移动机器人到目标位置
2. 使用 `/getstate` 接口获取当前位姿
3. 记录 TARGET_POSE 和 GRASP_POSE

### 7.2 相机配置
✅ **已解决**：USD 文件中的相机已正确命名为 `wrist_1` 和 `wrist_2`，与配置完全匹配。

**相机类型说明**：
- **原项目（真机）**：使用 Intel RealSense 深度相机（如 D435），但**只使用 RGB 彩色流**，不使用深度信息
- **仿真环境**：使用 Isaac Sim 虚拟 RGB 相机，功能等效
- **配置命名**：`REALSENSE_CAMERAS` 命名来自硬件类型，但实际只使用 RGB 图像

**当前状态**：
- ✅ USD 文件相机路径：`/World/franka/panda_hand/wrist_1`、`/World/franka/panda_hand/wrist_2`
- ✅ 配置键名：`wrist_1`、`wrist_2`
- ✅ 完全匹配，**不需要任何修改**

### 7.3 任务对象管理
⚠️ **注意**：新 USD 文件中的任务对象（gear_medium、gear_base、gear_large）已经定义，不需要在代码中创建。

**当前实现**：
- `isaac_sim_ram_env_enhanced.py` 中的 `_add_task_objects()` 方法为空
- 任务对象由 USD 文件定义，服务器自动加载
- **不需要修改**，只需确保 USD 文件中的对象路径正确

### 7.4 物理属性
⚠️ **注意**：gear 对象的物理属性（质量、摩擦系数等）可能与 RAM 不同，需要根据实际情况调整。

**当前实现**：
- 物理属性在 USD 文件中定义
- 如果需要在运行时调整，可以通过服务器接口（如果实现了）

---

## 八、总结

### 7.1 影响程度评估

| 文件/模块 | 影响程度 | 必须修改 | 建议修改 | 不需要修改 |
|---------|---------|---------|---------|-----------|
| `isaac_sim_server.py` 启动参数 | 🟢 低 | | | ✅ |
| `config.py` 位姿配置 | 🔴 高 | ✅ | | |
| `config.py` 相机配置 | 🟢 低 | | | ✅ |
| `wrapper.py` | 🟡 中 | | ✅ | |
| `isaac_sim_ram_env_enhanced.py` | 🟡 中 | | ✅ | |
| 基础环境类 | 🟢 低 | | | ✅ |
| 服务器核心功能 | 🟢 低 | | | ✅ |
| 训练框架 | 🟢 低 | | | ✅ |

### 8.2 迁移难度评估

**总体难度**：🟡 **中等**

**主要原因**：
1. ✅ 核心架构通用，不需要修改
2. ✅ USD 文件路径已正确，相机命名已匹配（不需要修改）
3. ⚠️ 需要更新位姿配置（需要实际测量）
4. ⚠️ 需要更新任务特定代码和文档

### 8.3 建议的迁移策略

1. **最小化修改**：保持键名和代码结构不变，只更新配置和路径
2. **分步迁移**：先完成服务器启动和基本功能测试，再更新任务特定代码
3. **保持兼容**：如果可能，保留 RAM 插入任务的配置，通过参数切换

### 8.4 结论

**任务变更会影响项目代码，但影响范围非常有限**：

✅ **必须修改**：
- 位姿配置（TARGET_POSE、GRASP_POSE 等，需要根据实际测量值设置）

🟡 **建议修改**：
- 任务特定代码（wrapper.py、isaac_sim_ram_env_enhanced.py）中的提示文本和注释
- 文档更新

🟢 **不需要修改**：
- ✅ 服务器启动参数（USD 文件路径已正确匹配）
- ✅ 相机配置（USD 文件相机命名已匹配，键名一致）
- ✅ 基础环境类
- ✅ 服务器核心功能
- ✅ 训练框架

**总体评估**：迁移工作量**较小**，主要是位姿配置更新和少量代码调整。USD 文件已正确配置，相机路径和命名完全匹配，大大简化了迁移工作。

---

## 九、关于任务文件夹组织

### 9.1 是否可以将 Gear Assembly 任务单独放到一个文件夹？

✅ **可以，且推荐这样做**

**理由**：
1. **项目结构一致性**：查看 `examples/experiments/` 目录，每个任务都有自己的文件夹：
   - `ram_insertion/` - RAM 插入任务
   - `usb_pickup_insertion/` - USB 拾取插入任务
   - `object_handover/` - 物体交接任务
   - `egg_flip/` - 翻蛋任务

2. **代码组织清晰**：每个任务文件夹包含：
   - `config.py` - 任务配置
   - `wrapper.py` - 任务特定环境包装器
   - `run_actor.sh` - Actor 启动脚本
   - `run_learner.sh` - Learner 启动脚本
   - 其他任务特定文件（如 USD 文件、环境实现等）

3. **便于维护**：独立文件夹便于：
   - 管理任务特定的配置和代码
   - 避免不同任务之间的配置冲突
   - 便于版本控制和代码复用

### 9.2 建议的文件夹结构

```
examples/experiments/
├── gear_assembly/              # 新建文件夹
│   ├── __init__.py
│   ├── config.py               # Gear 组装任务配置
│   ├── wrapper.py              # Gear 组装任务包装器
│   ├── isaac_sim_gear_env.py  # Isaac Sim 环境实现（可选）
│   ├── HIL_franka_gear.usda   # USD 场景文件
│   ├── run_actor.sh
│   └── run_learner.sh
├── ram_insertion/              # 原有 RAM 插入任务
│   ├── config.py
│   ├── wrapper.py
│   ├── isaac_sim_ram_env_enhanced.py
│   ├── create_ram_scene_usd.py
│   └── ...
└── ...
```

### 9.3 迁移步骤建议

1. **创建新文件夹**：
   ```bash
   mkdir -p examples/experiments/gear_assembly
   ```

2. **复制和修改文件**：
   - 从 `ram_insertion/` 复制 `config.py`、`wrapper.py`、`run_actor.sh`、`run_learner.sh`
   - 修改配置中的位姿、任务名称等
   - 将 `HIL_franka_gear.usda` 移动到新文件夹

3. **更新导入路径**：
   - 如果 `config.py` 中有相对导入，需要更新路径
   - 确保 `TrainConfig.get_environment()` 中的导入路径正确

4. **测试**：
   - 测试环境初始化
   - 测试训练流程

**结论**：将 Gear Assembly 任务单独放到 `gear_assembly/` 文件夹是**推荐的做法**，符合项目结构，便于维护和管理。
