# Isaac Sim Server 联调测试文档

## 测试概述

本文档详细描述 Isaac Sim Server 的联调测试流程，包括：
1. 仿真场景搭建和 USD 文件创建
2. Isaac Sim 应用控制（启动、加载场景、运行仿真、重置环境、结束仿真、关闭应用）
3. Isaac Sim Server 对外接口测试

---

## 一、仿真场景搭建测试

### 测试目标
验证能够成功创建包含机器人、相机和任务对象的 USD 场景文件。

### 测试步骤

#### 步骤 1.1：准备测试环境

**准备条件**：
- ✅ Isaac Sim 已安装（推荐版本：2023.1.1 或更高）
- ✅ Python 环境已配置（Python 3.10）
- ✅ 已安装必要的依赖包：
  ```bash
  pip install numpy scipy absl-py
  ```
- ✅ 已激活 Isaac Sim 的 Python 环境或已配置 Isaac Sim Python 路径

**执行步骤**：
```bash
# 检查 Isaac Sim 安装路径
ls ~/.local/share/ov/pkg/isaac_sim-*/

# 检查 Python 环境
python --version  # 应该显示 Python 3.10

# 检查依赖包
python -c "import numpy, scipy, absl; print('Dependencies OK')"
```

**期望输出**：
- Isaac Sim 安装路径存在
- Python 版本为 3.10
- 依赖包导入成功，输出 "Dependencies OK"

---

#### 步骤 1.2：运行场景创建脚本

**准备条件**：
- ✅ 步骤 1.1 已完成
- ✅ 场景创建脚本路径：`examples/experiments/ram_insertion/create_ram_scene_usd.py`
- ✅ 输出目录有写入权限

**执行步骤**：
```bash
# 方法 1：使用 Isaac Sim 的 Python 环境（推荐）
cd /path/to/isaac-sim
./python.sh /path/to/hil-serl/examples/experiments/ram_insertion/create_ram_scene_usd.py \
    --output_path=/path/to/ram_insertion_scene.usd

# 方法 2：如果已配置 Isaac Sim Python 路径
cd /path/to/hil-serl
python examples/experiments/ram_insertion/create_ram_scene_usd.py \
    --output_path=./ram_insertion_scene.usd
```

**期望输出**：
```
[INFO] Adding Franka robot...
[INFO] Franka robot added at /World/franka
[INFO] Adding cameras...
[INFO] Camera 'wrist_1' added at /World/franka/panda_hand/wrist_1
[INFO] Camera 'wrist_2' added at /World/franka/panda_hand/wrist_2
[INFO] Creating RAM stick...
[INFO] RAM stick created at /World/ram_stick
[INFO] Creating motherboard slot...
[INFO] Motherboard slot created at /World/motherboard/slot
[INFO] Creating RAM holder...
[INFO] RAM holder created at /World/ram_holder
[INFO] Saving USD scene to: /path/to/ram_insertion_scene.usd
[INFO] USD scene saved successfully
```

**验证检查**：
- ✅ USD 文件已创建
- ✅ 文件大小 > 0（通常 > 1MB）
- ✅ 无错误信息输出

---

#### 步骤 1.3：验证 USD 文件内容

**准备条件**：
- ✅ 步骤 1.2 已完成，USD 文件已创建

**执行步骤**：
```bash
# 方法 1：使用 USD 工具检查（如果已安装）
usdcat /path/to/ram_insertion_scene.usd | head -50

# 方法 2：在 Isaac Sim 中打开并检查
# 启动 Isaac Sim GUI，打开 USD 文件，检查场景内容
```

**期望输出**：
- USD 文件可以正常打开
- 场景包含以下对象：
  - ✅ `/World/franka` - Franka 机器人
  - ✅ `/World/franka/panda_hand/wrist_1` - 相机 1
    - 位置：相对于 `panda_hand` 为 `[0, 0, 0.05]`（z 方向向上 5cm）
    - 分辨率：1280x720
  - ✅ `/World/franka/panda_hand/wrist_2` - 相机 2
    - 位置：相对于 `panda_hand` 为 `[0, 0.05, 0]`（y 方向向前 5cm）
    - 分辨率：1280x720
  - ✅ `/World/ram_stick` - RAM 条
  - ✅ `/World/motherboard/slot` - 主板插槽
  - ✅ `/World/ram_holder` - RAM 支架
  - ✅ `/World/defaultGroundPlane` - 地面平面

**验证检查**：
- ✅ 所有必需对象都存在
- ✅ 对象位置和尺寸正确
- ✅ 相机已正确附加到机器人末端执行器

---

## 二、Isaac Sim 应用控制测试

### 测试目标
验证能够通过代码控制 Isaac Sim 应用的完整生命周期：启动、加载场景、运行仿真、重置环境、结束仿真、关闭应用。

### 测试步骤

#### 步骤 2.1：测试 SimulationApp 启动

**准备条件**：
- ✅ Isaac Sim 已安装
- ✅ Python 环境已配置
- ✅ 已安装必要的依赖包

**执行步骤**：
创建测试脚本 `test_simulation_app.py`：
```python
from omni.isaac.kit import SimulationApp

# 测试无头模式启动
config = {"headless": True}
simulation_app = SimulationApp(config)

print("[INFO] SimulationApp started successfully")
print(f"[INFO] Is running: {simulation_app.is_running()}")

# 关闭应用
simulation_app.close()
print("[INFO] SimulationApp closed successfully")
```

运行测试：
```bash
# 使用 Isaac Sim Python 环境
cd /path/to/isaac-sim
./python.sh /path/to/test_simulation_app.py
```

**期望输出**：
```
[INFO] SimulationApp started successfully
[INFO] Is running: True
[INFO] SimulationApp closed successfully
```

**验证检查**：
- ✅ SimulationApp 启动成功
- ✅ `is_running()` 返回 `True`
- ✅ 应用可以正常关闭
- ✅ 无错误或警告信息

---

#### 步骤 2.2：测试 USD 场景加载

**准备条件**：
- ✅ 步骤 2.1 已完成
- ✅ USD 场景文件已创建（步骤 1.2）

**执行步骤**：
创建测试脚本 `test_usd_loading.py`：
```python
from omni.isaac.kit import SimulationApp
import omni.usd

# 启动 SimulationApp
config = {"headless": True}
simulation_app = SimulationApp(config)

# 加载 USD 场景
usd_path = "/path/to/ram_insertion_scene.usd"
print(f"[INFO] Loading USD scene from: {usd_path}")

try:
    usd_context = omni.usd.get_context()
    usd_context.open_stage(usd_path)
    print("[INFO] USD scene loaded successfully")
    
    # 验证场景内容
    from omni.isaac.core.utils.stage import get_current_stage
    from pxr import UsdGeom
    
    stage = get_current_stage()
    
    # 检查机器人
    robot_prim = stage.GetPrimAtPath("/World/franka")
    if robot_prim.IsValid():
        print("[INFO] Robot prim found at /World/franka")
    else:
        print("[ERROR] Robot prim not found!")
    
    # 检查相机
    camera1_prim = stage.GetPrimAtPath("/World/franka/panda_hand/wrist_1")
    camera2_prim = stage.GetPrimAtPath("/World/franka/panda_hand/wrist_2")
    if camera1_prim.IsValid() and camera2_prim.IsValid():
        print("[INFO] Cameras found")
    else:
        print("[ERROR] Cameras not found!")
    
    # 检查任务对象
    ram_prim = stage.GetPrimAtPath("/World/ram_stick")
    if ram_prim.IsValid():
        print("[INFO] RAM stick found")
    else:
        print("[ERROR] RAM stick not found!")
        
except Exception as e:
    print(f"[ERROR] Failed to load USD scene: {e}")
    raise

# 关闭应用
simulation_app.close()
print("[INFO] Test completed")
```

运行测试：
```bash
cd /path/to/isaac-sim
./python.sh /path/to/test_usd_loading.py
```

**期望输出**：
```
[INFO] Loading USD scene from: /path/to/ram_insertion_scene.usd
[INFO] USD scene loaded successfully
[INFO] Robot prim found at /World/franka
[INFO] Cameras found
[INFO] RAM stick found
[INFO] Test completed
```

**验证检查**：
- ✅ USD 场景加载成功
- ✅ 所有必需对象都存在于场景中
- ✅ 无错误信息

---

#### 步骤 2.3：测试 World 创建和物理引擎初始化

**准备条件**：
- ✅ 步骤 2.2 已完成

**执行步骤**：
创建测试脚本 `test_world.py`：
```python
from omni.isaac.kit import SimulationApp
import omni.usd
from omni.isaac.core import World

# 启动 SimulationApp
config = {"headless": True}
simulation_app = SimulationApp(config)

# 加载 USD 场景
usd_path = "/path/to/ram_insertion_scene.usd"
usd_context = omni.usd.get_context()
usd_context.open_stage(usd_path)

# 创建 World
print("[INFO] Creating World...")
world = World(stage_units_in_meters=1.0)
world.set_physics_dt(1.0 / 60.0)  # 60 Hz
print("[INFO] World created successfully")
print(f"[INFO] Physics DT: {world.get_physics_dt()}")

# 初始化世界
print("[INFO] Resetting world...")
world.reset()
print("[INFO] World reset successfully")

# 关闭应用
simulation_app.close()
print("[INFO] Test completed")
```

运行测试：
```bash
cd /path/to/isaac-sim
./python.sh /path/to/test_world.py
```

**期望输出**：
```
[INFO] Creating World...
[INFO] World created successfully
[INFO] Physics DT: 0.016666666666666666
[INFO] Resetting world...
[INFO] World reset successfully
[INFO] Test completed
```

**验证检查**：
- ✅ World 创建成功
- ✅ 物理引擎 DT 设置正确（1/60 ≈ 0.0167）
- ✅ 世界重置成功
- ✅ 无错误信息

---

#### 步骤 2.4：测试机器人对象获取

**准备条件**：
- ✅ 步骤 2.3 已完成

**执行步骤**：
创建测试脚本 `test_robot.py`：
```python
from omni.isaac.kit import SimulationApp
import omni.usd
from omni.isaac.core import World
from omni.isaac.franka import Franka

# 启动 SimulationApp
config = {"headless": True}
simulation_app = SimulationApp(config)

# 加载 USD 场景
usd_path = "/path/to/ram_insertion_scene.usd"
usd_context = omni.usd.get_context()
usd_context.open_stage(usd_path)

# 创建 World
world = World(stage_units_in_meters=1.0)
world.set_physics_dt(1.0 / 60.0)
world.reset()

# 获取机器人对象
robot_prim_path = "/World/franka"
print(f"[INFO] Getting robot from prim path: {robot_prim_path}")

try:
    # 方法 1：从 scene 获取
    franka = world.scene.get(robot_prim_path)
    
    if franka is None:
        # 方法 2：创建 Franka 对象包装
        print("[INFO] Robot not in scene, creating wrapper...")
        franka = world.scene.add(
            Franka(
                prim_path=robot_prim_path,
                name="franka",
            )
        )
    
    print(f"[INFO] Robot object obtained: {type(franka)}")
    
    # 测试获取关节位置
    joint_positions = franka.get_joint_positions()
    print(f"[INFO] Joint positions: {joint_positions}")
    print(f"[INFO] Number of joints: {len(joint_positions)}")
    
    # 测试获取末端执行器位姿
    from omni.isaac.core.utils.stage import get_current_stage
    from pxr import UsdGeom
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    
    stage = get_current_stage()
    ee_prim = stage.GetPrimAtPath("/World/franka/panda_hand")
    xform = UsdGeom.Xformable(ee_prim)
    world_transform = xform.ComputeLocalToWorldTransform(0)
    
    position = np.array(world_transform.ExtractTranslation())
    rotation_matrix = world_transform.ExtractRotationMatrix()
    rotation = R.from_matrix(rotation_matrix).as_quat()
    
    print(f"[INFO] End-effector position: {position}")
    print(f"[INFO] End-effector rotation (quat): {rotation}")
    
except Exception as e:
    print(f"[ERROR] Failed to get robot: {e}")
    raise

# 关闭应用
simulation_app.close()
print("[INFO] Test completed")
```

运行测试：
```bash
cd /path/to/isaac-sim
./python.sh /path/to/test_robot.py
```

**期望输出**：
```
[INFO] Getting robot from prim path: /World/franka
[INFO] Robot object obtained: <class 'omni.isaac.franka.franka.Franka'>
[INFO] Joint positions: [0. 0. 0. 0. 0. 0. 0.]
[INFO] Number of joints: 7
[INFO] End-effector position: [0.58812412 -0.0357859  0.27843494]
[INFO] End-effector rotation (quat): [0. 0. 1. 0.]
[INFO] Test completed
```

**验证检查**：
- ✅ 机器人对象获取成功
- ✅ 关节数量为 7（Franka 有 7 个关节）
- ✅ 可以获取关节位置
- ✅ 可以获取末端执行器位姿
- ✅ 无错误信息

---

#### 步骤 2.5：测试相机对象获取

**准备条件**：
- ✅ 步骤 2.4 已完成

**执行步骤**：
创建测试脚本 `test_cameras.py`：
```python
from omni.isaac.kit import SimulationApp
import omni.usd
from omni.isaac.core import World
from omni.isaac.sensor import Camera

# 启动 SimulationApp
config = {"headless": True}
simulation_app = SimulationApp(config)

# 加载 USD 场景
usd_path = "/path/to/ram_insertion_scene.usd"
usd_context = omni.usd.get_context()
usd_context.open_stage(usd_path)

# 创建 World
world = World(stage_units_in_meters=1.0)
world.set_physics_dt(1.0 / 60.0)
world.reset()

# 获取相机对象
camera_prim_paths = [
    "/World/franka/panda_hand/wrist_1",
    "/World/franka/panda_hand/wrist_2"
]

cameras = {}
for cam_prim_path in camera_prim_paths:
    cam_name = cam_prim_path.split("/")[-1]
    print(f"[INFO] Getting camera '{cam_name}' from prim path: {cam_prim_path}")
    
    try:
        camera = world.scene.get(cam_prim_path)
        
        if camera is None:
            print(f"[INFO] Camera not in scene, creating wrapper...")
            camera = world.scene.add(
                Camera(
                    prim_path=cam_prim_path,
                    name=cam_name,
                )
            )
        
        cameras[cam_name] = camera
        print(f"[INFO] Camera '{cam_name}' obtained: {type(camera)}")
        
        # 验证相机位置（相对于 panda_hand）
        from omni.isaac.core.utils.stage import get_current_stage
        from pxr import UsdGeom
        import numpy as np
        
        stage = get_current_stage()
        cam_prim = stage.GetPrimAtPath(cam_prim_path)
        if cam_prim.IsValid():
            xform = UsdGeom.Xformable(cam_prim)
            # 获取相对于父节点（panda_hand）的局部变换
            local_transform = xform.GetLocalTransformation()
            local_pos = np.array(local_transform.ExtractTranslation())
            print(f"[INFO] Camera '{cam_name}' local position (relative to panda_hand): {local_pos}")
            
            # 验证位置是否符合预期
            expected_positions = {
                "wrist_1": np.array([0, 0, 0.05]),  # z 方向向上 5cm
                "wrist_2": np.array([0, 0.05, 0]),  # y 方向向前 5cm
            }
            if cam_name in expected_positions:
                expected_pos = expected_positions[cam_name]
                pos_diff = np.linalg.norm(local_pos - expected_pos)
                if pos_diff < 0.01:  # 允许 1cm 误差
                    print(f"[INFO] Camera '{cam_name}' position is correct (diff: {pos_diff:.4f} m)")
                else:
                    print(f"[WARNING] Camera '{cam_name}' position may be incorrect (diff: {pos_diff:.4f} m)")
        
        # 测试获取图像
        world.step(render=False)  # 需要步进一次才能获取图像
        rgb = camera.get_rgba()[:, :, :3]
        print(f"[INFO] Camera '{cam_name}' image shape: {rgb.shape}")
        print(f"[INFO] Camera '{cam_name}' image dtype: {rgb.dtype}")
        
    except Exception as e:
        print(f"[ERROR] Failed to get camera '{cam_name}': {e}")
        raise

print(f"[INFO] Successfully loaded {len(cameras)} cameras: {list(cameras.keys())}")

# 关闭应用
simulation_app.close()
print("[INFO] Test completed")
```

运行测试：
```bash
cd /path/to/isaac-sim
./python.sh /path/to/test_cameras.py
```

**期望输出**：
```
[INFO] Getting camera 'wrist_1' from prim path: /World/franka/panda_hand/wrist_1
[INFO] Camera 'wrist_1' obtained: <class 'omni.isaac.sensor.camera.Camera'>
[INFO] Camera 'wrist_1' image shape: (720, 1280, 3)
[INFO] Camera 'wrist_1' image dtype: uint8
[INFO] Getting camera 'wrist_2' from prim path: /World/franka/panda_hand/wrist_2
[INFO] Camera 'wrist_2' obtained: <class 'omni.isaac.sensor.camera.Camera'>
[INFO] Camera 'wrist_2' image shape: (720, 1280, 3)
[INFO] Camera 'wrist_2' image dtype: uint8
[INFO] Successfully loaded 2 cameras: ['wrist_1', 'wrist_2']
[INFO] Test completed
```

**验证检查**：
- ✅ 所有相机对象获取成功
- ✅ 相机位置正确：
  - `wrist_1`: 相对于 `panda_hand` 为 `[0, 0, 0.05]`（z 方向向上 5cm）
  - `wrist_2`: 相对于 `panda_hand` 为 `[0, 0.05, 0]`（y 方向向前 5cm）
- ✅ 可以获取相机图像
- ✅ 图像尺寸正确（1280x720）
- ✅ 图像数据类型为 uint8
- ✅ 无错误信息

---

#### 步骤 2.6：测试仿真循环运行

**准备条件**：
- ✅ 步骤 2.5 已完成

**执行步骤**：
创建测试脚本 `test_simulation_loop.py`：
```python
from omni.isaac.kit import SimulationApp
import omni.usd
from omni.isaac.core import World
import time
import numpy as np

# 启动 SimulationApp
config = {"headless": True}
simulation_app = SimulationApp(config)

# 加载 USD 场景
usd_path = "/path/to/ram_insertion_scene.usd"
usd_context = omni.usd.get_context()
usd_context.open_stage(usd_path)

# 创建 World
world = World(stage_units_in_meters=1.0)
sim_hz = 60.0
world.set_physics_dt(1.0 / sim_hz)
world.reset()

# 获取机器人
from omni.isaac.franka import Franka
franka = world.scene.get("/World/franka")
if franka is None:
    franka = world.scene.add(Franka(prim_path="/World/franka", name="franka"))

print("[INFO] Starting simulation loop...")
print(f"[INFO] Simulation frequency: {sim_hz} Hz")
print(f"[INFO] Physics DT: {1.0 / sim_hz} s")

# 运行仿真循环（10 秒）
num_steps = int(10 * sim_hz)  # 10 秒
start_time = time.time()

for i in range(num_steps):
    world.step(render=False)
    
    # 每 60 步打印一次状态（约每秒一次）
    if i % 60 == 0:
        joint_positions = franka.get_joint_positions()
        print(f"[INFO] Step {i}: Joint positions: {joint_positions[:3]}...")  # 只打印前3个

elapsed_time = time.time() - start_time
expected_time = num_steps / sim_hz
print(f"[INFO] Simulation completed")
print(f"[INFO] Steps executed: {num_steps}")
print(f"[INFO] Elapsed time: {elapsed_time:.2f} s")
print(f"[INFO] Expected time: {expected_time:.2f} s")
print(f"[INFO] Average step time: {elapsed_time / num_steps * 1000:.2f} ms")

# 关闭应用
simulation_app.close()
print("[INFO] Test completed")
```

运行测试：
```bash
cd /path/to/isaac-sim
./python.sh /path/to/test_simulation_loop.py
```

**期望输出**：
```
[INFO] Starting simulation loop...
[INFO] Simulation frequency: 60.0 Hz
[INFO] Physics DT: 0.016666666666666666 s
[INFO] Step 0: Joint positions: [0. 0. 0.]...
[INFO] Step 60: Joint positions: [0. 0. 0.]...
[INFO] Step 120: Joint positions: [0. 0. 0.]...
...
[INFO] Simulation completed
[INFO] Steps executed: 600
[INFO] Elapsed time: 10.02 s
[INFO] Expected time: 10.00 s
[INFO] Average step time: 16.70 ms
[INFO] Test completed
```

**验证检查**：
- ✅ 仿真循环运行正常
- ✅ 步进频率接近 60 Hz（平均步进时间约 16.7 ms）
- ✅ 可以持续运行多步
- ✅ 无错误信息

---

#### 步骤 2.7：测试环境重置

**准备条件**：
- ✅ 步骤 2.6 已完成

**执行步骤**：
创建测试脚本 `test_reset.py`：
```python
from omni.isaac.kit import SimulationApp
import omni.usd
from omni.isaac.core import World
from omni.isaac.franka import Franka
import numpy as np

# 启动 SimulationApp
config = {"headless": True}
simulation_app = SimulationApp(config)

# 加载 USD 场景
usd_path = "/path/to/ram_insertion_scene.usd"
usd_context = omni.usd.get_context()
usd_context.open_stage(usd_path)

# 创建 World
world = World(stage_units_in_meters=1.0)
world.set_physics_dt(1.0 / 60.0)

# 获取机器人
franka = world.scene.get("/World/franka")
if franka is None:
    franka = world.scene.add(Franka(prim_path="/World/franka", name="franka"))

# 初始重置
print("[INFO] Initial reset...")
world.reset()
initial_joint_pos = franka.get_joint_positions().copy()
print(f"[INFO] Initial joint positions: {initial_joint_pos}")

# 运行几步仿真
print("[INFO] Running simulation for 100 steps...")
for i in range(100):
    world.step(render=False)

# 检查关节位置是否改变
after_sim_joint_pos = franka.get_joint_positions()
print(f"[INFO] Joint positions after simulation: {after_sim_joint_pos}")
print(f"[INFO] Joint positions changed: {not np.allclose(initial_joint_pos, after_sim_joint_pos)}")

# 重置环境
print("[INFO] Resetting environment...")
world.reset()
reset_joint_pos = franka.get_joint_positions()
print(f"[INFO] Joint positions after reset: {reset_joint_pos}")

# 验证重置是否成功
if np.allclose(initial_joint_pos, reset_joint_pos, atol=1e-3):
    print("[INFO] Reset successful: Joint positions returned to initial state")
else:
    print("[WARNING] Reset may not have fully restored initial state")
    print(f"[INFO] Difference: {np.abs(initial_joint_pos - reset_joint_pos)}")

# 关闭应用
simulation_app.close()
print("[INFO] Test completed")
```

运行测试：
```bash
cd /path/to/isaac-sim
./python.sh /path/to/test_reset.py
```

**期望输出**：
```
[INFO] Initial reset...
[INFO] Initial joint positions: [0. 0. 0. 0. 0. 0. 0.]
[INFO] Running simulation for 100 steps...
[INFO] Joint positions after simulation: [0.001 0.002 ...]
[INFO] Joint positions changed: True
[INFO] Resetting environment...
[INFO] Joint positions after reset: [0. 0. 0. 0. 0. 0. 0.]
[INFO] Reset successful: Joint positions returned to initial state
[INFO] Test completed
```

**验证检查**：
- ✅ 初始重置成功
- ✅ 仿真运行后关节位置改变
- ✅ 重置后关节位置恢复到初始状态
- ✅ 无错误信息

---

## 三、Isaac Sim Server 对外接口测试

### 测试目标
验证 Isaac Sim Server 的所有 HTTP 接口和 WebSocket 接口正常工作。

### 测试步骤

#### 步骤 3.1：启动 Isaac Sim Server

**准备条件**：
- ✅ USD 场景文件已创建（步骤 1.2）
- ✅ 所有依赖包已安装：
  ```bash
  pip install flask flask-socketio requests numpy scipy opencv-python
  ```
- ✅ 服务器代码路径：`serl_robot_infra/robot_servers/isaac_sim_server.py`

**执行步骤**：
```bash
cd /path/to/hil-serl

# 启动服务器（无头模式）
python serl_robot_infra/robot_servers/isaac_sim_server.py \
    --flask_url=127.0.0.1 \
    --flask_port=5001 \
    --headless=True \
    --sim_width=1280 \
    --sim_height=720 \
    --sim_hz=60.0 \
    --usd_path=/path/to/ram_insertion_scene.usd \
    --robot_prim_path=/World/franka \
    --camera_prim_paths=/World/franka/panda_hand/wrist_1,/World/franka/panda_hand/wrist_2
```

**期望输出**：
```
[INFO] Loading USD scene from: /path/to/ram_insertion_scene.usd
[INFO] USD scene loaded successfully
[INFO] Getting robot from prim path: /World/franka
[INFO] Robot found at /World/franka
[INFO] Getting cameras from prim paths: ['/World/franka/panda_hand/wrist_1', '/World/franka/panda_hand/wrist_2']
[INFO] Camera 'wrist_1' found at /World/franka/panda_hand/wrist_1
[INFO] Camera 'wrist_2' found at /World/franka/panda_hand/wrist_2
[INFO] Successfully loaded 2 cameras: ['wrist_1', 'wrist_2']
[INFO] Isaac Sim Server initialized
[INFO] Starting Flask server on http://127.0.0.1:5001
 * Running on http://127.0.0.1:5001
```

**验证检查**：
- ✅ 服务器启动成功
- ✅ USD 场景加载成功
- ✅ 机器人和相机对象获取成功
- ✅ Flask 服务器监听在指定端口
- ✅ 无错误信息

---

#### 步骤 3.2：测试健康检查接口

**准备条件**：
- ✅ 步骤 3.1 已完成，服务器正在运行

**执行步骤**：
```bash
# 测试健康检查接口
curl http://127.0.0.1:5001/health
```

**期望输出**：
```json
{
  "status": "healthy",
  "simulation_running": true,
  "num_cameras": 2
}
```

**验证检查**：
- ✅ 返回状态码 200
- ✅ 返回 JSON 格式数据
- ✅ `status` 字段为 "healthy"
- ✅ `simulation_running` 为 `true`

---

#### 步骤 3.3：测试状态查询接口

**准备条件**：
- ✅ 步骤 3.2 已完成

**执行步骤**：
```bash
# 测试获取所有状态
curl -X POST http://127.0.0.1:5001/getstate

# 测试获取位姿
curl -X POST http://127.0.0.1:5001/getpos

# 测试获取速度
curl -X POST http://127.0.0.1:5001/getvel

# 测试获取力
curl -X POST http://127.0.0.1:5001/getforce

# 测试获取力矩
curl -X POST http://127.0.0.1:5001/gettorque

# 测试获取关节位置
curl -X POST http://127.0.0.1:5001/getq

# 测试获取关节速度
curl -X POST http://127.0.0.1:5001/getdq

# 测试获取雅可比矩阵
curl -X POST http://127.0.0.1:5001/getjacobian

# 测试获取夹爪位置
curl -X POST http://127.0.0.1:5001/get_gripper
```

**期望输出**：

`/getstate` 响应示例：
```json
{
  "pose": [0.588, -0.036, 0.278, 0.0, 0.0, 1.0, 0.0],
  "vel": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "force": [0.0, 0.0, 0.0],
  "torque": [0.0, 0.0, 0.0],
  "q": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "dq": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "jacobian": [[...], [...], ...],
  "gripper_pos": 1.0
}
```

`/getpos` 响应示例：
```json
{
  "pose": [0.588, -0.036, 0.278, 0.0, 0.0, 1.0, 0.0]
}
```

**验证检查**：
- ✅ 所有接口返回状态码 200
- ✅ 返回 JSON 格式数据
- ✅ 数据格式正确：
  - `pose`: 7 维数组（xyz + quat）
  - `vel`: 6 维数组
  - `force`: 3 维数组
  - `torque`: 3 维数组
  - `q`: 7 维数组（7 个关节）
  - `dq`: 7 维数组
  - `jacobian`: 6x7 矩阵
  - `gripper_pos`: 浮点数（0.0-1.0）

---

#### 步骤 3.4：测试控制命令接口

**准备条件**：
- ✅ 步骤 3.3 已完成

**执行步骤**：
```bash
# 测试设置位姿
curl -X POST http://127.0.0.1:5001/pose \
    -H "Content-Type: application/json" \
    -d '{"arr": [0.588, -0.036, 0.278, 0.0, 0.0, 1.0, 0.0]}'

# 测试关闭夹爪
curl -X POST http://127.0.0.1:5001/close_gripper

# 测试打开夹爪
curl -X POST http://127.0.0.1:5001/open_gripper

# 测试移动夹爪到指定位置
curl -X POST http://127.0.0.1:5001/move_gripper \
    -H "Content-Type: application/json" \
    -d '{"gripper_pos": 0.5}'

# 测试清除错误（占位符）
curl -X POST http://127.0.0.1:5001/clearerr

# 测试更新参数（占位符）
curl -X POST http://127.0.0.1:5001/update_param \
    -H "Content-Type: application/json" \
    -d '{"param": "value"}'

# 测试关节复位（可选）
curl -X POST http://127.0.0.1:5001/jointreset

# 测试场景重置
curl -X POST http://127.0.0.1:5001/reset_scene
```

**期望输出**：

`/pose` 响应：
```
Moved
```

`/close_gripper` 响应：
```
Closed
```

`/open_gripper` 响应：
```
Opened
```

`/move_gripper` 响应：
```
Moved Gripper
```

`/reset_scene` 响应：
```json
{
  "status": "success",
  "message": "Scene reset completed"
}
```

**验证检查**：
- ✅ 所有接口返回状态码 200
- ✅ 控制命令执行成功
- ✅ 可以通过 `/getstate` 验证状态变化：
  - 位姿命令后，`pose` 值改变
  - 夹爪命令后，`gripper_pos` 值改变
  - 场景重置后，所有状态恢复到初始值

---

#### 步骤 3.5：测试位姿控制功能

**准备条件**：
- ✅ 步骤 3.4 已完成

**执行步骤**：
创建测试脚本 `test_pose_control.py`：
```python
import requests
import time
import numpy as np

url = "http://127.0.0.1:5001"

# 获取初始位姿
print("[INFO] Getting initial pose...")
response = requests.post(url + "/getpos")
initial_pose = np.array(response.json()["pose"])
print(f"[INFO] Initial pose: {initial_pose}")

# 设置新位姿（稍微移动）
target_pose = initial_pose.copy()
target_pose[0] += 0.01  # x 方向移动 1cm
target_pose[1] += 0.01  # y 方向移动 1cm

print(f"[INFO] Setting target pose: {target_pose}")
response = requests.post(
    url + "/pose",
    json={"arr": target_pose.tolist()}
)
print(f"[INFO] Response: {response.text}")

# 等待机器人移动
print("[INFO] Waiting for robot to move...")
time.sleep(2.0)

# 获取新位姿
print("[INFO] Getting new pose...")
response = requests.post(url + "/getpos")
new_pose = np.array(response.json()["pose"])
print(f"[INFO] New pose: {new_pose}")

# 验证位姿是否改变
pose_diff = np.linalg.norm(new_pose[:3] - initial_pose[:3])
print(f"[INFO] Position difference: {pose_diff:.4f} m")

if pose_diff > 0.005:  # 至少移动 5mm
    print("[INFO] Pose control test PASSED")
else:
    print("[WARNING] Pose control may not be working correctly")
```

运行测试：
```bash
python test_pose_control.py
```

**期望输出**：
```
[INFO] Getting initial pose...
[INFO] Initial pose: [0.588 -0.036  0.278  0.    0.    1.    0.   ]
[INFO] Setting target pose: [0.598 -0.026  0.278  0.    0.    1.    0.   ]
[INFO] Response: Moved
[INFO] Waiting for robot to move...
[INFO] Getting new pose...
[INFO] New pose: [0.597 -0.027  0.278  0.    0.    1.    0.   ]
[INFO] Position difference: 0.0141 m
[INFO] Pose control test PASSED
```

**验证检查**：
- ✅ 位姿命令发送成功
- ✅ 机器人移动到目标位姿
- ✅ 位置变化符合预期（误差 < 1cm）
- ✅ 无错误信息

---

#### 步骤 3.6：测试夹爪控制功能

**准备条件**：
- ✅ 步骤 3.5 已完成

**执行步骤**：
创建测试脚本 `test_gripper_control.py`：
```python
import requests
import time

url = "http://127.0.0.1:5001"

# 获取初始夹爪位置
print("[INFO] Getting initial gripper position...")
response = requests.post(url + "/get_gripper")
initial_gripper = response.json()["gripper"]
print(f"[INFO] Initial gripper position: {initial_gripper}")

# 关闭夹爪
print("[INFO] Closing gripper...")
response = requests.post(url + "/close_gripper")
print(f"[INFO] Response: {response.text}")

time.sleep(1.0)

# 获取夹爪位置
response = requests.post(url + "/get_gripper")
closed_gripper = response.json()["gripper"]
print(f"[INFO] Gripper position after close: {closed_gripper}")

# 打开夹爪
print("[INFO] Opening gripper...")
response = requests.post(url + "/open_gripper")
print(f"[INFO] Response: {response.text}")

time.sleep(1.0)

# 获取夹爪位置
response = requests.post(url + "/get_gripper")
opened_gripper = response.json()["gripper"]
print(f"[INFO] Gripper position after open: {opened_gripper}")

# 验证
if closed_gripper < 0.2 and opened_gripper > 0.8:
    print("[INFO] Gripper control test PASSED")
else:
    print("[WARNING] Gripper control may not be working correctly")
    print(f"[INFO] Expected: closed < 0.2, opened > 0.8")
    print(f"[INFO] Actual: closed = {closed_gripper}, opened = {opened_gripper}")
```

运行测试：
```bash
python test_gripper_control.py
```

**期望输出**：
```
[INFO] Getting initial gripper position...
[INFO] Initial gripper position: 1.0
[INFO] Closing gripper...
[INFO] Response: Closed
[INFO] Gripper position after close: 0.0
[INFO] Opening gripper...
[INFO] Response: Opened
[INFO] Gripper position after open: 1.0
[INFO] Gripper control test PASSED
```

**验证检查**：
- ✅ 夹爪关闭命令成功
- ✅ 夹爪打开命令成功
- ✅ 夹爪位置值正确（关闭 < 0.2，打开 > 0.8）
- ✅ 无错误信息

---

#### 步骤 3.7：测试场景重置接口

**准备条件**：
- ✅ 步骤 3.6 已完成

**执行步骤**：
创建测试脚本 `test_reset_scene.py`：
```python
import requests
import time
import numpy as np

url = "http://127.0.0.1:5001"

# 获取初始状态
print("[INFO] Getting initial state...")
response = requests.post(url + "/getstate")
initial_state = response.json()
initial_pose = np.array(initial_state["pose"])
initial_gripper = initial_state["gripper_pos"]
print(f"[INFO] Initial pose: {initial_pose[:3]}")
print(f"[INFO] Initial gripper: {initial_gripper}")

# 改变状态
print("[INFO] Changing robot state...")
target_pose = initial_pose.copy()
target_pose[0] += 0.05
requests.post(url + "/pose", json={"arr": target_pose.tolist()})
requests.post(url + "/close_gripper")

time.sleep(2.0)

# 获取改变后的状态
response = requests.post(url + "/getstate")
changed_state = response.json()
changed_pose = np.array(changed_state["pose"])
changed_gripper = changed_state["gripper_pos"]
print(f"[INFO] Changed pose: {changed_pose[:3]}")
print(f"[INFO] Changed gripper: {changed_gripper}")

# 重置场景
print("[INFO] Resetting scene...")
response = requests.post(url + "/reset_scene")
print(f"[INFO] Response: {response.json()}")

time.sleep(2.0)

# 获取重置后的状态
response = requests.post(url + "/getstate")
reset_state = response.json()
reset_pose = np.array(reset_state["pose"])
reset_gripper = reset_state["gripper_pos"]
print(f"[INFO] Reset pose: {reset_pose[:3]}")
print(f"[INFO] Reset gripper: {reset_gripper}")

# 验证重置
pose_diff = np.linalg.norm(reset_pose[:3] - initial_pose[:3])
gripper_diff = abs(reset_gripper - initial_gripper)

print(f"[INFO] Pose difference: {pose_diff:.4f} m")
print(f"[INFO] Gripper difference: {gripper_diff:.4f}")

if pose_diff < 0.01 and gripper_diff < 0.1:
    print("[INFO] Scene reset test PASSED")
else:
    print("[WARNING] Scene reset may not have fully restored initial state")
```

运行测试：
```bash
python test_reset_scene.py
```

**期望输出**：
```
[INFO] Getting initial state...
[INFO] Initial pose: [0.588 -0.036  0.278]
[INFO] Initial gripper: 1.0
[INFO] Changing robot state...
[INFO] Changed pose: [0.638 -0.036  0.278]
[INFO] Changed gripper: 0.0
[INFO] Resetting scene...
[INFO] Response: {'status': 'success', 'message': 'Scene reset completed'}
[INFO] Reset pose: [0.588 -0.036  0.278]
[INFO] Reset gripper: 1.0
[INFO] Pose difference: 0.0000 m
[INFO] Gripper difference: 0.0000
[INFO] Scene reset test PASSED
```

**验证检查**：
- ✅ 场景重置命令成功
- ✅ 位姿恢复到初始值（误差 < 1cm）
- ✅ 夹爪恢复到初始状态
- ✅ 无错误信息

---

#### 步骤 3.8：测试 WebSocket 图像传输（如果支持）

**准备条件**：
- ✅ 步骤 3.7 已完成
- ✅ `flask-socketio` 已安装
- ✅ `python-socketio` 已安装

**执行步骤**：
创建测试脚本 `test_websocket_images.py`：
```python
import socketio
import cv2
import numpy as np
import time

# 创建 SocketIO 客户端
sio = socketio.Client()

# 连接事件
@sio.on('connect')
def on_connect():
    print("[INFO] WebSocket connected")

@sio.on('disconnect')
def on_disconnect():
    print("[INFO] WebSocket disconnected")

@sio.on('image')
def on_image(data):
    """接收图像数据"""
    # data 应该是字典：{"camera_key": base64_encoded_image, ...}
    print(f"[INFO] Received image data from {len(data)} cameras")
    for cam_key, img_data in data.items():
        print(f"[INFO] Camera '{cam_key}': {len(img_data)} bytes")

# 连接到服务器
url = "http://127.0.0.1:5001"
print(f"[INFO] Connecting to {url}...")
try:
    sio.connect(url)
    print("[INFO] Connected successfully")
    
    # 等待接收图像（10 秒）
    print("[INFO] Waiting for images (10 seconds)...")
    time.sleep(10)
    
    # 断开连接
    sio.disconnect()
    print("[INFO] Disconnected")
    
except Exception as e:
    print(f"[ERROR] Failed to connect: {e}")
```

运行测试：
```bash
pip install python-socketio
python test_websocket_images.py
```

**期望输出**：
```
[INFO] Connecting to http://127.0.0.1:5001...
[INFO] Connected successfully
[INFO] WebSocket connected
[INFO] Received image data from 2 cameras
[INFO] Camera 'wrist_1': 45678 bytes
[INFO] Camera 'wrist_2': 45231 bytes
[INFO] Received image data from 2 cameras
[INFO] Camera 'wrist_1': 45678 bytes
[INFO] Camera 'wrist_2': 45231 bytes
...
[INFO] Disconnected
[INFO] WebSocket disconnected
```

**验证检查**：
- ✅ WebSocket 连接成功
- ✅ 能够接收图像数据
- ✅ 图像数据来自所有配置的相机
- ✅ 图像数据大小合理（JPEG 压缩后约 40-50KB）
- ✅ 无错误信息

---

#### 步骤 3.9：测试接口性能和延迟

**准备条件**：
- ✅ 步骤 3.8 已完成

**执行步骤**：
创建测试脚本 `test_latency.py`：
```python
import requests
import time
import statistics

url = "http://127.0.0.1:5001"

# 测试状态查询延迟
print("[INFO] Testing state query latency...")
latencies = []
for i in range(100):
    start = time.time()
    response = requests.post(url + "/getstate", timeout=1.0)
    latency = (time.time() - start) * 1000  # 转换为毫秒
    latencies.append(latency)
    if i % 20 == 0:
        print(f"[INFO] Request {i}: {latency:.2f} ms")

avg_latency = statistics.mean(latencies)
median_latency = statistics.median(latencies)
min_latency = min(latencies)
max_latency = max(latencies)

print(f"\n[INFO] State query latency statistics:")
print(f"[INFO]   Average: {avg_latency:.2f} ms")
print(f"[INFO]   Median: {median_latency:.2f} ms")
print(f"[INFO]   Min: {min_latency:.2f} ms")
print(f"[INFO]   Max: {max_latency:.2f} ms")

# 目标：平均延迟 < 50ms
if avg_latency < 50:
    print("[INFO] Latency test PASSED")
else:
    print(f"[WARNING] Latency may be too high: {avg_latency:.2f} ms")
```

运行测试：
```bash
python test_latency.py
```

**期望输出**：
```
[INFO] Testing state query latency...
[INFO] Request 0: 12.34 ms
[INFO] Request 20: 11.89 ms
[INFO] Request 40: 12.56 ms
[INFO] Request 60: 11.23 ms
[INFO] Request 80: 12.78 ms

[INFO] State query latency statistics:
[INFO]   Average: 12.15 ms
[INFO]   Median: 12.01 ms
[INFO]   Min: 10.23 ms
[INFO]   Max: 15.67 ms
[INFO] Latency test PASSED
```

**验证检查**：
- ✅ 平均延迟 < 50ms（本地网络）
- ✅ 延迟稳定（标准差小）
- ✅ 无超时错误

---

#### 步骤 3.10：测试错误处理

**准备条件**：
- ✅ 步骤 3.9 已完成

**执行步骤**：
```bash
# 测试无效的位姿命令（缺少参数）
curl -X POST http://127.0.0.1:5001/pose \
    -H "Content-Type: application/json" \
    -d '{}'

# 测试无效的位姿命令（错误的数组长度）
curl -X POST http://127.0.0.1:5001/pose \
    -H "Content-Type: application/json" \
    -d '{"arr": [1, 2, 3]}'

# 测试不存在的接口
curl -X POST http://127.0.0.1:5001/nonexistent
```

**期望输出**：

无效位姿命令响应（状态码 400 或 500）：
```json
{
  "error": "Invalid request format"
}
```

不存在接口响应（状态码 404）：
```
404 Not Found
```

**验证检查**：
- ✅ 错误请求返回适当的错误状态码
- ✅ 错误信息清晰
- ✅ 服务器不会崩溃
- ✅ 服务器继续正常运行

---

#### 步骤 3.11：测试服务器关闭

**准备条件**：
- ✅ 所有接口测试已完成

**执行步骤**：
```bash
# 在运行服务器的终端按 Ctrl+C
# 或发送 SIGTERM 信号
kill <server_pid>
```

**期望输出**：
```
^C
[INFO] Shutting down...
[INFO] Stopping simulation loop...
[INFO] Closing Isaac Sim Server...
[INFO] SimulationApp closed
```

**验证检查**：
- ✅ 服务器优雅关闭
- ✅ 仿真循环停止
- ✅ Isaac Sim 应用关闭
- ✅ 资源正确释放
- ✅ 无错误信息

---

## 四、测试总结

### 测试结果记录

| 测试阶段 | 测试项 | 状态 | 备注 |
|---------|--------|------|------|
| 一、场景搭建 | 1.1 准备测试环境 | ⬜ | |
| | 1.2 运行场景创建脚本 | ⬜ | |
| | 1.3 验证 USD 文件内容 | ⬜ | |
| 二、应用控制 | 2.1 测试 SimulationApp 启动 | ⬜ | |
| | 2.2 测试 USD 场景加载 | ⬜ | |
| | 2.3 测试 World 创建 | ⬜ | |
| | 2.4 测试机器人对象获取 | ⬜ | |
| | 2.5 测试相机对象获取 | ⬜ | |
| | 2.6 测试仿真循环运行 | ⬜ | |
| | 2.7 测试环境重置 | ⬜ | |
| 三、接口测试 | 3.1 启动 Isaac Sim Server | ⬜ | |
| | 3.2 测试健康检查接口 | ⬜ | |
| | 3.3 测试状态查询接口 | ⬜ | |
| | 3.4 测试控制命令接口 | ⬜ | |
| | 3.5 测试位姿控制功能 | ⬜ | |
| | 3.6 测试夹爪控制功能 | ⬜ | |
| | 3.7 测试场景重置接口 | ⬜ | |
| | 3.8 测试 WebSocket 图像传输 | ⬜ | |
| | 3.9 测试接口性能和延迟 | ⬜ | |
| | 3.10 测试错误处理 | ⬜ | |
| | 3.11 测试服务器关闭 | ⬜ | |

### 已知问题和解决方案

记录测试过程中发现的问题和解决方案：

1. **问题**：
   - **解决方案**：

2. **问题**：
   - **解决方案**：

---

## 五、下一步测试计划

完成 Isaac Sim Server 测试后，下一步测试：

1. **Gym 环境接口测试**
   - 测试 `IsaacSimFrankaEnv` 的 `reset()` 和 `step()` 方法
   - 测试观察空间和动作空间
   - 测试奖励计算

2. **完整训练流程测试**
   - 测试数据收集
   - 测试策略训练
   - 测试 Actor-Learner 通信

---

## 附录：测试脚本汇总

所有测试脚本应保存在 `docs/isaac sim虚拟环境改造/test_scripts/` 目录下：

- `test_simulation_app.py` - SimulationApp 启动测试
- `test_usd_loading.py` - USD 场景加载测试
- `test_world.py` - World 创建测试
- `test_robot.py` - 机器人对象获取测试
- `test_cameras.py` - 相机对象获取测试
- `test_simulation_loop.py` - 仿真循环测试
- `test_reset.py` - 环境重置测试
- `test_pose_control.py` - 位姿控制测试
- `test_gripper_control.py` - 夹爪控制测试
- `test_reset_scene.py` - 场景重置测试
- `test_websocket_images.py` - WebSocket 图像传输测试
- `test_latency.py` - 接口延迟测试
