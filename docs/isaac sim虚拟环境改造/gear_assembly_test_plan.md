# Gear Assembly 任务 Isaac Sim Server 测试计划

## 测试概述

本文档基于 `isaac_sim_server_integration_test.md`，针对 **gear_assembly** 任务进行适配和优化。

**主要修改**：
1. ✅ **跳过场景创建步骤**：gear_assembly 已有 USD 文件（`HIL_franka_gear.usda`）
2. ✅ **更新 USD 文件路径**：使用 `examples/experiments/gear_assembly/HIL_franka_gear.usda`
3. ✅ **更新对象名称**：从 RAM 相关对象改为 Gear 相关对象
4. ✅ **更新相机路径**：验证相机路径是否正确

---

## 一、USD 场景文件验证（替代场景创建）

### 测试目标
验证 `HIL_franka_gear.usda` 文件包含所有必需的对象（机器人、相机、任务对象）。

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

#### 步骤 1.2：验证 USD 文件存在

**准备条件**：
- ✅ 步骤 1.1 已完成

**执行步骤**：
```bash
# 检查 USD 文件是否存在
ls -lh examples/experiments/gear_assembly/HIL_franka_gear.usda

# 检查文件大小
du -h examples/experiments/gear_assembly/HIL_franka_gear.usda
```

**期望输出**：
- USD 文件存在
- 文件大小 > 0（通常 > 100KB）

**验证检查**：
- ✅ USD 文件已存在
- ✅ 文件大小合理
- ✅ 文件路径正确

---

#### 步骤 1.3：验证 USD 文件内容

**准备条件**：
- ✅ 步骤 1.2 已完成，USD 文件存在

**执行步骤**：
创建测试脚本 `test_gear_usd_validation.py`：
```python
from omni.isaac.kit import SimulationApp
import omni.usd
from omni.isaac.core.utils.stage import get_current_stage
from pxr import UsdGeom

# 启动 SimulationApp
config = {"headless": True}
simulation_app = SimulationApp(config)

# 加载 USD 场景
usd_path = "examples/experiments/gear_assembly/HIL_franka_gear.usda"
print(f"[INFO] Loading USD scene from: {usd_path}")

try:
    usd_context = omni.usd.get_context()
    usd_context.open_stage(usd_path)
    print("[INFO] USD scene loaded successfully")
    
    # 获取 stage
    stage = get_current_stage()
    
    # 检查机器人
    robot_prim = stage.GetPrimAtPath("/World/franka")
    if robot_prim.IsValid():
        print("[INFO] ✅ Robot prim found at /World/franka")
    else:
        print("[ERROR] ❌ Robot prim not found at /World/franka")
    
    # 检查相机
    camera1_prim = stage.GetPrimAtPath("/World/franka/panda_hand/wrist_1")
    camera2_prim = stage.GetPrimAtPath("/World/franka/panda_hand/wrist_2")
    if camera1_prim.IsValid():
        print("[INFO] ✅ Camera 'wrist_1' found at /World/franka/panda_hand/wrist_1")
    else:
        print("[ERROR] ❌ Camera 'wrist_1' not found")
    if camera2_prim.IsValid():
        print("[INFO] ✅ Camera 'wrist_2' found at /World/franka/panda_hand/wrist_2")
    else:
        print("[ERROR] ❌ Camera 'wrist_2' not found")
    
    # 检查任务对象
    gear_medium_prim = stage.GetPrimAtPath("/World/factory_gear_medium")
    gear_base_prim = stage.GetPrimAtPath("/World/factory_gear_base")
    gear_large_prim = stage.GetPrimAtPath("/World/factory_gear_large")
    
    if gear_medium_prim.IsValid():
        print("[INFO] ✅ Gear medium found at /World/factory_gear_medium")
    else:
        print("[ERROR] ❌ Gear medium not found")
    
    if gear_base_prim.IsValid():
        print("[INFO] ✅ Gear base found at /World/factory_gear_base")
    else:
        print("[ERROR] ❌ Gear base not found")
    
    if gear_large_prim.IsValid():
        print("[INFO] ✅ Gear large found at /World/factory_gear_large")
    else:
        print("[ERROR] ❌ Gear large not found")
    
    print("[INFO] USD scene validation completed")
        
except Exception as e:
    print(f"[ERROR] Failed to load USD scene: {e}")
    import traceback
    traceback.print_exc()
    raise

# 关闭应用
simulation_app.close()
print("[INFO] Test completed")
```

运行测试：
```bash
# 使用 Isaac Sim 的 Python 环境
cd /path/to/isaac-sim
./python.sh /path/to/hil-serl/test_gear_usd_validation.py

# 或者如果已配置 Isaac Sim Python 路径
cd /path/to/hil-serl
python test_gear_usd_validation.py
```

**期望输出**：
```
[INFO] Loading USD scene from: examples/experiments/gear_assembly/HIL_franka_gear.usda
[INFO] USD scene loaded successfully
[INFO] ✅ Robot prim found at /World/franka
[INFO] ✅ Camera 'wrist_1' found at /World/franka/panda_hand/wrist_1
[INFO] ✅ Camera 'wrist_2' found at /World/franka/panda_hand/wrist_2
[INFO] ✅ Gear medium found at /World/factory_gear_medium
[INFO] ✅ Gear base found at /World/factory_gear_base
[INFO] ✅ Gear large found at /World/factory_gear_large
[INFO] USD scene validation completed
[INFO] Test completed
```

**验证检查**：
- ✅ USD 文件可以正常打开
- ✅ 场景包含以下对象：
  - ✅ `/World/franka` - Franka 机器人
  - ✅ `/World/franka/panda_hand/wrist_1` - 相机 1
  - ✅ `/World/franka/panda_hand/wrist_2` - 相机 2
  - ✅ `/World/factory_gear_medium` - Gear medium（待安装的齿轮）
  - ✅ `/World/factory_gear_base` - Gear base（底座）
  - ✅ `/World/factory_gear_large` - Gear large（大齿轮）
- ✅ 所有必需对象都存在
- ✅ 无错误信息

---

## 二、Isaac Sim 应用控制测试

### 测试目标
验证能够通过代码控制 Isaac Sim 应用的完整生命周期：启动、加载场景、运行仿真、重置环境、结束仿真、关闭应用。

**注意**：此部分与原始测试文档相同，只需将 USD 文件路径改为 `examples/experiments/gear_assembly/HIL_franka_gear.usda`。

### 测试步骤

#### 步骤 2.1：测试 SimulationApp 启动

**与原始文档相同**，无需修改。

---

#### 步骤 2.2：测试 USD 场景加载

**修改点**：将 USD 文件路径改为 gear_assembly 的 USD 文件。

创建测试脚本 `test_gear_usd_loading.py`：
```python
from omni.isaac.kit import SimulationApp
import omni.usd
from omni.isaac.core.utils.stage import get_current_stage
from pxr import UsdGeom

# 启动 SimulationApp
config = {"headless": True}
simulation_app = SimulationApp(config)

# 加载 USD 场景（修改为 gear_assembly 的 USD 文件）
usd_path = "examples/experiments/gear_assembly/HIL_franka_gear.usda"
print(f"[INFO] Loading USD scene from: {usd_path}")

try:
    usd_context = omni.usd.get_context()
    usd_context.open_stage(usd_path)
    print("[INFO] USD scene loaded successfully")
    
    # 验证场景内容
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
    
    # 检查任务对象（修改为 gear 相关对象）
    gear_medium_prim = stage.GetPrimAtPath("/World/factory_gear_medium")
    gear_base_prim = stage.GetPrimAtPath("/World/factory_gear_base")
    gear_large_prim = stage.GetPrimAtPath("/World/factory_gear_large")
    
    if gear_medium_prim.IsValid():
        print("[INFO] Gear medium found")
    else:
        print("[ERROR] Gear medium not found!")
    
    if gear_base_prim.IsValid():
        print("[INFO] Gear base found")
    else:
        print("[ERROR] Gear base not found!")
    
    if gear_large_prim.IsValid():
        print("[INFO] Gear large found")
    else:
        print("[ERROR] Gear large not found!")
        
except Exception as e:
    print(f"[ERROR] Failed to load USD scene: {e}")
    raise

# 关闭应用
simulation_app.close()
print("[INFO] Test completed")
```

**期望输出**：
```
[INFO] Loading USD scene from: examples/experiments/gear_assembly/HIL_franka_gear.usda
[INFO] USD scene loaded successfully
[INFO] Robot prim found at /World/franka
[INFO] Cameras found
[INFO] Gear medium found
[INFO] Gear base found
[INFO] Gear large found
[INFO] Test completed
```

---

#### 步骤 2.3-2.6：其他测试步骤

**修改点**：将所有测试脚本中的 USD 文件路径改为：
```python
usd_path = "examples/experiments/gear_assembly/HIL_franka_gear.usda"
```

**对象名称修改**：
- 将 `ram_stick` 改为 `factory_gear_medium`
- 将 `motherboard` 改为 `factory_gear_base`
- 将 `ram_holder` 改为 `factory_gear_large`（如果需要）

**其他内容保持不变**。

---

## 三、Isaac Sim Server 对外接口测试

### 测试目标
验证 Isaac Sim Server 的所有 HTTP 接口和 WebSocket 接口正常工作。

### 测试步骤

#### 步骤 3.1：启动 Isaac Sim Server

**准备条件**：
- ✅ USD 场景文件已存在（`examples/experiments/gear_assembly/HIL_franka_gear.usda`）
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
    --usd_path=examples/experiments/gear_assembly/HIL_franka_gear.usda \
    --robot_prim_path=/World/franka \
    --camera_prim_paths=/World/franka/panda_hand/wrist_1,/World/franka/panda_hand/wrist_2 \
    --config_module=experiments.gear_assembly.config
```

**关键参数说明**：
- `--usd_path`: gear_assembly 的 USD 文件路径
- `--robot_prim_path`: `/World/franka`（与 ram_insertion 相同）
- `--camera_prim_paths`: `/World/franka/panda_hand/wrist_1,/World/franka/panda_hand/wrist_2`（与 ram_insertion 相同）
- `--config_module`: `experiments.gear_assembly.config`（用于加载 `IMAGE_CROP` 配置）

**期望输出**：
```
[INFO] Loading USD scene from: examples/experiments/gear_assembly/HIL_franka_gear.usda
[INFO] USD scene loaded successfully
[INFO] Getting robot from prim path: /World/franka
[INFO] Robot found at /World/franka
[INFO] Getting cameras from prim paths: ['/World/franka/panda_hand/wrist_1', '/World/franka/panda_hand/wrist_2']
[INFO] Camera 'wrist_1' found at /World/franka/panda_hand/wrist_1
[INFO] Camera 'wrist_2' found at /World/franka/panda_hand/wrist_2
[INFO] Successfully loaded 2 cameras: ['wrist_1', 'wrist_2']
[INFO] Loaded IMAGE_CROP config: ['wrist_1', 'wrist_2']
[INFO] Isaac Sim Server initialized
[INFO] Starting Flask server on http://127.0.0.1:5001
 * Running on http://127.0.0.1:5001
```

**验证检查**：
- ✅ 服务器启动成功
- ✅ USD 场景加载成功
- ✅ 机器人和相机对象获取成功
- ✅ `IMAGE_CROP` 配置加载成功
- ✅ Flask 服务器监听在指定端口
- ✅ 无错误信息

---

#### 步骤 3.2-3.7：接口测试

**与原始文档相同**，无需修改。

所有 HTTP 接口测试（`/health`、`/getstate`、`/pose`、`/close_gripper`、`/open_gripper` 等）都可以直接使用，因为接口是通用的，不依赖任务特定的对象。

---

## 四、测试计划适配总结

### 4.1 可以直接使用的部分

| 测试部分 | 状态 | 说明 |
|---------|------|------|
| **SimulationApp 启动测试** | ✅ 直接使用 | 不依赖任务 |
| **World 创建测试** | ✅ 直接使用 | 不依赖任务 |
| **机器人对象获取测试** | ✅ 直接使用 | 机器人路径相同 |
| **相机对象获取测试** | ✅ 直接使用 | 相机路径相同 |
| **仿真循环测试** | ✅ 直接使用 | 不依赖任务 |
| **环境重置测试** | ✅ 直接使用 | 不依赖任务 |
| **HTTP 接口测试** | ✅ 直接使用 | 接口是通用的 |
| **WebSocket 图像测试** | ✅ 直接使用 | 接口是通用的 |

### 4.2 需要修改的部分

| 测试部分 | 修改内容 | 修改位置 |
|---------|---------|---------|
| **场景创建测试** | ⚠️ **跳过** | gear_assembly 已有 USD 文件 |
| **USD 文件路径** | ✅ 修改为 `HIL_franka_gear.usda` | 所有测试脚本 |
| **对象名称验证** | ✅ 修改为 gear 相关对象 | 步骤 1.3、2.2 |
| **服务器启动参数** | ✅ 修改 USD 路径和 config_module | 步骤 3.1 |

### 4.3 对象路径对照表

| 对象类型 | ram_insertion | gear_assembly |
|---------|--------------|---------------|
| **机器人** | `/World/franka` | `/World/franka` ✅ 相同 |
| **相机 1** | `/World/franka/panda_hand/wrist_1` | `/World/franka/panda_hand/wrist_1` ✅ 相同 |
| **相机 2** | `/World/franka/panda_hand/wrist_2` | `/World/franka/panda_hand/wrist_2` ✅ 相同 |
| **任务对象 1** | `/World/ram_stick` | `/World/factory_gear_medium` |
| **任务对象 2** | `/World/motherboard` | `/World/factory_gear_base` |
| **任务对象 3** | `/World/ram_holder` | `/World/factory_gear_large` |

---

## 五、快速测试检查清单

### 5.1 准备阶段

- [ ] Isaac Sim 已安装
- [ ] Python 环境已配置
- [ ] 依赖包已安装
- [ ] USD 文件存在：`examples/experiments/gear_assembly/HIL_franka_gear.usda`

### 5.2 基础测试

- [ ] USD 文件验证测试通过
- [ ] SimulationApp 启动测试通过
- [ ] USD 场景加载测试通过
- [ ] World 创建测试通过
- [ ] 机器人对象获取测试通过
- [ ] 相机对象获取测试通过

### 5.3 服务器测试

- [ ] Isaac Sim Server 启动成功
- [ ] 健康检查接口测试通过
- [ ] 状态查询接口测试通过
- [ ] 控制命令接口测试通过
- [ ] 位姿控制功能测试通过
- [ ] 夹爪控制功能测试通过
- [ ] 场景重置功能测试通过
- [ ] WebSocket 图像传输测试通过

---

## 六、测试执行建议

### 6.1 测试顺序

1. **第一阶段：基础验证**
   - 步骤 1.1-1.3：USD 文件验证
   - 步骤 2.1-2.4：Isaac Sim 应用控制测试

2. **第二阶段：服务器接口测试**
   - 步骤 3.1：启动服务器
   - 步骤 3.2-3.7：接口功能测试

3. **第三阶段：集成测试**
   - 测试 Gym 环境连接
   - 测试完整控制流程

### 6.2 测试脚本路径

建议将测试脚本保存在：
```
docs/isaac sim虚拟环境改造/test_scripts/gear_assembly/
```

### 6.3 测试记录

建议记录：
- 测试执行时间
- 测试结果（通过/失败）
- 发现的问题
- 解决方案

---

## 七、结论

✅ **测试文档可以作为测试计划执行**，但需要进行以下适配：

1. ✅ **跳过场景创建步骤**（gear_assembly 已有 USD 文件）
2. ✅ **修改 USD 文件路径**（所有测试脚本）
3. ✅ **修改对象名称验证**（步骤 1.3、2.2）
4. ✅ **修改服务器启动参数**（步骤 3.1）

**大部分测试内容可以直接使用**，因为：
- 机器人路径相同（`/World/franka`）
- 相机路径相同（`/World/franka/panda_hand/wrist_1`、`wrist_2`）
- HTTP 接口是通用的（不依赖任务对象）
- WebSocket 接口是通用的

**建议**：按照本测试计划执行测试，重点关注 USD 文件验证和服务器启动测试。
