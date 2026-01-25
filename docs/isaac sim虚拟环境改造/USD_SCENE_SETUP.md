# RAM 插入任务 USD 场景文件创建指南

## 概述

根据新的架构设计，RAM 插入任务的所有对象（机器人、相机、任务对象）都应该在 USD 场景文件中定义，由 `isaac_sim_server.py` 加载。

## 需要完成的工作

### 1. 创建 USD 场景文件

#### 方法 A：使用 Python 脚本创建（推荐）

已提供脚本：`create_ram_scene_usd.py`

**使用步骤**：

1. **在 Isaac Sim 环境中运行脚本**：
   ```bash
   # 方法 1：在 Isaac Sim 的 Python 环境中运行
   cd /path/to/isaac-sim
   ./python.sh /path/to/hil-serl/examples/experiments/ram_insertion/create_ram_scene_usd.py \
       --output_path=/path/to/ram_insertion_scene.usd
   
   # 方法 2：如果 Isaac Sim 的 Python 在系统路径中
   python create_ram_scene_usd.py --output_path=./ram_insertion_scene.usd
   ```

2. **验证生成的 USD 文件**：
   - 在 Isaac Sim 中打开生成的 USD 文件
   - 检查所有对象是否正确创建
   - 验证物理属性是否正确

#### 方法 B：在 Isaac Sim 中手动创建

1. **启动 Isaac Sim**
2. **创建新场景**：`File > New Stage`
3. **添加 Franka 机器人**：
   - `Window > Isaac Utils > Add Robot`
   - 选择 Franka Panda
   - 设置 prim_path: `/World/franka`
4. **添加相机**：
   - 在 `/World/franka/panda_hand` 下添加两个相机
   - 设置 prim_path: `/World/franka/panda_hand/wrist_1` 和 `/World/franka/panda_hand/wrist_2`
   - 配置相机参数（分辨率 1280x720）
5. **添加任务对象**（见下方详细说明）
6. **保存场景**：`File > Save As` 保存为 USD 文件

### 2. USD 场景文件必须包含的内容

#### 2.1 机器人（必需）

- **Prim 路径**：`/World/franka`
- **类型**：Franka Panda 机器人
- **位置**：基座位置 `[0, 0, 0]`（可根据需要调整）
- **要求**：必须包含完整的机器人模型和关节定义

#### 2.2 相机（必需）

- **Prim 路径**：
  - `/World/franka/panda_hand/wrist_1`
  - `/World/franka/panda_hand/wrist_2`
- **类型**：Camera
- **分辨率**：1280x720
- **位置**：相对于 `panda_hand` 的位置
  - `wrist_1`: `[0, 0, 0.05]`（相对于末端执行器）
  - `wrist_2`: `[0, 0.05, 0]`
- **要求**：必须正确配置相机参数，确保能够捕获图像

#### 2.3 RAM 条（必需）

- **Prim 路径**：`/World/ram_stick`
- **类型**：Cube 或 Mesh
- **尺寸**（DDR4 标准）：
  - 长度：0.13335m (133.35mm)
  - 宽度：0.030m (30mm)
  - 高度：0.0038m (3.8mm)
- **初始位置**：`[0.586, -0.220, 0.273]`（对应 GRASP_POSE）
- **物理属性**：
  - 质量：0.04kg (40g)
  - 摩擦系数：0.4（静摩擦），0.35（动摩擦）
  - 弹性：0.1
  - 刚体：是
  - 碰撞：启用
- **视觉材质**：深绿色（PCB 颜色）

#### 2.4 主板插槽（必需）

- **Prim 路径**：`/World/motherboard/slot`
- **类型**：Cube 或 Mesh
- **尺寸**：
  - 长度：0.133m (133mm)
  - 宽度：0.0305m (30.5mm，略宽于 RAM 条)
  - 深度：0.006m (6mm)
- **位置**：`[0.588, -0.036, 0.275]`（对应 TARGET_POSE，插槽底部）
- **物理属性**：
  - 运动学：是（固定不动）
  - 摩擦系数：0.25（静摩擦），0.2（动摩擦）
  - 弹性：0.05
  - 碰撞：启用
- **视觉材质**：深色

#### 2.5 RAM 支架（必需）

- **Prim 路径**：`/World/ram_holder`
- **类型**：Cube
- **尺寸**：
  - 长度：0.05m (50mm)
  - 宽度：0.02m (20mm)
  - 高度：0.01m (10mm)
- **位置**：`[0.586, -0.220, 0.263]`（GRASP_POSE 下方 1cm）
- **物理属性**：
  - 运动学：是（固定不动）
  - 摩擦系数：0.6（静摩擦），0.5（动摩擦，高摩擦防止滑动）
  - 碰撞：启用

#### 2.6 地面平面（必需）

- **Prim 路径**：`/World/defaultGroundPlane`
- **类型**：Ground Plane
- **要求**：由 Isaac Sim 自动创建，或手动添加

#### 2.7 光照（可选但推荐）

- **Prim 路径**：`/World/env_light`
- **类型**：Dome Light（环境光）
- **强度**：1.0（可根据需要调整）

### 3. 关键配置参数

根据 `config.py` 中的配置：

```python
TARGET_POSE = [0.588, -0.036, 0.278, π, 0, 0]  # 插入位置
GRASP_POSE = [0.586, -0.220, 0.273, π, 0, 0]     # 抓取位置
```

**对象位置设置**：
- RAM 条初始位置：`GRASP_POSE[:3]` = `[0.586, -0.220, 0.273]`
- 主板插槽位置：`TARGET_POSE[:3]` = `[0.588, -0.036, 0.278]`（插槽底部）
- RAM 支架位置：`GRASP_POSE[:3] - [0, 0, 0.01]` = `[0.586, -0.220, 0.263]`

### 4. 验证清单

创建 USD 文件后，需要验证：

- [ ] Franka 机器人正确加载（prim_path: `/World/franka`）
- [ ] 两个相机正确配置（prim_path: `/World/franka/panda_hand/wrist_1` 和 `wrist_2`）
- [ ] RAM 条尺寸正确（133.35mm × 30mm × 3.8mm）
- [ ] RAM 条物理属性正确（质量 0.04kg，摩擦系数 0.4）
- [ ] 主板插槽尺寸正确（133mm × 30.5mm × 6mm）
- [ ] 主板插槽设置为运动学（固定不动）
- [ ] RAM 支架位置正确（在 GRASP_POSE 下方）
- [ ] RAM 支架设置为运动学（固定不动）
- [ ] 所有对象位置与配置一致
- [ ] 碰撞检测启用
- [ ] 物理材质正确配置

### 5. 使用 USD 文件

创建 USD 文件后，使用 `isaac_sim_server.py` 加载：

```bash
python serl_robot_infra/robot_servers/isaac_sim_server.py \
    --flask_url=0.0.0.0 \
    --flask_port=5001 \
    --headless=True \
    --usd_path=/path/to/ram_insertion_scene.usd \
    --robot_prim_path=/World/franka \
    --camera_prim_paths=/World/franka/panda_hand/wrist_1,/World/franka/panda_hand/wrist_2 \
    --config_module=experiments.ram_insertion.config
```

### 6. 可能遇到的问题和解决方案

#### 问题 1：机器人无法加载
- **原因**：prim_path 不正确或机器人模型未正确导入
- **解决**：检查 USD 文件中机器人 prim 路径，确保与 `--robot_prim_path` 参数一致

#### 问题 2：相机无法获取图像
- **原因**：相机 prim_path 不正确或相机未正确配置
- **解决**：检查相机 prim 路径，确保相机已正确添加到场景中

#### 问题 3：任务对象位置不正确
- **原因**：对象位置与配置不一致
- **解决**：检查 USD 文件中对象位置，确保与 `TARGET_POSE` 和 `GRASP_POSE` 一致

#### 问题 4：物理属性不正确
- **原因**：物理材质或刚体属性未正确设置
- **解决**：在 Isaac Sim 中打开 USD 文件，检查物理属性设置

### 7. 下一步工作

完成 USD 文件创建后，还需要：

1. **测试场景加载**：
   - 使用 `isaac_sim_server.py` 加载 USD 文件
   - 验证所有对象正确加载

2. **测试物理仿真**：
   - 验证 RAM 条可以正确掉落
   - 验证碰撞检测正常工作

3. **测试相机**：
   - 验证相机可以捕获图像
   - 验证图像质量符合要求

4. **测试控制**：
   - 验证机器人可以正确控制
   - 验证 IK 求解器正常工作

5. **优化场景**（可选）：
   - 添加更精确的几何模型
   - 优化物理参数
   - 添加域随机化支持

## 参考文档

- Isaac Sim USD 文档：https://docs.omniverse.nvidia.com/isaacsim/latest/core_api_tutorials/tutorial_core_adding_objects.html
- USD 格式规范：https://openusd.org/
- RAM 插入任务设置计划：`docs/isaac sim虚拟环境改造/ram_insertion_sim_setup_plan.md`
