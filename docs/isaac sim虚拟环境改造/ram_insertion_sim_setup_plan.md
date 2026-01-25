# RAM 插入任务仿真环境高保真度搭建方案

## 一、目标与原则

### 1.1 目标
构建高保真度的 Isaac Sim 仿真环境，使仿真数据与真实任务数据足够接近，支持：
- 仿真到真实的策略迁移
- 在仿真环境中进行大规模数据收集
- 减少真实机器人的使用时间

### 1.2 关键原则
1. **几何精度**：对象尺寸、位置与真实环境一致
2. **物理真实性**：质量、摩擦、刚度等物理参数匹配真实
3. **接触建模**：精确模拟 RAM 条与插槽的接触和插入过程
4. **视觉一致性**：相机视角、光照、材质与真实环境相似
5. **控制等效性**：仿真中的控制行为与真实阻抗控制等效

---

## 二、真实环境关键特征分析

### 2.1 几何特征

#### RAM 条（DDR4 标准）
- **尺寸**：约 133.35mm × 30mm × 3.8mm（长×宽×高）
- **金手指**：底部有 288 个接触点（DDR4）
- **重量**：约 30-50g
- **抓取位置**：通常在 RAM 条中部偏上

#### 主板 RAM 插槽
- **插槽深度**：约 5-6mm
- **插槽宽度**：约 30mm（与 RAM 条宽度匹配）
- **插槽长度**：约 133mm
- **接触点**：288 个金手指接触点
- **插入力**：需要约 20-30N 的力才能完全插入

#### RAM 支架
- **尺寸**：约 50mm × 20mm × 10mm
- **功能**：支撑 RAM 条，便于抓取
- **位置**：固定在 GRASP_POSE 位置

### 2.2 物理参数

#### 真实阻抗控制参数
```python
COMPLIANCE_PARAM = {
    "translational_stiffness": 2000,  # N/m
    "translational_damping": 89,      # N·s/m
    "rotational_stiffness": 150,      # N·m/rad
    "rotational_damping": 7,          # N·m·s/rad
}

PRECISION_PARAM = {
    "translational_stiffness": 2000,
    "translational_damping": 89,
    "rotational_stiffness": 250,      # 更高的旋转刚度
    "rotational_damping": 9,
}
```

#### 材料属性（估计值）
- **RAM 条**：
  - 密度：约 1.5-2.0 g/cm³（PCB + 组件）
  - 摩擦系数：0.3-0.5（与插槽接触）
  - 弹性模量：约 10-20 GPa（PCB 材料）

- **主板插槽**：
  - 摩擦系数：0.2-0.4
  - 刚度：高（固定结构）

- **夹爪**：
  - 抓取力：约 20-50N
  - 摩擦系数：0.6-0.8（橡胶材质）

### 2.3 任务空间约束

```python
TARGET_POSE = [0.588, -0.036, 0.278, π, 0, 0]  # 插入位置
GRASP_POSE = [0.586, -0.220, 0.273, π, 0, 0]   # 抓取位置
RESET_POSE = TARGET_POSE + [0, 0, 0.05, 0, 0.05, 0]  # 重置位置

# 探索边界框（非常小，说明任务精度要求高）
ABS_POSE_LIMIT_LOW = TARGET_POSE - [0.03, 0.02, 0.01, 0.01, 0.1, 0.4]
ABS_POSE_LIMIT_HIGH = TARGET_POSE + [0.03, 0.02, 0.05, 0.01, 0.1, 0.4]
```

**关键观察**：
- XY 方向精度要求：±3cm, ±2cm
- Z 方向精度要求：±1cm（非常严格）
- 旋转精度要求：±0.1 rad（约 ±5.7°）

---

## 三、仿真环境搭建详细方案

### 3.1 几何模型创建

#### 方案 A：使用精确的 USD 模型（推荐）

**步骤 1：创建或导入精确模型**

```python
# 文件：serl_robot_infra/franka_env/envs/isaac_sim_ram_env.py

def _add_task_objects(self):
    """添加任务相关对象（高保真度版本）"""
    from omni.isaac.core.utils.prims import create_prim
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.utils.stage import get_current_stage
    from pxr import UsdPhysics, UsdShade, Gf
    
    stage = get_current_stage()
    
    # 方法 1：从 USD 文件导入（如果有精确模型）
    # add_reference_to_stage(
    #     usd_path="/path/to/ram_stick.usd",
    #     prim_path="/World/ram_stick"
    # )
    
    # 方法 2：使用程序化创建精确几何体
    self._create_ram_stick_precise()
    self._create_motherboard_slot_precise()
    self._create_ram_holder_precise()
```

#### RAM 条精确模型

```python
def _create_ram_stick_precise(self):
    """创建精确的 RAM 条模型"""
    from omni.isaac.core.utils.prims import create_prim
    from omni.isaac.core.utils.stage import get_current_stage
    from pxr import UsdPhysics, UsdGeom, Gf
    
    stage = get_current_stage()
    
    # RAM 条尺寸（DDR4 标准，单位：米）
    RAM_LENGTH = 0.13335    # 133.35mm
    RAM_WIDTH = 0.030       # 30mm
    RAM_HEIGHT = 0.0038     # 3.8mm
    
    # 创建 RAM 条主体
    ram_position = self.config.GRASP_POSE[:3].copy()
    
    # 使用 Mesh 创建更精确的形状（如果有）
    # 或者使用多个 Box 组合创建带金手指的模型
    
    # 简化版本：使用精确尺寸的 Box
    self.ram_stick = create_prim(
        prim_path="/World/ram_stick",
        prim_type="Cube",
        position=ram_position,
        scale=np.array([RAM_LENGTH, RAM_WIDTH, RAM_HEIGHT]),
        orientation=Gf.Quatf(1, 0, 0, 0),  # 无旋转
    )
    
    # 添加金手指区域（底部接触区域）
    # 金手指厚度约 0.001m
    finger_position = ram_position.copy()
    finger_position[2] -= RAM_HEIGHT / 2 - 0.0005  # 底部
    
    finger_prim = create_prim(
        prim_path="/World/ram_stick/fingers",
        prim_type="Cube",
        position=finger_position,
        scale=np.array([RAM_LENGTH, RAM_WIDTH, 0.001]),
    )
    
    # 配置物理属性
    self._configure_ram_physics()
```

#### 主板插槽精确模型

```python
def _create_motherboard_slot_precise(self):
    """创建精确的主板 RAM 插槽模型"""
    from omni.isaac.core.utils.prims import create_prim
    from pxr import UsdPhysics, Gf
    
    slot_position = self.config.TARGET_POSE[:3].copy()
    
    # 插槽尺寸
    SLOT_LENGTH = 0.133    # 133mm
    SLOT_WIDTH = 0.0305   # 略宽于 RAM 条，30.5mm
    SLOT_DEPTH = 0.006    # 6mm 深度
    
    # 创建插槽（使用 Mesh 或组合几何体）
    self.motherboard = create_prim(
        prim_path="/World/motherboard",
        prim_type="Xform",
        position=slot_position,
    )
    
    # 插槽主体（固定）
    slot_prim = create_prim(
        prim_path="/World/motherboard/slot",
        prim_type="Cube",
        position=slot_position + np.array([0, 0, -SLOT_DEPTH/2]),
        scale=np.array([SLOT_LENGTH, SLOT_WIDTH, SLOT_DEPTH]),
    )
    
    # 配置为固定（运动学）
    self._configure_slot_physics()
```

### 3.2 物理属性配置

#### RAM 条物理属性

```python
def _configure_ram_physics(self):
    """配置 RAM 条的物理属性"""
    from omni.isaac.core.utils.stage import get_current_stage
    from pxr import UsdPhysics, UsdShade
    
    stage = get_current_stage()
    ram_prim = stage.GetPrimAtPath("/World/ram_stick")
    
    if not ram_prim.IsValid():
        return
    
    # 1. 添加刚体属性
    rigid_body = UsdPhysics.RigidBodyAPI.Apply(ram_prim, "RigidBody")
    
    # 2. 设置质量（约 30-50g）
    mass_api = UsdPhysics.MassAPI.Apply(ram_prim, "MassAPI")
    mass_api.CreateMassAttr().Set(0.04)  # 40g
    
    # 3. 设置惯性（简化：均匀分布）
    # 对于长方体：I = (1/12) * m * (h² + w²)
    # RAM 条：133.35mm × 30mm × 3.8mm
    inertia = Gf.Vec3f(
        0.04 * (0.0038**2 + 0.030**2) / 12,   # Ixx
        0.04 * (0.13335**2 + 0.0038**2) / 12,  # Iyy
        0.04 * (0.13335**2 + 0.030**2) / 12    # Izz
    )
    mass_api.CreateDiagonalInertiaAttr().Set(inertia)
    
    # 4. 添加碰撞形状
    collision_api = UsdPhysics.CollisionAPI.Apply(ram_prim, "Collision")
    
    # 5. 创建物理材质
    material_path = "/World/ram_material"
    material = UsdShade.Material.Define(stage, material_path)
    
    # 设置摩擦系数
    physics_material = UsdPhysics.MaterialAPI.Apply(
        material.GetPrim(), "PhysicsMaterial"
    )
    physics_material.CreateStaticFrictionAttr().Set(0.4)  # 静摩擦
    physics_material.CreateDynamicFrictionAttr().Set(0.35)  # 动摩擦
    physics_material.CreateRestitutionAttr().Set(0.1)  # 恢复系数（低）
    
    # 应用材质到 RAM 条
    UsdShade.MaterialBindingAPI(ram_prim).Bind(material)
```

#### 插槽物理属性

```python
def _configure_slot_physics(self):
    """配置插槽的物理属性（固定）"""
    from omni.isaac.core.utils.stage import get_current_stage
    from pxr import UsdPhysics, UsdShade
    
    stage = get_current_stage()
    slot_prim = stage.GetPrimAtPath("/World/motherboard/slot")
    
    if not slot_prim.IsValid():
        return
    
    # 1. 添加刚体属性
    rigid_body = UsdPhysics.RigidBodyAPI.Apply(slot_prim, "RigidBody")
    
    # 2. 设置为运动学（固定不动）
    rigid_body.CreateKinematicEnabledAttr().Set(True)
    
    # 3. 添加碰撞形状
    collision_api = UsdPhysics.CollisionAPI.Apply(slot_prim, "Collision")
    
    # 4. 创建物理材质（低摩擦，便于插入）
    material_path = "/World/slot_material"
    material = UsdShade.Material.Define(stage, material_path)
    
    physics_material = UsdPhysics.MaterialAPI.Apply(
        material.GetPrim(), "PhysicsMaterial"
    )
    physics_material.CreateStaticFrictionAttr().Set(0.25)  # 低摩擦
    physics_material.CreateDynamicFrictionAttr().Set(0.2)
    physics_material.CreateRestitutionAttr().Set(0.05)  # 几乎无弹性
    
    UsdShade.MaterialBindingAPI(slot_prim).Bind(material)
```

### 3.3 接触和约束建模

#### 插入过程的接触建模

```python
def _setup_insertion_contact(self):
    """设置插入过程的接触建模"""
    from omni.isaac.core.utils.stage import get_current_stage
    from pxr import UsdPhysics
    
    stage = get_current_stage()
    
    # 方法 1：使用接触传感器检测插入
    # 在插槽底部添加接触传感器
    contact_sensor_prim = stage.DefinePrim(
        "/World/motherboard/contact_sensor",
        "PhysicsContactSensor"
    )
    
    # 方法 2：使用距离检测
    # 检测 RAM 条底部与插槽底部的距离
    # 当距离 < 阈值时，认为插入成功
    
    # 方法 3：使用约束（插入后锁定）
    # 当检测到完全插入时，创建固定约束
```

#### 夹爪抓取约束优化

```python
def _attach_ram_to_gripper(self):
    """建立 RAM 条与夹爪的约束（优化版本）"""
    from omni.isaac.core.utils.stage import get_current_stage
    from pxr import UsdPhysics, Gf
    
    stage = get_current_stage()
    ram_prim = stage.GetPrimAtPath("/World/ram_stick")
    gripper_prim = stage.GetPrimAtPath("/World/franka/panda_hand")
    
    if not ram_prim.IsValid() or not gripper_prim.IsValid():
        return
    
    # 使用固定约束（Fixed Joint）
    constraint_path = "/World/ram_grasp_constraint"
    constraint_prim = stage.DefinePrim(constraint_path, "PhysicsFixedJoint")
    
    if constraint_prim.IsValid():
        constraint = UsdPhysics.FixedJoint(constraint_prim)
        
        # 设置约束的两个刚体
        constraint.CreateBody0Rel().SetTargets([ram_prim.GetPath()])
        constraint.CreateBody1Rel().SetTargets([gripper_prim.GetPath()])
        
        # 设置约束位置（在 RAM 条抓取点）
        # 计算抓取点相对于夹爪的局部位置
        grasp_offset = Gf.Vec3f(0, 0, 0.02)  # 夹爪中心到抓取点的偏移
        constraint.CreateLocalPos0Attr().Set(grasp_offset)
        constraint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
        
        self.ram_grasp_constraint = constraint_prim
```

### 3.4 机器人控制等效性

#### 阻抗控制等效实现

```python
def _setup_impedance_control(self):
    """设置与真实阻抗控制等效的仿真控制"""
    # 真实环境使用阻抗控制：
    # F = K * (x_target - x) - D * v
    # 
    # 在仿真中，可以通过以下方式等效：
    # 1. 使用位置控制器 + 阻尼
    # 2. 使用力控模式（如果 Isaac Sim 支持）
    # 3. 使用关节空间 PD 控制
    
    # 方法：使用 RMPFlowController 并调整参数
    if self.controller is not None:
        # 调整控制器参数以匹配阻抗控制行为
        # 注意：这需要根据实际 API 调整
        pass
```

#### 控制参数映射

```python
# 真实阻抗控制参数 → 仿真控制参数映射
SIM_CONTROL_CONFIG = {
    # 位置控制增益（对应刚度）
    "position_gain": 2000.0,  # 对应 translational_stiffness
    
    # 速度控制增益（对应阻尼）
    "velocity_gain": 89.0,    # 对应 translational_damping
    
    # 旋转控制增益
    "orientation_gain": 150.0,  # 对应 rotational_stiffness
    "angular_velocity_gain": 7.0,  # 对应 rotational_damping
    
    # 最大速度限制（对应 clip 参数）
    "max_linear_velocity": 0.1,  # m/s
    "max_angular_velocity": 0.5,  # rad/s
}
```

### 3.5 相机和视觉配置

#### 相机精确配置

```python
def _add_cameras(self):
    """添加相机（精确配置）"""
    from omni.isaac.sensor import Camera
    
    # 相机配置（与真实 RealSense 相机匹配）
    camera_configs = {
        "wrist_1": {
            "position": np.array([0, 0, 0.05]),  # 相对于末端执行器
            "orientation": np.array([0, 0, 0, 1]),  # quaternion
            "fov": 69.4,  # RealSense D435 水平 FOV
            "resolution": (1280, 720),
            "focal_length": 1.93,  # mm（等效）
        },
        "wrist_2": {
            "position": np.array([0, 0.05, 0]),
            "orientation": np.array([0, 0, 0, 1]),
            "fov": 69.4,
            "resolution": (1280, 720),
            "focal_length": 1.93,
        },
    }
    
    for cam_key, cam_config in camera_configs.items():
        camera = self.world.scene.add(
            Camera(
                prim_path=f"/World/franka/panda_hand/{cam_key}",
                name=cam_key,
                position=cam_config["position"],
                orientation=cam_config["orientation"],
                resolution=cam_config["resolution"],
                # 如果 API 支持，设置 FOV
                # fov=cam_config["fov"],
            )
        )
        self.cameras[cam_key] = camera
```

#### 光照和材质配置

```python
def _setup_lighting(self):
    """设置光照（匹配真实环境）"""
    from omni.isaac.core.utils.prims import create_prim
    from pxr import UsdLux
    
    # 添加环境光
    env_light = create_prim(
        prim_path="/World/env_light",
        prim_type="DomeLight",
    )
    
    # 设置环境光强度（匹配真实工作空间）
    light_prim = env_light.GetPrim()
    UsdLux.DomeLight(light_prim).CreateIntensityAttr().Set(1.0)
    UsdLux.DomeLight(light_prim).CreateColorAttr().Set(
        Gf.Vec3f(1.0, 1.0, 1.0)  # 白色光
    )
    
    # 添加定向光（模拟工作灯）
    dir_light = create_prim(
        prim_path="/World/directional_light",
        prim_type="DistantLight",
    )
    UsdLux.DistantLight(dir_light.GetPrim()).CreateIntensityAttr().Set(2.0)
```

### 3.6 域随机化配置

#### 提高仿真到真实迁移的随机化

```python
class IsaacSimEnvConfig(DefaultEnvConfig):
    """Isaac Sim 环境配置（带域随机化）"""
    
    # 域随机化参数
    DOMAIN_RANDOMIZATION = {
        # 物理参数随机化
        "mass_randomization": {
            "enabled": True,
            "ram_mass_range": (0.03, 0.05),  # 30-50g
            "friction_range": (0.3, 0.5),     # 摩擦系数范围
        },
        
        # 视觉随机化
        "visual_randomization": {
            "enabled": True,
            "lighting_intensity_range": (0.8, 1.2),
            "material_color_variation": 0.1,
        },
        
        # 几何随机化
        "geometry_randomization": {
            "enabled": True,
            "ram_size_variation": 0.001,  # ±1mm
            "slot_position_variation": 0.002,  # ±2mm
        },
        
        # 相机随机化
        "camera_randomization": {
            "enabled": True,
            "position_noise": 0.001,  # ±1mm
            "orientation_noise": 0.01,  # ±0.01 rad
        },
    }
```

---

## 四、实施步骤

### 阶段 1：基础几何模型（1-2 天）

1. **创建精确的 RAM 条模型**
   - 使用标准 DDR4 尺寸
   - 添加金手指区域
   - 配置物理属性

2. **创建主板插槽模型**
   - 精确的插槽尺寸
   - 固定约束
   - 接触表面

3. **创建 RAM 支架模型**
   - 支撑结构
   - 固定约束

### 阶段 2：物理属性配置（1-2 天）

1. **配置质量、惯性**
   - RAM 条：40g，精确惯性
   - 插槽：固定，大质量

2. **配置摩擦和材质**
   - RAM 条：摩擦系数 0.4
   - 插槽：摩擦系数 0.25
   - 创建物理材质

3. **配置碰撞形状**
   - 精确的碰撞几何
   - 接触检测

### 阶段 3：接触和约束（1-2 天）

1. **实现插入检测**
   - 距离检测
   - 接触传感器
   - 插入成功判断

2. **优化抓取约束**
   - 固定约束位置
   - 约束刚度

3. **实现释放机制**
   - 约束删除
   - 物理状态重置

### 阶段 4：控制等效性（2-3 天）

1. **实现阻抗控制等效**
   - 位置控制器参数调整
   - 阻尼配置
   - 速度限制

2. **测试控制行为**
   - 与真实环境对比
   - 调整参数

### 阶段 5：视觉配置（1 天）

1. **相机精确配置**
   - 位置、角度
   - FOV、分辨率
   - 图像裁剪

2. **光照配置**
   - 环境光
   - 定向光
   - 材质属性

### 阶段 6：域随机化（1-2 天）

1. **实现随机化参数**
   - 物理参数
   - 视觉参数
   - 几何参数

2. **测试随机化效果**
   - 验证多样性
   - 调整范围

### 阶段 7：验证和调优（2-3 天）

1. **数据对比**
   - 仿真 vs 真实观察分布
   - 动作分布
   - 奖励分布

2. **策略迁移测试**
   - 仿真训练的策略在真实环境测试
   - 调整参数

---

## 五、关键参数参考表

### 5.1 几何参数

| 对象 | 参数 | 真实值 | 仿真值 | 单位 |
|------|------|--------|--------|------|
| RAM 条 | 长度 | 133.35 | 0.13335 | m |
| RAM 条 | 宽度 | 30 | 0.030 | m |
| RAM 条 | 高度 | 3.8 | 0.0038 | m |
| RAM 条 | 质量 | 30-50 | 0.04 | kg |
| 插槽 | 长度 | 133 | 0.133 | m |
| 插槽 | 宽度 | 30.5 | 0.0305 | m |
| 插槽 | 深度 | 6 | 0.006 | m |

### 5.2 物理参数

| 参数 | 真实值 | 仿真值 | 说明 |
|------|--------|--------|------|
| RAM 摩擦系数 | 0.3-0.5 | 0.4 | 与插槽接触 |
| 插槽摩擦系数 | 0.2-0.4 | 0.25 | 低摩擦便于插入 |
| RAM 恢复系数 | 0.1-0.2 | 0.1 | 低弹性 |
| 插入力 | 20-30 | - | N（通过接触力实现） |

### 5.3 控制参数

| 参数 | 真实值 | 仿真等效 | 说明 |
|------|--------|----------|------|
| 位置增益 | 2000 | 2000 | N/m |
| 速度增益 | 89 | 89 | N·s/m |
| 旋转增益 | 150-250 | 150-250 | N·m/rad |
| 角速度增益 | 7-9 | 7-9 | N·m·s/rad |

---

## 六、验证方法

### 6.1 几何验证

```python
def verify_geometry():
    """验证几何模型精度"""
    # 1. 测量对象尺寸
    # 2. 验证位置精度
    # 3. 检查碰撞形状
    pass
```

### 6.2 物理验证

```python
def verify_physics():
    """验证物理参数"""
    # 1. 测试自由落体（验证质量）
    # 2. 测试滑动（验证摩擦）
    # 3. 测试插入力（验证接触）
    pass
```

### 6.3 数据分布验证

```python
def verify_data_distribution():
    """验证数据分布一致性"""
    # 1. 收集仿真数据
    # 2. 收集真实数据
    # 3. 对比观察分布、动作分布
    # 4. 使用统计检验（KS test, etc.）
    pass
```

### 6.4 策略迁移验证

```python
def verify_policy_transfer():
    """验证策略迁移效果"""
    # 1. 在仿真中训练策略
    # 2. 在真实环境测试
    # 3. 对比成功率
    # 4. 调整参数直到迁移成功
    pass
```

---

## 七、常见问题与解决方案

### 7.1 插入力不准确

**问题**：仿真中插入太容易或太难

**解决方案**：
1. 调整摩擦系数
2. 添加插入阻力（使用力场或约束）
3. 调整接触刚度

### 7.2 视觉不一致

**问题**：仿真图像与真实图像差异大

**解决方案**：
1. 调整光照强度
2. 匹配材质属性
3. 添加域随机化
4. 使用真实感渲染（如果可用）

### 7.3 控制行为不一致

**问题**：仿真中机器人行为与真实不一致

**解决方案**：
1. 调整控制器增益
2. 添加延迟模拟
3. 添加噪声
4. 使用力控模式（如果支持）

---

## 八、总结

本方案提供了构建高保真度 RAM 插入任务仿真环境的详细步骤。关键要点：

1. **精确几何**：使用标准尺寸创建模型
2. **真实物理**：匹配质量、摩擦、刚度等参数
3. **接触建模**：精确模拟插入过程
4. **控制等效**：使仿真控制行为与真实阻抗控制等效
5. **域随机化**：提高仿真到真实的迁移能力

通过遵循本方案，可以构建出与真实环境高度一致的仿真环境，支持有效的仿真到真实迁移。

---

# RAM 插入任务 - Isaac Sim 仿真环境搭建指南

本文档详细描述了在 NVIDIA Isaac Sim 仿真环境中搭建 RAM 插入任务训练系统的完整流程。与真实机器人环境不同，仿真环境无需物理硬件，但需要正确配置 Isaac Sim 和相关的仿真资源。

## 目录

1. [系统要求](#系统要求)
2. [软件环境安装](#软件环境安装)
3. [Isaac Sim 环境搭建](#isaac-sim-环境搭建)
4. [仿真场景配置](#仿真场景配置)
5. [训练配置适配](#训练配置适配)
6. [训练流程](#训练流程)
7. [常见问题排查](#常见问题排查)

---

## 系统要求

### 硬件要求（计算设备）

1. **GPU**
   - NVIDIA GPU（推荐 RTX 3060 或更高）
   - 显存：至少 8GB（推荐 16GB 或更多）
   - 支持 CUDA 11.8 或更高版本
   - 用于 Isaac Sim 的实时渲染和物理仿真

2. **CPU**
   - 多核 CPU（推荐 8 核或更多）
   - 用于 Python 训练代码和 JAX 计算

3. **内存**
   - 至少 16GB RAM（推荐 32GB 或更多）
   - 用于存储训练数据和模型

4. **存储**
   - 至少 50GB 可用空间
   - 用于 Isaac Sim 安装、模型文件和训练数据

5. **操作系统**
   - Linux（推荐 Ubuntu 20.04/22.04）
   - 或 Windows 10/11（需要 WSL2）

### 软件要求

1. **NVIDIA Isaac Sim**
   - 版本：2023.1.1 或更高（推荐最新稳定版）
   - 需要 NVIDIA Omniverse Launcher

2. **Python 环境**
   - Python 3.10
   - Conda 或 Miniconda

3. **CUDA 和 cuDNN**
   - CUDA 11.8 或更高
   - 对应版本的 cuDNN

---

## 软件环境安装

### 步骤 1：安装 NVIDIA Isaac Sim

1. **下载并安装 NVIDIA Omniverse Launcher**
   - 访问：https://www.nvidia.com/en-us/omniverse/
   - 下载并安装 Omniverse Launcher
   - 创建 NVIDIA 账户（如需要）

2. **通过 Omniverse Launcher 安装 Isaac Sim**
   - 打开 Omniverse Launcher
   - 在 "Exchange" 标签页搜索 "Isaac Sim"
   - 点击 "Install" 安装 Isaac Sim
   - 等待安装完成（可能需要较长时间，取决于网络速度）

3. **验证安装**
   ```bash
   # Isaac Sim 通常安装在以下位置
   ~/.local/share/ov/pkg/isaac_sim-<version>/
   ```

### 步骤 2：设置 Python 环境

1. **创建 Conda 环境**
   ```bash
   conda create -n hilserl python=3.10
   conda activate hilserl
   ```

2. **安装 JAX**
   - 对于 GPU（推荐）：
     ```bash
     pip install --upgrade "jax[cuda12_pip]==0.4.35" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
     ```
   - 对于 CPU（不推荐，速度较慢）：
     ```bash
     pip install --upgrade "jax[cpu]"
     ```

3. **安装 HIL-SERL 核心库**
   ```bash
   cd serl_launcher
   pip install -e .
   pip install -r requirements.txt
   cd ..
   ```

4. **安装 Isaac Sim Python 包**
   ```bash
   # 获取 Isaac Sim 的 Python 路径
   ISAAC_SIM_PATH=~/.local/share/ov/pkg/isaac_sim-<version>
   
   # 安装 Isaac Sim Python 包到当前环境
   cd $ISAAC_SIM_PATH
   ./python.sh -m pip install -e python
   ```

5. **安装其他依赖**
   ```bash
   pip install gymnasium
   pip install numpy scipy
   pip install opencv-python
   pip install pyyaml
   ```

### 步骤 3：安装 Isaac Sim 扩展（如需要）

如果需要使用 Isaac Sim 的特定功能或扩展：

1. **通过 Omniverse Launcher 安装扩展**
   - 在 "Exchange" 中搜索相关扩展
   - 例如：Isaac Lab、Isaac Manipulation 等

2. **或通过代码安装**
   ```python
   # 在 Isaac Sim 脚本中
   import omni.ext
   # 加载所需扩展
   ```

---

## Isaac Sim 环境搭建

### 步骤 4：创建仿真场景

1. **启动 Isaac Sim**
   ```bash
   # 方法 1：通过 Omniverse Launcher 启动
   # 在 Launcher 中点击 Isaac Sim 的 "Launch" 按钮
   
   # 方法 2：通过命令行启动
   ~/.local/share/ov/pkg/isaac_sim-<version>/isaac-sim.sh
   ```

2. **创建新场景**
   - 在 Isaac Sim 中：`File > New Stage`
   - 或使用代码创建场景（见下方）

3. **添加 Franka 机器人**
   ```python
   # 在 Isaac Sim Python 脚本中
   from omni.isaac.core import World
   from omni.isaac.franka import Franka
   
   # 创建世界
   world = World(stage_units_in_meters=1.0)
   
   # 添加 Franka 机器人
   franka = world.scene.add(
       Franka(
           prim_path="/World/franka",
           name="franka",
           position=np.array([0, 0, 0]),
       )
   )
   ```

### 步骤 5：添加任务对象

1. **添加主板模型**
   ```python
   # 导入主板 3D 模型
   # 方法 1：从 USD 文件导入
   from omni.isaac.core.utils.stage import add_reference_to_stage
   
   motherboard_usd_path = "/path/to/motherboard.usd"
   add_reference_to_stage(
       usd_path=motherboard_usd_path,
       prim_path="/World/motherboard"
   )
   
   # 方法 2：使用基本几何体创建简化模型
   from omni.isaac.core.utils.prims import create_prim
   create_prim(
       prim_path="/World/motherboard",
       prim_type="Xform",
       position=np.array([0.588, -0.036, 0.25])  # 对应 TARGET_POSE 位置
   )
   ```

2. **添加 RAM 条模型**
   ```python
   # 导入 RAM 条 3D 模型
   ram_usd_path = "/path/to/ram_stick.usd"
   add_reference_to_stage(
       usd_path=ram_usd_path,
       prim_path="/World/ram_stick"
   )
   
   # 设置初始位置（对应 GRASP_POSE）
   ram_prim = create_prim(
       prim_path="/World/ram_stick",
       position=np.array([0.586, -0.220, 0.273])  # GRASP_POSE 位置
   )
   ```

3. **添加 RAM 支架**
   ```python
   # 创建支架（可以使用基本几何体）
   from omni.isaac.core.utils.prims import create_prim
   create_prim(
       prim_path="/World/ram_holder",
       prim_type="Cube",
       position=np.array([0.586, -0.220, 0.268]),  # GRASP_POSE 下方
       scale=np.array([0.05, 0.02, 0.01])
   )
   ```

### 步骤 6：配置物理属性

1. **设置碰撞属性**
   ```python
   from omni.physx import get_physx_interface
   
   # 为对象添加碰撞形状
   # 确保 RAM 条和主板插槽有正确的碰撞检测
   ```

2. **配置物理材质**
   ```python
   # 设置摩擦系数、弹性等物理属性
   # 确保插入过程有真实的物理反馈
   ```

3. **设置重力和其他物理参数**
   ```python
   world.set_physics_dt(1.0/60.0)  # 60 Hz 物理更新
   world.scene.add_default_ground_plane()
   ```

### 步骤 7：配置相机

1. **添加手腕相机**
   ```python
   from omni.isaac.sensor import Camera
   
   # 相机 1：安装在手腕上
   camera1 = world.scene.add(
       Camera(
           prim_path="/World/franka/panda_hand/camera1",
           name="wrist_1",
           position=np.array([0, 0, 0.05]),  # 相对于手腕的偏移
           resolution=(1280, 720),
       )
   )
   
   # 相机 2：安装在手腕上（不同角度）
   camera2 = world.scene.add(
       Camera(
           prim_path="/World/franka/panda_hand/camera2",
           name="wrist_2",
           position=np.array([0, 0.05, 0]),
           resolution=(1280, 720),
       )
   )
   ```

2. **配置相机参数**
   ```python
   # 设置相机内参、视野角等
   camera1.set_focal_length(24.0)  # 根据实际相机调整
   camera1.set_horizontal_aperture(20.955)  # mm
   ```

---

## 仿真场景配置

### 步骤 8：创建 Gym 环境接口

1. **创建 Isaac Sim 环境包装器**
   
   需要创建一个包装器，将 Isaac Sim 的接口适配到 HIL-SERL 的 Gym 环境接口：

   ```python
   # 新建文件：serl_robot_infra/franka_env/envs/isaac_sim_env.py
   import gymnasium as gym
   import numpy as np
   from omni.isaac.core import World
   from omni.isaac.franka import Franka
   
   class IsaacSimFrankaEnv(gym.Env):
       def __init__(self, config):
           self.config = config
           self.world = World(stage_units_in_meters=1.0)
           
           # 初始化机器人、对象、相机等
           self._setup_scene()
           
           # 定义动作和观察空间
           self.action_space = gym.spaces.Box(
               low=-1, high=1, shape=(7,), dtype=np.float32
           )
           self.observation_space = self._get_observation_space()
           
       def _setup_scene(self):
           # 添加机器人、对象等
           pass
           
       def reset(self, **kwargs):
           # 重置环境
           self.world.reset()
           obs = self._get_obs()
           return obs, {}
           
       def step(self, action):
           # 执行动作
           self._apply_action(action)
           self.world.step(render=True)
           obs = self._get_obs()
           reward = self._compute_reward(obs)
           done = self._check_done(obs)
           return obs, reward, done, False, {}
   ```

2. **实现关键方法**
   - `_apply_action()`: 将动作转换为机器人控制命令
   - `_get_obs()`: 获取观察（包括相机图像和状态）
   - `_compute_reward()`: 计算奖励（可以使用奖励分类器）
   - `_check_done()`: 检查是否完成

### 步骤 9：配置环境参数

1. **位姿配置**
   
   在仿真环境中，位姿配置与真实环境相同：

   ```python
   # 在 config.py 中
   TARGET_POSE = np.array([0.588, -0.036, 0.278, np.pi, 0, 0])
   GRASP_POSE = np.array([0.586, -0.220, 0.273, np.pi, 0, 0])
   RESET_POSE = TARGET_POSE + np.array([0, 0, 0.05, 0, 0.05, 0])
   ABS_POSE_LIMIT_LOW = TARGET_POSE - np.array([0.03, 0.02, 0.01, 0.01, 0.1, 0.4])
   ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array([0.03, 0.02, 0.05, 0.01, 0.1, 0.4])
   ```

2. **相机配置**
   
   仿真环境中的相机配置：

   ```python
   # 不需要真实的序列号，使用虚拟相机
   REALSENSE_CAMERAS = {
       "wrist_1": {
           "serial_number": "virtual_camera_1",  # 虚拟标识
           "dim": (1280, 720),
           "exposure": 40000,
       },
       "wrist_2": {
           "serial_number": "virtual_camera_2",
           "dim": (1280, 720),
           "exposure": 40000,
       },
   }
   ```

3. **移除真实硬件相关配置**
   - 不需要 `SERVER_URL`（没有 Flask 服务器）
   - 不需要 `COMPLIANCE_PARAM` 和 `PRECISION_PARAM`（Isaac Sim 使用自己的物理引擎）
   - 不需要 `LOAD_PARAM`（末端执行器质量在 Isaac Sim 中配置）

---

## 训练配置适配

### 步骤 10：修改训练配置

1. **创建仿真环境配置**
   
   在 `examples/experiments/ram_insertion/config.py` 中：

   ```python
   class IsaacSimEnvConfig(DefaultEnvConfig):
       # 移除 SERVER_URL
       # SERVER_URL = "http://127.0.0.2:5000/"  # 不需要
       
       # 使用虚拟相机
       REALSENSE_CAMERAS = {
           "wrist_1": {
               "serial_number": "virtual_camera_1",
               "dim": (1280, 720),
               "exposure": 40000,
           },
           "wrist_2": {
               "serial_number": "virtual_camera_2",
               "dim": (1280, 720),
               "exposure": 40000,
           },
       }
       
       # 位姿配置（与真实环境相同）
       TARGET_POSE = np.array([0.588, -0.036, 0.278, np.pi, 0, 0])
       GRASP_POSE = np.array([0.586, -0.220, 0.273, np.pi, 0, 0])
       RESET_POSE = TARGET_POSE + np.array([0, 0, 0.05, 0, 0.05, 0])
       ABS_POSE_LIMIT_LOW = TARGET_POSE - np.array([0.03, 0.02, 0.01, 0.01, 0.1, 0.4])
       ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array([0.03, 0.02, 0.05, 0.01, 0.1, 0.4])
       
       RANDOM_RESET = True
       RANDOM_XY_RANGE = 0.02
       RANDOM_RZ_RANGE = 0.05
       ACTION_SCALE = (0.01, 0.06, 1)
       DISPLAY_IMAGE = True
       MAX_EPISODE_LENGTH = 100
   ```

2. **修改环境获取函数**
   
   ```python
   class TrainConfig(DefaultTrainingConfig):
       # ... 其他配置 ...
       
       def get_environment(self, fake_env=True, save_video=False, classifier=False):
           # 注意：fake_env=True 表示使用仿真环境
           env = IsaacSimRAMEnv(  # 使用 Isaac Sim 环境
               fake_env=True,
               save_video=save_video,
               config=IsaacSimEnvConfig(),
           )
           # ... 其他包装器 ...
           return env
   ```

### 步骤 11：适配训练脚本

1. **修改训练脚本以支持 Isaac Sim**
   
   在训练脚本中，确保使用 `fake_env=True`：

   ```python
   # 在 train_rlpd.py 或类似脚本中
   env = config.get_environment(
       fake_env=True,  # 使用仿真环境
       save_video=FLAGS.save_video,
       classifier=True,
   )
   ```

2. **移除 SpaceMouse 干预（可选）**
   
   在仿真环境中，可以：
   - 保留 SpaceMouse 支持（如果已连接）
   - 或使用键盘/鼠标进行干预
   - 或完全自动化训练（无干预）

---

## 训练流程

### 步骤 12：奖励分类器训练（仿真环境）

1. **收集分类器数据**
   ```bash
   cd examples
   python record_success_fail.py --exp_name ram_insertion --successes_needed 200
   ```
   
   **注意**：在仿真环境中，可以通过程序化方式生成更多数据，而无需手动操作。

2. **训练分类器**
   ```bash
   cd examples/experiments/ram_insertion
   python ../../train_reward_classifier.py --exp_name ram_insertion
   ```

### 步骤 13：演示数据收集（仿真环境）

1. **录制演示**
   ```bash
   cd examples
   python record_demos.py --exp_name ram_insertion --successes_needed 20
   ```
   
   **注意**：在仿真环境中，可以使用程序化演示生成，或使用 SpaceMouse（如果已连接）。

### 步骤 14：策略训练

1. **启动训练**
   ```bash
   # 终端 1：Actor
   cd examples/experiments/ram_insertion
   bash run_actor.sh
   
   # 终端 2：Learner
   bash run_learner.sh
   ```

2. **监控训练**
   - 在 Isaac Sim 窗口中观察机器人行为
   - 查看训练日志和指标
   - 根据需要调整参数

---

## 仿真环境的优势与注意事项

### 优势

1. **无需物理硬件**
   - 不需要真实的 Franka 机器人
   - 不需要 RealSense 相机
   - 不需要 SpaceMouse（可选）

2. **可扩展性**
   - 可以并行运行多个仿真实例
   - 可以快速重置和重试
   - 可以程序化生成大量训练数据

3. **安全性**
   - 不会损坏真实硬件
   - 可以测试危险操作
   - 可以快速迭代

4. **可重复性**
   - 完全确定性的仿真（如需要）
   - 可以保存和加载场景状态
   - 便于调试和复现

### 注意事项

1. **仿真到真实的差距（Sim-to-Real Gap）**
   - 物理参数可能与真实环境不同
   - 视觉渲染可能与真实相机不同
   - 需要域随机化来提高泛化能力

2. **性能要求**
   - 需要强大的 GPU 进行实时渲染
   - 多实例并行训练需要更多资源

3. **模型准备**
   - 需要准备 3D 模型（主板、RAM 条等）
   - 需要正确配置物理属性
   - 需要调整碰撞检测

4. **调试复杂性**
   - Isaac Sim 的调试可能比真实环境复杂
   - 需要熟悉 Isaac Sim 的 API 和工具

---

## 常见问题排查

### 问题 1：Isaac Sim 无法启动

**解决方案**：
- 检查 GPU 驱动是否最新
- 确认 CUDA 版本兼容
- 检查系统资源（内存、显存）

### 问题 2：Python 环境找不到 Isaac Sim 模块

**解决方案**：
```bash
# 确保正确安装 Isaac Sim Python 包
ISAAC_SIM_PATH=~/.local/share/ov/pkg/isaac_sim-<version>
cd $ISAAC_SIM_PATH
./python.sh -m pip install -e python

# 在代码中设置路径
import sys
sys.path.append("$ISAAC_SIM_PATH")
```

### 问题 3：仿真速度太慢

**解决方案**：
- 降低渲染质量
- 减少物理更新频率
- 使用无头模式（headless mode）
- 优化场景复杂度

### 问题 4：物理行为不真实

**解决方案**：
- 检查碰撞检测设置
- 调整物理材质参数
- 验证模型尺寸和比例
- 调整物理引擎参数

### 问题 5：相机图像质量差

**解决方案**：
- 调整相机内参
- 改善场景光照
- 调整渲染设置
- 使用域随机化

---

## 总结

在 Isaac Sim 中搭建 RAM 插入任务训练系统的关键步骤：

1. ✅ 安装 Isaac Sim 和 Python 环境
2. ✅ 创建仿真场景（机器人、对象、相机）
3. ✅ 配置物理属性和碰撞检测
4. ✅ 创建 Gym 环境接口适配器
5. ✅ 修改训练配置以使用仿真环境
6. ✅ 进行奖励分类器训练
7. ✅ 收集演示数据
8. ✅ 启动策略训练

通过仿真环境，可以在没有物理硬件的情况下进行训练，并可以快速迭代和实验。

---

## 相关资源

- [NVIDIA Isaac Sim 官方文档](https://docs.isaacsim.omniverse.nvidia.com/)
- [Isaac Sim GitHub](https://github.com/NVIDIA-Omniverse/IsaacSim)
- [HIL-SERL 项目 README](../README.md)
- [真实机器人训练指南](./ram_insertion_training_guide.md)

---

# Isaac Sim 力/力矩传感器实现方案

## 一、原项目力/力矩传感器分析

### 1.1 传感器数量
- **使用数量**：**1个**力/力矩测量
- **数据格式**：6维向量（3维力 + 3维力矩）
- **数据来源**：`msg.K_F_ext_hat_K`（Franka ROS 消息）

### 1.2 测量位置
- **位置**：**末端执行器（end-effector）**
- **坐标系**：**刚度坐标系（Stiffness Frame）**，通常与末端执行器坐标系一致
- **含义**：`K_F_ext_hat_K` 表示在刚度坐标系中估计的外部力/力矩

### 1.3 数据来源机制
- **Franka 机器人没有直接的力/力矩传感器**
- 通过**关节力矩传感器**（每个关节都有）+ **动力学模型**计算得出
- Franka 控制器通过 `franka_state_controller/franka_states` 话题发布
- 这是 Franka 的内置功能，通过关节力矩反推末端执行器处的力/力矩

### 1.4 代码位置
```python
# franka_server.py 第 166-167 行
self.force = np.array(list(msg.K_F_ext_hat_K)[:3])   # 前3维：力 [fx, fy, fz]
self.torque = np.array(list(msg.K_F_ext_hat_K)[3:])  # 后3维：力矩 [tx, ty, tz]
```

### 1.5 在训练中的使用
- **观察空间**：包含 `tcp_force` 和 `tcp_torque`（3维向量各一个）
- **网络处理**：通过 `SERLObsWrapper` 展平，通过 `EncodingWrapper` 投影到潜在空间（64维）
- **实际使用**：所有 SAC agent 都设置了 `use_proprio=True`，力/力矩数据会输入到 Actor 和 Critic 网络

---

## 二、Isaac Sim 实现方案

### 2.1 方案对比

#### 方案 A：使用接触力计算
- **原理**：从接触力计算末端执行器处的净力/力矩
- **优点**：直接反映接触情况
- **缺点**：需要启用接触报告，实现复杂

#### 方案 B：使用 Isaac Sim 力/力矩传感器 API
- **原理**：在末端执行器上添加力/力矩传感器
- **优点**：API 直接，使用简单
- **缺点**：需要确认 Isaac Sim 是否支持，可能增加计算开销

#### 方案 C：从关节力矩反推（**推荐**）
- **原理**：使用关节力矩和雅可比矩阵计算末端执行器力/力矩
- **优点**：
  1. 与 Franka 的实现方式一致（从关节力矩计算）
  2. 实现简单，不需要额外的传感器设置
  3. 精度较高，利用已有的雅可比矩阵
  4. 兼容性好，适用于大多数 Isaac Sim 版本
- **缺点**：需要确保关节力矩数据可用

### 2.2 方案 C 实现原理

#### 数学原理
使用虚功原理（Virtual Work Principle）：
- 关节力矩：`τ = [τ1, τ2, ..., τ7]`（7维）
- 雅可比矩阵：`J`（6×7，从关节空间到末端执行器空间）
- 末端执行器力/力矩：`wrench = [fx, fy, fz, tx, ty, tz]`（6维）

关系式：
```
τ = J^T * wrench
```

因此，从关节力矩反推末端执行器力/力矩：
```
wrench = (J^T)^+ * τ
```
其中 `(J^T)^+` 是 `J^T` 的伪逆（Moore-Penrose pseudoinverse）。

#### 实现步骤
1. 获取关节力矩：`tau = franka.get_applied_joint_actions()` 或 `get_joint_efforts()`
2. 获取雅可比矩阵：`jacobian = self.state_cache["jacobian"]`（已在 `_update_state_fast()` 中计算）
3. 计算伪逆：`jacobian_T_pinv = np.linalg.pinv(jacobian.T)`
4. 计算力/力矩：`wrench = jacobian_T_pinv @ tau`
5. 分离力和力矩：`force = wrench[:3]`, `torque = wrench[3:]`

---

## 三、实现代码

### 3.1 在 `isaac_sim_server.py` 中添加方法

```python
def _compute_end_effector_wrench(self):
    """
    计算末端执行器处的力/力矩（在末端执行器坐标系中）
    
    方法：从关节力矩反推（与 Franka 实现一致）
    原理：使用虚功原理，wrench = (J^T)^+ * tau
    
    Returns:
        force: np.ndarray[3] - 末端执行器处的力 [fx, fy, fz]
        torque: np.ndarray[3] - 末端执行器处的力矩 [tx, ty, tz]
    """
    try:
        # 1. 获取关节力矩
        tau = None
        
        # 方法1：尝试使用 get_applied_joint_actions()
        if hasattr(self.franka, 'get_applied_joint_actions'):
            try:
                tau = self.franka.get_applied_joint_actions()
            except:
                pass
        
        # 方法2：尝试使用 get_joint_efforts()
        if tau is None and hasattr(self.franka, 'get_joint_efforts'):
            try:
                tau = self.franka.get_joint_efforts()
            except:
                pass
        
        # 方法3：尝试从关节状态获取
        if tau is None and hasattr(self.franka, 'get_joint_states'):
            try:
                joint_states = self.franka.get_joint_states()
                if hasattr(joint_states, 'effort'):
                    tau = np.array(joint_states.effort)
            except:
                pass
        
        # 如果无法获取关节力矩，返回零值
        if tau is None or len(tau) != 7:
            return np.zeros(3), np.zeros(3)
        
        # 2. 获取雅可比矩阵
        jacobian = self.state_cache.get("jacobian")
        if jacobian is None or jacobian.shape != (6, 7):
            return np.zeros(3), np.zeros(3)
        
        # 3. 计算伪逆
        jacobian_T = jacobian.T  # (7, 6)
        jacobian_T_pinv = np.linalg.pinv(jacobian_T)  # (6, 7)
        
        # 4. 计算末端执行器力/力矩
        wrench = jacobian_T_pinv @ tau  # (6,)
        
        # 5. 分离力和力矩
        force = wrench[:3]   # [fx, fy, fz]
        torque = wrench[3:]  # [tx, ty, tz]
        
        return force, torque
        
    except Exception as e:
        print(f"[WARNING] Failed to compute end-effector wrench: {e}")
        return np.zeros(3), np.zeros(3)
```

### 3.2 在 `_update_state_fast()` 中调用

```python
def _update_state_fast(self):
    """
    快速更新状态（不加锁，减少延迟）
    """
    try:
        # ... 现有代码（更新 pose, q, dq, jacobian） ...
        
        # 计算末端执行器处的力/力矩
        force, torque = self._compute_end_effector_wrench()
        self.state_cache["force"] = force
        self.state_cache["torque"] = torque
        
    except Exception as e:
        print(f"[WARNING] Error in _update_state_fast: {e}")
```

---

## 四、验证和测试

### 4.1 验证清单
- [ ] 关节力矩数据可用（`tau` 不为 None 且长度为 7）
- [ ] 雅可比矩阵正确计算（形状为 (6, 7)）
- [ ] 力/力矩值在合理范围内（通常 < 100N, < 10Nm）
- [ ] 无接触时力/力矩接近零
- [ ] 有接触时力/力矩反映接触情况

### 4.2 测试方法
1. **静态测试**：机器人静止时，力/力矩应该接近零
2. **接触测试**：末端执行器接触物体时，应该检测到非零力/力矩
3. **对比测试**：与真实 Franka 机器人的力/力矩数据对比（如果可用）

### 4.3 可能的问题和解决方案

#### 问题1：无法获取关节力矩
- **原因**：Isaac Sim API 可能不同
- **解决**：尝试多种方法获取关节力矩，或使用接触力计算作为备选

#### 问题2：力/力矩值异常大
- **原因**：关节力矩单位可能不匹配，或雅可比矩阵计算错误
- **解决**：检查单位转换，验证雅可比矩阵计算

#### 问题3：力/力矩始终为零
- **原因**：关节力矩获取失败，或关节力矩本身为零
- **解决**：检查关节力矩获取方法，确认机器人是否在运动

---

## 五、参考文档

- Franka ROS 文档：`K_F_ext_hat_K` 的含义和计算方式
- Isaac Sim 文档：关节力矩获取 API
- 虚功原理：机器人学基础理论

---

## 六、后续优化

1. **添加滤波**：对力/力矩数据进行低通滤波，减少噪声
2. **单位转换**：确保力/力矩单位与真实环境一致（N, Nm）
3. **坐标系转换**：如果需要，可以转换到其他坐标系
4. **性能优化**：缓存伪逆矩阵，减少计算开销
