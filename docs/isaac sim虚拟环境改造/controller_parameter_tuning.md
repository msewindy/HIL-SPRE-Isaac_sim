# 控制器参数调优方案（可选）

## 一、问题理解

### 1.1 核心目标
**减少仿真环境和真实环境的动力学特性差异**，使仿真环境的控制行为与真实阻抗控制等效。

### 1.2 问题背景
- **真实环境**：使用**阻抗控制（Impedance Control）**，有明确的刚度（stiffness）和阻尼（damping）参数
- **仿真环境**：使用 **RMPFlowController**（Isaac Sim 推荐的控制器），但当前使用默认参数
- **目标**：调优 RMPFlowController 的参数，使其行为与真实阻抗控制一致

### 1.3 优先级说明

**⚠️ 注意：此任务为可选优化项**

根据项目使用场景：
- **仿真环境目的**：低成本验证项目过程可实施性（训练流程、数据收集、策略训练）
- **不进行策略迁移**：仿真中训练的策略不会直接用于真实环境
- **真实环境会重新训练**：真实场景会重新运行原项目的完整训练过程

**因此，控制器参数调优可以降低优先级或暂时忽略。**

#### 建议的处理方式：
1. **优先使用默认参数**：RMPFlowController 的默认参数通常已经足够好
2. **基本验证即可**：确保控制器能正常工作（机器人能响应命令、运动平滑）
3. **仅在必要时调优**：如果默认参数导致异常行为（抖动、不稳定），再进行简单调整
4. **不追求完全一致**：不需要与真实环境的阻抗参数完全匹配

### 1.4 为什么重要？（仅当需要策略迁移时）
1. **Sim-to-Real 迁移**：如果仿真和真实的控制行为不一致，在仿真中训练的策略可能无法直接迁移到真实环境
2. **训练数据一致性**：仿真和真实的观察分布、动作分布应该相似
3. **任务成功率**：对于精确操作任务（如 RAM 插入），控制行为的微小差异可能导致任务失败

**注意**：如果不需要策略迁移，这些因素不影响验证流程的可行性。

---

## 二、真实环境阻抗控制参数

### 2.1 RAM 插入任务参数

#### COMPLIANCE_PARAM（默认/抓取阶段）
```python
COMPLIANCE_PARAM = {
    "translational_stiffness": 2000,  # N/m - 平移刚度
    "translational_damping": 89,      # N·s/m - 平移阻尼
    "rotational_stiffness": 150,      # N·m/rad - 旋转刚度
    "rotational_damping": 7,          # N·m·s/rad - 旋转阻尼
    "translational_Ki": 0,            # 积分项（通常为0）
    "rotational_Ki": 0,               # 积分项（通常为0）
    # 位置限制（clip）
    "translational_clip_x": 0.0075,
    "translational_clip_y": 0.0016,
    "translational_clip_z": 0.0055,
    # ... 其他 clip 参数
}
```

#### PRECISION_PARAM（精确操作阶段）
```python
PRECISION_PARAM = {
    "translational_stiffness": 2000,  # N/m - 与 COMPLIANCE 相同
    "translational_damping": 89,      # N·s/m - 与 COMPLIANCE 相同
    "rotational_stiffness": 250,      # N·m/rad - 更高的旋转刚度
    "rotational_damping": 9,          # N·m·s/rad - 更高的旋转阻尼
    "translational_Ki": 0.0,
    "rotational_Ki": 0.0,
    # 位置限制（更宽松）
    "translational_clip_x": 0.1,
    "translational_clip_y": 0.1,
    "translational_clip_z": 0.1,
    # ... 其他 clip 参数
}
```

### 2.2 参数含义

#### 刚度（Stiffness）
- **平移刚度**：2000 N/m - 末端执行器在位置误差上的恢复力系数
- **旋转刚度**：150-250 N·m/rad - 末端执行器在姿态误差上的恢复力矩系数
- **物理意义**：刚度越大，机器人对位置/姿态误差的响应越强

#### 阻尼（Damping）
- **平移阻尼**：89 N·s/m - 末端执行器在速度上的阻尼系数
- **旋转阻尼**：7-9 N·m·s/rad - 末端执行器在角速度上的阻尼系数
- **物理意义**：阻尼越大，机器人运动越平滑，但响应可能变慢

#### 位置限制（Clip）
- **作用**：限制位置误差的最大值，防止过度响应
- **COMPLIANCE_PARAM**：较小的限制（毫米级），用于精确控制
- **PRECISION_PARAM**：较大的限制（厘米级），用于快速移动

---

## 三、Isaac Sim RMPFlowController 分析

### 3.1 RMPFlowController 简介
- **RMPFlow**：Riemannian Motion Policy（黎曼运动策略）
- **特点**：基于几何的轨迹规划和控制方法
- **优势**：处理复杂约束、避免碰撞、平滑轨迹
- **参数**：通常包括目标权重、障碍物权重、速度限制等

### 3.2 当前实现
```python
self.controller = RMPFlowController(
    name="franka_controller",
    robot_articulation=self.franka,
    end_effector_prim_path=ee_prim_path,
)
```
**问题**：使用默认参数，可能与真实阻抗控制行为不一致

### 3.3 RMPFlowController 可配置参数
根据 Isaac Sim 文档，RMPFlowController 可能支持以下参数：
- `target_attraction_gain`：目标吸引力增益（类似刚度）
- `damping_gain`：阻尼增益
- `obstacle_repulsion_gain`：障碍物排斥增益
- `max_speed`：最大速度限制
- `max_acceleration`：最大加速度限制
- `smoothing_factor`：平滑因子

**注意**：具体参数名称和可用性取决于 Isaac Sim 版本，需要查阅对应版本的文档。

---

## 四、实现方案

### 4.1 方案 A：直接配置 RMPFlowController 参数（推荐）

如果 RMPFlowController 支持刚度/阻尼参数，直接配置：

```python
def _setup_controller(self, impedance_params=None):
    """
    设置机器人控制器，配置阻抗控制参数
    
    Args:
        impedance_params: Dict - 阻抗控制参数（可选）
            {
                "translational_stiffness": 2000,
                "translational_damping": 89,
                "rotational_stiffness": 150,
                "rotational_damping": 7,
            }
    """
    if impedance_params is None:
        # 使用默认参数（COMPLIANCE_PARAM）
        impedance_params = {
            "translational_stiffness": 2000,
            "translational_damping": 89,
            "rotational_stiffness": 150,
            "rotational_damping": 7,
        }
    
    try:
        from omni.isaac.manipulators.controllers import RMPFlowController
        
        # 尝试配置参数（根据实际 API 调整）
        controller_kwargs = {
            "name": "franka_controller",
            "robot_articulation": self.franka,
            "end_effector_prim_path": ee_prim_path,
        }
        
        # 尝试映射阻抗参数到 RMPFlowController 参数
        # 注意：参数名称可能不同，需要根据实际 API 调整
        if hasattr(RMPFlowController, 'set_stiffness'):
            # 如果支持直接设置刚度
            controller_kwargs["stiffness"] = impedance_params["translational_stiffness"]
        elif hasattr(RMPFlowController, 'target_attraction_gain'):
            # 如果使用吸引力增益（可能需要缩放）
            controller_kwargs["target_attraction_gain"] = impedance_params["translational_stiffness"] / 100.0
        
        if hasattr(RMPFlowController, 'set_damping'):
            controller_kwargs["damping"] = impedance_params["translational_damping"]
        elif hasattr(RMPFlowController, 'damping_gain'):
            controller_kwargs["damping_gain"] = impedance_params["translational_damping"] / 10.0
        
        self.controller = RMPFlowController(**controller_kwargs)
        
        # 如果创建后可以设置参数
        if hasattr(self.controller, 'set_stiffness'):
            self.controller.set_stiffness(
                translational=impedance_params["translational_stiffness"],
                rotational=impedance_params["rotational_stiffness"]
            )
        if hasattr(self.controller, 'set_damping'):
            self.controller.set_damping(
                translational=impedance_params["translational_damping"],
                rotational=impedance_params["rotational_damping"]
            )
        
        print(f"[INFO] Initialized RMPFlowController with impedance parameters")
        print(f"  Translational: K={impedance_params['translational_stiffness']}, D={impedance_params['translational_damping']}")
        print(f"  Rotational: K={impedance_params['rotational_stiffness']}, D={impedance_params['rotational_damping']}")
        
    except Exception as e:
        print(f"[WARNING] Failed to configure RMPFlowController with impedance params: {e}")
        # 回退到默认配置
        # ...
```

### 4.2 方案 B：使用自定义阻抗控制器

如果 RMPFlowController 不支持直接配置阻抗参数，实现自定义控制器：

```python
class ImpedanceController:
    """
    自定义阻抗控制器，模拟真实环境的阻抗控制行为
    """
    def __init__(self, franka, impedance_params):
        self.franka = franka
        self.stiffness_trans = impedance_params["translational_stiffness"]
        self.damping_trans = impedance_params["translational_damping"]
        self.stiffness_rot = impedance_params["rotational_stiffness"]
        self.damping_rot = impedance_params["rotational_damping"]
        
        self.target_pose = None
        self.target_velocity = np.zeros(6)
    
    def set_target_pose(self, position, orientation):
        """设置目标位姿"""
        self.target_pose = np.concatenate([position, orientation])
        self.target_velocity = np.zeros(6)
    
    def compute(self):
        """
        计算关节目标（基于阻抗控制）
        
        原理：
        1. 计算位置/姿态误差
        2. 计算速度误差
        3. 使用 PD 控制：tau = K * error + D * velocity_error
        4. 通过雅可比矩阵转换为关节力矩
        5. 转换为关节位置目标
        """
        # 获取当前状态
        current_pose = self._get_current_pose()
        current_velocity = self._get_current_velocity()
        jacobian = self._get_jacobian()
        
        # 计算误差
        pose_error = self.target_pose - current_pose
        velocity_error = self.target_velocity - current_velocity
        
        # 分离平移和旋转误差
        trans_error = pose_error[:3]
        rot_error = pose_error[3:]
        trans_vel_error = velocity_error[:3]
        rot_vel_error = velocity_error[3:]
        
        # 计算力/力矩（PD 控制）
        force = self.stiffness_trans * trans_error + self.damping_trans * trans_vel_error
        torque = self.stiffness_rot * rot_error + self.damping_rot * rot_vel_error
        wrench = np.concatenate([force, torque])
        
        # 转换为关节力矩：tau = J^T * wrench
        joint_torques = jacobian.T @ wrench
        
        # 转换为关节位置目标（简化：使用当前位置 + 增量）
        # 注意：这需要根据实际需求调整
        current_joint_pos = self.franka.get_joint_positions()
        joint_vel_target = np.linalg.pinv(jacobian) @ velocity_error
        joint_pos_target = current_joint_pos + joint_vel_target * dt
        
        return joint_pos_target
```

### 4.3 方案 C：参数映射和调优

如果 RMPFlowController 的参数与阻抗参数不完全对应，需要建立映射关系：

```python
def map_impedance_to_rmpflow_params(impedance_params):
    """
    将阻抗控制参数映射到 RMPFlowController 参数
    
    注意：这是经验映射，需要通过实验验证和调优
    """
    rmpflow_params = {}
    
    # 平移刚度 -> 目标吸引力增益
    # 经验公式：gain = stiffness / scale_factor
    rmpflow_params["target_attraction_gain"] = impedance_params["translational_stiffness"] / 100.0
    
    # 平移阻尼 -> 阻尼增益
    rmpflow_params["damping_gain"] = impedance_params["translational_damping"] / 10.0
    
    # 旋转刚度 -> 旋转吸引力增益（如果支持）
    if "rotational_stiffness" in impedance_params:
        rmpflow_params["rotational_attraction_gain"] = impedance_params["rotational_stiffness"] / 10.0
    
    # 旋转阻尼 -> 旋转阻尼增益（如果支持）
    if "rotational_damping" in impedance_params:
        rmpflow_params["rotational_damping_gain"] = impedance_params["rotational_damping"] / 1.0
    
    return rmpflow_params
```

---

## 五、验证和调优方法

### 5.1 对比测试

#### 测试 1：阶跃响应
- **真实环境**：发送阶跃位置命令，记录位置、速度、力/力矩响应
- **仿真环境**：发送相同命令，记录相同数据
- **对比**：响应时间、超调量、稳态误差

#### 测试 2：正弦跟踪
- **真实环境**：发送正弦位置命令，记录跟踪误差
- **仿真环境**：发送相同命令，记录跟踪误差
- **对比**：相位延迟、幅值误差

#### 测试 3：接触响应
- **真实环境**：末端执行器接触物体，记录力/力矩响应
- **仿真环境**：相同场景，记录力/力矩响应
- **对比**：接触力大小、响应速度

### 5.2 参数调优流程

1. **初始参数**：使用真实环境的阻抗参数作为初始值
2. **单参数扫描**：逐个调整参数，观察对行为的影响
3. **多参数优化**：使用优化算法（如贝叶斯优化）寻找最优参数
4. **验证**：在多个测试场景中验证参数效果

### 5.3 评估指标

- **位置跟踪误差**：RMS 误差、最大误差
- **速度响应**：响应时间、超调量
- **力/力矩响应**：接触力大小、响应时间
- **任务成功率**：在 RAM 插入任务中的成功率

---

## 六、实施步骤

### 步骤 1：查阅 Isaac Sim 文档
- 确认 RMPFlowController 支持的参数
- 了解参数的含义和取值范围
- 查看示例代码

### 步骤 2：实现参数配置
- 在 `_setup_controller()` 中添加参数配置
- 支持从配置文件读取参数
- 添加参数验证

### 步骤 3：对比测试
- 实现测试脚本
- 在真实和仿真环境中运行相同测试
- 收集数据并对比

### 步骤 4：参数调优
- 根据对比结果调整参数
- 迭代优化直到行为一致

### 步骤 5：任务验证
- 在 RAM 插入任务中验证
- 对比任务成功率
- 必要时进一步调优

---

## 七、注意事项

### 7.1 参数单位
- **真实环境**：N/m, N·s/m, N·m/rad, N·m·s/rad
- **仿真环境**：可能使用不同的单位或归一化值
- **需要转换**：确保单位一致

### 7.2 参数范围
- RMPFlowController 的参数可能有取值范围限制
- 需要将真实环境的参数映射到有效范围内

### 7.3 版本差异
- 不同版本的 Isaac Sim 可能有不同的 API
- 需要根据实际版本调整代码

### 7.4 性能影响
- 某些参数可能影响仿真性能
- 需要在准确性和性能之间平衡

---

## 八、参考文档

- Isaac Sim RMPFlowController 文档
- 阻抗控制原理：机器人学教材
- 真实环境参数：`examples/experiments/ram_insertion/config.py`
- 控制等效性要求：`docs/isaac sim虚拟环境改造/ram_insertion_sim_setup_plan.md`
