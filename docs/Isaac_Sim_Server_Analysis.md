# Isaac Sim Server Logic Analysis

本文档详细梳理了 `serl_robot_infra/robot_servers/isaac_sim_server.py` 的逻辑，并针对用户关于分辨率、裁剪和启动参数的问题进行了解答。

## 1. 用户问题解答 (Q&A)

### Q1: `config = {"width": width, "height": height}` 中的 `width`/`height` 有什么用？
**回答**: 
这两个参数（默认 1280x720）用于设置 **Isaac Sim 仿真器本身的主窗口分辨率**（Viewport）。
*   它们决定了你在屏幕上看到的 Isaac Sim 窗口的大小。
*   它们**不直接决定**相机传感器（Camera Sensor）的渲染分辨率，除非相机明确设置为使用 viewport 分辨率。在当前代码中，它们主要影响 UI 体验。

### Q2: 相机图像尺寸强制为 128x128，后续的裁剪还有意义吗？
**回答**: 
**已修复（Fixed）：**
*   **代码更新**: 我们已修改 `isaac_sim_server.py`，将相机初始化分辨率设置为 **1280x720**，以匹配真实相机配置和 `config.py` 中的裁剪逻辑。
*   **缩放逻辑**: 启用了 `cv2.resize`。现在的流程是：
    1.  Sim 渲染 **1280x720** 图像。
    2.  应用 `IMAGE_CROP`（例如裁剪出 300x300 的区域）。
    3.  `cv2.resize` 将裁剪后的图像缩放到 **128x128**。
    4.  发送给 Agent。
*   **结论**: 现在的逻辑是自洽的。裁剪用于聚焦 ROI，Resize 用于满足 RL 输入要求。

### Q3: USD 文件中未指定像素大小，渲染出来的尺寸是多少？
**回答**: 
在 `isaac_sim_server.py` 中，代码显式地**覆盖**了 USD 中的任何默认设置。
修改后：
```python
camera = self.Camera(
    prim_path=cam_prim_path,
    name=cam_name,
    resolution=(1280, 720),  # <--- 已修改为 720p
)
```
因此，运行时渲染分辨率确定是 **1280x720**。

---

## 2. Isaac Sim Server 逻辑梳理

### 2.1 启动与 USD 加载流程

1.  **启动 SimulationApp**:
    *   入口: `IsaacSimServer.__init__`
    *   首先通过 `SimulationApp(config)` 启动 Isaac Sim 内核。这是必须的第一步，否则无法导入任何 `omni.*` 或 `pxr.*` 模块。
    *   参数 `width`/`height` 设定了仿真器窗口大小。

2.  **导入核心模块**:
    *   在 SimulationApp 启动后，才导入 `World`, `Camera`, `Franka` 等核心类。兼容 Isaac Sim 2023.x 和 4.x (isaacsim vs omni.isaac)。

3.  **加载 USD 场景**:
    *   使用 `omni.usd.get_context().open_stage(usd_path)` 加载指定的 `.usd` 文件。该文件包含了环境、光照、物体等。

4.  **初始化 World**:
    *   创建 `World` 实例，并设置物理时间步长 (`sim_hz`，默认 60Hz)。

5.  **加载机器人 (Franka)**:
    *   尝试从 Scene 中获取机器人 Prim。
    *   如果 Prim 存在但未注册为 Articulation，则创建一个 `Franka` (或 `Articulation`) 包装器并添加到 Scene 中。
    *   **关键操作**: 在物理 step 之前，调用 `set_joint_positions` 将机器人瞬移到 `initial_q`，防止初始状态下的剧烈物理碰撞。

6.  **加载相机**:
    *   遍历 `camera_prim_paths`。
    *   为每个路径创建一个 `Camera` 对象，并设置分辨率为 **(1280, 720)**。
    *   调用 `camera.initialize()` 并添加到 Scene。

7.  **物理与动力学调优**:
    *   `_tune_robot_dynamics()`: 调整关节的刚度 (Stiffness)、阻尼 (Damping) 和摩擦力，以防止仿真中的机械臂抖动。
    *   `_optimize_physics_precision()`: 降低碰撞体的 `ContactOffset` 和 `RestOffset`，提高接触精度，允许精密装配。

### 2.2 外部接口 (HTTP & WebSocket)

Server 使用 Flask 和 Flask-SocketIO 提供服务。

#### **HTTP 接口 (控制命令)**
*   **POST /pose**: 接收 `[x, y, z, qx, qy, qz, qw]`，更新目标位姿。
*   **POST /move_gripper**: 接收 `{"gripper_pos": float}` (0~1)，控制夹爪。
*   **POST /open_gripper**: 完全打开夹爪 (1.0)。
*   **POST /close_gripper**: 完全关闭夹爪 (0.0)。
*   **POST /reset_scene**: 重置整个 USD 场景（物理、物体位置、机器人状态）。
*   **POST /jointreset**: 强制关节复位 (Teleport)，用于当物理卡死时将机器人重置到安全位置。
*   **POST /update_param**: 更新参数（Stub，当前返回 "Updated"）。
*   **POST /clearerr**: 清除错误（Stub）。

#### **HTTP 接口 (状态查询)**
*   **POST /getstate**: **[Core]** 获取所有机器人状态（Pose, Vel, Force, Torque, Joint Q/DQ, Jacobian, Gripper）。
    *   **频率**: 由 Client 端 (`isaac_sim_env.py`) 的控制频率决定（通常 10Hz - 20Hz）。
*   **POST /getpos**: 仅获取末端位姿。
*   **POST /getvel**: 仅获取末端速度。
*   **POST /getforce**: 仅获取末端力。
*   **POST /gettorque**: 仅获取末端扭矩。
*   **POST /getq**: 仅获取关节位置。
*   **POST /getdq**: 仅获取关节速度。
*   **POST /getjacobian**: 仅获取雅可比矩阵。
*   **POST /get_gripper**: 仅获取夹爪位置。
*   **GET /health**: 健康检查，返回 `{"status": "healthy"}`。

#### **WebSocket 接口 (图像流)**
*   **Event 'connect'**: 客户端连接时触发。
*   **数据推送**:
    *   Server 端有一个后台线程 `_run_socket_process`。
    *   **频率**: 约为 `sim_hz` (60Hz) 或受限于渲染速度。
    *   **逻辑**: 
        1. 从 Isaac Sim 获取最新图像 (RGBA)。
        2. 转换为 RGB。
        3. **[问题点]** 应用 `image_crop` (基于 Config)。
        4. 通过 WebSocket 发送二进制 JPEG 数据。

### 2.3 仿真运行逻辑 (Execution Loop)

主循环在 `run()` 方法中：

1.  **Step Simulation**:
    *   调用 `self.world.step(render=True)` 进行一次物理仿真和渲染。
  
2.  **处理图像 (`_process_images`)**:
    *   从所有 Camera 获取 `current_frame` (RGBA)。
    *   执行裁剪（Crop）和格式转换。
    *   将最新帧放入 `image_queues`。

3.  **处理动作 (`_process_actions`)**:
    *   从 `action_queue` 取出最新的控制指令（由 Flask 线程放入）。
    *   如果是 **Pose 指令**: 使用 IK Solver (如 RMPFlow 或 Lula) 计算关节位置目标，并应用到机器人 (`apply_action`)。
    *   如果是 **Gripper 指令**: 使用简单 PD 控制器或 ArticulationAction 控制夹爪开合。

4.  **更新状态缓存 (`_update_state_cache`)**:
    *   读取机器人的当前关节位置、速度、末端位姿、雅可比矩阵等。
    *   更新 `state_cache` 供 `/getstate` 接口使用。

5.  **推送图像**:
    *   通过 WebSocket 广播最新图像帧。

### 2.4 其他逻辑

*   **Mock Dependencies**: 在加载 `config.py` 时，为了避免 Sim 环境缺少 `jax` 等库导致报错，使用了一个 Mock 系统 (`MockModules`) 来模拟缺失的依赖。
*   **Physics UI 延迟加载**: 为了避免 Isaac Sim 启动时的崩溃问题，PhysX UI 扩展是在 World Reset 之后才加载的。
