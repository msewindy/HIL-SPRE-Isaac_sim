"""
Isaac Sim Flask Server

独立进程运行，管理 Isaac Sim 仿真环境，提供 HTTP 和 WebSocket 接口

使用方法：
    python isaac_sim_server.py \
        --flask_url=0.0.0.0 \
        --flask_port=5001 \
        --headless=True \
        --sim_width=1280 \
        --sim_height=720 \
        --usd_path=/path/to/your/scene.usd \
        --robot_prim_path=/World/franka \
        --camera_prim_paths=/World/franka/panda_hand/wrist_1,/World/franka/panda_hand/wrist_2
"""

# 必须在导入其他 Isaac Sim 模块之前启动 SimulationApp
# 必须在导入其他 Isaac Sim 模块之前启动 SimulationApp
try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp

import numpy as np
import os
import threading
import time
import queue
import copy
import cv2
from typing import Dict, Optional, Callable
from absl import app, flags
# from scipy.spatial.transform import Rotation as R # [CRITICAL] REMOVED to prevent crash in Isaac Sim 2023+
from flask import Flask, request, jsonify
try:
    from flask_socketio import SocketIO
except ImportError:
    print("[WARNING] flask-socketio not installed. WebSocket support will be limited.")
    SocketIO = None
import base64
import logging

# [FIX] 抑制 Flask/Werkzeug 的刷屏日志
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

FLAGS = flags.FLAGS
flags.DEFINE_string("flask_url", "127.0.0.1", "URL for the flask server to run on.")
flags.DEFINE_integer("flask_port", 5001, "Port for the flask server to run on.")
flags.DEFINE_bool("headless", False, "Run Isaac Sim in headless mode.")
flags.DEFINE_integer("sim_width", 1280, "Simulation window width.")
flags.DEFINE_integer("sim_height", 720, "Simulation window height.")
flags.DEFINE_float("sim_hz", 60.0, "Simulation frequency (Hz).")
flags.DEFINE_string("config_module", None, "Optional: Python module path for IMAGE_CROP config (e.g., 'experiments.ram_insertion.config').")
flags.DEFINE_string("usd_path", None, "Path to the USD scene file (required). Scene should include robot, cameras, and task objects.")
flags.DEFINE_string("robot_prim_path", "/World/franka", "Prim path of the robot in the USD scene.")
flags.DEFINE_list("camera_prim_paths", ["/World/franka/panda_hand/wrist_1", "/World/franka/panda_hand/wrist_2"], "List of camera prim paths in the USD scene.")


class IsaacSimServer:
    """
    管理 Isaac Sim 仿真环境
    
    关键设计：
    1. 状态缓存：持续更新状态，快速响应查询
    2. 图像队列：丢弃旧帧，只保留最新帧（参考原项目 VideoCapture）
    3. 线程安全：使用锁保护共享状态
    4. WebSocket 推送：实时推送图像数据
    """
    
    def __init__(
        self,
        headless: bool = False,
        width: int = 1280,
        height: int = 720,
        sim_hz: float = 60.0,
        image_crop: Optional[Dict[str, Callable]] = None,
        socketio=None,
        usd_path: Optional[str] = None,
        robot_prim_path: str = "/World/franka",
        camera_prim_paths: Optional[list] = None,
    ):
        """
        初始化 Isaac Sim 服务器
        
        Args:
            headless: 是否无头模式
            width: 仿真窗口宽度
            height: 仿真窗口高度
            sim_hz: 仿真频率（Hz）
            image_crop: 图像裁剪配置字典 {camera_key: crop_function}
            socketio: Flask-SocketIO 实例
            usd_path: USD 场景文件路径（必须提供）
            robot_prim_path: 机器人 prim 路径（在 USD 场景中）
            camera_prim_paths: 相机 prim 路径列表（在 USD 场景中）
        """
        if usd_path is None:
            raise ValueError("usd_path must be provided. Please specify the path to your USD scene file.")
        
        # 启动 Isaac Sim
        config = {
            "headless": headless,
            "width": width,
            "height": height,
            # [UI] Revert to default experience to prevent crash
            # "experience": ... 
        }
        self.simulation_app = SimulationApp(config)
        
        # [UI] Physics Visualization is now loaded in _load_cameras() or at end of __init__
        # to ensure stability (loading AFTER world reset).
        
        # [OPTIMIZATION] 设置初始关节位姿，防止机械臂在空中乱舞或过度跌落
        # 这是一个比较接近目标的常规抓取姿态
        self.initial_q = np.array([0.0, -0.4, 0.0, -2.4, 0.0, 2.0, 0.8, 0.0, 0.0])
        
        # 频率配置
        self.sim_hz = sim_hz
        self.control_hz = 10.0  # 控制频率（由客户端决定）
        
        # 现在可以导入其他模块
        # 适配 Isaac Sim 5.1+ (使用 isaacsim 命名空间)
        try:
            from isaacsim.core.api import World
            from isaacsim.sensors.camera import Camera
            from isaacsim.core.utils.stage import get_current_stage
            print("[INFO] Imported core modules from isaacsim namespace")
        except ImportError:
            from omni.isaac.core import World
            from omni.isaac.sensor import Camera
            from omni.isaac.core.utils.stage import get_current_stage
            print("[INFO] Imported core modules from omni.isaac namespace")

        # Franka 包装器 (omni.isaac.franka 已被标记为 deprecated 但暂时仍是最方便的)
        try:
            from omni.isaac.franka import Franka
        except ImportError:
             print("[WARN] omni.isaac.franka not found. Robot wrapper might be missing features.")
             # Fallback to generic Articulation if Franka class is gone (future proofing)
             from omni.isaac.core.articulations import Articulation as Franka
             
        from pxr import UsdGeom
        import omni.usd
        
        self.World = World
        self.Franka = Franka
        self.Camera = Camera
        self.get_current_stage = get_current_stage
        self.UsdGeom = UsdGeom
        
        # 加载 USD 场景文件（必须在创建 World 之前）
        print(f"[INFO] Loading USD scene from: {usd_path}")
        try:
            # 使用 omni.usd 加载 USD 文件
            usd_context = omni.usd.get_context()
            usd_context.open_stage(usd_path)
            print(f"[INFO] USD scene loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load USD scene from {usd_path}: {e}")
        
        # 创建世界（加载 USD 后）
        self.world = World(stage_units_in_meters=1.0)
        if hasattr(self.world, 'set_physics_dt'):
            self.world.set_physics_dt(1.0 / self.sim_hz)
        elif hasattr(self.world, 'set_simulation_dt'):
            self.world.set_simulation_dt(physics_dt=1.0 / self.sim_hz)
        else:
            print("[WARNING] Could not set physics DT: World object missing set_physics_dt/set_simulation_dt")
        
        # 从场景中获取机器人对象
        self.robot_prim_path = robot_prim_path
        print(f"[INFO] Getting robot from prim path: {robot_prim_path}")
        try:
            # 使用 world.scene.get() 获取已存在的对象
            # API 变更适配: Scene.get可能不存在，且在fresh scene中对象未注册
            self.franka = None
            if hasattr(self.world.scene, "get_object"):
                self.franka = self.world.scene.get_object("franka") # 尝试通过名称获取
            elif hasattr(self.world.scene, "get"):
                self.franka = self.world.scene.get(robot_prim_path)
            if self.franka is None:
                # 尝试使用 Franka 类来获取（如果 prim 存在但未注册到 scene）
                stage = self.get_current_stage()
                prim = stage.GetPrimAtPath(robot_prim_path)
                if prim.IsValid():
                    # Prim 存在，尝试创建 Franka 对象来包装它
                    print(f"[INFO] Prim found at {robot_prim_path}, creating Franka wrapper...")
                    self.franka = self.world.scene.add(
                        Franka(
                            prim_path=robot_prim_path,
                            name="franka",
                        )
                    )
                else:
                    raise RuntimeError(f"Robot prim not found at {robot_prim_path} in USD scene")
            else:
                print(f"[INFO] Robot found at {robot_prim_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to get robot from {robot_prim_path}: {e}")
        
        # 从场景中获取相机对象 (将在 reset 后延迟加载)
        self.cameras = {}
        self.camera_prim_paths = camera_prim_paths or [
            "/World/franka/panda_hand/wrist_1",
            "/World/franka/panda_hand/wrist_2"
        ]
        
        # 图像裁剪配置
        self.image_crop = image_crop or {}
        
        # 状态变量（线程安全）
        self.state_lock = threading.Lock()
        self.state_cache = {
            "pose": np.zeros(7),      # xyz + quat
            "vel": np.zeros(6),
            "force": np.zeros(3),
            "torque": np.zeros(3),
            "q": np.zeros(7),
            "dq": np.zeros(7),
            "jacobian": np.zeros((6, 7)),
            "gripper_pos": 0.0,
        }
        
        # 图像队列（每个相机一个队列，maxsize=1，只保留最新帧）
        self.image_queues = {}
        self.image_lock = threading.Lock()
        
        # 动作指令队列（线程安全，用于从 Flask 线程传递指令到仿真主线程）
        self.action_queue = queue.Queue()
        
        # WebSocket 客户端列表（线程安全）
        self.ws_clients = []
        self.ws_lock = threading.Lock()
        self.socketio = socketio  # Flask-SocketIO 实例
        
        # IK 控制器和夹爪控制（初始化）
        self._setup_controller()
        
        # 雅可比矩阵计算标志（用于日志记录）
        self._jacobian_method_logged = False
        
        # 初始化世界
        self.world.reset()
        
        # [CRITICAL] 强制设置初始关节位置（瞬间移动）
        # 必须在物理步进之前设置，确保从一个合理的姿态开始 Warmup
        if hasattr(self, 'initial_q'):
            print(f"[INFO] Teleporting robot to initial joint positions for stability...")
            self.franka.set_joint_positions(self.initial_q)
            # 给模拟器一点时间感知状态
            for _ in range(5):
                 self.world.step(render=False)
        
        # 加载相机（必须在 world.reset() 之后，确保渲染环境就绪）
        self._load_cameras()

        # [UI] Manually load Physics UI extension (Delayed Load for Stability)
        # Moving this AFTER world.reset() / load_cameras to prevent startup race conditions/crashes
        try:
            from omni.isaac.core.utils.extensions import enable_extension
            if not self._is_extension_enabled("omni.physx.ui"):
                 enable_extension("omni.physx.ui")
                 print("[INFO] Enabled omni.physx.ui extension (Delayed Load)")
            
            # Enable visualization settings
            import carb.settings
            settings = carb.settings.get_settings()
            
            # Enable global physics visualization
            settings.set_bool("/physics/visualization/enabled", True)
            
            # Enable collision shapes visualization
            settings.set_bool("/physics/visualization/collisionShapes", True)
            
            # Additional useful debug settings
            settings.set_bool("/physics/visualization/joints", False) # Set to True if needed
            
        except Exception as e:
            print(f"[WARNING] Failed to enable Physics UI: {e}")

        if self.image_crop:
            print("[INFO] Image cropping enabled.")
            
        # [NEW] Setup Force/Torque Sensor
        self._setup_force_sensor()
        
        # [GRASP] Magnetic Grasping State (Initialized)
        self.grasp_joint = None
        self.grasp_obj_path = "/World/factory_gear_medium" 
        
        # [GRASP] Parameters (Refined for 4cm Gear)
        self.grasp_threshold_deg = 4.5     # Alignment < 4.5 deg (Relaxed)
        self.grasp_dist_threshold = 0.02   # Center dist < 2cm
        self.grasp_gear_diameter = 0.04    # Gear Diameter 5cm
        self.grasp_width_tolerance = 0.01 # +/- 0.5cm tolerance (Strict)
        
        self.last_gripper_cmd = 0.0 # Track command to detect opening intent
        
        # 6. Tune Scene & Robot Dynamics
        # self._tune_scene_physics() # Assume already tuned based on previous tasks
        self._tune_robot_dynamics()

        # [GRIP] 场景物理属性调优 (摩擦力 & 质量) - 解决滑冰和抓不住的问题
        try:
            self._tune_scene_physics()
        except Exception as e:
            print(f"[WARNING] Failed to tune scene physics: {e}")

        # [PRECISION] 针对精密装配任务，全局优化物理接触参数
        self._optimize_physics_precision()

        # [STABILITY] 机械臂动力学参数调优 (防止晃动)
        self._tune_robot_dynamics()

        # [GRIP] 场景物理属性调优 (摩擦力 & 质量) - 解决滑冰和抓不住的问题
        try:
            self._tune_scene_physics()
        except Exception as e:
            print(f"[ERROR] Failed to tune scene physics: {e}")

    def _tune_scene_physics(self):
        """
        [Reverted] Physics tuning removed as per user request.
        Using Magnetic Grasping instead.
        """
        pass

    def _tune_robot_dynamics(self):
        """
        自动调整机械臂关节的 Stiffness(刚度), Damping(阻尼), Armature(电枢惯量) 和 Friction(摩擦)。
        解决默认参数过软导致的机械臂晃动问题。
        """
        print("[INFO] Tuning robot dynamics (Stiffness/Damping) for stability...")
        stage = self.get_current_stage()
        from pxr import UsdPhysics, PhysxSchema, Usd

        # 定义增益配置
        # Root joints (1-4) need high stiffness
        # Wrist joints (5-7) need higher stiffness to reduce end-effector vibration
        gains = {
            "default": {"stiffness": 400.0, "damping": 40.0},
            "root":    {"stiffness": 600.0, "damping": 60.0}, # Joints 1-4
            "wrist":   {"stiffness": 400.0, "damping": 40.0}, # Joints 5-7 (Increased from 200/20 to reduce vibration)
            "finger":  {"stiffness": 1000.0, "damping": 100.0}, # Gripper
        }
        
        # 通用稳定性参数
        armature_val = 0.01
        friction_val = 0.1

        # 获取机器人 Prim
        robot_prim = stage.GetPrimAtPath(self.robot_prim_path)
        if not robot_prim.IsValid():
            print(f"[WARNING] Robot prim not found at {self.robot_prim_path}, skip dynamics tuning.")
            return

        # 遍历机器人下的所有关节
        for prim in Usd.PrimRange(robot_prim):
            if prim.IsA(UsdPhysics.Joint):
                name = prim.GetName()
                
                # 确定增益配置
                gain = gains["default"]
                if "joint1" in name or "joint2" in name or "joint3" in name or "joint4" in name:
                    gain = gains["root"]
                elif "joint5" in name or "joint6" in name or "joint7" in name:
                    gain = gains["wrist"]
                elif "finger" in name:
                    gain = gains["finger"]

                # 1. 设置 Stiffness & Damping (通过 DriveAPI)
                # 通常 drive 是 "angular" 或 "linear"
                # 我们尝试获取 "angular" drive (旋转关节)
                drive_api = UsdPhysics.DriveAPI.Get(prim, "angular")
                if not drive_api:
                     drive_api = UsdPhysics.DriveAPI.Apply(prim, "angular")
                
                if drive_api:
                    drive_api.CreateStiffnessAttr().Set(gain["stiffness"])
                    drive_api.CreateDampingAttr().Set(gain["damping"])
                
                # linear drive (比如夹爪)
                if "finger" in name:
                    drive_api_lin = UsdPhysics.DriveAPI.Get(prim, "linear")
                    if not drive_api_lin:
                        drive_api_lin = UsdPhysics.DriveAPI.Apply(prim, "linear")
                    if drive_api_lin:
                        drive_api_lin.CreateStiffnessAttr().Set(gain["stiffness"])
                        drive_api_lin.CreateDampingAttr().Set(gain["damping"])

                # 2. 设置 Armature & Friction (通过 PhysxJointAPI)
                physx_joint_api = PhysxSchema.PhysxJointAPI.Apply(prim)
                physx_joint_api.CreateArmatureAttr().Set(armature_val)
                physx_joint_api.CreateJointFrictionAttr().Set(friction_val)

                # print(f"[DEBUG] Tuned joint {name}: P={gain['stiffness']}, D={gain['damping']}, Armature={armature_val}")

        print("[INFO] Robot dynamics tuned successfully.")

    def _optimize_physics_precision(self):
        """
        全局降低 Contact Offset 和 Rest Offset，消除“力场盾”效应，
        使机械臂能进行毫米级精密操作。
        """
        print("[INFO] Optimizing physics scenes for precision assembly...")
        stage = self.get_current_stage()
        from pxr import PhysxSchema, UsdPhysics

        # 遍历舞台上所有 Prim
        for prim in stage.Traverse():
            # [FIX] 增加求解器迭代次数 (Solver Iterations)
            # 找到 PhysicsScene 并增强其求解精度，这对刚性抓取至关重要
            if prim.IsA(UsdPhysics.Scene):
                try:
                    physx_scene_api = PhysxSchema.PhysxSceneAPI.Apply(prim)
                    # 默认是 4，提高到 8 或 16 可以显著增加接触刚度
                    physx_scene_api.CreateSolverPositionIterationsAttr().Set(8)
                    physx_scene_api.CreateSolverVelocityIterationsAttr().Set(1)
                    print(f"[INFO] Enhanced physics scene solver iterations for {prim.GetPath()}")
                except Exception as e:
                     print(f"[WARN] Failed to set solver iterations: {e}")

            # 检查是否有 CollisionAPI (即是否是碰撞体)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                # 获取或应用 PhysxCollisionAPI
                physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                
                # [FIX] Set RestOffset FIRST to 0.0, then ContactOffset to 0.001
                # Because standard rule is ContactOffset > RestOffset.
                # If we set ContactOffset first (0.001) while RestOffset is default (e.g. 0.02),
                # it violates 0.001 > 0.02, causing Error.
                
                # 设置 Rest Offset (静止间距) -> 0mm
                physx_collision_api.CreateRestOffsetAttr().Set(0.0)
                
                # 设置 Contact Offset (接触阈值) -> 0.1mm (从 1mm 降低)
                # 消除 "力场盾" 效应，允许抓取时紧密接触
                physx_collision_api.CreateContactOffsetAttr().Set(0.0001)
                
                # print(f"[DEBUG] Optimized collision for: {prim.GetPath()}")
        
        print("[INFO] precision assembly optimization completed: ContactOffset=1mm, RestOffset=0mm")
        
        # 标记运行状态
        self.running = True

        # [FIX] 初始化状态缓存并设置控制器目标，防止启动时机械臂下坠
        print("[INFO] Initializing robot state and controller target...")
        
        # [DEBUG] 打印机器人基座位置和姿态
        from scipy.spatial.transform import Rotation as R
        try:
            robot_prim_path = self.robot_prim_path
            robot_prim = self.get_current_stage().GetPrimAtPath(robot_prim_path)
            if robot_prim.IsValid():
                xform = self.UsdGeom.Xformable(robot_prim)
                base_transform = xform.ComputeLocalToWorldTransform(0)
                base_pos = base_transform.ExtractTranslation()
                base_rot_mat = base_transform.ExtractRotationMatrix()
                base_quat = R.from_matrix(base_rot_mat).as_quat()
                print(f"[INFO] Robot Base World Pose: pos={np.round(base_pos, 3)}, quat_xyzw={np.round(base_quat, 3)}")
            else:
                print(f"[WARNING] Robot prim not found at {robot_prim_path}")
        except Exception as e:
             print(f"[DEBUG] Failed to get robot base pose: {e}")

        # 1. 强制更新一次状态，获取当前 USD 中的位姿
        self._update_state_fast()
        
        # 2. 将当前位姿设为控制器目标
        current_pose = self.state_cache["pose"]
        print(f"[DEBUG] Current EE World Pose: {np.round(current_pose, 3)}")
        # 检查位姿是否有效 (非全0)
        if np.any(current_pose):
            # [IMPROVEMENT] 尝试从配置模块加载初始位姿 (RESET_POSE)
            # 这里的逻辑是：如果用户提供了配置文件，优先使用配置文件中的 RESET_POSE 作为初始目标
            # 这样可以在预热期间直接将机器人拉到任务初始位置，避免"先下坠再拉起"
            target_pose = current_pose
            
            if FLAGS.config_module:
                try:
                    import importlib
                    import sys
                    from unittest.mock import MagicMock
                    import importlib.util
                    
                    # 定义一个上下文管理器来模拟缺失的模块
                    class MockModules:
                        def __init__(self, modules):
                            self.modules = modules
                            self.mocked_names = []
                            
                        def __enter__(self):
                            for mod_name in self.modules:
                                # 检查模块是否已经存在于路径中
                                if mod_name in sys.modules:
                                    continue
                                    
                                spec = importlib.util.find_spec(mod_name)
                                if spec is None:
                                    # 真正缺失，需要 mock
                                    # 确保父包也存在
                                    parts = mod_name.split('.')
                                    for i in range(1, len(parts) + 1):
                                        name = '.'.join(parts[:i])
                                        if name not in sys.modules and importlib.util.find_spec(name) is None:
                                            mock_obj = MagicMock()
                                            # 设置为包，允许 . 访问
                                            mock_obj.__path__ = []
                                            sys.modules[name] = mock_obj
                                            if name not in self.mocked_names:
                                                self.mocked_names.append(name)
                        
                        def __exit__(self, exc_type, exc_val, exc_tb):
                            # 清理我们添加的 mock
                            for name in reversed(self.mocked_names):
                                if name in sys.modules:
                                    del sys.modules[name]

                    # 列出可能缺失的库
                    mock_list = [
                        "jax", "jax.numpy", "jax.random", "flax", "einops", "pynput", "gym", "gymnasium",
                        "serl_launcher", "serl_launcher.wrappers", "serl_launcher.networks",
                        "franka_env", "franka_env.envs", "franka_env.utils",
                        "experiments.ram_insertion", "experiments.gear_assembly.wrapper"
                    ]
                    
                    self._log(f"[INFO] Loading config module '{FLAGS.config_module}' with mocked dependencies...")
                    
                    with MockModules(mock_list):
                        cfg_module = importlib.import_module(FLAGS.config_module)
                        
                        # 尝试获取 IsaacSimEnvConfig 或 EnvConfig
                        env_config = None
                        if hasattr(cfg_module, "IsaacSimEnvConfig"):
                            env_config = cfg_module.IsaacSimEnvConfig
                        elif hasattr(cfg_module, "EnvConfig"):
                            env_config = cfg_module.EnvConfig
                        
                        if env_config and hasattr(env_config, "RESET_POSE"):
                            config_pose = np.array(env_config.RESET_POSE)
                            
                            # 只有当 RESET_POSE 是 numpy 数组时才使用 (Mock对象会被忽略)
                            if isinstance(config_pose, (np.ndarray, list)):
                                config_pose = np.array(config_pose)
                                self._log(f"[DEBUG] Raw config RESET_POSE: {config_pose}")
                                # 转换欧拉角到四元数 (xyz + rxryrz -> xyz + qxqyqzqw)
                                # 注意：RESET_POSE 通常是 [x, y, z, rx, ry, rz]
                                if config_pose.shape == (6,):
                                     from scipy.spatial.transform import Rotation as R
                                     pos = config_pose[:3]
                                     euler = config_pose[3:].astype(float)
                                     # [FIX] 切换到 intrinsic xyz (内旋)，这在机器人配置中更为常见
                                     quat = R.from_euler("xyz", euler).as_quat()
                                     target_pose = np.concatenate([pos, quat])
                                     # 日志显示为 WXYZ 格式
                                     quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])
                                     pose_display = np.concatenate([pos, quat_wxyz])
                                     self._log(f"[INFO] Successfully parsed RESET_POSE from config (intrinsic xyz): {np.round(pose_display, 3)} [WXYZ]")
                                elif config_pose.shape == (7,):
                                     target_pose = config_pose
                                     # 假设输入是 XYZW，转换为 WXYZ 显示
                                     quat_xyzw = target_pose[3:]
                                     quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
                                     pose_display = np.concatenate([target_pose[:3], quat_wxyz])
                                     self._log(f"[INFO] Successfully parsed RESET_POSE from config (Quat): {np.round(pose_display, 3)} [WXYZ]")
                            else:
                                self._log(f"[WARNING] RESET_POSE found but is unexpected type: {type(config_pose)}")
                        else:
                            self._log("[WARNING] Config loaded but RESET_POSE not found")
                        
                        if env_config:
                             self._log(f"[VERIFY] Loaded Config Class: {env_config.__name__}")
                             
                        if 'target_pose' in locals():
                             # 转换为 WXYZ 显示
                             quat_xyzw = target_pose[3:]
                             quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
                             pose_display = np.concatenate([target_pose[:3], quat_wxyz])
                             self._log(f"[VERIFY] Server Initial RESET_POSE: {np.round(pose_display, 4)} [WXYZ]")
                            
                except Exception as e:
                    self._log(f"[WARNING] Failed to load RESET_POSE from config module: {e}")
                    import traceback
                    traceback.print_exc()
            
            # 转换为 WXYZ 显示
            quat_xyzw = target_pose[3:]
            quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
            pose_display = np.concatenate([target_pose[:3], quat_wxyz])
            self._log(f"[INFO] Setting initial controller target: {np.round(pose_display, 3)} [WXYZ]")
            
            # [RESET LOGIC] Store valid target pose for future resets
            self.default_reset_pose = target_pose
            
            # 注意：set_pose 会将指令放入队列，在 run() 的第一次循环中被处理
            self.set_pose(target_pose)
        else:
            self._log("[WARNING] Could not determine initial pose, robot might fall!")
        
        # 调试计数器
        self.frame_count = 0
        self.last_log_time = time.time()

    def _is_extension_enabled(self, ext_id: str) -> bool:
        """Check if an extension is enabled."""
        try:
            import omni.kit.app
            manager = omni.kit.app.get_app().get_extension_manager()
            return manager.is_extension_enabled(ext_id)
        except ImportError:
            return False
    
    def _log(self, msg):
        from datetime import datetime
        # 增加毫秒级时间戳，方便与客户端对齐日志
        print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] [SERVER] {msg}")
    
    def _load_cameras(self):
        """加载相机（在 world.reset() 后调用）"""
        self._log(f"[INFO] Getting cameras from prim paths: {self.camera_prim_paths}")
        for cam_prim_path in self.camera_prim_paths:
            try:
                # 从 prim 路径提取相机名称
                cam_name = cam_prim_path.split("/")[-1]
                camera = None
                
                # 尝试从 scene 获取
                if hasattr(self.world.scene, "get_object"):
                    camera = self.world.scene.get_object(cam_name)
                elif hasattr(self.world.scene, "get"):
                    camera = self.world.scene.get(cam_prim_path)
                
                if camera is None:
                    # 尝试创建 Camera 包装器
                    stage = self.get_current_stage()
                    prim = stage.GetPrimAtPath(cam_prim_path)
                    if prim.IsValid():
                        self._log(f"[INFO] Camera prim found at {cam_prim_path}, initializing Camera wrapper...")
                        # [FIX] Set to 720p to match RealSense and Cropping config
                        camera = self.Camera(
                            prim_path=cam_prim_path,
                            name=cam_name,
                            resolution=(1280, 720), 
                        )
                        # 初始化相机
                        camera.initialize()
                        # 添加到 scene
                        self.world.scene.add(camera)
                    else:
                        self._log(f"[WARNING] Camera prim not found at {cam_prim_path}, skipping...")
                        continue
                
                if camera is not None:
                    # 如果是从 scene 获取的，不需要额外操作
                    self.cameras[cam_name] = camera
                    self._log(f"[INFO] Camera '{cam_name}' loaded")
                    
            except Exception as e:
                self._log(f"[WARNING] Failed to load camera {cam_prim_path}: {e}")

        if len(self.cameras) == 0:
            self._log("[WARNING] No cameras loaded. Image capture will be disabled.")
        else:
            self._log(f"[INFO] Loaded {len(self.cameras)} cameras: {list(self.cameras.keys())}")

    def _setup_controller(self):
        """
        设置机器人控制器（IK 求解器）
        
        尝试使用 RMPFlowController，如果不可用则使用 IK 求解器
        参考 isaac_sim_env.py 中的 _init_controller() 实现
        """
        self.controller = None
        self.ik_solver = None
        
        try:
            # 方法1：尝试使用 RMPFlowController（推荐）
            from omni.isaac.manipulators.controllers import RMPFlowController
            
            # 确定末端执行器路径
            if hasattr(self.franka, 'end_effector'):
                ee_prim_path = self.franka.end_effector.prim_path
            else:
                ee_prim_path = "/World/franka/panda_hand"
            
            self.controller = RMPFlowController(
                name="franka_controller",
                robot_articulation=self.franka,
                end_effector_prim_path=ee_prim_path,
            )
            self._log("[INFO] Initialized RMPFlowController for robot control")
            
        except (ImportError, AttributeError, Exception) as e:
            self._log(f"[WARN] RMPFlowController (omni.isaac.manipulators) not available: {e}")
            
            # 尝试使用 omni.isaac.motion_generation (Isaac Sim 2023.1.1+ / 4.0+)
            try:
                from omni.isaac.motion_generation import RmpFlow, ArticulationMotionPolicy
                from omni.isaac.motion_generation.interface_config_loader import load_supported_motion_policy_config
                
                # 为 Franka 加载默认配置
                config = load_supported_motion_policy_config("Franka", "RMPflow")
                if config:
                    rmpflow = RmpFlow(**config)
                    self.controller = ArticulationMotionPolicy(self.franka, rmpflow)
                    self._log("[INFO] Initialized ArticulationMotionPolicy(RmpFlow) for robot control")
                else:
                    raise ImportError("Franka RMPflow config not found")
                    
            except (ImportError, AttributeError, Exception) as e_mg:
                 self._log(f"[WARN] omni.isaac.motion_generation RmpFlow not available: {e_mg}")
                 self._log("[INFO] Falling back to IK solver or direct joint control")
            
            try:
                # 方法2：尝试使用 IK 求解器
                from omni.isaac.manipulators import InverseKinematicsSolver
                
                # 确定末端执行器路径
                if hasattr(self.franka, 'end_effector'):
                    ee_prim_path = self.franka.end_effector.prim_path
                else:
                    ee_prim_path = "/World/franka/panda_hand"
                
                # 创建 IK 求解器
                # 注意：需要根据实际 API 调整参数
                self.ik_solver = InverseKinematicsSolver(
                    robot_articulation=self.franka,
                    end_effector_prim_path=ee_prim_path,
                )
                self._log("[INFO] Initialized InverseKinematicsSolver for robot control")
                
            except (ImportError, AttributeError, Exception) as e2:
                self._log(f"[WARN] InverseKinematicsSolver (omni.isaac.manipulators) not available: {e2}")
                
                 # 尝试使用 omni.isaac.motion_generation.LulaKinematicsSolver
                try:
                    from omni.isaac.motion_generation import LulaKinematicsSolver
                    from omni.isaac.motion_generation.interface_config_loader import load_supported_lula_kinematics_solver_config
                    
                    config = load_supported_lula_kinematics_solver_config("Franka")
                    if config:
                        self.ik_solver = LulaKinematicsSolver(**config)
                        self._log("[INFO] Initialized LulaKinematicsSolver for robot control")
                    else:
                        raise ImportError("Franka Lula kinematics config not found")
                        
                except (ImportError, AttributeError, Exception) as e3:
                    self._log(f"[WARN] LulaKinematicsSolver not available: {e3}")
                    self._log("[WARN] Robot pose control will not work without controller or IK solver")
                    self._log("[WARN] Please ensure 'omni.isaac.motion_generation' extension is enabled")
                # 如果都不可用，控制器和IK求解器保持为 None
    
    def _compute_jacobian(self) -> Optional[np.ndarray]:
        """
        计算雅可比矩阵
        
        尝试多种方法获取雅可比矩阵：
        1. 使用 franka.get_jacobian()（如果可用）
        2. 使用 root_physx_view.get_jacobians()（如果可用）
        3. 使用 Isaac Sim 的其他 API
        
        Returns:
            jacobian: np.ndarray[6, 7] - 雅可比矩阵，如果无法获取则返回 None
        """
        try:
            # 方法1：尝试使用 franka 对象的 get_jacobian 方法
            if hasattr(self.franka, 'get_jacobian'):
                try:
                    jacobian = self.franka.get_jacobian()
                    if jacobian is not None:
                        # 确保形状正确 (6, 7)
                        jacobian = np.array(jacobian)
                        if jacobian.shape == (6, 7):
                            if not self._jacobian_method_logged:
                                self._log("[INFO] Using franka.get_jacobian() to compute Jacobian")
                                self._jacobian_method_logged = True
                            return jacobian
                        elif jacobian.shape == (7, 6):
                            # 如果是转置，转置回来
                            if not self._jacobian_method_logged:
                                self._log("[INFO] Using franka.get_jacobian() to compute Jacobian (transposed)")
                                self._jacobian_method_logged = True
                            return jacobian.T
                        elif jacobian.size == 42:  # 42 = 6 * 7
                            # 如果是展平的数组，重塑
                            if not self._jacobian_method_logged:
                                self._log("[INFO] Using franka.get_jacobian() to compute Jacobian (reshaped)")
                                self._jacobian_method_logged = True
                            return jacobian.reshape(6, 7)
                except Exception as e:
                    pass
            
            # 方法2：尝试使用 root_physx_view.get_jacobians()
            if hasattr(self.franka, 'root_physx_view'):
                try:
                    # 获取末端执行器的索引
                    if hasattr(self.franka, 'end_effector'):
                        ee_index = self.franka.end_effector.prim_path
                    else:
                        ee_index = "/World/franka/panda_hand"
                    
                    # 获取雅可比矩阵
                    jacobians = self.franka.root_physx_view.get_jacobians()
                    if jacobians is not None:
                        # jacobians 可能是字典或数组
                        if isinstance(jacobians, dict):
                            # 如果是字典，尝试找到末端执行器的雅可比
                            if ee_index in jacobians:
                                jacobian = np.array(jacobians[ee_index])
                            elif len(jacobians) > 0:
                                # 使用第一个可用的雅可比
                                jacobian = np.array(list(jacobians.values())[0])
                            else:
                                return None
                        else:
                            # 如果是数组，直接使用
                            jacobian = np.array(jacobians)
                        
                        # 确保形状正确
                        if jacobian.shape == (6, 7):
                            if not self._jacobian_method_logged:
                                self._log("[INFO] Using root_physx_view.get_jacobians() to compute Jacobian")
                                self._jacobian_method_logged = True
                            return jacobian
                        elif jacobian.shape == (7, 6):
                            if not self._jacobian_method_logged:
                                self._log("[INFO] Using root_physx_view.get_jacobians() to compute Jacobian (transposed)")
                                self._jacobian_method_logged = True
                            return jacobian.T
                        elif jacobian.size == 42:
                            if not self._jacobian_method_logged:
                                self._log("[INFO] Using root_physx_view.get_jacobians() to compute Jacobian (reshaped)")
                                self._jacobian_method_logged = True
                            return jacobian.reshape(6, 7)
                except Exception as e:
                    pass
            
            # 方法3：尝试使用 articulation 的 API
            if hasattr(self.franka, 'articulation'):
                try:
                    articulation = self.franka.articulation
                    if hasattr(articulation, 'get_jacobian'):
                        jacobian = articulation.get_jacobian()
                        if jacobian is not None:
                            jacobian = np.array(jacobian)
                            if jacobian.shape == (6, 7):
                                return jacobian
                            elif jacobian.shape == (7, 6):
                                return jacobian.T
                            elif jacobian.size == 42:
                                return jacobian.reshape(6, 7)
                except Exception as e:
                    pass
            
            # 方法4：尝试使用 Isaac Sim 的物理视图 API
            try:
                # 尝试通过物理视图获取
                if hasattr(self.franka, '_articulation_view'):
                    view = self.franka._articulation_view
                    if hasattr(view, 'get_jacobians'):
                        jacobians = view.get_jacobians()
                        if jacobians is not None:
                            jacobian = np.array(jacobians)
                            if jacobian.shape == (6, 7):
                                return jacobian
                            elif jacobian.shape == (7, 6):
                                return jacobian.T
                            elif jacobian.size == 42:
                                return jacobian.reshape(6, 7)
            except Exception as e:
                pass
            
            # 方法5：尝试使用 Isaac Sim 的动力学 API（如果可用）
            try:
                # 某些版本的 Isaac Sim 可能提供动力学 API
                if hasattr(self.franka, 'get_dof_jacobians'):
                    jacobians = self.franka.get_dof_jacobians()
                    if jacobians is not None:
                        # 通常返回的是所有 DOF 的雅可比，需要提取末端执行器的
                        jacobian = np.array(jacobians)
                        if len(jacobian.shape) == 3:
                            # 如果是 (n_dofs, 6, 7) 形状，取最后一个（通常是末端执行器）
                            jacobian = jacobian[-1]
                        if jacobian.shape == (6, 7):
                            return jacobian
                        elif jacobian.shape == (7, 6):
                            return jacobian.T
                        elif jacobian.size == 42:
                            return jacobian.reshape(6, 7)
            except Exception as e:
                pass
            
            # 如果所有方法都失败，返回 None
            if not self._jacobian_method_logged:
                self._log("[WARNING] Failed to compute Jacobian matrix using all available methods")
                self._log("[WARNING] End-effector velocity will be set to zero")
                self._jacobian_method_logged = True
            return None
            
        except Exception as e:
            # 只在第一次失败时打印错误
            if not self._jacobian_method_logged:
                self._log(f"[WARNING] Exception while computing Jacobian: {e}")
                self._jacobian_method_logged = True
            return None
    
    def _setup_force_sensor(self):
        """
        在运行时为机械臂末端添加力/力矩传感器 (PhysxArticulationForceSensorAPI)。
        无需修改 USD 文件，直接在内存中应用 API。
        """
        print("[INFO] Setting up Force/Torque Sensor...")
        stage = self.get_current_stage()
        
        # 尝试找到连接 Hand 的关节
        # 通常是 panda_hand_joint (Fixed) 或 panda_joint7 (Revolute)
        # 我们希望测量的是末端(Hand + Gripper + Object)受到的力，所以通常放在 Hand 的父关节上
        
        # 路径可能是: /World/franka/panda_hand_joint (如果是打平的结构)
        # 或者: /World/franka/panda_link8/panda_hand_joint
        
        potential_paths = [
            f"{self.robot_prim_path}/panda_hand_joint",
            f"{self.robot_prim_path}/panda_link8/panda_hand_joint",
            f"{self.robot_prim_path}/panda_joint7", # Fallback
        ]
        
        sensor_prim = None
        for path in potential_paths:
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                sensor_prim = prim
                print(f"[INFO] Found sensor joint prim at: {path}")
                break
        
        if sensor_prim:
            # 应用力传感器 API
            # PhysxSchema.PhysxArticulationForceSensorAPI
            try:
                self.force_sensor_api = PhysxSchema.PhysxArticulationForceSensorAPI.Apply(sensor_prim)
                self.force_sensor_api.CreateSensorEnabledAttr().Set(True)
                self.force_sensor_api.CreateWorldFrameEnabledAttr().Set(False) # Local Frame (EE)
                print("[INFO] Force/Torque Sensor API applied and enabled.")
            except Exception as e:
                print(f"[ERROR] Failed to apply Force Sensor API: {e}")
                self.force_sensor_api = None
        else:
            print("[WARNING] Could not find suitable joint for Force Sensor.")
            self.force_sensor_api = None

    def _compute_end_effector_wrench(self):
        """
        计算末端执行器处的力/力矩。
        优先使用 PhysxArticulationForceSensorAPI (如果已安装)。
        否则回退到 Jacobian 估算。
        """
        # 1. 尝试从传感器读取 (Sim2Real 对齐：Local Frame, Total Wrench)
        if hasattr(self, 'force_sensor_api') and self.force_sensor_api:
            try:
                # GetForceAttr() 返回 Gf.Vec3f
                force_gf = self.force_sensor_api.GetForceAttr().Get()
                torque_gf = self.force_sensor_api.GetTorqueAttr().Get()
                
                if force_gf is not None:
                    force = np.array(force_gf)
                    torque = np.array(torque_gf)
                    return force, torque
            except Exception as e:
                pass # Fallback to Jacobian

        # 2. 回退到 Jacobian 方法 (World Frame, Estimated from Tau)
        try:
            # ... (Original Jacobian logic) ...
            # 1. 获取关节力矩
            tau = None
            
            # 方法1：尝试使用 get_applied_joint_actions()
            if hasattr(self.franka, 'get_applied_joint_actions'):
                try:
                    tau = self.franka.get_applied_joint_actions()
                    if tau is not None:
                        tau = np.array(tau)
                except Exception as e:
                    pass
            
            # 方法2：尝试使用 get_joint_efforts()
            if tau is None and hasattr(self.franka, 'get_joint_efforts'):
                try:
                    tau = self.franka.get_joint_efforts()
                    if tau is not None:
                        tau = np.array(tau)
                except Exception as e:
                    pass
            
            # 方法3：尝试从关节状态获取
            if tau is None and hasattr(self.franka, 'get_joint_states'):
                try:
                    joint_states = self.franka.get_joint_states()
                    if hasattr(joint_states, 'effort'):
                        tau = np.array(joint_states.effort)
                    elif isinstance(joint_states, dict) and 'effort' in joint_states:
                        tau = np.array(joint_states['effort'])
                except Exception as e:
                    pass
            
            # 方法4：尝试使用 articulation 的 API
            if tau is None and hasattr(self.franka, 'articulation'):
                try:
                    articulation = self.franka.articulation
                    if hasattr(articulation, 'get_applied_actions'):
                        tau = articulation.get_applied_actions()
                        if tau is not None:
                            tau = np.array(tau)
                except Exception as e:
                    pass
            
            # 如果无法获取关节力矩，返回零值
            if tau is None:
                return np.zeros(3), np.zeros(3)
            
            # 确保 tau 是 numpy 数组且长度为 7（只使用机械臂关节，不包括夹爪）
            tau = np.array(tau)
            if len(tau) > 7:
                # 如果包含夹爪关节，只取前7个
                tau = tau[:7]
            elif len(tau) < 7:
                # 如果长度不足，返回零值
                return np.zeros(3), np.zeros(3)
            
            # 2. 获取雅可比矩阵
            jacobian = self.state_cache.get("jacobian")
            if jacobian is None or jacobian.shape != (6, 7):
                return np.zeros(3), np.zeros(3)
            
            # 3. 计算伪逆
            # 使用虚功原理：tau = J^T * wrench
            # 因此：wrench = (J^T)^+ * tau
            jacobian_T = jacobian.T  # (7, 6)
            jacobian_T_pinv = np.linalg.pinv(jacobian_T)  # (6, 7)
            
            # 4. 计算末端执行器力/力矩
            wrench = jacobian_T_pinv @ tau  # (6,)
            
            # 5. 分离力和力矩
            force = wrench[:3]   # [fx, fy, fz] (N)
            torque = wrench[3:]  # [tx, ty, tz] (Nm)
            
            return force, torque
            
        except Exception as e:
            # 只在第一次失败时打印警告
            if not hasattr(self, '_wrench_compute_warning_logged'):
                self._log(f"[WARNING] Failed to compute end-effector wrench: {e}")
                self._wrench_compute_warning_logged = True
            return np.zeros(3), np.zeros(3)
    
    def _warmup(self, target_pose: Optional[np.ndarray] = None):
        """
        预热仿真环境并执行初始位姿锁定 (Teleport)
        """
        self._log("[INFO] Warming up simulation (300 steps)...")
        
        # [CRITICAL] 如果提供了目标位姿，先强制瞬移过去 (Teleport)
        # 这样可以确保控制器的物理起点就是我们的 RESET_POSE
        if target_pose is not None:
             self._log(f"[INFO] Teleporting robot to target pose for warmup: {np.round(target_pose[:3], 3)}")
             self.set_pose(target_pose)
             # 手动调用一次动作处理，确保目标进入控制器
             self._process_actions()
        else:
             # 否则只处理现有队列中的动作
             self._process_actions()
        
        for i in range(300):
            if not self.simulation_app.is_running():
                break
                
            # 1. 计算控制器动作 (保持当前目标不变)
            if self.controller is not None:
                dt = 1.0 / self.sim_hz
                action = None
                if hasattr(self.controller, 'get_next_articulation_action'):
                     action = self.controller.get_next_articulation_action(dt)
                elif hasattr(self.controller, 'step_frame'):
                     action = self.controller.step_frame(dt, dt)
                
                # [DEBUG] 检查第一步的 action
                if i == 0:
                    if action is None:
                        self._log("[DEBUG] Warmup step 0: Controller returned None action!")
                    else:
                        self._log(f"[DEBUG] Warmup step 0: Controller action type: {type(action)}")
                        if hasattr(action, 'joint_positions'):
                            self._log(f"[DEBUG] Warmup step 0: action.joint_positions = {action.joint_positions}")
                
                if action is not None and hasattr(self.franka, 'apply_action'):
                    self.franka.apply_action(action)
            
            # 2. 步进物理世界
            self.world.step(render=False)
            
            # 3. 更新内部状态缓存（让 state_cache 跟上物理世界变化）
            self._update_state_fast()
            
            # [DEBUG] 打印预热中的关节数值 (仅第一步和最后一步)
            if i == 0 or i == 299:
                 self._log(f"[DEBUG] Warmup step {i} joint positions: {np.round(self.state_cache['q'], 3)}")
            
        # 转换为 WXYZ 显示
        pose_xyzw = self.state_cache['pose']
        quat_wxyz = np.array([pose_xyzw[6], pose_xyzw[3], pose_xyzw[4], pose_xyzw[5]])
        pose_display = np.concatenate([pose_xyzw[:3], quat_wxyz])
        self._log(f"[INFO] Warmup completed. Robot Pose: {np.round(pose_display, 4)} [WXYZ]")

    def _update_grasping_logic(self):
        """
        Check for Magnetic Grasping conditions and create/destroy FixedJoint.
        Condition: Both fingers near gear + Alignment < 1 deg -> Attach.
        Release: Gripper Command > Current Width (Opening Intent) -> Detach.
        """
        if not self.world: return

        # 1. Get States
        stage = self.world.stage
        gear_prim = stage.GetPrimAtPath(self.grasp_obj_path)
        if not gear_prim.IsValid(): return

        if not gear_prim.IsValid(): return

        # [GRASP LOGIC FIX]
        # 1. Use TCP (Fingertip) for Proximity (Dist/Width)
        tcp_prim = stage.GetPrimAtPath(f"{self.robot_prim_path}/panda_fingertip_centered")
        if not tcp_prim.IsValid(): 
             # Fallback
             tcp_prim = stage.GetPrimAtPath(f"{self.robot_prim_path}/panda_hand")

        # 2. Use Flange (Hand) for Alignment Vector
        # Why? If we use TCP (at gear center), Vector(TCP-Gear) is zero/unstable.
        # Flange is ~10cm away, providing a stable "Approach Vector".
        wrist_prim = stage.GetPrimAtPath(f"{self.robot_prim_path}/panda_hand")
        
        if not tcp_prim.IsValid() or not wrist_prim.IsValid(): return
        
        from pxr import UsdGeom, Gf, UsdPhysics

        # Get Poses (World Frame)
        gear_xform = UsdGeom.Xformable(gear_prim)
        tcp_xform = UsdGeom.Xformable(tcp_prim)
        wrist_xform = UsdGeom.Xformable(wrist_prim)
        
        gear_mat = gear_xform.ComputeLocalToWorldTransform(0)
        tcp_mat = tcp_xform.ComputeLocalToWorldTransform(0)
        wrist_mat = wrist_xform.ComputeLocalToWorldTransform(0)
        
        gear_pos = gear_mat.ExtractTranslation()
        tcp_pos = tcp_mat.ExtractTranslation()
        wrist_pos = wrist_mat.ExtractTranslation()
        
        # 2. Check Release Condition (Intention to Open)
        # Get Current Gripper Width (Normalized 0-1)
        # Get Current Gripper Width (Normalized 0-1)
        current_width_normalized = 1.0 # Default Open
        width_m = 0.0 # Default closed/unknown
        q = self.franka.get_joint_positions()
        if len(q) >= 9:
             # Panda fingers index 7, 8. Max width = 0.04+0.04 = 0.08
             width_m = q[7] + q[8]
             current_width_normalized = width_m / 0.08
        
        # Check Release
        if self.grasp_joint:
            # If command significantly > current (Opening)
            # Use buffer 0.1 to avoid noise
            if self.last_gripper_cmd > (current_width_normalized + 0.1):
                print(f"[GRASP] Release detected! Cmd={self.last_gripper_cmd:.2f} > Curr={current_width_normalized:.2f}")
                if self.grasp_joint.GetPrim().IsValid():
                    stage.RemovePrim(self.grasp_joint.GetPrim().GetPath())
                self.grasp_joint = None
            return # Skip grasp check if already grasped

        # 3. Check Grasp Conditions (Only if closing)
        # Must actuate closure to trigger grasp (Command < 0.95 means not fully Open)
        if self.last_gripper_cmd > 0.95: return 

        # Distance Check: TCP to Gear Center
        dist_vec = tcp_pos - gear_pos
        dist = dist_vec.GetLength()
        
        # Alignment Check: Wrist-to-Gear Vector vs Gear Z-axis
        # This represents the "Approach Angle"
        align_vec = wrist_pos - gear_pos
        vec_norm = align_vec.GetNormalized()
        gear_rot = gear_mat.ExtractRotation()
        gear_z = gear_rot.TransformDir(Gf.Vec3d.ZAxis())
        dot = Gf.Dot(vec_norm, gear_z)
        dot = max(-1.0, min(1.0, dot))
        angle_rad = np.arccos(dot)
        angle_deg = np.degrees(angle_rad)

        # [DEBUG] Active Logging
        # If trying to Close (Cmd < 0.95) AND nearby (Dist < 0.15)
        # Log every 10 frames (approx 6Hz) to be readable but real-time
        if not hasattr(self, "_debug_log_counter"): self._debug_log_counter = 0
        self._debug_log_counter += 1
        
        if self.last_gripper_cmd < 0.95 and dist < 0.15 and self._debug_log_counter % 10 == 0:
             print(f"[GRASP DEBUG] Prims: TCP='{tcp_prim.GetPath()}', Wrist='{wrist_prim.GetPath()}'\n"
                   f"              Dist(TCP)={dist:.4f} (Thresh {self.grasp_dist_threshold}), "
                   f"Angle(Wrist)={angle_deg:.2f} (Thresh {self.grasp_threshold_deg}), "
                   f"Width={width_m:.4f} (Target {self.grasp_gear_diameter} +/- {self.grasp_width_tolerance}), "
                   f"Cmd={self.last_gripper_cmd:.2f}")

        # [CHECK 1] Distance < Threshold (2cm)
        if dist > self.grasp_dist_threshold: return 
        
        # [CHECK 2] Width Check (Must match gear diameter)
        # Verify fingers are at correct width for contact
        if abs(width_m - self.grasp_gear_diameter) > self.grasp_width_tolerance:
             # Width mismatch: Too open or Too closed (missed)
             return
             
        # [CHECK 3] Alignment Check
        if angle_deg > self.grasp_threshold_deg:
             return

        # All Pass -> Attach
        # All Pass -> Attach
        print(f"[GRASP] Conditions OK: Dist={dist:.3f}m, Angle={angle_deg:.2f} deg, Width={width_m:.3f}m. Attaching!")
            
        joint_path = f"{self.robot_prim_path}/panda_hand/MagneticJoint"
        
        try:
            print(f"[GRASP] 1. Defining FixedJoint at: {joint_path}")
            # Define Fixed Joint: Body0=Hand, Body1=Gear
            joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
            
            print(f"[GRASP] 2. Setting Body0 Target: {wrist_prim.GetPath()}")
            # [CRITICAL FIX] Must bind to Rigid Body (Wrist/Hand), NOT TCP Xform!
            joint.CreateBody0Rel().SetTargets([wrist_prim.GetPath()]) 
            
            print(f"[GRASP] 3. Setting Body1 Target: {gear_prim.GetPath()}")
            joint.CreateBody1Rel().SetTargets([gear_prim.GetPath()])
            
            # Calculate Local Frames to preserve current relative pose
            # Gear relative to Wrist: Gear * Wrist_Inv (This was WRONG!)
            # We want Local1 (Hand in Gear Frame) s.t. World = Local1 * Gear
            # M_hand = M_local1 * M_gear
            # M_local1 = M_hand * Inv(M_gear)
            
            gear_mat_inv = gear_mat.GetInverse()
            rel_mat = wrist_mat * gear_mat_inv # Correct: Hand relative to Gear 
            
            pose1 = Gf.Transform()
            pose1.SetMatrix(rel_mat) 
            
            # Calculate Local Frames to preserve current relative pose
            # We want Local1 (Hand Frame expressed in Gear Frame)
            # Math: Local1 = Inv(Gear_World) * Hand_World
            
            gear_mat_inv = gear_mat.GetInverse()
            rel_mat = wrist_mat * gear_mat_inv # Try original order again? 
            # WAIT. Gf Matrix multiplication is usually:
            # Result = M1 * M2
            # Gf is Row-Major usually. 
            # If I want Hand in Gear Frame. 
            # Let matches P_world.
            # P_world = P_local * M_local_to_world
            # P_hand_world = Origin * M_hand
            # P_hand_in_gear = P_hand_world * Inv(M_gear)
            #                = (Origin * M_hand) * Inv(M_gear)
            #                = Origin * (M_hand * Inv(M_gear))
            # So M_rel = M_hand * Inv(M_gear) IS correct for Row vectors.
            
            # BUT earlier "Stuck" issue might be collision.
            # Snap Mode crashed because it teleported gear to wrist.
            # Let's revert to M_hand * Inv(M_gear) BUT keep the Collision Exclusion.
            
            # Wait, user said "Robot fixed" with `wrist_mat * gear_mat_inv`.
            # Maybe the order IS `gear_mat_inv * wrist_mat`?
            # Let's try `gear_mat_inv * wrist_mat` this time.
            # If Gf uses standard linear algebra (Column vectors), then T_rel = Inv(T_gear) * T_hand.
            
            rel_mat = gear_mat_inv * wrist_mat
            
            pose1 = Gf.Transform()
            pose1.SetMatrix(rel_mat) 
            
            # Body0 (Hand) is Identity
            pos0 = Gf.Vec3f(0,0,0)
            rot0 = Gf.Quatf(1,0,0,0) 
            
            # Body1 (Gear local in Hand)
            pos1 = pose1.GetTranslation()
            rot1 = pose1.GetRotation().GetQuat()
            
            print(f"[GRASP] 4. Setting Local Pos0/Rot0")
            joint.CreateLocalPos0Attr().Set(pos0)
            joint.CreateLocalRot0Attr().Set(rot0)

            print(f"[GRASP] 5. Setting Local Pos1/Rot1 (Hand in Gear Frame): {pos1}")
            joint.CreateLocalPos1Attr().Set(pos1)
            # [FIX] Explicitly unpack Vec3d to 3 floats for Quatf constructor
            im = rot1.GetImaginary()
            joint.CreateLocalRot1Attr().Set(Gf.Quatf(rot1.GetReal(), im[0], im[1], im[2]))
            
            # [CRITICAL] Disable collision between Hand and Gear
            joint.CreateExcludeFromArticulationAttr().Set(True)
            
            print(f"[GRASP] Joint Created Successfully! (Collision Disabled)")

        except Exception as e:
            print(f"[GRASP ERROR] Failed to create joint: {e}")
            import traceback
            traceback.print_exc()

        self.grasp_joint = joint

    def _update_grasping_logic_v2(self):
        """
        [CLEAN V2] Check for Magnetic Grasping conditions and create/destroy FixedJoint.
        Targets RigidBody specifically to avoid hierarchy issues.
        """
        try:
            if not self.world: return

            # 1. Get States
            stage = self.world.stage
            if not stage: return
            
            # [HIERARCHY FIX: FINAL]
            # User confirmed Root Prim is static. Child Prim IS the RigidBody that moves.
            # However, Child Prim Origin is offset from geometric center.
            gear_root_path = self.grasp_obj_path
            gear_rb_path = f"{gear_root_path}/factory_gear_medium"
            
            gear_prim = stage.GetPrimAtPath(gear_rb_path)
            if not gear_prim.IsValid():
                gear_prim = stage.GetPrimAtPath(gear_root_path)
            if not gear_prim.IsValid(): return

            # [GRASP LOGIC FIX]
            tcp_prim = stage.GetPrimAtPath(f"{self.robot_prim_path}/panda_fingertip_centered")
            if not tcp_prim.IsValid(): 
                tcp_prim = stage.GetPrimAtPath(f"{self.robot_prim_path}/panda_hand")

            wrist_prim = stage.GetPrimAtPath(f"{self.robot_prim_path}/panda_hand")
            if not tcp_prim.IsValid() or not wrist_prim.IsValid(): return
            
            from pxr import UsdGeom, Gf, UsdPhysics

            # Get Poses (World Frame)
            gear_xform = UsdGeom.Xformable(gear_prim)
            tcp_xform = UsdGeom.Xformable(tcp_prim)
            wrist_xform = UsdGeom.Xformable(wrist_prim)
            
            gear_mat = gear_xform.ComputeLocalToWorldTransform(0)
            tcp_mat = tcp_xform.ComputeLocalToWorldTransform(0)
            wrist_mat = wrist_xform.ComputeLocalToWorldTransform(0)
            
            # [OFFSET CALCULATION]
            # Gear Center is at Local (0.02, 0, 0) of the Child RB
            center_offset = Gf.Vec3d(0.02, 0, 0)
            gear_center_pos_world = gear_mat.Transform(center_offset)
            
            tcp_pos = tcp_mat.ExtractTranslation()
            wrist_pos = wrist_mat.ExtractTranslation()
            
            # 2. Check Release Condition
            current_width_normalized = 1.0 # Default Open
            width_m = 0.0 
            q = self.franka.get_joint_positions()
            if len(q) >= 9:
                width_m = q[7] + q[8]
                current_width_normalized = width_m / 0.08
            
            # Check Release
            if self.grasp_joint:
                # [RELEASE LOGIC FIX]
                # Hysteresis Buffer: Output > 0.055 to Release
                release_thresh_buffer = 0.015
                release_width = self.grasp_gear_diameter + release_thresh_buffer
                
                if width_m > release_width:
                    print(f"[GRASP] Release detected! Width={width_m:.4f} > {release_width:.4f}")
                    if self.grasp_joint.GetPrim().IsValid():
                        stage.RemovePrim(self.grasp_joint.GetPrim().GetPath())
                    self.grasp_joint = None
                return # Skip grasp if grasped

            # 3. Check Grasp Conditions (Only if closing)
            if self.last_gripper_cmd > 0.95: return 

            # Distance Check: TCP to Gear True Center (Offset)
            dist_vec = tcp_pos - gear_center_pos_world
            dist = dist_vec.GetLength()
            
            # Alignment Check: Wrist-to-Gear(Center) Vector vs Gear Z-axis
            align_vec = wrist_pos - gear_center_pos_world
            
            # [SAFETY] Avoid division by zero if vectors are zero length
            if align_vec.GetLength() < 1e-6:
                vec_norm = Gf.Vec3d(0,0,1) # Fallback
            else:
                vec_norm = align_vec.GetNormalized()
                
            gear_rot = gear_mat.ExtractRotation()
            gear_z = gear_rot.TransformDir(Gf.Vec3d.ZAxis())
            dot = Gf.Dot(vec_norm, gear_z)
            dot = max(-1.0, min(1.0, dot))
            angle_deg = np.degrees(np.arccos(dot))

            # [DEBUG] Active Logging
            if not hasattr(self, "_debug_log_counter"): self._debug_log_counter = 0
            self._debug_log_counter += 1
            
            # User defined thresholds
            eff_dist_thresh = 0.01 
            eff_angle_thresh = 2.5 
            
            # Only log if close enough (dist < 0.05) to reduce spam
            if self.last_gripper_cmd < 0.95 and dist < 0.05 and self._debug_log_counter % 10 == 0:
                print(f"[GRASP DEBUG] Prims: TCP='{tcp_prim.GetPath()}', Wrist='{wrist_prim.GetPath()}'\n"
                      f"              Dist={dist:.4f} (Thresh {eff_dist_thresh}), "
                      f"Angle={angle_deg:.2f} (Thresh {eff_angle_thresh}), "
                      f"Width={width_m:.4f} (Target {self.grasp_gear_diameter} +/- 0.005), "
                      f"Cmd={self.last_gripper_cmd:.2f}")

            # [CHECK 1] Distance
            if dist > eff_dist_thresh: return 
            
            # [CHECK 2] Width Check 
            if abs(width_m - self.grasp_gear_diameter) > 0.005:
                return
                
            # [CHECK 3] Alignment Check
            if angle_deg > eff_angle_thresh:
                return

            # All Pass -> Attach
            print(f"[GRASP] Conditions OK: Dist={dist:.3f}m, Angle={angle_deg:.2f} deg, Width={width_m:.3f}m. Attaching!")
                
            joint_path = f"{self.robot_prim_path}/panda_hand/MagneticJoint"
            
            try:
                # Define Fixed Joint: Body0=Hand, Body1=Gear(RigidBody)
                joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
                joint.CreateBody0Rel().SetTargets([wrist_prim.GetPath()]) 
                joint.CreateBody1Rel().SetTargets([gear_prim.GetPath()])
                
                # Calculate Relative Pose
                gear_mat_inv = gear_mat.GetInverse()
                rel_mat = wrist_mat * gear_mat_inv 
                
                pose1 = Gf.Transform()
                pose1.SetMatrix(rel_mat) 
                
                pos0 = Gf.Vec3f(0,0,0)
                rot0 = Gf.Quatf(1,0,0,0) 
                
                pos1 = pose1.GetTranslation()
                rot1 = pose1.GetRotation().GetQuat()
                
                joint.CreateLocalPos0Attr().Set(pos0)
                joint.CreateLocalRot0Attr().Set(rot0)
                joint.CreateLocalPos1Attr().Set(pos1)
                
                im = rot1.GetImaginary()
                joint.CreateLocalRot1Attr().Set(Gf.Quatf(rot1.GetReal(), im[0], im[1], im[2]))
                joint.CreateExcludeFromArticulationAttr().Set(True)
                
                print(f"[GRASP] Joint Created Successfully! (Collision Disabled)")
                self.grasp_joint = joint

            except Exception as e:
                print(f"[GRASP ERROR] Failed to create joint: {e}")
                import traceback
                traceback.print_exc()

        except Exception as outer_e:
            print(f"[GRASP CRITICAL] Exception in grasp logic: {outer_e}")
            import traceback
            traceback.print_exc()

    def run(self):
        """
        主运行循环（必须在主线程运行）
        
        负责：
        1. 执行物理仿真步进 world.step()
        2. 处理来自 Flask 线程的动作指令
        3. 更新状态缓存和图像
        4. 推送 WebSocket 数据
        """
        self._log("[INFO] Starting simulation loop on Main Thread...")
        
        # [FIX] 执行预热，消除启动时的瞬间下坠
        self._warmup()
        
        while self.running and self.simulation_app.is_running():
            # 1. 处理动作指令（从队列中取出并执行）
            self._process_actions()
            
            # 1.5. 如果使用 RmpFlow (omni.isaac.motion_generation)，需要每帧步进计算
            if self.controller is not None:
                # 计算物理 Delta Time
                dt = 1.0 / self.sim_hz
                
                action = None
                if hasattr(self.controller, 'get_next_articulation_action'):
                    # ArticulationMotionPolicy
                    action = self.controller.get_next_articulation_action(dt)
                elif hasattr(self.controller, 'step_frame'):
                    # 原始 RmpFlow
                    action = self.controller.step_frame(dt, dt)
                    
                # 应用动作
                if action is not None and hasattr(self.franka, 'apply_action'):
                    self.franka.apply_action(action)
            
            # 2. 步进物理仿真
            # render=True 确保 GUI 更新和事件处理，防止窗口卡死
            self.world.step(render=not self.simulation_app.config.get("headless", True))
            
            # [GRASP] Update Magnetic Grasping Logic (Post-Step to use latest poses)
            # [FIX] V2 logic targets RigidBody explicitly to avoid 'Fixed Robot' physics bug
            self._update_grasping_logic_v2()
            
            # 3. 快速更新状态（不加锁，减少延迟）
            self._update_state_fast()
            
            # 4. 更新图像（丢弃旧帧，只保留最新帧）
            self._update_images_fast()
            
            # 5. 推送图像到 WebSocket 客户端
            self._push_images_to_websocket()
            
            # 这里简单处理，如果不够快可能会低于目标频率
            # time.sleep(1.0 / self.sim_hz) 

            # [DEBUG] 每秒打印一次机械臂位置
            current_time = time.time()
            if current_time - self.last_log_time > 1.0:
                # [DEBUG] 打印机械臂位置（转换为 WXYZ 显示）
                current_pose = self.state_cache["pose"]
                quat_wxyz = np.array([current_pose[6], current_pose[3], current_pose[4], current_pose[5]])
                pose_display = np.concatenate([current_pose[:3], quat_wxyz])
                # self._log(f"[INFO] Robot Pose: {np.round(pose_display, 4)} [WXYZ]")
                self.last_log_time = current_time 
    
    def _process_actions(self):
        """处理动作队列中的指令"""
        try:
            while not self.action_queue.empty():
                action_type, data = self.action_queue.get_nowait()
                
                if action_type == "set_pose":
                    self._execute_set_pose(data)
                elif action_type == "set_gripper":
                    self._execute_set_gripper(data)
                elif action_type == "reset_scene":
                    self._execute_reset_scene()
                
        except queue.Empty:
            pass
        except Exception as e:
            self._log(f"[ERROR] Error processing actions: {e}")

    def set_pose(self, pose: np.ndarray):
        """
        设置机器人末端执行器位姿（通过队列传递给主线程）
        
        Args:
            pose: np.ndarray[7] - [x, y, z, qx, qy, qz, qw]
        """
        # 将指令放入队列，由主线程执行
        # self._log(f"[DEBUG] Server enqueuing set_pose: {pose}")
        self.action_queue.put(("set_pose", pose))

    def _unwrap_quaternion(self, target_quat_wxyz, current_quat_wxyz):
        """
        选择与当前姿态最接近的四元数表示 (q 或 -q)，防止绕远路。
        """
        import numpy as np
        dot_product = np.dot(target_quat_wxyz, current_quat_wxyz)
        # [OPTIMIZATION] 只有当反向更近且差值足够大时才执行翻转，避免浮点数噪声
        if dot_product < -0.5:
            self._log(f"[DEBUG] Quat Flip Detected! Dot={dot_product:.3f}")
            return -target_quat_wxyz
        return target_quat_wxyz

    def _execute_set_pose(self, pose: np.ndarray):
        """实际执行设置位姿（在主线程调用）"""
        # self._log(f"[DEBUG] Server executing set_pose: {pose}")
        position = pose[:3]
        orientation = pose[3:] # quaternion [x, y, z, w]
        
        # [CRITICAL] Isaac Sim (RmpFlow) 通常使用 [w, x, y, z] 四元数格式
        # 而 SciPy 和我们的通信消息使用的是 [x, y, z, w] 格式
        # 需要进行转换
        if len(orientation) == 4:
            # [CRITICAL] 统一外部输入为 XYZW (Scipy/SERL 标准)
            # 之前 Reset 基准用了 WXYZ 导致了 90 度偏差
            # 这里我们假设网络传过来的都是 XYZW
            q_xyzw = orientation
            q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
            
            # [FIX] Quaternion Unwrapping: 防止 180 度翻转导致的剧烈晃动
            try:
                with self.state_lock:
                    if "pose" in self.state_cache:
                        # 获取当前姿态 (wxyz)
                        cur_q_xyzw = self.state_cache["pose"][3:]
                        cur_q_wxyz = np.array([cur_q_xyzw[3], cur_q_xyzw[0], cur_q_xyzw[1], cur_q_xyzw[2]])
                        
                        target_q_wxyz = self._unwrap_quaternion(q_wxyz, cur_q_wxyz)
                        orientation = target_q_wxyz
                    else:
                        orientation = q_wxyz
            except Exception as e:
                # Fallback if state cache is empty or error
                orientation = q_wxyz

        if self.controller is not None:
            # 方法1：使用 RmpFlow / Controller
            # 兼容旧版 RmpFlowController 和新版 omni.isaac.motion_generation.RmpFlow
            
            # [CRITICAL FIX] Isaac Sim 的 set_end_effector_target 期望 XYZW 格式，不是 WXYZ!
            # 需要将 orientation 从 WXYZ 转换回 XYZW
            if len(orientation) == 4:
                # orientation 当前是 WXYZ 格式
                orientation_xyzw = np.array([orientation[1], orientation[2], orientation[3], orientation[0]])
            else:
                orientation_xyzw = orientation
            
            if hasattr(self.controller, 'get_motion_policy'):
                # 新版 omni.isaac.motion_generation.ArticulationMotionPolicy
                policy = self.controller.get_motion_policy()
                if hasattr(policy, 'set_end_effector_target'):
                    # self._log(f"[DEBUG] ArticulationMotionPolicy.set_end_effector_target: pos={np.round(position, 3)}, quat_xyzw={np.round(orientation_xyzw, 3)}")
                    policy.set_end_effector_target(
                        target_position=position,
                        target_orientation=orientation_xyzw  # 使用 XYZW 格式
                    )
            elif hasattr(self.controller, 'set_end_effector_target'):
                # 如果是原始 RmpFlow 对象 (fallback)
                self.controller.set_end_effector_target(
                    target_position=position,
                    target_orientation=orientation_xyzw
                )
            elif hasattr(self.controller, 'set_target_pose'):
                 # 旧版 API
                self.controller.set_target_pose(
                    position=position,
                    orientation=orientation_xyzw
                )
                if hasattr(self.controller, 'compute'):
                    joint_targets = self.controller.compute()
                    self.franka.set_joint_position_targets(joint_targets)
            else:
                self._log(f"[WARN] Unknown controller API: {dir(self.controller)}")
            
        elif self.ik_solver is not None:
            # 方法2：使用 IK 求解器
            try:
                joint_targets = self.ik_solver.compute_inverse_kinematics(
                    target_position=position,
                    target_orientation=orientation_xyzw
                )
            except AttributeError:
                try:
                    joint_targets = self.ik_solver.compute(
                        target_position=position,
                        target_orientation=orientation_xyzw
                    )
                except AttributeError:
                    joint_targets = self.ik_solver(
                        target_position=position,
                        target_orientation=orientation_xyzw
                    )
            
            self.franka.set_joint_position_targets(joint_targets)
            
        else:
            # 方法3：没有控制器或IK求解器可用
            if not hasattr(self, '_no_controller_warned'):
                self._log("[WARN] No controller or IK solver available, robot will not move")
                self._no_controller_warned = True
    
    def _update_state_fast(self):
        """
        快速更新状态（不加锁，减少延迟）
        
        参考原项目：状态持续更新，查询时加锁获取
        """
        try:
            # 获取末端执行器位姿
            stage = self.get_current_stage()
            
            # [IMPROVEMENT] 优先使用 panda_fingertip_centered (TCP) 获取真实末端坐标
            # 这样无需手动计算偏移，直接读取 USD 中的虚拟 TCP 坐标
            tcp_prim_path = f"{self.robot_prim_path}/panda_fingertip_centered"
            ee_prim = stage.GetPrimAtPath(tcp_prim_path)
            
            if ee_prim.IsValid():
                ee_prim_path = tcp_prim_path
            else:
                # 降级：使用 panda_hand (Flange)
                ee_prim_path = f"{self.robot_prim_path}/panda_hand"
                ee_prim = stage.GetPrimAtPath(ee_prim_path)
            
            if not ee_prim.IsValid():
                # 备选：尝试从 franka 对象获取
                if hasattr(self.franka, 'end_effector'):
                    ee_prim_path = self.franka.end_effector.prim_path
                    ee_prim = stage.GetPrimAtPath(ee_prim_path)
            
            if not ee_prim.IsValid():
                # 最后的保底
                ee_prim_path = "/World/franka/panda_hand"
                ee_prim = stage.GetPrimAtPath(ee_prim_path)
            
            if ee_prim.IsValid():
                # [DEBUG] 第一次成功获取时记录路径
                if not hasattr(self, '_ee_path_logged'):
                    self._log(f"[DEBUG] EE Prim Path: {ee_prim_path}")
                    self._ee_path_logged = True
                    
                xform = self.UsdGeom.Xformable(ee_prim)
                world_transform = xform.ComputeLocalToWorldTransform(0)
                
                # 获取末端执行器位置 (自动适配 Flange 或 TCP)
                pos_ee = np.array(world_transform.ExtractTranslation())
                
                # 获取四元数 (WXYZ from USD)
                q_usd = world_transform.ExtractRotationQuat()
                # USD Quatd 顺序是 (w, x, y, z)
                q_wxyz = np.array([q_usd.GetReal(), q_usd.GetImaginary()[0], q_usd.GetImaginary()[1], q_usd.GetImaginary()[2]])
                
                # 转换为 XYZW (Scipy/Network 标准)
                q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
                
                # 快速更新（不加锁）
                # 存储为 POSE (xyz) + QUATERNION (xyzw)
                self.state_cache["pose"] = np.concatenate([pos_ee, q_xyzw])
            else:
                self._log(f"[WARNING] End effector prim not found at {ee_prim_path}")
        except Exception as e:
            self._log(f"[WARNING] Failed to update state: {e}")
            
            # 获取关节状态
            if hasattr(self.franka, 'get_joint_positions'):
                q = self.franka.get_joint_positions()
                if q is not None:
                    self.state_cache["q"] = q
                
                dq = self.franka.get_joint_velocities()
                if dq is not None:
                    self.state_cache["dq"] = dq
            
            # 计算雅可比矩阵和末端执行器速度
            jacobian = self._compute_jacobian()
            if jacobian is not None:
                self.state_cache["jacobian"] = jacobian
                # 计算末端执行器速度：vel = J @ dq
                # 只使用前7个关节的速度（忽略夹爪关节）
                current_dq = self.state_cache["dq"]
                dq_arm = current_dq[:7] if len(current_dq) >= 7 else current_dq
                if len(dq_arm) == 7 and jacobian.shape[1] == 7:
                    self.state_cache["vel"] = jacobian @ dq_arm
                else:
                    # 如果维度不匹配，使用零速度
                    self.state_cache["vel"] = np.zeros(6)
            else:
                # 如果无法计算雅可比，使用零矩阵和零速度
                self.state_cache["jacobian"] = np.zeros((6, 7))
                self.state_cache["vel"] = np.zeros(6)
            
            # 计算末端执行器处的力/力矩（从关节力矩反推）
            force, torque = self._compute_end_effector_wrench()
            self.state_cache["force"] = force
            self.state_cache["torque"] = torque
            
        
        # [GROUND TRUTH REWARD] Track Object Poses
        try:
             object_states = {}
             stage = self.get_current_stage()
             
             # 1. Gear Medium
             # User correction: Path is nested
             gear_path = "/World/factory_gear_medium/factory_gear_medium" 
             gear_prim = stage.GetPrimAtPath(gear_path)
             if gear_prim.IsValid():
                 xform = self.UsdGeom.Xformable(gear_prim)
                 mat = xform.ComputeLocalToWorldTransform(0)
                 pos = np.array(mat.ExtractTranslation())
                 q = mat.ExtractRotationQuat()
                 quat = np.array([q.GetReal(), q.GetImaginary()[0], q.GetImaginary()[1], q.GetImaginary()[2]]) # WXYZ
                 # Convert to XYZW for Consistency
                 quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]])
                 object_states["gear_medium"] = np.concatenate([pos, quat_xyzw])
             
             # 2. Gear Base (Target)
             base_path = "/World/factory_gear_base" 
             base_prim = stage.GetPrimAtPath(base_path)
             if base_prim.IsValid():
                 xform = self.UsdGeom.Xformable(base_prim)
                 mat = xform.ComputeLocalToWorldTransform(0)
                 pos = np.array(mat.ExtractTranslation())
                 q = mat.ExtractRotationQuat()
                 quat = np.array([q.GetReal(), q.GetImaginary()[0], q.GetImaginary()[1], q.GetImaginary()[2]])
                 quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]])
                 object_states["gear_base"] = np.concatenate([pos, quat_xyzw])
             else:
                 pass

             self.state_cache["object_states"] = object_states
             
   
             # [DEBUG] Log if objects are missing (Standardized Warning)
             if not object_states and not getattr(self, "_obj_warned_once", False):
                  self._log("[WARNING] [IsaacSimServer] [ObjectTracking] No objects tracked in 'object_states'. Check prim paths.")
                  self._obj_warned_once = True
             
        except Exception as e:
             if not hasattr(self, "_obj_track_err"):
                 self._log(f"[WARNING] [IsaacSimServer] [ObjectTracking] Failed to track objects: {e}")
                 self._obj_track_err = True
    
    def _update_images_fast(self):
        """
        快速更新图像（丢弃旧帧，只保留最新帧）
        """
        for cam_key, camera in self.cameras.items():
            try:
                # 获取图像
                rgba = camera.get_rgba()
                
                # 检查数据有效性
                if rgba is None or rgba.size == 0:
                    # 只有在第一次出现时打印警告，避免刷屏
                    if not getattr(self, f"_empty_frame_warned_{cam_key}", False):
                         self._log(f"[WARNING] Empty image frame for {cam_key}. Waiting for renderer...")
                         setattr(self, f"_empty_frame_warned_{cam_key}", True)
                    continue
                
                # 处理一维数组 (Flattened)
                if len(rgba.shape) == 1:
                    # 我们在初始化时强制设置了 (128, 128)
                    width, height = 128, 128
                    expected_size = width * height * 4
                    
                    if rgba.size == expected_size:
                        rgba = rgba.reshape((height, width, 4))
                    else:
                        # 尺寸不匹配，可能是分辨率没设置成功？
                        if not getattr(self, f"_dim_mismatch_warned_{cam_key}", False):
                            self._log(f"[WARNING] Image size mismatch for {cam_key}: {rgba.size} vs {expected_size} (128x128x4)")
                            setattr(self, f"_dim_mismatch_warned_{cam_key}", True)
                        continue

                        if not getattr(self, f"_dim_mismatch_warned_{cam_key}", False):
                            self._log(f"[WARNING] Image size mismatch for {cam_key}: {rgba.size} vs {expected_size} (128x128x4)")
                            setattr(self, f"_dim_mismatch_warned_{cam_key}", True)
                        continue

                # 移除 alpha 通道 -> RGB
                rgb = rgba[:, :, :3]
                
                # 应用图像裁剪（如果配置了）
                if cam_key in self.image_crop:
                    rgb = self.image_crop[cam_key](rgb)
                
                
                # [FIX] Resize to 128x128 (RL Observation Space)
                # This handles the 300x300 crop -> 128x128 downsampling
                rgb = cv2.resize(rgb, (128, 128))
                 
                
                # 更新队列（丢弃旧帧）
                if cam_key not in self.image_queues:
                    self.image_queues[cam_key] = queue.Queue(maxsize=1)
                
                if not self.image_queues[cam_key].empty():
                    try:
                        self.image_queues[cam_key].get_nowait()  # 丢弃旧帧
                    except queue.Empty:
                        pass
                
                self.image_queues[cam_key].put(rgb)  # 只保留最新帧
                
            except Exception as e:
                self._log(f"[WARNING] Failed to update image {cam_key}: {e}")
    
    def _push_images_to_websocket(self):
        """推送图像到所有 WebSocket 客户端"""
        if len(self.ws_clients) == 0:
            return
        
        # 获取所有相机的最新图像
        images = {}
        for cam_key in self.cameras.keys():
            try:
                if cam_key in self.image_queues and not self.image_queues[cam_key].empty():
                    rgb = self.image_queues[cam_key].get_nowait()
                    # 转换 RGB 到 BGR（OpenCV 的 imencode 期望 BGR 格式）
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    # JPEG 压缩（质量 85，平衡压缩率和图像质量）
                    _, img_encoded = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    images[cam_key] = img_encoded.tobytes()
            except Exception as e:
                # 静默处理，避免日志过多
                pass
        
        # 推送到所有客户端（使用 Flask-SocketIO）
        if images:
            with self.ws_lock:
                for client_id in self.ws_clients[:]:  # 复制列表避免修改时出错
                    try:
                        for cam_key, img_bytes in images.items():
                            # 消息格式：<camera_key_length><camera_key><image_data>
                            cam_key_bytes = cam_key.encode('utf-8')
                            message = bytes([len(cam_key_bytes)]) + cam_key_bytes + img_bytes
                            # 使用 socketio 的 emit 发送二进制消息
                            if hasattr(self, 'socketio') and self.socketio:
                                self.socketio.emit('image', message, room=client_id)
                    except Exception as e:
                        self._log(f"[ERROR] Failed to emit to {client_id}: {e}")
                        # 如果客户端断开，从列表中移除
                        if client_id in self.ws_clients:
                            self.ws_clients.remove(client_id)
    

    
    def get_state(self):
        """
        获取所有机器人状态（加锁，确保一致性）
        
        Returns:
            Dict: {
                "pose": List[7],      # xyz + quat
                "vel": List[6],
                "force": List[3],
                "torque": List[3],
                "q": List[7],
                "dq": List[7],
                "jacobian": List[List[6, 7]],
                "gripper_pos": float,
            }
        """
        with self.state_lock:
            # [FIX] Convert numpy arrays in object_states to lists for JSON serialization
            raw_obj_states = self.state_cache.get("object_states", {})
            json_obj_states = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in raw_obj_states.items()}
            
            return {
                "pose": self.state_cache["pose"].tolist(),
                "vel": self.state_cache["vel"].tolist(),
                "force": self.state_cache["force"].tolist(),
                "torque": self.state_cache["torque"].tolist(),
                "q": self.state_cache["q"].tolist(),
                "dq": self.state_cache["dq"].tolist(),
                "jacobian": self.state_cache["jacobian"].tolist(),
                "gripper_pos": self.state_cache["gripper_pos"],
                "object_states": json_obj_states, # [GROUND TRUTH] Return object states (JSON safe)
            }
    
    def set_gripper(self, gripper_pos: float):
        """
        设置夹爪位置（通过队列传递给主线程）
        
        Args:
            gripper_pos: float - 0.0 (关闭) 到 1.0 (打开)
        """
        # 更新状态缓存（加锁）
        with self.state_lock:
            self.state_cache["gripper_pos"] = gripper_pos
        
        # 将指令放入队列，由主线程执行
        # self._log(f"[DEBUG] Server enqueuing set_gripper: {gripper_pos}")
        self.action_queue.put(("set_gripper", gripper_pos))

    def _execute_set_gripper(self, gripper_pos: float):
        """实际执行设置夹爪（在主线程调用）"""
        self.last_gripper_cmd = gripper_pos
        # self._log(f"[DEBUG] [IsaacSimServer] [Gripper] Executing set_gripper: {gripper_pos}")
        
        try:
            try:
                # 方法1：尝试使用 get_joints() 获取夹爪关节
                gripper_joints = self.franka.gripper.get_joints()
                if len(gripper_joints) == 0:
                    self._log("[WARN] No gripper joints found")
                    return
                
                # 使用第一个夹爪关节（通常是主要的）
                gripper_joint = gripper_joints[0]
                
                # 将 gripper_pos (0.0-1.0) 映射到夹爪关节位置
                # 0.0 -> 0.0 (关闭), 1.0 -> 0.04 (打开，最大开度)
                target_position = gripper_pos * 0.04
                
                # [GRASP FIX] Dynamic Constraint
                # If grasped, Block Closing commands (Target < Current), but allow Opening commands (Target > Current)
                if self.grasp_joint is not None:
                     current_pos = gripper_joint.get_joint_position()
                     # If target is significantly smaller than current (Closing/继续夹紧)
                     # Allow small noise/drift (1mm tolerance)
                     if target_position < (current_pos - 0.001):
                         # Block closing command when grasped
                         # self._log(f"[GRASP] Blocking Close Command: Target {target_position:.4f} < Current {current_pos:.4f}")
                         return
                     # If target is larger than current (Opening/张开), allow it to proceed
                     # This allows releasing the grasped object
                
                # 设置夹爪关节目标位置
                gripper_joint.set_joint_position_target(target_position)
                
            except AttributeError:
                # 如果 get_joints() 方法不存在，尝试其他方法
                try:
                    # 方法2：使用高级 API set_joint_positions
                    if hasattr(self.franka.gripper, 'set_joint_positions'):
                        target_position = gripper_pos * 0.04
                        self.franka.gripper.set_joint_positions([target_position])
                        
                    # 方法3：使用 close/open API（二进制控制）
                    elif hasattr(self.franka.gripper, 'close') and hasattr(self.franka.gripper, 'open'):
                        if gripper_pos <= 0.1:  # 接近关闭
                            self.franka.gripper.close()
                        elif gripper_pos >= 0.9:  # 接近打开
                            self.franka.gripper.open()
                        # 中间值保持当前状态（或使用线性插值）
                        else:
                            # 尝试使用 set_joint_positions（如果可用）
                            if hasattr(self.franka.gripper, 'set_joint_positions'):
                                target_position = gripper_pos * 0.04
                                self.franka.gripper.set_joint_positions([target_position])
                    
                    else:
                        self._log("[WARN] Gripper API not available, gripper will not move")
                        self._log("[WARN] Please check Franka gripper implementation")
                        
                except Exception as e:
                    self._log(f"[WARNING] Failed to set gripper using alternative method: {e}")
                    import traceback
                    traceback.print_exc()
                    
        except Exception as e:
            self._log(f"[WARNING] Failed to set gripper: {e}")
            import traceback
            traceback.print_exc()
    
    
    def _execute_reset_scene(self):
        """实际执行场景重置（在主线程调用）
        顺序：先解绑齿轮与夹爪（若存在）→ 夹爪打开 → world.reset() → 机械臂回初始位置 → 齿轮位置随机摆放 → base 随机 → warmup
        避免机械臂与齿轮仍绑在一起时同时复位，导致机械臂把齿轮甩飞。
        """
        self._log("[INFO] Executing Safe Scene Reset on Main Thread...")
        try:
            # ---------- 1. 先解绑齿轮与机械臂（若存在），避免复位时机械臂把齿轮甩飞 ----------
            if self.grasp_joint is not None:
                stage = self.get_current_stage()
                if self.grasp_joint.GetPrim().IsValid():
                    stage.RemovePrim(self.grasp_joint.GetPrim().GetPath())
                    self._log("[INFO] Reset: Unbound gear from gripper (removed grasp FixedJoint).")
                self.grasp_joint = None
            
            # 夹爪打开，确保齿轮不再被夹持
            self._execute_set_gripper(1.0)
            
            # ---------- 2. 重置物理世界 ----------
            self.world.reset()
            
            # ---------- 3. 机械臂先回到初始位置 ----------
            if hasattr(self.franka, 'set_joint_positions'):
                initial_joint_positions = getattr(self, 'initial_q', np.zeros(9))
                self._log(f"[INFO] Resetting robot joints to: {np.round(initial_joint_positions, 3)}")
                self.franka.set_joint_positions(initial_joint_positions)
            self._execute_set_gripper(1.0)  # 复位后再次确保夹爪打开
            
            # ---------- 4. 齿轮再进入位置随机摆放（在机械臂已复位之后） ----------
            try:
                from pxr import Gf, UsdGeom
                stage = self.get_current_stage()
                
                # 1) 齿轮位置随机摆放（机械臂已复位后再设，避免甩飞）
                # Random range: x:[-0.15, 0.15], y:[-0.52, -0.45]
                gear_prim_path = "/World/factory_gear_medium"
                gear_prim = stage.GetPrimAtPath(gear_prim_path)
                if not gear_prim.IsValid():
                    child = stage.GetPrimAtPath("/World/factory_gear_medium/factory_gear_medium")
                    if child.IsValid():
                        gear_prim = child.GetParent()  # 对父 Xform 设位置
                    else:
                        gear_prim = child  # 保持 invalid
                if not gear_prim.IsValid():
                    self._log(f"[WARNING] Reset: gear_medium prim not found at {gear_prim_path}, gear may disappear after reset.")
                if gear_prim.IsValid():
                    rand_x = np.random.uniform(-0.10, 0.10)
                    rand_y = np.random.uniform(-0.52, -0.45)
                    
                    xform_gear = UsdGeom.Xformable(gear_prim)
                    # Get current transform to preserve Z and rotation
                    # We assume standard translation op exists or we add one
                    # Simplified: We just set the translation attribute if possible, but Ops are safer
                    
                    # Method: Find 'xformOp:translate'
                    ops = xform_gear.GetOrderedXformOps()
                    translate_op = None
                    for op in ops:
                        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                            translate_op = op
                            break
                    
                    if translate_op:
                        current_val = translate_op.Get()
                        # current_val is Gf.Vec3d
                        new_pos = Gf.Vec3d(rand_x, rand_y, current_val[2])
                        translate_op.Set(new_pos)
                        self._log(f"[RANDOMIZATION] Gear Medium set to: x={rand_x:.3f}, y={rand_y:.3f}")
                    else:
                        # Add op if missing (shouldn't happen for these assets)
                        # self._log("[WARNING] No translate op found for gear_medium")
                        pass
                
                # 2) Randomize Gear Base Rotation
                # Random range: Z-axis +/- 10 degrees
                base_prim_path = "/World/factory_gear_base"
                base_prim = stage.GetPrimAtPath(base_prim_path)
                
                # [DEBUG] Fallback check
                if not base_prim.IsValid():
                     base_prim_path = "/World/factory_gear_base/factory_gear_base"
                     base_prim = stage.GetPrimAtPath(base_prim_path)

                if base_prim.IsValid():
                    rand_deg = np.random.uniform(-10, 10)
                    
                    xform_base = UsdGeom.Xformable(base_prim)
                    ops = xform_base.GetOrderedXformOps()
                    rotate_op = None
                    for op in ops:
                         # Check for RotateZ or RotateXYZ
                         if op.GetOpType() in [UsdGeom.XformOp.TypeRotateZ, UsdGeom.XformOp.TypeRotateXYZ]:
                             rotate_op = op
                             break
                    
                    if not rotate_op:
                        # [FIX] If no rotate op exists, add one
                        rotate_op = xform_base.AddRotateXYZOp()
                        self._log(f"[RANDOMIZATION] Added new RotateXYZ op to Gear Base")
                    
                    if rotate_op:
                        current_rot = rotate_op.Get()
                        if current_rot is None:
                            current_rot = Gf.Vec3d(0, 0, 0)
                            
                        if isinstance(current_rot, (float, int)): # RotateZ is scalar
                            new_rot = current_rot + rand_deg
                        else: # RotateXYZ is Vec3
                            # Assuming Z is index 2
                            new_rot = Gf.Vec3d(current_rot[0], current_rot[1], current_rot[2] + rand_deg)
                        
                        rotate_op.Set(new_rot)
                        self._log(f"[RANDOMIZATION] Gear Base rotated by: {rand_deg:.2f} deg")
                        
                        # [FIX] Update FixedJoint Constraint
                        # Since the base contains a FixedJoint to static world, rotating the base 
                        # enables the joint to fight back unless we rotate the joint's world anchor too.
                        try:
                             # Search for FixedJoint under the rigid body
                             # Rigid body is at /World/factory_gear_base/factory_gear_base
                             # NOTE: We just found base_prim, let's use that as root to search or assume standard structure
                             # If base_prim is the Xform, the RigidBody is likely inside or IT IS the rigid body.
                             # Let's search under base_prim first, then base_prim_path/factory_gear_base
                             
                             joint_prim = None
                             
                             # Search 1: Under base_prim itself
                             for child in base_prim.GetChildren():
                                 if child.GetTypeName() == "PhysicsFixedJoint":
                                     joint_prim = child
                                     break
                             
                             # Search 2: If base_prim is the wrapper Xform, check child "factory_gear_base"
                             if not joint_prim:
                                 nested_rb = stage.GetPrimAtPath(f"{base_prim_path}/factory_gear_base")
                                 if nested_rb.IsValid():
                                     for child in nested_rb.GetChildren():
                                         if child.GetTypeName() == "PhysicsFixedJoint":
                                             joint_prim = child
                                             break
                                             
                             if joint_prim:
                                     # Assuming Body1 is World (implicit or explicit)
                                     # We need to rotate 'physics:localRot1' (Frame in Body1) by rand_deg
                                     
                                     # Create Delta Quat (Z-axis rotation)
                                     # Gf.Rotation takes degrees
                                     delta_rot = Gf.Rotation(Gf.Vec3d(0,0,1), rand_deg)
                                     delta_quat = delta_rot.GetQuat() # Gf.Quatd
                                     
                                     # Get current localRot1
                                     # Note: Attribute might be Quatf or Quatd
                                     rot1_attr = joint_prim.GetAttribute("physics:localRot1")
                                     current_quat = rot1_attr.Get() # Gf.Quatf or Gf.Quatd
                                     
                                     if current_quat is None:
                                         # Default to Identity if not set
                                         current_quat = Gf.Quatf(1.0)
                                     
                                     # Convert delta to same type (Quatf usually)
                                     # Gf.Quatf constructor from Gf.Quatd
                                     try:
                                         delta_quat_f = Gf.Quatf(delta_quat)
                                         # Multiply: New = Delta * Old (Apply rotation in World Frame)
                                         new_quat = delta_quat_f * current_quat
                                         rot1_attr.Set(new_quat)
                                         self._log(f"[RANDOMIZATION] FixedJoint 'localRot1' updated for base rotation")
                                     except:
                                         # Fallback for double precision
                                         delta_quat_d = Gf.Quatd(delta_quat)
                                         new_quat = delta_quat_d * current_quat
                                         rot1_attr.Set(new_quat)
                                         self._log(f"[RANDOMIZATION] FixedJoint 'localRot1' updated (double)")
                                         
                        except Exception as e_joint:
                             self._log(f"[WARNING] Failed to update FixedJoint: {e_joint}")
                else:
                    # [DEBUG] Log failure to find base prim
                    self._log(f"[WARNING] Could not find factory_gear_base at {base_prim_path}")
                    try:
                        world = stage.GetPrimAtPath("/World")
                        children = [p.GetName() for p in world.GetChildren()]
                        self._log(f"[DEBUG] Children of /World: {children}")
                    except:
                        pass
            
            except Exception as e_rand:
                self._log(f"[WARNING] Domain Randomization failed: {e_rand}")
            
            # [STABILITY] Warmup physics to let the simulator digest the teleportation
            # Like in __init__, we need to step the physics to ensure the robot is actually at the target
            self._log("[INFO] Warming up physics after reset (10 steps)...")
            for _ in range(10):
                self.world.step(render=False)
            
            # [CRITICAL] 强制更新状态缓存，确保客户端立刻读到的是 Reset 后的状态
            self._update_state_fast()
            self._log(f"[INFO] Post-Reset State Synced. Pose: {np.round(self.state_cache['pose'], 3)}")
            
            # [RESET LOGIC] 自动运动到 Config 定义的 Reset Pose
            # 否则机械臂会停留在 initial_q (软复位后的默认姿态)
            if hasattr(self, 'default_reset_pose'):
                 self._log("[INFO] Auto-moving to configured RESET_POSE...")
                 self.set_pose(self.default_reset_pose)
            
            self._log("[INFO] Scene reset completed successfully")
            
        except Exception as e:
            self._log(f"[ERROR] Failed to reset scene on main thread: {e}")
            import traceback
            traceback.print_exc()

    def close(self):
        """关闭服务器"""
        self.running = False
        if self.simulation_app is not None:
            self.simulation_app.close()
        self._log("[INFO] Isaac Sim Server closed")


def load_image_crop_config(config_module: str) -> Optional[Dict[str, Callable]]:
    """
    从配置模块加载 IMAGE_CROP 配置
    
    Args:
        config_module: Python 模块路径（如 'experiments.ram_insertion.config'）
    
    Returns:
        IMAGE_CROP 字典，如果不存在则返回 None
    """
    try:
        import importlib
        module = importlib.import_module(config_module)
        if hasattr(module, 'EnvConfig'):
            env_config = module.EnvConfig()
            if hasattr(env_config, 'IMAGE_CROP'):
                return env_config.IMAGE_CROP
        return None
    except Exception as e:
        print(f"[WARNING] Failed to load IMAGE_CROP config from {config_module}: {e}")
        return None


def main(_):
    """主函数：启动 Flask 服务器"""
    # 检查 USD 文件路径
    if FLAGS.usd_path is None:
        raise ValueError("--usd_path must be provided. Please specify the path to your USD scene file.")
    
    # 加载图像裁剪配置（如果提供了）
    image_crop = None
    if FLAGS.config_module:
        image_crop = load_image_crop_config(FLAGS.config_module)
        if image_crop:
            print(f"[INFO] Loaded IMAGE_CROP config: {list(image_crop.keys())}") # Keep print here as it's outside class
    
    # 初始化 Isaac Sim 服务器
    isaac_sim_server = IsaacSimServer(
        headless=FLAGS.headless,
        width=FLAGS.sim_width,
        height=FLAGS.sim_height,
        sim_hz=FLAGS.sim_hz,
        image_crop=image_crop,
        usd_path=FLAGS.usd_path,
        robot_prim_path=FLAGS.robot_prim_path,
        camera_prim_paths=FLAGS.camera_prim_paths,
    )
    
    # 创建 Flask 应用
    webapp = Flask(__name__)
    if SocketIO is not None:
        socketio = SocketIO(webapp, cors_allowed_origins="*", async_mode='threading')
    else:
        socketio = None
        print("[WARNING] WebSocket support disabled (flask-socketio not installed)") # Keep print here as it's outside class
    
    # 将 socketio 传递给服务器
    isaac_sim_server.socketio = socketio
    
    # ========== 控制命令路由 ==========
    
    @webapp.route("/pose", methods=["POST"])
    def pose():
        """发送末端执行器位姿命令"""
        arr = request.json["arr"]
        # [NET] Received /pose raw: [x,y,z, qx,qy,qz,qw] (XYZW format)
        if len(arr) > 0:
             # 转换为 WXYZ 显示以保持一致性
             quat_xyzw = np.array(arr[3:7])
             quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
             # isaac_sim_server._log(f"[NET] Received pose: {np.round(arr[:3], 3)} quat(WXYZ): {np.round(quat_wxyz, 3)}")
        pos_array = np.array(arr)
        isaac_sim_server.set_pose(pos_array)
        return "Moved"
    
    @webapp.route("/close_gripper", methods=["POST"])
    def close_gripper():
        """关闭夹爪"""
        isaac_sim_server.set_gripper(0.0)
        return "Closed"
    
    @webapp.route("/open_gripper", methods=["POST"])
    def open_gripper():
        """打开夹爪"""
        isaac_sim_server.set_gripper(1.0)
        return "Opened"
    
    @webapp.route("/move_gripper", methods=["POST"])
    def move_gripper():
        """
        移动夹爪到指定位置
        
        Request:
            JSON: {"gripper_pos": float}  # 0.0-1.0
        """
        gripper_pos = request.json["gripper_pos"]
        isaac_sim_server.set_gripper(gripper_pos)
        return "Moved Gripper"
    
    @webapp.route("/clearerr", methods=["POST"])
    def clear():
        """清除错误（仿真中可能不需要）"""
        return "Clear"
    
    @webapp.route("/update_param", methods=["POST"])
    def update_param():
        """更新参数（仿真中可能不需要）"""
        return "Updated"
    
    @webapp.route("/jointreset", methods=["POST"])
    def joint_reset():
        """
        执行关节复位 (Teleport)
        当逆运动学或物理引擎无法到达目标位置时，强制重置关节状态。
        """
        try:
            # 使用预定义的安全姿态 (对应 warmup pose)
            # [0.0, -0.4, 0.0, -2.4, 0.0, 2.0, 0.8, 0.0, 0.0]
            # 注意：set_joint_positions 需要所有关节 (包括 gripper)
            safe_joint_pos = np.array([0.0, -0.4, 0.0, -2.4, 0.0, 2.0, 0.8, 0.04, 0.04])
            
            # 或者从请求中获取 (如果提供了)
            if request.json and "joints" in request.json:
                safe_joint_pos = np.array(request.json["joints"])
            
            isaac_sim_server.franka.set_joint_positions(safe_joint_pos)
            
            # 同时重置这个位置对应的控制器目标，防止重置后立即弹回
            # 这需要正向运动学计算当前 EE pose，但简化起见，我们可以在客户端 reset 后再次发送目标
            
            isaac_sim_server._log(f"[INFO] Forced Joint Reset (Teleport) executed.")
            return jsonify({"status": "success"})
        except Exception as e:
            isaac_sim_server._log(f"[ERROR] Joint Reset Failed: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @webapp.route("/reset_scene", methods=["POST"])
    def reset_scene():
        """
        重置整个 USD 场景
        
        功能：
        1. 重置物理世界（world.reset()）
        2. 重置所有对象到初始位置
        3. 清除所有约束
        4. 重置机器人到初始状态
        
        用途：
        - 用于快速重置仿真环境
        - 可以通过手柄按钮触发
        """
        try:
            # [FIX] 线程安全：不要直接在 Flask 线程重置世界，而是放入队列由主线程安全处理
            isaac_sim_server.action_queue.put(("reset_scene", None))
            
            print("[INFO] Scene reset request enqueued")
            return jsonify({"status": "success", "message": "Scene reset request enqueued"})
            
        except Exception as e:
            print(f"[ERROR] Failed to reset scene: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({"status": "error", "message": str(e)}), 500
    
    # ========== 状态查询路由 ==========
    
    @webapp.route("/getstate", methods=["POST"])
    def get_state():
        """
        获取所有机器人状态（最常用）
        
        Response:
            JSON: {
                "pose": List[7],
                "vel": List[6],
                "force": List[3],
                "torque": List[3],
                "q": List[7],
                "dq": List[7],
                "jacobian": List[List[6, 7]],
                "gripper_pos": float,
            }
        """
        return jsonify(isaac_sim_server.get_state())
    
    @webapp.route("/getpos", methods=["POST"])
    def get_pos():
        """获取末端执行器位姿"""
        with isaac_sim_server.state_lock:
            return jsonify({"pose": isaac_sim_server.state_cache["pose"].tolist()})
    
    @webapp.route("/getvel", methods=["POST"])
    def get_vel():
        """获取末端执行器速度"""
        with isaac_sim_server.state_lock:
            return jsonify({"vel": isaac_sim_server.state_cache["vel"].tolist()})
    
    @webapp.route("/getforce", methods=["POST"])
    def get_force():
        """获取末端执行器力"""
        with isaac_sim_server.state_lock:
            return jsonify({"force": isaac_sim_server.state_cache["force"].tolist()})
    
    @webapp.route("/gettorque", methods=["POST"])
    def get_torque():
        """获取末端执行器力矩"""
        with isaac_sim_server.state_lock:
            return jsonify({"torque": isaac_sim_server.state_cache["torque"].tolist()})
    
    @webapp.route("/getq", methods=["POST"])
    def get_q():
        """获取关节位置"""
        with isaac_sim_server.state_lock:
            return jsonify({"q": isaac_sim_server.state_cache["q"].tolist()})
    
    @webapp.route("/getdq", methods=["POST"])
    def get_dq():
        """获取关节速度"""
        with isaac_sim_server.state_lock:
            return jsonify({"dq": isaac_sim_server.state_cache["dq"].tolist()})
    
    @webapp.route("/getjacobian", methods=["POST"])
    def get_jacobian():
        """获取雅可比矩阵"""
        with isaac_sim_server.state_lock:
            return jsonify({"jacobian": isaac_sim_server.state_cache["jacobian"].tolist()})
    
    @webapp.route("/get_gripper", methods=["POST"])
    def get_gripper():
        """获取夹爪位置"""
        with isaac_sim_server.state_lock:
            return jsonify({"gripper": isaac_sim_server.state_cache["gripper_pos"]})
    
    # ========== 健康检查路由 ==========
    
    @webapp.route("/health", methods=["GET"])
    def health():
        """
        健康检查
        
        Response:
            JSON: {
                "status": "healthy",
                "simulation_running": bool,
            }
        """
        return jsonify({
            "status": "healthy",
            "simulation_running": isaac_sim_server.running,
        })
    
    # ========== WebSocket 事件处理 ==========
    
    if socketio is not None:
        @socketio.on('connect')
        def handle_connect():
            """WebSocket 连接建立"""
            print(f"[INFO] WebSocket client connected: {request.sid}")
            with isaac_sim_server.ws_lock:
                if request.sid not in isaac_sim_server.ws_clients:
                    isaac_sim_server.ws_clients.append(request.sid)
        
        @socketio.on('disconnect')
        def handle_disconnect():
            """WebSocket 连接断开"""
            print(f"[INFO] WebSocket client disconnected: {request.sid}")
            with isaac_sim_server.ws_lock:
                if request.sid in isaac_sim_server.ws_clients:
                    isaac_sim_server.ws_clients.remove(request.sid)
    
    # 启动 Flask 服务器（在独立线程中）
    def run_flask():
        try:
            print(f"[INFO] Starting Flask server on {FLAGS.flask_url}:{FLAGS.flask_port}")
            if socketio is not None:
                socketio.run(
                    webapp,
                    host=FLAGS.flask_url,
                    port=FLAGS.flask_port,
                    allow_unsafe_werkzeug=True,
                    use_reloader=False # 重要：禁止 reloader，否则会启动子进程导致 Isaac Sim 也是两份
                )
            else:
                webapp.run(
                    host=FLAGS.flask_url, 
                    port=FLAGS.flask_port,
                    threaded=True,
                    use_reloader=False
                )
        except Exception as e:
            print(f"[ERROR] Flask server failed: {e}")

    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()

    # 主线程运行 Simulation Loop (Isaac Sim 要求)
    try:
        isaac_sim_server.run()
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
    finally:
        isaac_sim_server.close()


if __name__ == "__main__":
    app.run(main)
