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
from omni.isaac.kit import SimulationApp

import numpy as np
import threading
import time
import queue
import copy
import cv2
from typing import Dict, Optional, Callable
from absl import app, flags
from scipy.spatial.transform import Rotation as R
from flask import Flask, request, jsonify
try:
    from flask_socketio import SocketIO
except ImportError:
    print("[WARNING] flask-socketio not installed. WebSocket support will be limited.")
    SocketIO = None
import base64

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
        }
        self.simulation_app = SimulationApp(config)
        
        # 频率配置
        self.sim_hz = sim_hz
        self.control_hz = 10.0  # 控制频率（由客户端决定）
        
        # 现在可以导入其他模块
        from omni.isaac.core import World
        from omni.isaac.franka import Franka
        from omni.isaac.sensor import Camera
        from omni.isaac.core.utils.stage import get_current_stage
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
        self.world.set_physics_dt(1.0 / self.sim_hz)
        
        # 从场景中获取机器人对象
        print(f"[INFO] Getting robot from prim path: {robot_prim_path}")
        try:
            # 使用 world.scene.get() 获取已存在的对象
            # 如果对象不存在，会返回 None
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
        
        # 从场景中获取相机对象
        self.cameras = {}
        if camera_prim_paths is None:
            camera_prim_paths = [
                "/World/franka/panda_hand/wrist_1",
                "/World/franka/panda_hand/wrist_2"
            ]
        
        print(f"[INFO] Getting cameras from prim paths: {camera_prim_paths}")
        for cam_prim_path in camera_prim_paths:
            try:
                # 从 prim 路径提取相机名称（最后一个路径组件）
                cam_name = cam_prim_path.split("/")[-1]
                camera = self.world.scene.get(cam_prim_path)
                if camera is None:
                    # 尝试使用 Camera 类来获取（如果 prim 存在但未注册到 scene）
                    stage = self.get_current_stage()
                    prim = stage.GetPrimAtPath(cam_prim_path)
                    if prim.IsValid():
                        print(f"[INFO] Camera prim found at {cam_prim_path}, creating Camera wrapper...")
                        camera = self.world.scene.add(
                            self.Camera(
                                prim_path=cam_prim_path,
                                name=cam_name,
                            )
                        )
                    else:
                        print(f"[WARNING] Camera prim not found at {cam_prim_path}, skipping...")
                        continue
                self.cameras[cam_name] = camera
                print(f"[INFO] Camera '{cam_name}' found at {cam_prim_path}")
            except Exception as e:
                print(f"[WARNING] Failed to get camera from {cam_prim_path}: {e}")
        
        if len(self.cameras) == 0:
            print("[WARNING] No cameras found in the scene. Image capture will not work.")
        else:
            print(f"[INFO] Successfully loaded {len(self.cameras)} cameras: {list(self.cameras.keys())}")
        
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
        
        # WebSocket 客户端列表（线程安全）
        self.ws_clients = []
        self.ws_lock = threading.Lock()
        self.socketio = socketio  # Flask-SocketIO 实例
        
        # IK 控制器和夹爪控制（初始化）
        self.controller = None
        self.ik_solver = None
        self._setup_controller()
        
        # 雅可比矩阵计算标志（用于日志记录）
        self._jacobian_method_logged = False
        
        # 初始化世界
        self.world.reset()
        
        # 启动仿真循环线程
        self.running = True
        self.sim_thread = threading.Thread(target=self._simulation_loop)
        self.sim_thread.daemon = True
        self.sim_thread.start()
        
        print("[INFO] Isaac Sim Server initialized")
    
    
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
            print("[INFO] Initialized RMPFlowController for robot control")
            
        except (ImportError, AttributeError, Exception) as e:
            print(f"[WARN] RMPFlowController not available: {e}")
            print("[INFO] Falling back to IK solver or direct joint control")
            
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
                print("[INFO] Initialized InverseKinematicsSolver for robot control")
                
            except (ImportError, AttributeError, Exception) as e2:
                print(f"[WARN] IK solver not available: {e2}")
                print("[WARN] Robot pose control will not work without controller or IK solver")
                print("[WARN] Please install omni.isaac.manipulators extension")
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
                                print("[INFO] Using franka.get_jacobian() to compute Jacobian")
                                self._jacobian_method_logged = True
                            return jacobian
                        elif jacobian.shape == (7, 6):
                            # 如果是转置，转置回来
                            if not self._jacobian_method_logged:
                                print("[INFO] Using franka.get_jacobian() to compute Jacobian (transposed)")
                                self._jacobian_method_logged = True
                            return jacobian.T
                        elif jacobian.size == 42:  # 42 = 6 * 7
                            # 如果是展平的数组，重塑
                            if not self._jacobian_method_logged:
                                print("[INFO] Using franka.get_jacobian() to compute Jacobian (reshaped)")
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
                                print("[INFO] Using root_physx_view.get_jacobians() to compute Jacobian")
                                self._jacobian_method_logged = True
                            return jacobian
                        elif jacobian.shape == (7, 6):
                            if not self._jacobian_method_logged:
                                print("[INFO] Using root_physx_view.get_jacobians() to compute Jacobian (transposed)")
                                self._jacobian_method_logged = True
                            return jacobian.T
                        elif jacobian.size == 42:
                            if not self._jacobian_method_logged:
                                print("[INFO] Using root_physx_view.get_jacobians() to compute Jacobian (reshaped)")
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
                print("[WARNING] Failed to compute Jacobian matrix using all available methods")
                print("[WARNING] End-effector velocity will be set to zero")
                self._jacobian_method_logged = True
            return None
            
        except Exception as e:
            # 只在第一次失败时打印错误
            if not self._jacobian_method_logged:
                print(f"[WARNING] Exception while computing Jacobian: {e}")
                self._jacobian_method_logged = True
            return None
    
    def _compute_end_effector_wrench(self):
        """
        计算末端执行器处的力/力矩（在末端执行器坐标系中）
        
        方法：从关节力矩反推（与 Franka 实现一致）
        原理：使用虚功原理，wrench = (J^T)^+ * tau
        
        参考：
        - 原项目：franka_server.py 从 K_F_ext_hat_K 获取力/力矩
        - Franka 通过关节力矩传感器 + 动力学模型计算
        - 这里使用相同原理：从关节力矩反推末端执行器力/力矩
        
        Returns:
            force: np.ndarray[3] - 末端执行器处的力 [fx, fy, fz]（单位：N）
            torque: np.ndarray[3] - 末端执行器处的力矩 [tx, ty, tz]（单位：Nm）
        """
        try:
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
                print(f"[WARNING] Failed to compute end-effector wrench: {e}")
                self._wrench_compute_warning_logged = True
            return np.zeros(3), np.zeros(3)
    
    def _simulation_loop(self):
        """
        仿真循环（在独立线程中运行，60 Hz）
        
        关键设计：
        1. 持续更新状态缓存（不加锁，快速更新）
        2. 持续更新图像队列（丢弃旧帧）
        3. 推送图像到 WebSocket 客户端
        4. 确保状态和图像始终是最新的
        """
        while self.running and self.simulation_app.is_running():
            # 步进物理仿真
            self.world.step(render=not self.simulation_app.config.get("headless", True))
            
            # 快速更新状态（不加锁，减少延迟）
            self._update_state_fast()
            
            # 更新图像（丢弃旧帧，只保留最新帧）
            self._update_images_fast()
            
            # 推送图像到 WebSocket 客户端
            self._push_images_to_websocket()
            
            # 控制频率
            time.sleep(1.0 / self.sim_hz)
    
    def _update_state_fast(self):
        """
        快速更新状态（不加锁，减少延迟）
        
        参考原项目：状态持续更新，查询时加锁获取
        """
        try:
            # 获取末端执行器位姿
            # 尝试从 franka 对象获取末端执行器路径
            if hasattr(self.franka, 'end_effector'):
                ee_prim_path = self.franka.end_effector.prim_path
            else:
                # 默认路径
                ee_prim_path = "/World/franka/panda_hand"
            
            stage = self.get_current_stage()
            ee_prim = stage.GetPrimAtPath(ee_prim_path)
            if not ee_prim.IsValid():
                # 尝试其他可能的路径
                ee_prim = stage.GetPrimAtPath("/World/franka/panda_link8")
            if not ee_prim.IsValid():
                ee_prim = stage.GetPrimAtPath("/World/franka/panda_hand")
            
            if ee_prim.IsValid():
                xform = self.UsdGeom.Xformable(ee_prim)
                world_transform = xform.ComputeLocalToWorldTransform(0)
                
                position = np.array(world_transform.ExtractTranslation())
                rotation_matrix = world_transform.ExtractRotationMatrix()
                rotation = R.from_matrix(rotation_matrix).as_quat()
                
                # 快速更新（不加锁）
                self.state_cache["pose"] = np.concatenate([position, rotation])
            else:
                print(f"[WARNING] End effector prim not found at {ee_prim_path}")
            
            # 获取关节状态
            if hasattr(self.franka, 'get_joint_positions'):
                self.state_cache["q"] = self.franka.get_joint_positions()
                self.state_cache["dq"] = self.franka.get_joint_velocities()
            
            # 计算雅可比矩阵和末端执行器速度
            jacobian = self._compute_jacobian()
            if jacobian is not None:
                self.state_cache["jacobian"] = jacobian
                # 计算末端执行器速度：vel = J @ dq
                # 只使用前7个关节的速度（忽略夹爪关节）
                dq_arm = self.state_cache["dq"][:7] if len(self.state_cache["dq"]) >= 7 else self.state_cache["dq"]
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
            
        except Exception as e:
            print(f"[WARNING] Failed to update state: {e}")
    
    def _update_images_fast(self):
        """
        快速更新图像（丢弃旧帧，只保留最新帧）
        
        参考原项目 VideoCapture 的设计：
        - 如果队列不为空，丢弃旧帧
        - 只保留最新帧
        
        处理流程：
        1. 获取原始图像
        2. 应用图像裁剪（如果配置了）
        3. 调整大小到 128x128（观察空间要求）
        4. 存入队列，供 WebSocket 传输
        """
        for cam_key, camera in self.cameras.items():
            try:
                # 获取图像
                rgba = camera.get_rgba()
                rgb = rgba[:, :, :3]  # 移除 alpha 通道
                
                # 应用图像裁剪（如果配置了）
                if cam_key in self.image_crop:
                    rgb = self.image_crop[cam_key](rgb)
                
                # 调整大小到 128x128（观察空间要求）
                # 在 server 端完成，减少传输数据大小
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
                print(f"[WARNING] Failed to update image {cam_key}: {e}")
    
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
                                self.socketio.emit('image', message, room=client_id, binary=True)
                    except Exception as e:
                        # 如果客户端断开，从列表中移除
                        if client_id in self.ws_clients:
                            self.ws_clients.remove(client_id)
    
    def set_pose(self, pose: np.ndarray):
        """
        设置机器人末端执行器位姿
        
        使用控制器或 IK 求解器将目标位姿转换为关节目标
        参考 isaac_sim_env.py 中的 _set_robot_pose() 实现
        
        Args:
            pose: np.ndarray[7] - [x, y, z, qx, qy, qz, qw]
        """
        position = pose[:3]
        orientation = pose[3:]  # quaternion [x, y, z, w]
        
        try:
            if self.controller is not None:
                # 方法1：使用 RMPFlowController（推荐）
                self.controller.set_target_pose(
                    position=position,
                    orientation=orientation
                )
                joint_targets = self.controller.compute()
                self.franka.set_joint_position_targets(joint_targets)
                
            elif self.ik_solver is not None:
                # 方法2：使用 IK 求解器
                # 注意：API 可能不同，需要根据实际 Isaac Sim 版本调整
                try:
                    # 尝试使用 compute_inverse_kinematics 方法
                    joint_targets = self.ik_solver.compute_inverse_kinematics(
                        target_position=position,
                        target_orientation=orientation
                    )
                except AttributeError:
                    # 如果方法名不同，尝试其他可能的 API
                    try:
                        # 尝试使用 compute 方法
                        joint_targets = self.ik_solver.compute(
                            target_position=position,
                            target_orientation=orientation
                        )
                    except AttributeError:
                        # 如果都不行，尝试直接调用
                        joint_targets = self.ik_solver(
                            target_position=position,
                            target_orientation=orientation
                        )
                
                self.franka.set_joint_position_targets(joint_targets)
                
            else:
                # 方法3：没有控制器或IK求解器可用
                print("[WARN] No controller or IK solver available, robot will not move")
                print("[WARN] Please ensure omni.isaac.manipulators extension is installed")
                # 不执行任何动作，保持当前位置
            
        except Exception as e:
            print(f"[WARNING] Failed to set robot pose: {e}")
            import traceback
            traceback.print_exc()
            # 如果设置失败，继续使用当前位置，不抛出异常
    
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
            return {
                "pose": self.state_cache["pose"].tolist(),
                "vel": self.state_cache["vel"].tolist(),
                "force": self.state_cache["force"].tolist(),
                "torque": self.state_cache["torque"].tolist(),
                "q": self.state_cache["q"].tolist(),
                "dq": self.state_cache["dq"].tolist(),
                "jacobian": self.state_cache["jacobian"].tolist(),
                "gripper_pos": self.state_cache["gripper_pos"],
            }
    
    def set_gripper(self, gripper_pos: float):
        """
        设置夹爪位置
        
        参考 isaac_sim_env.py 中的 _set_gripper() 实现
        使用关节控制方式，更可靠
        
        Args:
            gripper_pos: float - 0.0 (关闭) 到 1.0 (打开)
        """
        try:
            # 更新状态缓存（加锁）
            with self.state_lock:
                self.state_cache["gripper_pos"] = gripper_pos
            
            # 实际控制夹爪关节
            try:
                # 方法1：尝试使用 get_joints() 获取夹爪关节
                gripper_joints = self.franka.gripper.get_joints()
                if len(gripper_joints) == 0:
                    print("[WARN] No gripper joints found")
                    return
                
                # 使用第一个夹爪关节（通常是主要的）
                gripper_joint = gripper_joints[0]
                
                # 将 gripper_pos (0.0-1.0) 映射到夹爪关节位置
                # 0.0 -> 0.0 (关闭), 1.0 -> 0.04 (打开，最大开度)
                target_position = gripper_pos * 0.04
                
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
                        print("[WARN] Gripper API not available, gripper will not move")
                        print("[WARN] Please check Franka gripper implementation")
                        
                except Exception as e:
                    print(f"[WARNING] Failed to set gripper using alternative method: {e}")
                    import traceback
                    traceback.print_exc()
                    
        except Exception as e:
            print(f"[WARNING] Failed to set gripper: {e}")
            import traceback
            traceback.print_exc()
            # 即使失败也更新状态缓存（至少记录目标值）
    
    def close(self):
        """关闭服务器"""
        self.running = False
        if self.sim_thread.is_alive():
            self.sim_thread.join(timeout=1.0)
        if self.simulation_app is not None:
            self.simulation_app.close()
        print("[INFO] Isaac Sim Server closed")


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
            print(f"[INFO] Loaded IMAGE_CROP config: {list(image_crop.keys())}")
    
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
        print("[WARNING] WebSocket support disabled (flask-socketio not installed)")
    
    # 将 socketio 传递给服务器
    isaac_sim_server.socketio = socketio
    
    # ========== 控制命令路由 ==========
    
    @webapp.route("/pose", methods=["POST"])
    def pose():
        """
        发送末端执行器位姿命令
        
        Request:
            JSON: {"arr": [x, y, z, qx, qy, qz, qw]}
        
        Response:
            str: "Moved"
        """
        pos = np.array(request.json["arr"])
        isaac_sim_server.set_pose(pos)
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
        """执行关节复位"""
        # TODO: 实现关节复位
        return "Reset Joint"
    
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
            # 重置物理世界（这会重置所有物理对象到初始状态）
            isaac_sim_server.world.reset()
            
            # 重置机器人到初始关节位置（如果需要）
            # 注意：world.reset() 通常已经重置了机器人，但可以显式设置
            if hasattr(isaac_sim_server.franka, 'set_joint_positions'):
                # 设置到初始关节位置（通常是零位或配置的初始位置）
                initial_joint_positions = np.zeros(7)  # 可以根据需要调整
                isaac_sim_server.franka.set_joint_positions(initial_joint_positions)
            
            # 重置夹爪到打开状态
            isaac_sim_server.set_gripper(1.0)  # 打开夹爪
            
            # 清除所有约束（如果有）
            # 注意：world.reset() 通常已经清除了约束，但可以显式清除
            # 这里可以添加清除约束的逻辑（如果需要）
            
            print("[INFO] Scene reset completed")
            return jsonify({"status": "success", "message": "Scene reset completed"})
            
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
    
    # 启动服务器
    try:
        print(f"[INFO] Starting Flask server on {FLAGS.flask_url}:{FLAGS.flask_port}")
        if socketio is not None:
            socketio.run(
                webapp,
                host=FLAGS.flask_url,
                port=FLAGS.flask_port,
                allow_unsafe_werkzeug=True,
            )
        else:
            webapp.run(host=FLAGS.flask_url, port=FLAGS.flask_port)
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
    finally:
        isaac_sim_server.close()


if __name__ == "__main__":
    app.run(main)
