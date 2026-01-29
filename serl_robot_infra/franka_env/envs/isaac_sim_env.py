"""
Isaac Sim Franka 环境基础类

实现与 FrankaEnv 相同的 Gym 接口，通过 HTTP/WebSocket 连接到 isaac_sim_server

架构设计：
- isaac_sim_server.py: 独立进程，管理 Isaac Sim 仿真环境，提供 HTTP/WebSocket 接口
- isaac_sim_env.py: Gym 环境，通过 HTTP/WebSocket 连接到服务器，不直接使用 Isaac Sim API
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Tuple
import copy
import time
import cv2
import requests
import queue
import threading
from scipy.spatial.transform import Rotation

from franka_env.envs.franka_env import DefaultEnvConfig
from franka_env.utils.rotations import euler_2_quat


class IsaacSimFrankaEnv(gym.Env):
    """
    Isaac Sim Franka 环境基础类
    
    实现与 FrankaEnv 相同的接口，通过 HTTP/WebSocket 连接到 isaac_sim_server
    
    关键特点：
    - 通过 HTTP 发送控制命令和获取状态
    - 通过 WebSocket 接收图像数据
    - 不直接使用 Isaac Sim API
    - 与 FrankaEnv 保持接口一致性
    """
    
    def __init__(
        self,
        hz=10,
        fake_env=True,  # 始终为 True（仿真环境）
        save_video=False,
        config: DefaultEnvConfig = None,
    ):
        """
        初始化 Isaac Sim 环境
        
        Args:
            hz: 控制频率（Hz）
            fake_env: 始终为 True（表示仿真环境）
            save_video: 是否保存视频
            config: 环境配置（必须包含 SERVER_URL）
        """
        if config is None:
            raise ValueError("config must be provided")
        
        if not hasattr(config, 'SERVER_URL') or config.SERVER_URL is None:
            raise ValueError("config.SERVER_URL must be set. Please set SERVER_URL in your config class.")
        
        self.config = config
        self.hz = hz
        self.fake_env = fake_env
        self.save_video = save_video
        self.max_episode_length = config.MAX_EPISODE_LENGTH
        self.display_image = config.DISPLAY_IMAGE
        
        # 服务器 URL
        self.url = config.SERVER_URL
        if not self.url.endswith('/'):
            self.url += '/'
        
        # 动作缩放
        self.action_scale = config.ACTION_SCALE
        
        # 位姿配置
        self._TARGET_POSE = config.TARGET_POSE
        self._RESET_POSE = config.RESET_POSE
        self._REWARD_THRESHOLD = config.REWARD_THRESHOLD
        
        # 随机重置配置
        self.randomreset = config.RANDOM_RESET
        self.random_xy_range = config.RANDOM_XY_RANGE
        self.random_rz_range = config.RANDOM_RZ_RANGE
        
        # 转换 RESET_POSE 为四元数格式
        self.resetpos = np.concatenate(
            [config.RESET_POSE[:3], euler_2_quat(config.RESET_POSE[3:])]
        )
        
        # 边界框
        self.xyz_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[:3],
            config.ABS_POSE_LIMIT_HIGH[:3],
            dtype=np.float64,
        )
        self.rpy_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[3:],
            config.ABS_POSE_LIMIT_HIGH[3:],
            dtype=np.float64,
        )
        
        # 定义动作和观察空间
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )
        self.observation_space = self._get_observation_space()
        
        # 状态变量
        self.currpos = None
        self.currvel = None
        self.currforce = None
        self.currtorque = None
        self.curr_gripper_pos = None
        self.curr_path_length = 0
        self.terminate = False
        
        # 视频录制
        if self.save_video:
            self.recording_frames = []
        
        # 创建 HTTP 会话（连接池）
        self.session = requests.Session()
        self.session.timeout = 1.0  # 1秒超时
        
        # WebSocket 客户端和图像缓存
        self.ws_client = None
        self.image_cache = {}
        self.image_cache_lock = threading.Lock()
        self.ws_connected = False
        
        # 连接状态标志（用于抑制重复日志）
        self._connection_lost = False
        
        # 建立 WebSocket 连接（用于图像接收）
        self._connect_websocket()
        
        # [ROBUSTNESS] 强制等待服务器就绪并同步初始状态，防止起点跳变
        self._log("[INFO] Synchronizing with Server...")
        connected = False
        for i in range(20): # 最多等 10 秒
             self._update_currpos()
             if self.currpos is not None and not np.allclose(self.currpos[:3], [0.5, 0.0, 0.5]):
                  connected = True
                  break
             time.sleep(0.5)
        
        if not connected:
             self._log("[WARNING] Server not ready or state not synced. Robot might jump.")
        else:
             self._log(f"[INFO] Initial state synced at: {np.round(self.currpos[:3], 3)}")

        # 键盘监听（用于终止）
        if not fake_env:
            from pynput import keyboard
            self.terminate = False
            def on_press(key):
                if key == keyboard.Key.esc:
                    self.terminate = True
            self.listener = keyboard.Listener(on_press=on_press)
            self.listener.start()
        else:
            self.listener = None
        
        print(f"[INFO] Initialized Isaac Sim Franka Environment (connected to {self.url}) - SOURCE CODE VERIFIED")
    
    def _log(self, msg):
        from datetime import datetime
        # 增加毫秒级时间戳，方便与服务器对齐日志
        if hasattr(self, "name"):
             prefix = f"[{self.name}]"
        else:
             prefix = "[IsaacSimClient]"
        print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {prefix} {msg}")

    def _get_observation_space(self):
        """定义观察空间"""
        return gym.spaces.Dict({
            "state": gym.spaces.Dict({
                "tcp_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,)),
                "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                "gripper_pose": gym.spaces.Box(-1, 1, shape=(1,)),
                "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
            }),
            "images": gym.spaces.Dict({
                key: gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8)
                for key in self.config.REALSENSE_CAMERAS.keys()
            }),
        })
    
    def _connect_websocket(self):
        """建立 WebSocket 连接（用于接收图像）"""
        try:
            import socketio
            
            # 构建 WebSocket URL
            ws_url = self.url.replace("http://", "ws://").replace("https://", "wss://")
            # 移除末尾的 '/'
            if ws_url.endswith('/'):
                ws_url = ws_url[:-1]
            
            # 创建 SocketIO 客户端
            self.ws_client = socketio.Client()
            
            # 定义图像接收回调
            @self.ws_client.on('image')
            def on_image(data):
                """
                接收图像数据（二进制格式）
                
                注意：图像已在 server 端完成裁剪和调整大小（128x128），
                这里只需要解码 JPEG 并缓存即可。
                """
                try:
                    # 解析消息格式：<camera_key_length><camera_key><jpeg_data>
                    if len(data) < 1:
                        return
                    
                    cam_key_len = data[0]
                    if len(data) < 1 + cam_key_len:
                        return
                    
                    cam_key = data[1:1+cam_key_len].decode('utf-8')
                    jpeg_data = data[1+cam_key_len:]
                    
                    # 解码 JPEG 图像
                    # 注意：图像已在 server 端完成裁剪和调整大小（128x128）
                    img_array = np.frombuffer(jpeg_data, np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    
                    if img is None:
                        return
                    
                    # 转换 BGR 到 RGB（OpenCV 默认使用 BGR）
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # 更新缓存（线程安全）
                    # 图像已经是 128x128，无需进一步处理
                    with self.image_cache_lock:
                        self.image_cache[cam_key] = img
                        
                except Exception as e:
                    print(f"[WARNING] Failed to process image from {cam_key}: {e}")
            
            @self.ws_client.on('connect')
            def on_connect():
                self.ws_connected = True
                print("[INFO] WebSocket connected to Isaac Sim server")
            
            @self.ws_client.on('disconnect')
            def on_disconnect():
                self.ws_connected = False
                print("[WARNING] WebSocket disconnected from Isaac Sim server")
            
            # 连接到服务器
            self.ws_client.connect(ws_url, wait_timeout=5)
            self.ws_connected = True
            
        except ImportError:
            print("[WARNING] socketio not installed. WebSocket image streaming disabled.")
            print("[WARNING] Images will not be available. Install with: pip install python-socketio")
            self.ws_client = None
            self.ws_connected = False
        except Exception as e:
            print(f"[WARNING] Failed to connect WebSocket: {e}")
            print("[WARNING] Images will not be available. Make sure isaac_sim_server is running.")
            self.ws_client = None
            self.ws_connected = False
    
    def _send_pos_command(self, pos: np.ndarray):
        """发送位姿命令（HTTP POST）"""
        try:
            arr = np.array(pos).astype(np.float32)
            # [DIAGNOSTIC] Log sent data (转换为 WXYZ 显示)
            quat_wxyz = np.array([arr[6], arr[3], arr[4], arr[5]])
            # self._log(f"[NET] Sending Pose: {np.round(arr[:3], 3)} Quat(WXYZ): {np.round(quat_wxyz, 3)}")
            data = {"arr": arr.tolist()}
            self.session.post(self.url + "pose", json=data, timeout=0.5)
            
            if self._connection_lost:
                print("[INFO] Connection to Isaac Sim server restored.")
                self._connection_lost = False
                
        except Exception as e:
            if not self._connection_lost:
                print(f"[WARNING] Failed to send pose command: {e}")
                self._connection_lost = True
    
    def _send_gripper_command(self, pos: float, mode="binary"):
        """发送夹爪命令（HTTP POST）"""
        try:
            if mode == "binary":
                if pos <= -0.5:
                    self.session.post(self.url + "close_gripper", timeout=0.5)
                elif pos >= 0.5:
                    self.session.post(self.url + "open_gripper", timeout=0.5)
            elif mode == "continuous":
                # 连续控制模式
                gripper_pos = (pos + 1.0) / 2.0  # 从 [-1, 1] 映射到 [0, 1]
                self.session.post(
                    self.url + "move_gripper",
                    json={"gripper_pos": float(gripper_pos)},
                    timeout=0.5
                )
            
            if self._connection_lost:
                print("[INFO] Connection to Isaac Sim server restored.")
                self._connection_lost = False

        except Exception as e:
            if not self._connection_lost:
                print(f"[WARNING] Failed to send gripper command: {e}")
                self._connection_lost = True
    
    def _update_currpos(self):
        """更新状态（HTTP POST）"""
        try:
            response = self.session.post(self.url + "getstate", timeout=1.0)
            if response.status_code == 200:
                ps = response.json()
                
                # [DEBUG_DATA_COLLECTION] Check for None values
                if self.hz < 20: # Only print for low freq to avoid spam, or use counter
                     for k, v in ps.items():
                         if v is None:
                             print(f"[ERROR] Received None for key '{k}' in getstate response!")
                
                # [GROUND TRUTH REWARD] Parse object states
                self.object_states = ps.get("object_states", {})
                self.currpos = np.array(ps["pose"])
                
                self.currvel = np.array(ps["vel"])
                self.currforce = np.array(ps["force"])
                self.currtorque = np.array(ps["torque"])
                self.curr_gripper_pos = np.array([ps["gripper_pos"]])
                
                # [GROUND TRUTH REWARD] Parse object states
                self.object_states = ps.get("object_states", {})
                
                # [DIAGNOSTIC] Log received state occasionally
                # self._log(f"[STATE] Received pose: {np.round(self.currpos[:3], 3)}")
                
                if self._connection_lost:
                    print("[INFO] Connection to Isaac Sim server restored.")
                    self._connection_lost = False
            else:
                if not self._connection_lost:
                    print(f"[WARNING] Failed to get state: HTTP {response.status_code}")
                    self._connection_lost = True
        except Exception as e:
            if not self._connection_lost:
                print(f"[WARNING] Failed to update current position: {e}")
                self._connection_lost = True
            # 使用默认值
            if self.currpos is None:
                self.currpos = np.array([0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0])
            if self.currvel is None:
                self.currvel = np.zeros(6)
            if self.currforce is None:
                self.currforce = np.zeros(3)
            if self.currtorque is None:
                self.currtorque = np.zeros(3)
            if self.curr_gripper_pos is None:
                self.curr_gripper_pos = np.array([0.0])
    
    def _get_images(self) -> Dict[str, np.ndarray]:
        """从 WebSocket 缓存获取最新图像"""
        images = {}
        with self.image_cache_lock:
            for cam_key in self.config.REALSENSE_CAMERAS.keys():
                if cam_key in self.image_cache:
                    images[cam_key] = self.image_cache[cam_key].copy()
                else:
                    # 如果没有图像，返回黑色占位符
                    images[cam_key] = np.zeros((128, 128, 3), dtype=np.uint8)
        
        # [VISUALIZATION] Display images if configured
        if self.display_image:
            for cam_key, img_rgb in images.items():
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                cv2.imshow(f"IsaacSim_{cam_key}", img_bgr)
            cv2.waitKey(1)
            
        # 保存视频帧（如果需要）
        if self.save_video:
            if not hasattr(self, 'recording_frames'):
                self.recording_frames = []
            self.recording_frames.append(copy.deepcopy(images))
        
        return images
    
    def _get_obs(self) -> Dict:
        """获取观察"""
        images = self._get_images()
        state_obs = {
            "tcp_pose": self.currpos,
            "tcp_vel": self.currvel,
            "gripper_pose": self.curr_gripper_pos,
            "tcp_force": self.currforce,
            "tcp_torque": self.currtorque,
        }
        return copy.deepcopy({
            "images": images,
            "state": state_obs,
        })
    
    def reset(self, **kwargs) -> Tuple[Dict, Dict]:
        """
        重置环境
        
        Returns:
            observation: Dict[str, np.ndarray]
            info: Dict
        """
        # 保存视频（如果需要）
        if self.save_video and hasattr(self, 'recording_frames') and len(self.recording_frames) > 0:
            self.save_video_recording()
        
        # 更新参数（如果需要，仿真中可能不需要）
        try:
            if hasattr(self.config, 'COMPLIANCE_PARAM') and self.config.COMPLIANCE_PARAM:
                self.session.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM, timeout=0.5)
        except:
            pass
        
        # [ROBUSTNESS] 确保在重置前已获取有效的当前位姿，防止因连接延迟导致的“大跨度跳跃”
        max_retries = 30
        while self.currpos is None and max_retries > 0:
            self._log("[INFO] Waiting for valid state from server before reset...")
            self._update_currpos()
            time.sleep(0.2)
            max_retries -= 1
            
        if self.currpos is None:
            self._log("[ERROR] Could not connect to server after retries. Reset might fail.")
            # 使用一个安全的默认值
            self.currpos = np.array([0.5, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5])
        
        # 获取重置位姿
        reset_pose = self._get_reset_pose()
        
        # 移动到重置位姿
        self.interpolate_move(reset_pose, timeout=1.0)
        
        # [ROBUSTNESS] 强制等待物理收敛
        # 防止 interpolate_move 命令发完但物理未到达（例如 drift）导致后续 step 读取错误
        for _ in range(20): # 最多追加等待 2秒
            self._update_currpos()
            dist = np.linalg.norm(self.currpos[:3] - reset_pose[:3])
            # 旋转偏差 (1 - |q1 . q2|)
            q_curr = self.currpos[3:]
            q_target = reset_pose[3:]
            rot_err = 1.0 - np.abs(np.dot(q_curr, q_target))
            
            if dist < 0.02 and rot_err < 0.05:
                # self._log(f"[VERIFY] Reset Converged. Dist: {dist:.3f}, RotErr: {rot_err:.3f}")
                print(f"[VERIFY] Reset Converged. Dist: {dist:.3f}, RotErr: {rot_err:.3f}")
                break
            else:
                 pass
                 # print(f"[DEBUG] Reset Check: Dist={dist:.3f} (Threshold 0.02), RotDiff={rot_err:.3f}. Target={np.round(reset_pose[2], 3)}, Curr={np.round(self.currpos[2], 3)}")
                
            # 如果未收敛，补发指令并等待
            # print(f"[WARN] Reset not converged (Dist: {dist:.3f}, Rot: {rot_err:.3f}). Resending...")
            
            # [FALLBACK] 如果多次尝试后仍未收敛（例如还是很大），说明物理卡死或漂移过远
            # 在最后几次尝试时，尝试强制关节复位 (Teleport)
            if _ > 15 and dist > 0.1:
                 print(f"[WARN] Reset stubborn (Dist: {dist:.3f}). Triggering Force Teleport...")
                 try:
                     self.session.post(self.url + "jointreset", json={}, timeout=1.0)
                     time.sleep(0.5) # 等待传送生效
                 except Exception as e:
                     print(f"[ERROR] Force Teleport failed: {e}")
            
            self._send_pos_command(reset_pose)
            time.sleep(0.1)
            
        # 更新状态
        self._update_currpos()
        
        # [ROBUSTNESS] 再次强制同步，确保 RelativeFrame 拿到的是物理稳定后的 0.55
        # 很多时候 interpolate_move 刚结束，物理引擎还没来得及更新 Pose，导致立刻读取拿到的是旧值 (0.65)
        for _ in range(5):
            time.sleep(0.1)
            self._update_currpos()
            print(f"[DEBUG] Final Reset Obs Check: {np.round(self.currpos[:3], 3)}")
            
        self.curr_path_length = 0
        self.terminate = False
        
        # [DRIFT FIX] Initialize last_commanded_pose to the commanded reset pose (final target)
        # This prevents drift accumulation by tracking commanding pose instead of measured pose.
        self.last_commanded_pose = reset_pose.copy()

        # 获取观察
        obs = self._get_obs()
        
        return obs, {
            "succeed": False,
            "target_reset_pose": self.last_commanded_pose.copy() # [ANCHOR FIX] 传递理想重置位姿给 RelativeFrame
        }
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        执行一步动作
        
        Args:
            action: np.ndarray[7] - [x, y, z, rx, ry, rz, gripper]
        
        Returns:
            observation: Dict
            reward: float
            done: bool
            truncated: bool
            info: Dict
        """
        start_time = time.time()
        
        # 1. 处理动作
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # 2. 计算位置增量
        xyz_delta = action[:3] * self.action_scale[0]
        
        # 3. 计算旋转增量（rotvec）
        rot_delta = action[3:6] * self.action_scale[1]
        
        # 4. 更新位置 (使用 last_commanded_pose 以防止漂移)
        # Old: target_pos = self.currpos[:3] + xyz_delta
        if not hasattr(self, 'last_commanded_pose') or self.last_commanded_pose is None:
             # Fallback if reset wasn't called properly (shouldn't happen)
             print(f"[DEBUG-FAILURE] last_commanded_pose missing in step! Fallback to currpos: {np.round(self.currpos, 3)}")
             self.last_commanded_pose = self.currpos.copy()
             
        target_pos = self.last_commanded_pose[:3] + xyz_delta
        
        # 5. 更新姿态 (使用 rotvec 叠加在 last_commanded_pose 上)
        # Old: current_rot = Rotation.from_quat(self.currpos[3:])
        current_rot = Rotation.from_quat(self.last_commanded_pose[3:])
        delta_rot = Rotation.from_rotvec(rot_delta)
        target_rot = delta_rot * current_rot
        target_quat = target_rot.as_quat()

        if np.linalg.norm(action) < 1e-5:
           # self._log(f"[DEBUG] [Step] Zero Action Detected. TargetQuat: {np.round(target_quat, 3)}")
           pass
        
        # 6. 组合目标位姿
        target_pose = np.concatenate([target_pos, target_quat])
        
        # 7. 应用安全边界框
        target_pose = self.clip_safety_box(target_pose)
        
        # 8. 发送位姿命令
        # print(f"[DEBUG] Sending Pose Command: {target_pose}")
        self._send_pos_command(target_pose)
        
        # [DRIFT FIX] 更新 last_commanded_pose
        self.last_commanded_pose = target_pose.copy()
        
        # 9. 处理夹爪动作
        gripper_action = action[6] * self.action_scale[2]
        # print(f"[DEBUG] Sending Gripper Command: {gripper_action}")
        self._send_gripper_command(gripper_action, mode="continuous")
        
        # 10. 控制频率
        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))
        
        # 11. 更新状态
        self._update_currpos()
        self.curr_path_length += 1
        
        # 12. 获取观察
        obs = self._get_obs()
        
        # 13. 计算奖励
        reward = self.compute_reward(obs)
        
        # 14. 检查是否完成
        done = (
            self.curr_path_length >= self.max_episode_length
            or reward > 0
            or self.terminate
        )
        
        info = {"succeed": bool(reward)}
        return obs, float(reward), done, False, info
    
    def compute_reward(self, obs) -> bool:
        """
        计算奖励（基于位姿检查）
        
        Args:
            obs: 观察字典
        
        Returns:
            reward: float (0 或 1)
        """
        current_pose = obs["state"]["tcp_pose"]
        
        # 计算位姿差异
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
            return 1.0
        return 0.0
    
    def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        """
        裁剪到安全边界框内
        """
        # 裁剪位置
        # print(f"[DEBUG] Pre-Clip Pose: {np.round(pose[:3], 3)}")
        pose[:3] = np.clip(
            pose[:3],
            self.xyz_bounding_box.low,
            self.xyz_bounding_box.high
        )
        if np.any(pose[:3] == self.xyz_bounding_box.high) or np.any(pose[:3] == self.xyz_bounding_box.low):
             pass
             # print(f"[DEBUG] Safety Box Clipped Pose to: {np.round(pose[:3], 3)}")
        
        # 裁剪姿态 (处理欧拉角的不连续性)
        # [FIX] Disabled rotation clipping due to Euler singularity at Ry=pi (Gimbal Lock)
        # The conversion to Euler and back is unstable for 180 deg rotations and causes clamping errors.
        """
        from scipy.spatial.transform import Rotation as R
        euler = R.from_quat(pose[3:]).as_euler("xyz")
        
        low = self.rpy_bounding_box.low
        high = self.rpy_bounding_box.high
        
        for i in range(3):
            # [ROBUSTNESS] 如果限制范围包含 pi/-pi (例如 > 3.1), 
            # 那么在这个维度的裁剪会导致严重的抖动 (因为 pi 和 -pi 是同一个姿态)。
            # 我们仅对明显小于 pi 的限制进行强制裁剪。
            if high[i] > 3.1:
                continue
                
            val = euler[i]
            if val < low[i]:
                euler[i] = low[i]
            elif val > high[i]:
                euler[i] = high[i]
                
        pose[3:] = R.from_euler("xyz", euler).as_quat()
        """
        return pose
    
    def _get_reset_pose(self) -> np.ndarray:
        """
        获取重置位姿（支持随机化）
        
        Returns:
            reset_pose: np.ndarray[7] - [x, y, z, qx, qy, qz, qw]
        """
        reset_pose = self.resetpos.copy()
        
        if self.randomreset:
            # XY 平面随机化
            reset_pose[:2] += np.random.uniform(
                -self.random_xy_range,
                self.random_xy_range,
                (2,)
            )
            
            # Z 轴旋转随机化
            euler_random = self._RESET_POSE[3:].copy()
            euler_random[-1] += np.random.uniform(
                -self.random_rz_range,
                self.random_rz_range
            )
            reset_pose[3:] = euler_2_quat(euler_random)
        
        return reset_pose
    
    def interpolate_move(self, goal: np.ndarray, timeout: float):
        """移动到目标位置（位置线性插值，姿态球面线性插值 SLERP）"""
        if goal.shape == (6,):
            goal = np.concatenate([goal[:3], euler_2_quat(goal[3:])])
            
        steps = int(timeout * self.hz)
        if steps <= 1:
            self._send_pos_command(goal)
            return
            
        self._update_currpos()
        start_pose = self.currpos.copy()
        # [DIAGNOSTIC] Log interpolation range
        self._log(f"[DEBUG] Interp Start: {np.round(start_pose[:3], 3)} End: {np.round(goal[:3], 3)}")
        
        # 位置线性插值
        pos_path = np.linspace(start_pose[:3], goal[:3], steps)
        
        # 姿态球面线性插值 (SLERP)
        from scipy.spatial.transform import Rotation as R
        from scipy.spatial.transform import Slerp
        
        # [FIX] 确保两个四元数在同一半球（避免 Slerp 选择长路径）
        # q 和 -q 表示同一个旋转，但 Slerp 会选择最短路径
        # 如果点积为负，说明它们在不同半球，需要翻转其中一个
        start_quat = start_pose[3:].copy()
        goal_quat = goal[3:].copy()
        
        if np.dot(start_quat, goal_quat) < 0:
            goal_quat = -goal_quat
            self._log("[DEBUG] Flipped goal quaternion to ensure shortest Slerp path")
        
        # 确保输入是有效的四元数 [x, y, z, w]
        # 注意：euler_2_quat 已经更新为返回 XYZW
        key_rots = R.from_quat([start_quat, goal_quat])
        key_times = [0, 1]
        slerp = Slerp(key_times, key_rots)
        
        interp_times = np.linspace(0, 1, steps)
        quat_path = slerp(interp_times).as_quat()
        
        for i in range(steps):
            p = np.concatenate([pos_path[i], quat_path[i]])
            self._send_pos_command(p)
            time.sleep(1 / self.hz)
            
        self._update_currpos()
    
    def go_to_reset(self, joint_reset=False):
        """
        移动到重置位置
        
        基类实现：移动到重置位姿（支持随机化）
        子类可以重写此方法以添加任务特定的重置逻辑
        
        Args:
            joint_reset: 是否执行关节重置（Isaac Sim 中可能不需要）
        """
        # 更新参数（如果需要）
        try:
            if hasattr(self.config, 'PRECISION_PARAM') and self.config.PRECISION_PARAM:
                self.session.post(self.url + "update_param", json=self.config.PRECISION_PARAM, timeout=0.5)
        except:
            pass
        
        # 执行关节复位（如果需要）
        if joint_reset:
            try:
                self.session.post(self.url + "jointreset", timeout=0.5)
            except:
                pass
        
        # 获取重置位姿（支持随机化）
        reset_pose = self._get_reset_pose()
        # 转换为 WXYZ 显示
        quat_wxyz = np.array([reset_pose[6], reset_pose[3], reset_pose[4], reset_pose[5]])
        pose_display = np.concatenate([reset_pose[:3], quat_wxyz])
        print(f"[INFO] Moving to RESET_POSE: {np.round(pose_display, 3)} [WXYZ]")
        
        # 移动到重置位姿
        self.interpolate_move(reset_pose, timeout=1.0)
        
        # 更新参数（如果需要）
        try:
            if hasattr(self.config, 'COMPLIANCE_PARAM') and self.config.COMPLIANCE_PARAM:
                self.session.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM, timeout=0.5)
        except:
            pass
    
    def save_video_recording(self):
        """保存录制的视频"""
        try:
            if hasattr(self, 'recording_frames') and len(self.recording_frames) > 0:
                import os
                from datetime import datetime
                
                if not os.path.exists('./videos'):
                    os.makedirs('./videos')
                
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                
                for camera_key in self.recording_frames[0].keys():
                    video_path = f'./videos/isaac_sim_{camera_key}_{timestamp}.mp4'
                    
                    # 获取第一帧的形状
                    first_frame = self.recording_frames[0][camera_key]
                    height, width = first_frame.shape[:2]
                    
                    video_writer = cv2.VideoWriter(
                        video_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        10,
                        (width, height),
                    )
                    
                    for frame_dict in self.recording_frames:
                        if camera_key in frame_dict:
                            video_writer.write(frame_dict[camera_key])
                    
                    video_writer.release()
                    print(f"Saved video for camera {camera_key} at {video_path}")
                
                self.recording_frames.clear()
        except Exception as e:
            print(f"Failed to save video: {e}")
    
    def reset_scene(self):
        """
        重置整个 USD 场景
        
        通过 HTTP 接口调用服务器端的场景重置功能
        
        功能：
        - 重置物理世界到初始状态
        - 重置所有对象到初始位置
        - 重置机器人到初始状态
        - 打开夹爪
        """
        try:
            response = self.session.post(self.url + "reset_scene", timeout=2.0)
            if response.status_code == 200:
                print("[INFO] Scene reset successful")
                # 等待场景稳定 (加大等待时间，确保 Server 队列处理完毕)
                time.sleep(3.0)
                # 更新状态
                self._update_currpos()
                # [FIX] 同步 last_commanded_pose，防止 step() 继续使用旧的指令位置导致回跳
                if self.currpos is not None:
                    self.last_commanded_pose = self.currpos.copy()
                    print(f"[INFO] Client state synced to reset pose: {np.round(self.currpos, 3)}")
            else:
                print(f"[WARNING] Scene reset returned status {response.status_code}")
        except Exception as e:
            print(f"[WARNING] Failed to reset scene: {e}")
    
    def close(self):
        """关闭环境"""
        # 停止键盘监听
        if self.listener is not None:
            try:
                self.listener.stop()
            except:
                pass
        
        # 关闭 WebSocket 连接
        if self.ws_client is not None:
            try:
                self.ws_client.disconnect()
            except:
                pass
        
        # 关闭 HTTP 会话
        if hasattr(self, 'session'):
            try:
                self.session.close()
            except:
                pass
        
        # 保存视频
        if self.save_video and hasattr(self, 'recording_frames'):
            self.save_video_recording()
        
        print("[INFO] Isaac Sim Franka Environment closed")
