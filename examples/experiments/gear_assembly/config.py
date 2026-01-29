import os
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict

from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
    GripperCloseEnv
)
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.franka_env import DefaultEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig
from experiments.gear_assembly.wrapper import GearAssemblyEnv

# ========== Isaac Sim 支持 ==========
# 本文件支持真实环境和 Isaac Sim 仿真环境的配置
# 通过 fake_env 参数在 get_environment() 中切换

class EnvConfig(DefaultEnvConfig):
    SERVER_URL = "http://127.0.0.2:5000/"
    REALSENSE_CAMERAS = {
        "wrist_1": {
            "serial_number": "127122270146",
            "dim": (1280, 720),
            "exposure": 40000,
        },
        "wrist_2": {
            "serial_number": "127122270350",
            "dim": (1280, 720),
            "exposure": 40000,
        },
    }
    IMAGE_CROP = {
        "wrist_1": lambda img: img[150:450, 350:1100],
        "wrist_2": lambda img: img[100:500, 400:900],
    }
    # TODO: 根据实际测量值更新以下位姿配置
    TARGET_POSE = np.array([0.0, -0.60, 0.42, 0, np.pi, 0])
    GRASP_POSE = np.array([0.01, -0.75, 0.40, 0, np.pi, 0])
    # Safer RESET: Higher and closer to base to avoid elbow singularity/drift
    RESET_POSE = np.array([0.0, -0.4, 0.55, 0, np.pi, 0])
    ABS_POSE_LIMIT_LOW = TARGET_POSE - np.array([0.03, 0.02, 0.01, 0.01, 0.1, 0.4])
    ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array([0.03, 0.02, 0.05, 0.01, 0.1, 0.4])
    RANDOM_RESET = False
    RANDOM_XY_RANGE = 0.0
    RANDOM_RZ_RANGE = 0.0
    ACTION_SCALE = (0.01, 0.06, 1)
    DISPLAY_IMAGE = True
    MAX_EPISODE_LENGTH = 100
    COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0,
        "translational_clip_x": 0.0075,
        "translational_clip_y": 0.0016,
        "translational_clip_z": 0.0055,
        "translational_clip_neg_x": 0.002,
        "translational_clip_neg_y": 0.0016,
        "translational_clip_neg_z": 0.005,
        "rotational_clip_x": 0.01,
        "rotational_clip_y": 0.025,
        "rotational_clip_z": 0.005,
        "rotational_clip_neg_x": 0.01,
        "rotational_clip_neg_y": 0.025,
        "rotational_clip_neg_z": 0.005,
        "rotational_Ki": 0,
    }
    PRECISION_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 250,
        "rotational_damping": 9,
        "translational_Ki": 0.0,
        "translational_clip_x": 0.1,
        "translational_clip_y": 0.1,
        "translational_clip_z": 0.1,
        "translational_clip_neg_x": 0.1,
        "translational_clip_neg_y": 0.1,
        "translational_clip_neg_z": 0.1,
        "rotational_clip_x": 0.5,
        "rotational_clip_y": 0.5,
        "rotational_clip_z": 0.5,
        "rotational_clip_neg_x": 0.5,
        "rotational_clip_neg_y": 0.5,
        "rotational_clip_neg_z": 0.5,
        "rotational_Ki": 0.0,
    }


class IsaacSimEnvConfig(DefaultEnvConfig):
    """
    Isaac Sim 环境配置类
    
    与 EnvConfig 的区别：
    1. SERVER_URL 指向 isaac_sim_server（默认端口 5001）
    2. 移除 COMPLIANCE_PARAM、PRECISION_PARAM（Isaac Sim 使用自己的物理引擎）
    3. 移除 LOAD_PARAM（在 Isaac Sim 中直接配置）
    4. 使用虚拟相机标识（不需要真实序列号，字段值在 Isaac Sim 中不使用）
    5. 保留所有任务相关配置（位姿、边界框等）
    """
    
    # Isaac Sim 服务器 URL（指向 isaac_sim_server.py）
    SERVER_URL = "http://192.168.31.198:5001/"
    
    # 虚拟相机配置（Isaac Sim 使用虚拟相机）
    # 注意：对于 Isaac Sim 环境，只需要键名，字段值（serial_number、dim、exposure）不使用
    # Isaac Sim 服务器通过 camera_prim_paths 参数加载相机，不依赖这些字段
    REALSENSE_CAMERAS = {
        "wrist_1": {},  # 只需要键名，用于定义观察空间和图像键名
        "wrist_2": {},  # 字段值在 Isaac Sim 中不使用
    }
    
    # 图像裁剪配置（与真实环境相同）
    IMAGE_CROP = {
        # [FIX] Square Cropping (300x300) for Sim (Resized to 128x128 later)
        # Wrist 1 Center: (300, 725) -> Y[150:450], X[575:875]
        "wrist_1": lambda img: img[150:450, 575:875],
        # Wrist 2 Center: (300, 650) -> Y[150:450], X[500:800]
        "wrist_2": lambda img: img[150:450, 500:800],
    }
    
    # [OPTIMIZATION] 根据 Franka 约 855mm 的臂展和 USD 场景中物体的实际分布进行调整：
    # 1. TARGET_POSE: 组装目标位姿（大齿轮/底座位置 y=-0.6, z~0.41）
    TARGET_POSE = np.array([0.0, -0.60, 0.42, 0, np.pi, 0])
    # 2. GRASP_POSE: 抓取位姿（中齿轮初始位置 y=-0.75, z~0.40）。
    # 注意：y=-0.75 已经接近 Franka 的最大展弦比，建议不要再往外远了。
    GRASP_POSE = np.array([0.01, -0.75, 0.40, 0, np.pi, 0])
    
    # 3. RESET_POSE: Safer position
    # User requested [1, 0, 0, 0] quaternion, which corresponds to Rx=pi
    RESET_POSE = np.array([0.0, -0.4, 0.55, 0, np.pi, 0])
    
    # 4. 安全区 (Safety Box): 必须包含以上所有点，并留有足够扰动空间。
    # 设置为以 TARGET_POSE 为中心，±0.15m (X), ±0.20m (Y), ±0.25m (Z) 的大包络面。
    # [FIX] Widen limits to include RESET_POSE (y=-0.4)
    ABS_POSE_LIMIT_LOW = TARGET_POSE - np.array([0.2, 0.4, 0.01, 0.5, 0.5, 0.5])
    ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array([0.2, 0.4, 0.30, 0.5, 0.5, 0.5])
    
    # 随机重置配置
    RANDOM_RESET = False
    RANDOM_XY_RANGE = 0.0
    RANDOM_RZ_RANGE = 0.0
    
    # 动作缩放
    ACTION_SCALE = (0.01, 0.06, 1)
    
    # 其他配置
    DISPLAY_IMAGE = True
    MAX_EPISODE_LENGTH = 4800
    
    # 为了兼容性，定义空字典或默认值（如果基类需要）
    COMPLIANCE_PARAM: Dict[str, float] = {}
    PRECISION_PARAM: Dict[str, float] = {}
    RESET_PARAM: Dict[str, float] = {}
    LOAD_PARAM: Dict[str, float] = {
        "mass": 0.0,
        "F_x_center_load": [0.0, 0.0, 0.0],
        "load_inertia": [0, 0, 0, 0, 0, 0, 0, 0, 0]
    }
    REWARD_THRESHOLD: np.ndarray = np.array([0.01, 0.01, 0.01, 0.1, 0.1, 0.1])  # 默认阈值
    GRIPPER_SLEEP: float = 0.6
    JOINT_RESET_PERIOD: int = 0


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["wrist_1", "wrist_2"]
    classifier_keys = ["wrist_1", "wrist_2"]
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    buffer_period = 1000
    checkpoint_period = 5000
    steps_per_update = 50
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-continuous-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        """
        获取环境实例
        
        Args:
            fake_env: True 使用 Isaac Sim 仿真环境，False 使用真实环境
            save_video: 是否保存视频
            classifier: 是否使用奖励分类器
        
        Returns:
            env: Gym 环境实例
        """
        # ========== 环境选择逻辑 ==========
        if fake_env:
            # 使用 Isaac Sim 仿真环境
            try:
                from experiments.gear_assembly.isaac_sim_gear_env_enhanced import IsaacSimGearAssemblyEnvEnhanced
                env = IsaacSimGearAssemblyEnvEnhanced(
                    fake_env=True,  # 始终为 True（仿真环境）
                    save_video=save_video,
                    config=IsaacSimEnvConfig(),
                    enable_domain_randomization=False,  # 域随机化已关闭（根据项目需求）
                )
            except ImportError as e:
                raise ImportError(
                    f"Failed to import Isaac Sim environment: {e}\n"
                    "Please ensure Isaac Sim is installed and the environment classes are implemented.\n"
                    "Expected files: examples/experiments/gear_assembly/isaac_sim_gear_env_enhanced.py"
                )
        else:
            # 使用真实环境（原有逻辑）
            env = GearAssemblyEnv(
                fake_env=False,
                save_video=save_video,
                config=EnvConfig(),
            )
        
        # ========== 环境包装器（真实和仿真环境共用）==========
        # 1. 固定夹爪包装器（任务要求夹爪关闭）
        # [FIX] Disabled strict gripper closing to allow Gamepad control for demo recording
        # env = GripperCloseEnv(env)
        
        # 2. SpaceMouse 干预（真实环境必需，仿真环境可选）
        if not fake_env:
            # 真实环境：必需 SpaceMouse 进行干预
            env = SpacemouseIntervention(env)
        # 注意：仿真环境也可以使用 SpaceMouse（如果已连接）
        # 如果需要，可以取消下面的注释：
        else:
            # [新增] 仿真环境：使用手柄控制
            try:
                from franka_env.envs.wrappers import GamepadIntervention
                env = GamepadIntervention(env, joystick_id=0, sensitivity=0.2)
                print("[INFO] Using Gamepad for intervention in Simulation (Sensitivity=0.2)")
                # print("[INFO] Gamepad intervention disabled for stability testing")
            except ImportError:
                print("[WARNING] Gamepad wrapper not found, falling back to SpaceMouse or No-Intervention")
                # env = SpacemouseIntervention(env) # 如果想回退到 SpaceMouse
        
        # 3. 相对坐标系包装器
        env = RelativeFrame(env)
        
        # 4. 四元数转欧拉角包装器
        env = Quat2EulerWrapper(env)
        
        # 5. SERL 观察包装器
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        
        # 6. 动作分块包装器
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        
        # 7. 奖励分类器（如果需要）
        if classifier:
            classifier = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("classifier_ckpt/"),
            )

            def reward_func(obs):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                # [FIX] Use index -1 for gripper_pose as index 6 points to tcp_vel[0] due to Quat2EulerWrapper
                return int(sigmoid(classifier(obs)) > 0.85 and obs['state'][0, -1] > 0.04)

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        
        return env
