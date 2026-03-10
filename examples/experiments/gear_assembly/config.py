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
    
    # 图像裁剪配置（Isaac Sim 相机 1280x720，img 为 [H, W] = [720, 1280]）
    IMAGE_CROP = {
        # wrist_1: 720x720，按原图高做正方形截取（居中）
        # 原图中心 (640, 360)，取宽 720 居中 → col 280:1000, row 0:720
        "wrist_1": lambda img: img[0:720, 280:1000],
        # wrist_2: 500x500，严格以原图正中心 (640, 360) 为裁剪中心
        # col 390:890, row 110:610
        "wrist_2": lambda img: img[110:610, 390:890],
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
    ABS_POSE_LIMIT_LOW = TARGET_POSE - np.array([0.15, 0.4, 0.01, 0.5, 0.5, 0.5])
    ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array([0.15, 0.4, 0.30, 0.5, 0.5, 0.5])
    
    # 随机重置配置
    RANDOM_RESET = False
    RANDOM_XY_RANGE = 0.0
    RANDOM_RZ_RANGE = 0.0

    # Reset Scene：是否随机化齿轮位置与底座角度（True=在默认固定位置上随机偏移，False=每次重置到场景默认固定位置）
    RESET_RANDOMIZE_GEAR_AND_BASE = True
    # 随机化范围（仅当 RESET_RANDOMIZE_GEAR_AND_BASE=True 时生效）：(min, max)
    # 中齿轮位置 X/Y 随机偏移范围（米）
    GEAR_RESET_X_RANGE = (-0.10, 0.10)
    GEAR_RESET_Y_RANGE = (-0.035, 0.035)
    # 底座 Z 轴旋转随机偏移范围（度）
    GEAR_BASE_RESET_ANGLE_RANGE = (-10.0, 10.0)

    # 动作缩放
    # 为了手柄控制更稳更精准，统一下调动作增量（恒定系数，不做模式切换）
    # 平移/旋转都比之前更小，减轻远端与抓取后的抖动。
    ACTION_SCALE = (0.0012, 0.008, 1)
    
    # 其他配置
    DISPLAY_IMAGE = True
    MAX_EPISODE_LENGTH = 500
    
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

    # γ=0.995: 针对 ~283 步轨迹优化
    # 论文大部分任务用 γ=0.97 + 100 步 → Q(s_0)≈0.048
    # 我们用 γ=0.995 + 283 步 → Q(s_0)≈0.242（更强的信号传播）
    discount: float = 0.995

    def get_environment(self, fake_env=False, save_video=False, classifier=False, isaac_server_url=None, skip_server_connection=False):
        """
        获取环境实例
        
        Args:
            fake_env: True 使用 Isaac Sim 仿真环境，False 使用真实环境
            save_video: 是否保存视频
            classifier: 是否使用奖励分类器
            isaac_server_url: Isaac Sim 服务器 URL（可选，覆盖 config 中的 SERVER_URL）
        
        Returns:
            env: Gym 环境实例
        """
        if fake_env:
            try:
                from experiments.gear_assembly.isaac_sim_gear_env_enhanced import IsaacSimGearAssemblyEnvEnhanced
                isaac_config = IsaacSimEnvConfig()
                if isaac_server_url is not None:
                    isaac_config.SERVER_URL = isaac_server_url
                    if not isaac_config.SERVER_URL.endswith('/'):
                        isaac_config.SERVER_URL += '/'
                    print(f"[INFO] Using Isaac Sim server URL from command line: {isaac_config.SERVER_URL}")
                else:
                    print(f"[INFO] Using Isaac Sim server URL from config: {isaac_config.SERVER_URL}")
                env = IsaacSimGearAssemblyEnvEnhanced(
                    fake_env=True,
                    save_video=save_video,
                    config=isaac_config,
                    enable_domain_randomization=False,
                    skip_server_connection=skip_server_connection,
                )
            except ImportError as e:
                raise ImportError(
                    f"Failed to import Isaac Sim environment: {e}\n"
                    "Please ensure Isaac Sim is installed and the environment classes are implemented.\n"
                    "Expected files: examples/experiments/gear_assembly/isaac_sim_gear_env_enhanced.py"
                )
        else:
            env = GearAssemblyEnv(
                fake_env=False,
                save_video=save_video,
                config=EnvConfig(),
            )
        
        env = RelativeFrame(env)
        
        if not fake_env:
            env = SpacemouseIntervention(env)
        else:
            try:
                from franka_env.envs.wrappers import GamepadIntervention
                # 关键：将 GamepadIntervention 放在 RelativeFrame 外层，确保人控和策略都走同一坐标变换链
                env = GamepadIntervention(
                    env,
                    joystick_id=0,
                    sensitivity=1.0,
                    deadzone=0.08,
                )
                print("[INFO] Using Gamepad for intervention in Simulation (same transform path as policy)")
            except ImportError:
                print("[WARNING] Gamepad wrapper not found, falling back to SpaceMouse or No-Intervention")

        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        
        if classifier:
            classifier = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("classifier_ckpt/"),
            )

            def reward_func(obs):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                return int(sigmoid(classifier(obs)) > 0.85 and obs['state'][0, -1] > 0.04)

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        
        return env


class TrainPretrainConfig(TrainConfig):
    """
    针对长程任务优化的 RLPD 训练配置（预训练 + 动态采样比例 + 高折扣因子）。
    
    配合 train_rlpd_pretrain.py 使用。
    与 TrainConfig 共享环境配置和基础训练参数，新增预训练和长程优化参数。
    
    设计依据：
    - gear assembly 演示轨迹平均 ~548 步（远超 HIL-SERL 论文中的 ~100 步）
    - 纯 50/50 采样在长程稀疏奖励任务中效率极低
    - 需要预训练让 Bellman backup 有足够时间传播 Q 值
    - 默认 γ=0.97 的有效视野仅 ~33 步，对 548 步任务完全不够
    """

    # ==================== 折扣因子 ====================
    # 已统一到 TrainConfig 中 (γ=0.995)，此处不再单独覆盖

    # ==================== 预训练阶段参数 ====================
    # Learner 在 Actor 启动前，使用纯 demo 数据训练的步数。
    # 目的：让 Critic 建立完整的价值地图，Actor 学会基础策略。
    # γ=0.99 下，5000 步预训练可以让 Q 值传播约 200-300 步的有效距离。
    # 若效果不理想可增大到 10000。
    pretrain_steps: int = 5000

    # ==================== 动态采样比例参数 ====================
    # 正式训练开始时的 demo 数据采样比例（0.0 ~ 1.0）
    # 高比例减少早期脏数据（随机探索产生的无用数据）对训练的干扰
    initial_demo_ratio: float = 0.8

    # 最终的 demo 数据采样比例（退火目标值）
    # 0.5 = 标准 RLPD 的 50/50 比例
    final_demo_ratio: float = 0.5

    # 从 initial_demo_ratio 退火到 final_demo_ratio 所需的训练步数
    # 退火完成后，采样比例固定在 final_demo_ratio
    demo_ratio_anneal_steps: int = 20000
