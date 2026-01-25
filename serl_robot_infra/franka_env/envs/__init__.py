from franka_env.envs.franka_env import FrankaEnv, DefaultEnvConfig
from franka_env.envs.franka_wrench_env import FrankaWrenchEnv

# ========== Isaac Sim 环境导出 ==========
# 使用 try-except 处理 Isaac Sim 未安装或环境类未实现的情况
# 注意：IsaacSimRAMEnv 是任务特定的，已移动到 experiments/ram_insertion/，不再在此导出
try:
    from franka_env.envs.isaac_sim_env import IsaacSimFrankaEnv
    
    # 导出所有环境类
    __all__ = [
        "FrankaEnv",
        "DefaultEnvConfig",
        "FrankaWrenchEnv",
        "IsaacSimFrankaEnv",
    ]
except ImportError:
    # 如果 Isaac Sim 环境未实现或 Isaac Sim 未安装，只导出真实环境类
    __all__ = [
        "FrankaEnv",
        "DefaultEnvConfig",
        "FrankaWrenchEnv",
    ]