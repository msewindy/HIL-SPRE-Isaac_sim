#!/usr/bin/env python3
"""
检查BC训练后的动作分布，诊断BC是否学习到了"不动"策略
"""

import pickle as pkl
import numpy as np
import jax
import jax.numpy as jnp
from absl import app, flags
from flax.training import checkpoints
import os

from serl_launcher.agents.continuous.bc import BCAgent
from serl_launcher.utils.launcher import make_bc_agent
from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment.")
flags.DEFINE_string("bc_checkpoint_path", None, "Path to BC checkpoint.")
flags.DEFINE_string("demo_path", None, "Path to demo data for sampling observations.")
flags.DEFINE_integer("sample_size", 1000, "Number of observations to sample.")

devices = jax.local_devices()
num_devices = len(devices)
if num_devices == 1:
    sharding = jax.sharding.SingleDeviceSharding(devices[0])
else:
    from jax.sharding import Mesh, NamedSharding, PartitionSpec
    mesh = Mesh(devices, axis_names=('devices',))
    sharding = NamedSharding(mesh, PartitionSpec('devices'))


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


def print_yellow(x):
    return print("\033[93m {}\033[00m".format(x))


def print_red(x):
    return print("\033[91m {}\033[00m".format(x))


def main(_):
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    
    # 加载环境（仅用于获取observation space）
    env = config.get_environment(fake_env=True, save_video=False, classifier=False, skip_server_connection=True)
    
    # 创建BC agent
    bc_agent: BCAgent = make_bc_agent(
        seed=42,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=config.image_keys,
        encoder_type=config.encoder_type,
    )
    
    # 加载checkpoint
    latest_ckpt = checkpoints.latest_checkpoint(os.path.abspath(FLAGS.bc_checkpoint_path))
    if latest_ckpt is None:
        print_red(f"[ERROR] No checkpoint found in {FLAGS.bc_checkpoint_path}")
        return
    
    print_green(f"[INFO] Loading checkpoint from {latest_ckpt}")
    bc_ckpt = checkpoints.restore_checkpoint(latest_ckpt, bc_agent.state)
    bc_agent = bc_agent.replace(state=bc_ckpt)
    
    # 加载演示数据，采样观察
    print_green(f"[INFO] Loading demo data from {FLAGS.demo_path}")
    with open(FLAGS.demo_path, "rb") as f:
        transitions = pkl.load(f)
    
    # 采样观察
    sample_indices = np.random.choice(len(transitions), size=min(FLAGS.sample_size, len(transitions)), replace=False)
    sample_obs = [transitions[i]['observations'] for i in sample_indices]
    sample_actions_demo = np.array([transitions[i]['actions'] for i in sample_indices])
    
    # 将观察转换为batch格式
    # 注意：这里需要根据实际的观察格式来处理
    # 假设观察是dict格式，包含state和图像
    obs_batch = {}
    for key in sample_obs[0].keys():
        if isinstance(sample_obs[0][key], np.ndarray):
            obs_batch[key] = np.stack([obs[key] for obs in sample_obs])
        else:
            obs_batch[key] = [obs[key] for obs in sample_obs]
    
    # 采样BC动作（多次采样，取统计）
    print_green(f"[INFO] Sampling BC actions for {len(sample_obs)} observations...")
    rng = jax.random.PRNGKey(42)
    bc_actions_list = []
    
    for i in range(10):  # 采样10次
        rng, key = jax.random.split(rng)
        try:
            bc_actions = bc_agent.sample_actions(
                observations=jax.device_put(obs_batch, sharding),
                seed=key
            )
            bc_actions_np = np.asarray(jax.device_get(bc_actions))
            bc_actions_list.append(bc_actions_np)
        except Exception as e:
            print_red(f"[ERROR] Failed to sample actions: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # 计算统计
    bc_actions_all = np.concatenate(bc_actions_list, axis=0)
    bc_action_norms = np.linalg.norm(bc_actions_all, axis=-1)
    demo_action_norms = np.linalg.norm(sample_actions_demo, axis=-1)
    
    print_green(f"\n[BC ACTION DISTRIBUTION ANALYSIS]")
    print_green(f"=" * 60)
    
    # BC动作统计
    print_green(f"\n[BC输出动作统计]")
    print_green(f"  动作范数 - Mean: {np.mean(bc_action_norms):.6f}, Std: {np.std(bc_action_norms):.6f}")
    print_green(f"  动作范数 - Min: {np.min(bc_action_norms):.6f}, Max: {np.max(bc_action_norms):.6f}")
    print_green(f"  零动作数量: {np.sum(bc_action_norms < 1e-6)} ({np.sum(bc_action_norms < 1e-6)/len(bc_action_norms)*100:.2f}%)")
    print_green(f"  各维度均值: {np.mean(bc_actions_all, axis=0)}")
    print_green(f"  各维度标准差: {np.std(bc_actions_all, axis=0)}")
    
    # 演示动作统计
    print_green(f"\n[演示动作统计]")
    print_green(f"  动作范数 - Mean: {np.mean(demo_action_norms):.6f}, Std: {np.std(demo_action_norms):.6f}")
    print_green(f"  动作范数 - Min: {np.min(demo_action_norms):.6f}, Max: {np.max(demo_action_norms):.6f}")
    print_green(f"  零动作数量: {np.sum(demo_action_norms < 1e-6)} ({np.sum(demo_action_norms < 1e-6)/len(demo_action_norms)*100:.2f}%)")
    print_green(f"  各维度均值: {np.mean(sample_actions_demo, axis=0)}")
    print_green(f"  各维度标准差: {np.std(sample_actions_demo, axis=0)}")
    
    # 对比分析
    print_green(f"\n[对比分析]")
    norm_diff = abs(np.mean(bc_action_norms) - np.mean(demo_action_norms))
    print_green(f"  动作范数差异: {norm_diff:.6f}")
    
    # 诊断
    print_green(f"\n[诊断结果]")
    if np.mean(bc_action_norms) < 0.05:
        print_red(f"  ❌ BC输出动作范数过小 ({np.mean(bc_action_norms):.6f} < 0.05)")
        print_red(f"     BC策略学习到了'不动'策略")
        print_yellow(f"  [建议]")
        print_yellow(f"    1. 使用更激进的过滤: --filter_max_consecutive=5")
        print_yellow(f"    2. 检查演示数据质量（动作值是否太小）")
        print_yellow(f"    3. 考虑实现加权训练")
    elif np.mean(bc_action_norms) < 0.1:
        print_yellow(f"  ⚠️  BC输出动作范数较小 ({np.mean(bc_action_norms):.6f} < 0.1)")
        print_yellow(f"     BC策略可能不够活跃")
        print_yellow(f"  [建议] 尝试更激进的过滤或增加训练步数")
    else:
        print_green(f"  ✅ BC输出动作范数合理 ({np.mean(bc_action_norms):.6f} >= 0.1)")
        print_green(f"     BC策略学习到了有效的动作分布")
    
    if norm_diff > 0.1:
        print_yellow(f"  ⚠️  BC输出与演示数据差异较大 ({norm_diff:.6f} > 0.1)")
        print_yellow(f"     BC可能没有学习到演示数据的模式")
    else:
        print_green(f"  ✅ BC输出与演示数据差异较小 ({norm_diff:.6f} <= 0.1)")
    
    print_green(f"=" * 60)


if __name__ == "__main__":
    app.run(main)
