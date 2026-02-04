#!/usr/bin/env python3
"""
验证BC策略学习的有效性

方法：
1. 从演示数据中选取一些数据
2. 将这些数据的观察输入BC策略
3. 对比BC输出的action和演示数据的真实action
4. 计算MSE Loss、分布差异等指标

这与BC的loss计算方法一致，用于验证BC是否准确学习了演示数据。
"""

import pickle as pkl
import numpy as np
import jax
import jax.numpy as jnp
from absl import app, flags
from flax.training import checkpoints
import os
from typing import Dict, List

from serl_launcher.agents.continuous.bc import BCAgent
from serl_launcher.utils.launcher import make_bc_agent
from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment.")
flags.DEFINE_string("bc_checkpoint_path", None, "Path to BC checkpoint.")
flags.DEFINE_string("demo_path", None, "Path to demo data.")
flags.DEFINE_integer("sample_size", 1000, "Number of transitions to sample from demo data.")
flags.DEFINE_integer("num_samples", 10, "Number of action samples per observation (for distribution analysis).")

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


def prepare_observations(observations_list: List[Dict]) -> Dict:
    """
    将观察列表转换为batch格式
    
    Args:
        observations_list: List of observation dictionaries
    
    Returns:
        Batch format observations dictionary
    """
    # 获取第一个观察的键
    if len(observations_list) == 0:
        raise ValueError("Empty observations list")
    
    batch_obs = {}
    for key in observations_list[0].keys():
        if isinstance(observations_list[0][key], np.ndarray):
            # 如果是numpy数组，stack它们
            batch_obs[key] = np.stack([obs[key] for obs in observations_list])
        elif isinstance(observations_list[0][key], dict):
            # 如果是字典，递归处理
            batch_obs[key] = {}
            for sub_key in observations_list[0][key].keys():
                if isinstance(observations_list[0][key][sub_key], np.ndarray):
                    batch_obs[key][sub_key] = np.stack([obs[key][sub_key] for obs in observations_list])
                else:
                    batch_obs[key][sub_key] = [obs[key][sub_key] for obs in observations_list]
        else:
            # 其他类型，保持列表
            batch_obs[key] = [obs[key] for obs in observations_list]
    
    return batch_obs


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
    
    # 加载演示数据
    print_green(f"[INFO] Loading demo data from {FLAGS.demo_path}")
    with open(FLAGS.demo_path, "rb") as f:
        transitions = pkl.load(f)
    
    # 展平transitions（如果是嵌套列表）
    if isinstance(transitions, list) and len(transitions) > 0:
        if isinstance(transitions[0], list):
            # 嵌套列表：每个元素是一个轨迹
            flat_transitions = []
            for traj in transitions:
                flat_transitions.extend(traj)
            transitions = flat_transitions
    
    # 采样transitions
    sample_size = min(FLAGS.sample_size, len(transitions))
    sample_indices = np.random.choice(len(transitions), size=sample_size, replace=False)
    sample_transitions = [transitions[i] for i in sample_indices]
    
    print_green(f"[INFO] Sampling {sample_size} transitions from {len(transitions)} total transitions")
    
    # 提取观察和动作
    sample_obs = [t['observations'] for t in sample_transitions]
    demo_actions = np.array([t['actions'] for t in sample_transitions])
    
    # 准备batch格式的观察
    print_green(f"[INFO] Preparing batch observations...")
    batch_obs = prepare_observations(sample_obs)
    
    # 使用BC策略预测动作（使用mode）
    print_green(f"[INFO] Computing BC predictions (mode)...")
    rng = jax.random.PRNGKey(42)
    
    # 获取BC输出的分布
    dist = bc_agent.forward_policy(
        observations=jax.device_put(batch_obs, sharding),
        temperature=1.0,
    )
    
    # 计算mode（均值）
    bc_actions_mode = dist.mode()
    bc_actions_mode_np = np.asarray(jax.device_get(bc_actions_mode))
    
    # 计算MSE Loss（与BC训练时的计算一致）
    mse = ((bc_actions_mode_np - demo_actions) ** 2).sum(-1)
    mse_mean = np.mean(mse)
    mse_std = np.std(mse)
    
    # 计算log probability（演示动作在BC分布中的概率）
    log_probs = dist.log_prob(jax.device_put(demo_actions, sharding))
    log_probs_np = np.asarray(jax.device_get(log_probs))
    log_prob_mean = np.mean(log_probs_np)
    log_prob_std = np.std(log_probs_np)
    
    # 采样多个动作（用于分布分析）
    print_green(f"[INFO] Sampling {FLAGS.num_samples} actions per observation for distribution analysis...")
    bc_actions_samples = []
    for i in range(FLAGS.num_samples):
        rng, key = jax.random.split(rng)
        bc_actions_sample = bc_agent.sample_actions(
            observations=jax.device_put(batch_obs, sharding),
            seed=key,
            argmax=False,  # 使用采样
        )
        bc_actions_samples.append(np.asarray(jax.device_get(bc_actions_sample)))
    
    bc_actions_samples_np = np.array(bc_actions_samples)  # Shape: (num_samples, batch_size, action_dim)
    
    # 计算统计信息
    print_green(f"\n[BC LEARNING VALIDATION RESULTS]")
    print_green("=" * 80)
    
    # 1. MSE Loss（与BC训练时的计算一致）
    print_green(f"\n[1. MSE Loss (BC训练时的计算)]")
    print_green(f"   MSE Mean: {mse_mean:.6f}")
    print_green(f"   MSE Std: {mse_std:.6f}")
    print_green(f"   MSE Min: {np.min(mse):.6f}")
    print_green(f"   MSE Max: {np.max(mse):.6f}")
    
    # 判断标准
    if mse_mean < 0.01:
        print_green(f"   ✅ MSE很小，BC输出的action非常接近演示数据的action")
    elif mse_mean < 0.1:
        print_yellow(f"   ⚠️  MSE较小，BC输出的action接近演示数据的action，但仍有改进空间")
    else:
        print_red(f"   ❌ MSE较大，BC输出的action与演示数据的action差异较大")
    
    # 2. Log Probability
    print_green(f"\n[2. Log Probability (演示动作在BC分布中的概率)]")
    print_green(f"   Log Prob Mean: {log_prob_mean:.6f}")
    print_green(f"   Log Prob Std: {log_prob_std:.6f}")
    print_green(f"   Log Prob Min: {np.min(log_probs_np):.6f}")
    print_green(f"   Log Prob Max: {np.max(log_probs_np):.6f}")
    
    # 判断标准（log_prob越大越好，通常> -10表示概率较高）
    if log_prob_mean > -5:
        print_green(f"   ✅ Log Prob很大，演示动作在BC分布中概率很高")
    elif log_prob_mean > -10:
        print_yellow(f"   ⚠️  Log Prob中等，演示动作在BC分布中概率中等")
    else:
        print_red(f"   ❌ Log Prob很小，演示动作在BC分布中概率很低")
    
    # 3. Action分布对比
    print_green(f"\n[3. Action分布对比]")
    
    # BC输出的mode
    bc_action_norms_mode = np.linalg.norm(bc_actions_mode_np, axis=-1)
    demo_action_norms = np.linalg.norm(demo_actions, axis=-1)
    
    print_green(f"   [BC Mode输出]")
    print_green(f"     动作范数 - Mean: {np.mean(bc_action_norms_mode):.6f}, Std: {np.std(bc_action_norms_mode):.6f}")
    print_green(f"     动作范数 - Min: {np.min(bc_action_norms_mode):.6f}, Max: {np.max(bc_action_norms_mode):.6f}")
    print_green(f"     各维度均值: {np.mean(bc_actions_mode_np, axis=0)}")
    print_green(f"     各维度标准差: {np.std(bc_actions_mode_np, axis=0)}")
    
    print_green(f"   [演示数据]")
    print_green(f"     动作范数 - Mean: {np.mean(demo_action_norms):.6f}, Std: {np.std(demo_action_norms):.6f}")
    print_green(f"     动作范数 - Min: {np.min(demo_action_norms):.6f}, Max: {np.max(demo_action_norms):.6f}")
    print_green(f"     各维度均值: {np.mean(demo_actions, axis=0)}")
    print_green(f"     各维度标准差: {np.std(demo_actions, axis=0)}")
    
    # 计算差异
    norm_diff = abs(np.mean(bc_action_norms_mode) - np.mean(demo_action_norms))
    mean_diff = np.abs(np.mean(bc_actions_mode_np, axis=0) - np.mean(demo_actions, axis=0))
    
    print_green(f"   [差异]")
    print_green(f"     动作范数差异: {norm_diff:.6f}")
    print_green(f"     各维度均值差异: {mean_diff}")
    
    # 4. BC分布分析（通过采样）
    print_green(f"\n[4. BC分布分析（通过采样）]")
    
    # 计算每个观察的采样动作的均值和标准差
    bc_actions_samples_mean = np.mean(bc_actions_samples_np, axis=0)  # (batch_size, action_dim)
    bc_actions_samples_std = np.std(bc_actions_samples_np, axis=0)  # (batch_size, action_dim)
    
    print_green(f"   [采样动作统计]")
    print_green(f"     采样动作均值 - Mean: {np.mean(bc_actions_samples_mean, axis=0)}")
    print_green(f"     采样动作均值 - Std: {np.std(bc_actions_samples_mean, axis=0)}")
    print_green(f"     采样动作标准差 - Mean: {np.mean(bc_actions_samples_std, axis=0)}")
    print_green(f"     采样动作标准差 - Std: {np.std(bc_actions_samples_std, axis=0)}")
    
    # 对比mode和采样均值
    mode_vs_sample_mean_diff = np.abs(bc_actions_mode_np - bc_actions_samples_mean)
    print_green(f"   [Mode vs 采样均值差异]")
    print_green(f"     差异 - Mean: {np.mean(mode_vs_sample_mean_diff, axis=0)}")
    print_green(f"     差异 - Max: {np.max(mode_vs_sample_mean_diff, axis=0)}")
    
    # 5. 诊断结论
    print_green(f"\n[5. 诊断结论]")
    
    # 判断BC学习是否有效
    is_valid = True
    issues = []
    
    if mse_mean > 0.1:
        is_valid = False
        issues.append(f"MSE Loss过大 ({mse_mean:.6f} > 0.1)")
    
    if log_prob_mean < -10:
        is_valid = False
        issues.append(f"Log Probability过小 ({log_prob_mean:.6f} < -10)")
    
    if norm_diff > 0.1:
        is_valid = False
        issues.append(f"动作范数差异过大 ({norm_diff:.6f} > 0.1)")
    
    if np.mean(bc_action_norms_mode) < 0.05:
        is_valid = False
        issues.append(f"BC输出动作范数过小 ({np.mean(bc_action_norms_mode):.6f} < 0.05)")
    
    if is_valid:
        print_green(f"   ✅ BC策略学习有效")
        print_green(f"      - MSE Loss: {mse_mean:.6f} (合理)")
        print_green(f"      - Log Prob: {log_prob_mean:.6f} (合理)")
        print_green(f"      - 动作范数差异: {norm_diff:.6f} (合理)")
    else:
        print_red(f"   ❌ BC策略学习存在问题:")
        for issue in issues:
            print_red(f"      - {issue}")
        print_yellow(f"   [建议]")
        print_yellow(f"      1. 检查演示数据质量（动作值是否合理）")
        print_yellow(f"      2. 检查BC训练过程（loss是否收敛）")
        print_yellow(f"      3. 尝试增加训练步数或调整学习率")
    
    print_green("=" * 80)


if __name__ == "__main__":
    app.run(main)
