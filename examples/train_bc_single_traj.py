#!/usr/bin/env python3
"""
从演示数据中提取单条轨迹进行BC训练和验证

使用方法:
    python examples/train_bc_single_traj.py \
        --exp_name=gear_assembly \
        --demo_path=./demo_data/gear_assembly_5_demos_2026-02-04_09-01-56.pkl \
        --traj_index=0 \
        --bc_checkpoint_path=./checkpoints/bc_single_traj_test \
        --train_steps=5000 \
        --eval_n_trajs=3
"""

import glob
import time
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
import os
import pickle as pkl
import sys
from datetime import datetime
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics

from serl_launcher.agents.continuous.bc import BCAgent
from serl_launcher.utils.launcher import (
    make_bc_agent,
    make_wandb_logger,
)
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore
from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("bc_checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_integer("eval_n_trajs", 3, "Number of trajectories to evaluate.")
flags.DEFINE_integer("train_steps", 5_000, "Number of pretraining steps.")
flags.DEFINE_multi_string("demo_path", None, "Path to the demo data. If not set, uses demo_data/*.pkl")
flags.DEFINE_integer("traj_index", 0, "Index of trajectory to use for training (0-based).")
flags.DEFINE_boolean("use_sim", False, "Use Isaac Sim simulation environment for evaluation.")
flags.DEFINE_string("isaac_server_url", None, "Isaac Sim server URL (e.g., http://192.168.1.100:5001/).")
flags.DEFINE_boolean("debug", False, "Debug mode.")
flags.DEFINE_boolean("save_reference_traj", True, "Save reference trajectory for comparison.")
flags.DEFINE_integer("filter_max_consecutive", 10, "Maximum consecutive zero actions (first 6 dims) before filtering. Set to 0 to disable filtering.")
flags.DEFINE_boolean("enable_filtering", True, "Enable consecutive zero action filtering.")
flags.DEFINE_boolean("detailed_diagnosis", False, "Enable detailed diagnosis mode: print observations, actions, and comparisons at each step.")
flags.DEFINE_integer("diagnosis_print_freq", 10, "Print frequency for detailed diagnosis (every N steps).")
flags.DEFINE_string("log_file", None, "Path to log file. If not set, logs will be written to stdout and a default log file.")


devices = jax.local_devices()
num_devices = len(devices)
if num_devices == 1:
    sharding = jax.sharding.SingleDeviceSharding(devices[0])
else:
    from jax.sharding import Mesh, NamedSharding, PartitionSpec
    mesh = Mesh(devices, axis_names=('devices',))
    sharding = NamedSharding(mesh, PartitionSpec('devices'))


# 全局日志文件对象
log_file_obj = None

def setup_logging(log_file_path=None):
    """设置日志文件"""
    global log_file_obj
    
    if log_file_path is None:
        # 生成默认日志文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = f"bc_single_traj_{timestamp}.log"
    
    log_file_obj = open(log_file_path, 'w', encoding='utf-8')
    print(f"[LOG] 日志文件: {log_file_path}")
    return log_file_path

def log_print(*args, **kwargs):
    """同时打印到终端和日志文件"""
    message = ' '.join(str(arg) for arg in args)
    print(*args, **kwargs)
    if log_file_obj is not None:
        # 移除ANSI颜色代码
        clean_message = message
        import re
        clean_message = re.sub(r'\033\[[0-9;]*m', '', clean_message)
        log_file_obj.write(clean_message + '\n')
        log_file_obj.flush()

def print_green(x):
    message = "\033[92m {}\033[00m".format(x)
    log_print(message)

def print_yellow(x):
    message = "\033[93m {}\033[00m".format(x)
    log_print(message)


def filter_consecutive_zero_actions(transitions, max_consecutive=10):
    """
    过滤连续零动作（前6维都为0），保留间隔零动作
    
    Args:
        transitions: 演示数据列表
        max_consecutive: 连续零动作的最大数量，超过此数量只保留1个
    
    Returns:
        filtered: 过滤后的transitions列表
        zero_count: 零动作总数（前6维都为0）
        filtered_count: 被过滤掉的零动作数量
        stats: 统计信息字典
    """
    filtered = []
    consecutive_zeros = 0
    zero_count = 0
    filtered_count = 0
    non_zero_count = 0
    action_norms = []
    
    for transition in transitions:
        action = transition.get('actions', np.array([0]))
        action_arr = np.array(action)
        
        # 检查前6维是否都为0（使用小的阈值）
        if len(action_arr) >= 6:
            first_6d = action_arr[:6]
            is_zero_action = np.all(np.abs(first_6d) < 1e-6)
            action_norm = np.linalg.norm(first_6d)
        else:
            is_zero_action = np.linalg.norm(action_arr) < 1e-6
            action_norm = np.linalg.norm(action_arr)
        
        action_norms.append(action_norm)
        
        if is_zero_action:  # 前6维都为0
            zero_count += 1
            consecutive_zeros += 1
            if consecutive_zeros <= 1:  # 保留第一个零动作
                filtered.append(transition)
            elif consecutive_zeros > max_consecutive:
                # 如果连续超过max_consecutive个，再保留一个
                filtered.append(transition)
                consecutive_zeros = 0
            else:
                # 过滤掉中间的零动作
                filtered_count += 1
        else:  # 非零动作
            non_zero_count += 1
            consecutive_zeros = 0
            filtered.append(transition)
    
    stats = {
        "total": len(transitions),
        "zero_count": zero_count,
        "non_zero_count": non_zero_count,
        "filtered_count": filtered_count,
        "filtered": len(filtered),
        "zero_ratio_before": zero_count / len(transitions) if len(transitions) > 0 else 0,
        "zero_ratio_after": (zero_count - filtered_count) / len(filtered) if len(filtered) > 0 else 0,
        "action_norm_mean": np.mean(action_norms) if action_norms else 0,
        "action_norm_std": np.std(action_norms) if action_norms else 0,
    }
    
    return filtered, zero_count, filtered_count, stats


def find_successful_trajectories(transitions):
    """
    从transitions中找到所有成功轨迹
    
    Returns:
        list: 成功轨迹列表，每个轨迹是一个transition列表
    """
    trajectories = []
    current_traj = []
    
    for i, transition in enumerate(transitions):
        current_traj.append(transition)
        
        # 检查是否轨迹结束
        done = transition.get('dones', False)
        if done:
            # 检查是否成功
            reward = transition.get('rewards', 0.0)
            info = transition.get('infos', {})
            succeed = False
            if isinstance(info, dict):
                succeed = info.get('succeed', False)
            
            # 如果成功（reward > 0.5 或 succeed == True），保存轨迹
            if reward > 0.5 or succeed:
                trajectories.append(current_traj.copy())
                print_green(f"[INFO] 找到成功轨迹 #{len(trajectories)}: {len(current_traj)} 步, reward={reward}, succeed={succeed}")
            
            current_traj = []
    
    # 如果最后一个轨迹没有done标记，也检查一下
    if len(current_traj) > 0:
        last_transition = current_traj[-1]
        reward = last_transition.get('rewards', 0.0)
        info = last_transition.get('infos', {})
        succeed = False
        if isinstance(info, dict):
            succeed = info.get('succeed', False)
        if reward > 0.5 or succeed:
            trajectories.append(current_traj.copy())
            print_green(f"[INFO] 找到成功轨迹 #{len(trajectories)} (最后一条): {len(current_traj)} 步, reward={reward}, succeed={succeed}")
    
    return trajectories


def extract_initial_state(trajectory):
    """从轨迹中提取初始状态"""
    if len(trajectory) > 0:
        return trajectory[0]['observations']
    return None


def align_initial_state(env, demo_initial_obs, max_attempts=5, tolerance=0.01):
    """
    对齐环境的初始状态与演示数据的初始状态
    
    Args:
        env: 环境对象
        demo_initial_obs: 演示数据的初始观察
        max_attempts: 最大尝试次数
        tolerance: 位置对齐的容差（米）
    
    Returns:
        bool: 是否成功对齐
        float: 对齐后的状态差异
    """
    if demo_initial_obs is None or 'state' not in demo_initial_obs:
        return False, float('inf')
    
    demo_state = demo_initial_obs['state']
    if 'tcp_pose' not in demo_state:
        return False, float('inf')
    
    demo_tcp_pose = np.array(demo_state['tcp_pose'])
    demo_gripper_pose = demo_state.get('gripper_pose', None)
    
    # 获取当前环境状态
    if hasattr(env, '_get_obs'):
        current_obs = env._get_obs()
    else:
        # 如果环境不支持_get_obs，尝试通过reset获取
        current_obs, _ = env.reset()
    current_state = current_obs.get('state', {})
    current_tcp_pose = np.array(current_state.get('tcp_pose', np.zeros(7)))
    
    # 计算位置差异
    pos_diff = np.linalg.norm(demo_tcp_pose[:3] - current_tcp_pose[:3])
    
    log_print(f"[初始状态对齐] 演示初始TCP位置: {np.round(demo_tcp_pose[:3], 4)}")
    log_print(f"[初始状态对齐] 环境当前TCP位置: {np.round(current_tcp_pose[:3], 4)}")
    log_print(f"[初始状态对齐] 位置差异: {pos_diff:.6f} m")
    
    if pos_diff < tolerance:
        log_print(f"[初始状态对齐] ✓ 初始状态已对齐 (差异: {pos_diff:.6f} < {tolerance})")
        return True, pos_diff
    
    # 尝试移动到演示初始位置
    log_print(f"[初始状态对齐] 尝试对齐初始状态 (差异: {pos_diff:.6f} > {tolerance})...")
    
    for attempt in range(max_attempts):
        try:
            # 如果环境支持interpolate_move，使用它
            if hasattr(env, 'interpolate_move'):
                # 构建目标位姿（位置 + 四元数）
                target_pose = demo_tcp_pose.copy()
                env.interpolate_move(target_pose, timeout=2.0)
                
                # 等待物理收敛
                time.sleep(0.5)
                
                # 重新获取状态
                if hasattr(env, '_get_obs'):
                    current_obs = env._get_obs()
                else:
                    current_obs, _ = env.reset()
                current_state = current_obs.get('state', {})
                current_tcp_pose = np.array(current_state.get('tcp_pose', np.zeros(7)))
                
                # 重新计算差异
                pos_diff = np.linalg.norm(demo_tcp_pose[:3] - current_tcp_pose[:3])
                log_print(f"[初始状态对齐] 尝试 {attempt + 1}/{max_attempts}: 位置差异 = {pos_diff:.6f} m")
                
                if pos_diff < tolerance:
                    log_print(f"[初始状态对齐] ✓ 对齐成功 (差异: {pos_diff:.6f} < {tolerance})")
                    return True, pos_diff
            else:
                # 如果环境不支持interpolate_move，尝试通过step移动
                # 计算需要的动作（位置差）
                pos_error = demo_tcp_pose[:3] - current_tcp_pose[:3]
                action = np.zeros(7)
                action[:3] = pos_error  # 只移动位置
                if demo_gripper_pose is not None:
                    action[6] = demo_gripper_pose
                
                # 执行动作
                for _ in range(10):  # 执行10步
                    current_obs, _, _, _, _ = env.step(action)
                    current_state = current_obs.get('state', {})
                    current_tcp_pose = np.array(current_state.get('tcp_pose', np.zeros(7)))
                    pos_error = demo_tcp_pose[:3] - current_tcp_pose[:3]
                    if np.linalg.norm(pos_error) < tolerance:
                        break
                    action[:3] = pos_error
                
                pos_diff = np.linalg.norm(demo_tcp_pose[:3] - current_tcp_pose[:3])
                log_print(f"[初始状态对齐] 尝试 {attempt + 1}/{max_attempts}: 位置差异 = {pos_diff:.6f} m")
                
                if pos_diff < tolerance:
                    log_print(f"[初始状态对齐] ✓ 对齐成功 (差异: {pos_diff:.6f} < {tolerance})")
                    return True, pos_diff
                
        except Exception as e:
            log_print(f"[初始状态对齐] 尝试 {attempt + 1} 失败: {e}")
            continue
    
    log_print(f"[初始状态对齐] ⚠ 对齐失败，最终位置差异: {pos_diff:.6f} m (容差: {tolerance})")
    return False, pos_diff


def compare_observations(obs_bc, obs_ref, step):
    """
    对比BC评估时的观察和参考轨迹的观察
    
    Args:
        obs_bc: BC评估时的观察
        obs_ref: 参考轨迹的观察
        step: 当前步数
    
    Returns:
        dict: 观察差异统计
    """
    differences = {}
    
    # 对比state（本体感觉状态）
    if 'state' in obs_bc and 'state' in obs_ref:
        state_bc = np.array(obs_bc['state'])
        state_ref = np.array(obs_ref['state'])
        
        if state_bc.shape == state_ref.shape:
            state_diff = np.abs(state_bc - state_ref)
            state_norm_diff = np.linalg.norm(state_bc - state_ref)
            differences['state'] = {
                'norm_diff': float(state_norm_diff),
                'mean_diff': float(np.mean(state_diff)),
                'max_diff': float(np.max(state_diff)),
                'mean_bc': float(np.mean(state_bc)),
                'mean_ref': float(np.mean(state_ref)),
                'bc_state': state_bc.copy(),
                'ref_state': state_ref.copy(),
            }
        else:
            differences['state'] = {'error': f'Shape mismatch: BC {state_bc.shape} vs Ref {state_ref.shape}'}
    
    # 对比图像（如果存在）
    if 'images' in obs_bc and 'images' in obs_ref:
        image_diffs = {}
        for key in obs_bc['images'].keys():
            if key in obs_ref['images']:
                img_bc = np.array(obs_bc['images'][key])
                img_ref = np.array(obs_ref['images'][key])
                
                if img_bc.shape == img_ref.shape:
                    # 计算图像差异（像素值差异）
                    img_diff = np.abs(img_bc.astype(float) - img_ref.astype(float))
                    image_diffs[key] = {
                        'mean_diff': float(np.mean(img_diff)),
                        'max_diff': float(np.max(img_diff)),
                        'mse': float(np.mean(img_diff ** 2)),
                    }
                else:
                    image_diffs[key] = {'error': f'Shape mismatch: BC {img_bc.shape} vs Ref {img_ref.shape}'}
        differences['images'] = image_diffs
    
    return differences


def find_most_similar_demo_step(current_obs, reference_trajectory, search_range=None):
    """
    在演示数据中查找与当前观察最相似的步骤
    
    Args:
        current_obs: 当前观察
        reference_trajectory: 参考轨迹列表
        search_range: 搜索范围 (start, end)，如果为None则搜索全部
    
    Returns:
        dict: {
            'best_match_idx': 最相似步骤的索引,
            'similarity_score': 相似度分数（越小越相似）,
            'state_diff': state差异,
            'action_at_match': 该步骤对应的动作
        }
    """
    if reference_trajectory is None or len(reference_trajectory) == 0:
        return None
    
    if 'state' not in current_obs:
        return None
    
    current_state = np.array(current_obs['state'])
    
    start_idx = search_range[0] if search_range else 0
    end_idx = search_range[1] if search_range else len(reference_trajectory)
    
    best_match_idx = None
    best_similarity = float('inf')
    
    for i in range(start_idx, min(end_idx, len(reference_trajectory))):
        ref_transition = reference_trajectory[i]
        ref_obs = ref_transition.get('observations', {})
        
        if 'state' in ref_obs:
            ref_state = np.array(ref_obs['state'])
            if ref_state.shape == current_state.shape:
                # 计算state的L2距离作为相似度
                state_diff = np.linalg.norm(current_state - ref_state)
                if state_diff < best_similarity:
                    best_similarity = state_diff
                    best_match_idx = i
    
    if best_match_idx is not None:
        ref_transition = reference_trajectory[best_match_idx]
        ref_action = np.array(ref_transition.get('actions', np.zeros(7)))
        
        return {
            'best_match_idx': best_match_idx,
            'similarity_score': best_similarity,
            'state_diff': best_similarity,
            'action_at_match': ref_action,
        }
    
    return None


def eval(
    env,
    bc_agent: BCAgent,
    sampling_rng,
    reference_trajectory=None,
):
    """
    Evaluation loop for BC policy.
    如果提供了reference_trajectory，会比较BC策略输出和参考轨迹的差异。
    """
    success_counter = 0
    time_list = []
    action_errors = []
    observation_differences = []  # 存储观察差异
    
    # 提取演示数据的初始状态
    demo_initial_obs = None
    if reference_trajectory is not None and len(reference_trajectory) > 0:
        demo_initial_obs = extract_initial_state(reference_trajectory)
        log_print(f"[BC EVAL] 已提取演示数据初始状态")
    
    for episode in range(FLAGS.eval_n_trajs):
        print_green(f"[BC EVAL] Starting episode {episode + 1}/{FLAGS.eval_n_trajs}")
        
        # 重置环境
        obs, info = env.reset()
        
        # 如果提供了参考轨迹，对齐初始状态
        if demo_initial_obs is not None and episode == 0:
            print_yellow("[BC EVAL] 对齐初始状态与演示数据...")
            aligned, initial_state_diff = align_initial_state(env, demo_initial_obs, max_attempts=5, tolerance=0.01)
            
            if aligned:
                print_green(f"[BC EVAL] ✓ 初始状态对齐成功 (差异: {initial_state_diff:.6f} m)")
                # 重新获取对齐后的观察
                if hasattr(env, '_get_obs'):
                    obs = env._get_obs()
                else:
                    # 如果环境不支持_get_obs，使用当前obs
                    pass
            else:
                print_yellow(f"[BC EVAL] ⚠ 初始状态对齐失败 (差异: {initial_state_diff:.6f} m)，继续执行...")
            
            # 对比对齐后的状态
            if 'state' in obs and 'state' in demo_initial_obs:
                current_state = obs['state']
                demo_state = demo_initial_obs['state']
                
                if 'tcp_pose' in current_state and 'tcp_pose' in demo_state:
                    current_tcp = np.array(current_state['tcp_pose'])
                    demo_tcp = np.array(demo_state['tcp_pose'])
                    pos_diff = np.linalg.norm(current_tcp[:3] - demo_tcp[:3])
                    log_print(f"[BC EVAL] 对齐后状态对比:")
                    log_print(f"  当前TCP位置: {np.round(current_tcp[:3], 4)}")
                    log_print(f"  演示TCP位置: {np.round(demo_tcp[:3], 4)}")
                    log_print(f"  位置差异: {pos_diff:.6f} m")
        
        done = False
        start_time = time.time()
        step_count = 0
        episode_actions = []
        episode_states = []
        
        while not done:
            step_count += 1
            if step_count % 100 == 0:
                log_print(f"[BC EVAL] Episode {episode + 1}, Step {step_count}")
            
            rng, key = jax.random.split(sampling_rng)
            sampling_rng = rng

            try:
                # 如果提供了参考轨迹，先对比观察
                ref_obs = None
                obs_diff = None
                if reference_trajectory is not None and episode == 0 and step_count - 1 < len(reference_trajectory):
                    ref_obs = reference_trajectory[step_count - 1]['observations']
                    obs_diff = compare_observations(obs, ref_obs, step_count - 1)
                    observation_differences.append(obs_diff)
                
                # 使用argmax=True来使用分布的mode（均值）而不是采样
                actions = bc_agent.sample_actions(
                    observations=jax.device_put(obs), 
                    seed=key,
                    argmax=True
                )
                actions = np.asarray(jax.device_get(actions))
                episode_actions.append(actions.copy())
                
                # 详细诊断模式：打印每一步的观察和动作
                if FLAGS.detailed_diagnosis and episode == 0 and step_count % FLAGS.diagnosis_print_freq == 0:
                    log_print(f"\n{'='*80}")
                    log_print(f"[诊断] Step {step_count}")
                    log_print(f"{'='*80}")
                    
                    # 调试信息：检查参考轨迹是否可用
                    if reference_trajectory is None:
                        log_print(f"  [警告] reference_trajectory 为 None，无法进行对比")
                    elif step_count - 1 >= len(reference_trajectory):
                        log_print(f"  [警告] 当前步骤 {step_count - 1} 超出参考轨迹长度 {len(reference_trajectory)}，无法进行相同步骤对比")
                    else:
                        log_print(f"  [调试] 参考轨迹可用，长度: {len(reference_trajectory)}, 当前步骤: {step_count - 1}")
                    
                    # 1. 打印当前观察（State）
                    if 'state' in obs:
                        state = obs['state']
                        state_arr = np.array(state) if not isinstance(state, np.ndarray) else state
                        if len(state_arr) > 0:
                            log_print(f"  [观察] State形状: {state_arr.shape}")
                            log_print(f"  [观察] State (前10维): {np.round(state_arr[:min(10, len(state_arr))], 4)}")
                            if len(state_arr) > 10:
                                log_print(f"  [观察] State (后10维): {np.round(state_arr[-10:], 4)}")
                            log_print(f"  [观察] State统计: min={np.min(state_arr):.4f}, max={np.max(state_arr):.4f}, mean={np.mean(state_arr):.4f}")
                    
                    # 打印图像信息（如果存在）
                    if 'images' in obs:
                        log_print(f"  [观察] 图像键: {list(obs['images'].keys())}")
                        for img_key, img in obs['images'].items():
                            img_arr = np.array(img)
                            log_print(f"  [观察] {img_key}: shape={img_arr.shape}, dtype={img_arr.dtype}, "
                                  f"min={np.min(img_arr)}, max={np.max(img_arr)}, mean={np.mean(img_arr):.2f}")
                    
                    # 2. 打印策略输出的动作
                    log_print(f"  [动作] BC策略输出: {np.round(actions, 4)}")
                    log_print(f"  [动作] 动作范数: {np.linalg.norm(actions):.4f}")
                    
                    # 3. 如果提供了参考轨迹，对比相同步骤
                    if reference_trajectory is not None:
                        if step_count - 1 < len(reference_trajectory):
                            ref_action = np.array(reference_trajectory[step_count - 1]['actions'])
                            action_error = np.linalg.norm(actions - ref_action)
                            log_print(f"\n  [对比-相同步骤] Step {step_count - 1} (演示数据):")
                            if ref_obs is not None:
                                ref_state = ref_obs.get('state', None)
                                if ref_state is not None:
                                    ref_state_arr = np.array(ref_state) if not isinstance(ref_state, np.ndarray) else ref_state
                                    if len(ref_state_arr) > 0:
                                        log_print(f"    [观察] 演示State (前10维): {np.round(ref_state_arr[:min(10, len(ref_state_arr))], 4)}")
                            log_print(f"    [动作] 演示动作: {np.round(ref_action, 4)}")
                            log_print(f"    [动作] 演示动作范数: {np.linalg.norm(ref_action):.4f}")
                            log_print(f"    [差异] 动作误差: {action_error:.6f}")
                            log_print(f"    [差异] 动作差值: {np.round(actions - ref_action, 4)}")
                            log_print(f"    [差异] 动作差值范数: {np.linalg.norm(actions - ref_action):.6f}")
                            
                            if obs_diff and 'state' in obs_diff and 'error' not in obs_diff['state']:
                                log_print(f"    [差异] State差异范数: {obs_diff['state']['norm_diff']:.6f}")
                                log_print(f"    [差异] State差异均值: {obs_diff['state']['mean_diff']:.6f}")
                                log_print(f"    [差异] State差异最大: {obs_diff['state']['max_diff']:.6f}")
                        else:
                            log_print(f"\n  [对比-相同步骤] 当前步骤 {step_count - 1} 超出参考轨迹长度 {len(reference_trajectory)}，无法对比")
                    
                    # 4. 查找最相似的演示步骤
                    if reference_trajectory is not None:
                        # 搜索范围：当前步骤前后50步
                        search_start = max(0, step_count - 1 - 50)
                        search_end = min(len(reference_trajectory), step_count - 1 + 50)
                        similar_match = find_most_similar_demo_step(obs, reference_trajectory, (search_start, search_end))
                        
                        if similar_match:
                            log_print(f"\n  [对比-最相似步骤] Step {similar_match['best_match_idx']} (演示数据中最相似):")
                            log_print(f"    [相似度] State差异: {similar_match['similarity_score']:.6f}")
                            log_print(f"    [动作] 该步骤的演示动作: {np.round(similar_match['action_at_match'], 4)}")
                            
                            # 对比BC动作和相似步骤的演示动作
                            similar_action_error = np.linalg.norm(actions - similar_match['action_at_match'])
                            log_print(f"    [差异] BC动作 vs 相似步骤动作误差: {similar_action_error:.6f}")
                            
                            # 如果最相似步骤不是当前步骤，说明观察已经偏移
                            if similar_match['best_match_idx'] != step_count - 1:
                                step_offset = similar_match['best_match_idx'] - (step_count - 1)
                                print_yellow(f"    ⚠ 观察已偏移: 当前Step {step_count - 1} 最接近演示Step {similar_match['best_match_idx']} (偏移{step_offset}步)")
                    
                    log_print(f"{'='*80}\n")
                
                # 如果提供了参考轨迹，计算动作误差（非详细模式也计算）
                if reference_trajectory is not None and step_count - 1 < len(reference_trajectory):
                    ref_action = np.array(reference_trajectory[step_count - 1]['actions'])
                    action_error = np.linalg.norm(actions - ref_action)
                    action_errors.append(action_error)
                    
                    # 每50步打印一次动作误差和观察差异（非详细模式）
                    if not FLAGS.detailed_diagnosis and step_count % 50 == 0:
                        log_print(f"  [BC EVAL] Step {step_count}, Action error: {action_error:.6f}")
                        if obs_diff and 'state' in obs_diff and 'error' not in obs_diff['state']:
                            log_print(f"  [BC EVAL] Step {step_count}, State diff norm: {obs_diff['state']['norm_diff']:.6f}")
                
            except Exception as e:
                print_yellow(f"[BC EVAL] Error sampling actions: {e}")
                import traceback
                traceback.print_exc()
                actions = np.zeros(env.action_space.shape)
            
            obs, rew, done, truncated, info = env.step(actions)
            episode_states.append(obs)
            
            if done or truncated:
                break
        
        elapsed_time = time.time() - start_time
        time_list.append(elapsed_time)
        
        # 检查是否成功
        succeed = info.get("succeed", False) if isinstance(info, dict) else False
        if succeed or rew > 0.5:
            success_counter += 1
            print_green(f"[BC EVAL] Episode {episode + 1} SUCCESS (steps: {step_count}, time: {elapsed_time:.2f}s)")
        else:
            print_yellow(f"[BC EVAL] Episode {episode + 1} FAILED (steps: {step_count}, time: {elapsed_time:.2f}s)")
        
        # 如果提供了参考轨迹，比较轨迹长度和统计信息
        if reference_trajectory is not None and episode == 0:
            ref_length = len(reference_trajectory)
            log_print(f"\n[BC EVAL] ========== 参考轨迹对比 (Episode 1) ==========")
            log_print(f"  参考轨迹长度: {ref_length}, BC策略轨迹长度: {step_count}")
            
            if len(action_errors) > 0:
                log_print(f"\n  动作误差统计:")
                log_print(f"    平均: {np.mean(action_errors):.6f}")
                log_print(f"    最大: {np.max(action_errors):.6f}")
                log_print(f"    最小: {np.min(action_errors):.6f}")
                log_print(f"    标准差: {np.std(action_errors):.6f}")
                log_print(f"    中位数: {np.median(action_errors):.6f}")
            
            # 观察差异统计
            if len(observation_differences) > 0:
                log_print(f"\n  观察差异统计:")
                
                # State差异统计
                state_norms = [d['state']['norm_diff'] for d in observation_differences if 'state' in d and 'error' not in d['state']]
                if len(state_norms) > 0:
                    log_print(f"    State差异范数:")
                    log_print(f"      平均: {np.mean(state_norms):.6f}")
                    log_print(f"      最大: {np.max(state_norms):.6f}")
                    log_print(f"      最小: {np.min(state_norms):.6f}")
                    log_print(f"      标准差: {np.std(state_norms):.6f}")
                
                # 图像差异统计
                if 'images' in observation_differences[0]:
                    for img_key in observation_differences[0]['images'].keys():
                        img_mses = [d['images'][img_key]['mse'] for d in observation_differences 
                                   if 'images' in d and img_key in d['images'] and 'error' not in d['images'][img_key]]
                        if len(img_mses) > 0:
                            log_print(f"    Image {img_key} MSE:")
                            log_print(f"      平均: {np.mean(img_mses):.6f}")
                            log_print(f"      最大: {np.max(img_mses):.6f}")
                            log_print(f"      最小: {np.min(img_mses):.6f}")
            
            # 诊断建议
            if len(action_errors) > 0:
                mean_action_error = np.mean(action_errors)
                print(f"\n  [诊断] 动作误差分析:")
                if mean_action_error > 0.5:
                    print_yellow(f"    ⚠ 动作误差很大 ({mean_action_error:.6f})，可能原因：")
                    print_yellow(f"      1. 观察输入不匹配（环境状态与演示时不同）")
                    print_yellow(f"      2. BC模型未充分学习状态-动作映射")
                    print_yellow(f"      3. 环境初始状态不一致")
                    if len(observation_differences) > 0 and 'state' in observation_differences[0]:
                        state_norms = [d['state']['norm_diff'] for d in observation_differences if 'state' in d and 'error' not in d['state']]
                        if len(state_norms) > 0 and np.mean(state_norms) > 0.1:
                            print_yellow(f"      4. State差异较大 ({np.mean(state_norms):.6f})，说明环境状态不一致")
                elif mean_action_error > 0.1:
                    print_yellow(f"    ⚠ 动作误差中等 ({mean_action_error:.6f})，在可接受范围内")
                else:
                    print_green(f"    ✓ 动作误差较小 ({mean_action_error:.6f})，BC策略与参考轨迹接近")
    
    print_green(f"\n[BC EVAL] ========== 最终评估结果 ==========")
    print_green(f"  成功率: {success_counter}/{FLAGS.eval_n_trajs} ({success_counter/FLAGS.eval_n_trajs*100:.2f}%)")
    if len(time_list) > 0:
        print_green(f"  平均时间: {np.mean(time_list):.2f}s")
    
    # 如果所有episode都有动作误差数据，显示总体统计
    if len(action_errors) > 0:
        print_green(f"\n  总体动作误差统计 (所有episodes):")
        print_green(f"    平均: {np.mean(action_errors):.6f}")
        print_green(f"    最大: {np.max(action_errors):.6f}")
        print_green(f"    最小: {np.min(action_errors):.6f}")
        print_green(f"    标准差: {np.std(action_errors):.6f}")
        print_green(f"    中位数: {np.median(action_errors):.6f}")
    
    return success_counter, action_errors


def main(_):
    # 设置日志文件
    log_file_path = setup_logging(FLAGS.log_file)
    
    try:
        config = CONFIG_MAPPING[FLAGS.exp_name]()
        
        assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
        
        # 设置随机种子
        np.random.seed(FLAGS.seed)
        rng = jax.random.PRNGKey(FLAGS.seed)
        
        # 判断运行模式：
        # - 纯评估模式：train_steps == 0 且 eval_n_trajs > 0（只评估，不训练）
        # - 训练+评估模式：train_steps > 0 且 eval_n_trajs > 0（训练后评估）
        # - 纯训练模式：train_steps > 0 且 eval_n_trajs == 0（只训练，不评估）
        eval_only_mode = FLAGS.train_steps == 0 and FLAGS.eval_n_trajs > 0
        eval_mode = FLAGS.eval_n_trajs > 0
        
        if eval_mode:
            # 评估模式：需要实际环境来执行策略
            # 如果提供了 isaac_server_url，使用 Isaac Sim 仿真环境
            if FLAGS.isaac_server_url is not None:
                use_fake_env = True  # 使用 Isaac Sim 仿真环境
            else:
                use_fake_env = FLAGS.use_sim  # 根据 --use_sim 参数决定
            skip_server_connection = False  # 评估时需要连接环境
        else:
            # 训练模式：只需要环境空间定义，不需要实际连接
            use_fake_env = True  # 使用fake_env获取空间定义
            skip_server_connection = True  # 训练时跳过服务器连接，不需要实际环境
        
        # 使用 config.get_environment() 创建环境
        env = config.get_environment(
            fake_env=use_fake_env,
            save_video=False,  # 单轨迹训练不需要保存视频
            classifier=not use_fake_env,  # 评估时使用分类器（如果真实环境），训练时不用
            isaac_server_url=FLAGS.isaac_server_url,
            skip_server_connection=skip_server_connection,
        )
        env = RecordEpisodeStatistics(env)
        
        # 创建BC agent
        bc_agent = make_bc_agent(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
        )
        
        # replicate agent across devices
        bc_agent: BCAgent = jax.device_put(
            jax.tree.map(jnp.array, bc_agent), sharding
        )
        
        # ========== 评估模式：加载checkpoint和参考轨迹 ==========
        if eval_only_mode:
            assert FLAGS.bc_checkpoint_path is not None, "Must specify --bc_checkpoint_path for evaluation"
            
            # 加载checkpoint
            latest_ckpt = checkpoints.latest_checkpoint(os.path.abspath(FLAGS.bc_checkpoint_path))
            if latest_ckpt is None:
                raise ValueError(f"No checkpoint found in {FLAGS.bc_checkpoint_path}")
            
            print_green(f"[BC EVAL] Loading checkpoint from {latest_ckpt}")
            bc_ckpt = checkpoints.restore_checkpoint(
                latest_ckpt,
                bc_agent.state,
            )
            bc_agent = bc_agent.replace(state=bc_ckpt)
            
            # 尝试加载参考轨迹
            reference_trajectory = None
            ref_traj_path = os.path.join(FLAGS.bc_checkpoint_path, "reference_trajectory.pkl")
            if os.path.exists(ref_traj_path):
                print_green(f"[BC EVAL] Loading reference trajectory from {ref_traj_path}")
                with open(ref_traj_path, "rb") as f:
                    reference_trajectory = pkl.load(f)
                print_green(f"[BC EVAL] Reference trajectory loaded: {len(reference_trajectory)} transitions")
            else:
                print_yellow(f"[BC EVAL] Reference trajectory not found at {ref_traj_path}, will skip action error comparison")
            
            # 执行评估
            print_green(f"[BC EVAL] Starting evaluation with {FLAGS.eval_n_trajs} trajectories...")
            eval_rng = jax.random.PRNGKey(FLAGS.seed)
            eval_rng = jax.device_put(eval_rng, sharding)
            success_count, action_errors = eval(
                env,
                bc_agent,
                eval_rng,
                reference_trajectory=reference_trajectory,
            )
            
            print_green(f"\n[BC EVAL] 最终结果: {success_count}/{FLAGS.eval_n_trajs} 成功")
            if len(action_errors) > 0:
                print_green(f"[BC EVAL] 动作误差统计:")
                print_green(f"  平均: {np.mean(action_errors):.6f}")
                print_green(f"  最大: {np.max(action_errors):.6f}")
                print_green(f"  最小: {np.min(action_errors):.6f}")
                print_green(f"  标准差: {np.std(action_errors):.6f}")
            
            return
        
        # ========== 训练模式：加载演示数据并训练 ==========
        
        # 加载演示数据并提取单条轨迹
        if FLAGS.demo_path:
            demo_paths = FLAGS.demo_path
        else:
            demo_paths = glob.glob(os.path.join(os.getcwd(), "demo_data", "*.pkl"))
        
        assert len(demo_paths) > 0, f"No demo files found."
        
        print_green(f"[BC TRAINING] Loading demo data from {demo_paths[0]}...")
        with open(demo_paths[0], "rb") as f:
            transitions = pkl.load(f)
        
        # 找到所有成功轨迹
        trajectories = find_successful_trajectories(transitions)
        
        if len(trajectories) == 0:
            print_yellow("[ERROR] 未找到成功轨迹！")
            return
        
        if FLAGS.traj_index >= len(trajectories):
            print_yellow(f"[WARNING] 轨迹索引 {FLAGS.traj_index} 超出范围，使用索引 0")
            FLAGS.traj_index = 0
        
        # 选择要训练的轨迹
        selected_traj = trajectories[FLAGS.traj_index]
        original_traj_length = len(selected_traj)
        print_green(f"[BC TRAINING] 选择轨迹 {FLAGS.traj_index}，包含 {original_traj_length} 个transitions")
        
        # 统计原始数据（过滤前）
        print_green(f"\n[BC TRAINING] ========== 原始数据统计（过滤前）==========")
        original_actions = []
        original_zero_count = 0
        for transition in selected_traj:
            action = transition.get('actions', np.array([0]))
            action_arr = np.array(action)
            original_actions.append(action_arr)
            if len(action_arr) >= 6:
                first_6d = action_arr[:6]
                if np.all(np.abs(first_6d) < 1e-6):
                    original_zero_count += 1
        
        if len(original_actions) > 0:
            original_actions_np = np.array(original_actions)
            if original_actions_np.shape[1] >= 6:
                original_first_6d = original_actions_np[:, :6]
                original_action_norms = np.linalg.norm(original_first_6d, axis=-1)
            else:
                original_action_norms = np.linalg.norm(original_actions_np, axis=-1)
            
            print_green(f"  总transitions: {original_traj_length}")
            print_green(f"  前6维为零动作: {original_zero_count} ({original_zero_count/original_traj_length*100:.2f}%)")
            print_green(f"  前6维动作范数 - Mean: {np.mean(original_action_norms):.6f}, Std: {np.std(original_action_norms):.6f}")
            print_green(f"  前6维动作范数 - Min: {np.min(original_action_norms):.6f}, Max: {np.max(original_action_norms):.6f}")
            print_green(f"  各维度均值: {np.mean(original_actions_np, axis=0)}")
            print_green(f"  各维度标准差: {np.std(original_actions_np, axis=0)}")
            print_green(f"  各维度范围: [{np.min(original_actions_np, axis=0)}, {np.max(original_actions_np, axis=0)}]")
        
        # 应用过滤（如果启用）
        if FLAGS.enable_filtering and FLAGS.filter_max_consecutive > 0:
            print_green(f"[BC TRAINING] 应用过滤: 过滤连续零动作（前6维都为0，max_consecutive={FLAGS.filter_max_consecutive}）")
            filtered_traj, zero_count, filtered_count, stats = filter_consecutive_zero_actions(
                selected_traj, max_consecutive=FLAGS.filter_max_consecutive
            )
            
            print_green(f"\n[BC TRAINING] ========== 数据过滤统计 ==========")
            print_green(f"  过滤前: {stats['total']} transitions")
            print_green(f"  前6维为零动作: {stats['zero_count']} ({stats['zero_ratio_before']*100:.2f}%)")
            print_green(f"  过滤掉: {stats['filtered_count']} 零动作")
            print_green(f"  过滤后: {stats['filtered']} transitions")
            print_green(f"  零动作比例: {stats['zero_ratio_before']*100:.2f}% → {stats['zero_ratio_after']*100:.2f}%")
            print_green(f"  数据压缩率: {(1 - stats['filtered']/stats['total'])*100:.2f}%")
            print_green(f"  前6维动作范数 - Mean: {stats['action_norm_mean']:.6f}, Std: {stats['action_norm_std']:.6f}")
            
            selected_traj = filtered_traj
            print_green(f"[BC TRAINING] 过滤后轨迹长度: {len(selected_traj)} transitions")
            
            # 统计过滤后的数据（立即统计，不依赖replay buffer）
            print_green(f"\n[BC TRAINING] ========== 过滤后数据统计（立即验证）==========")
            filtered_actions = []
            filtered_zero_count = 0
            for transition in filtered_traj:
                action = transition.get('actions', np.array([0]))
                action_arr = np.array(action)
                filtered_actions.append(action_arr)
                if len(action_arr) >= 6:
                    first_6d = action_arr[:6]
                    if np.all(np.abs(first_6d) < 1e-6):
                        filtered_zero_count += 1
            
            if len(filtered_actions) > 0:
                filtered_actions_np = np.array(filtered_actions)
                if filtered_actions_np.shape[1] >= 6:
                    filtered_first_6d = filtered_actions_np[:, :6]
                    filtered_action_norms = np.linalg.norm(filtered_first_6d, axis=-1)
                else:
                    filtered_action_norms = np.linalg.norm(filtered_actions_np, axis=-1)
                
                print_green(f"  总transitions: {len(filtered_traj)}")
                print_green(f"  前6维为零动作: {filtered_zero_count} ({filtered_zero_count/len(filtered_traj)*100:.2f}%)")
                print_green(f"  前6维动作范数 - Mean: {np.mean(filtered_action_norms):.6f}, Std: {np.std(filtered_action_norms):.6f}")
                print_green(f"  前6维动作范数 - Min: {np.min(filtered_action_norms):.6f}, Max: {np.max(filtered_action_norms):.6f}")
                print_green(f"  各维度均值: {np.mean(filtered_actions_np, axis=0)}")
                print_green(f"  各维度标准差: {np.std(filtered_actions_np, axis=0)}")
                print_green(f"  各维度范围: [{np.min(filtered_actions_np, axis=0)}, {np.max(filtered_actions_np, axis=0)}]")
                
                # 对比过滤前后的变化
                print_green(f"\n[BC TRAINING] ========== 过滤效果对比 ==========")
                print_green(f"  数据量变化: {original_traj_length} → {len(filtered_traj)} (减少 {original_traj_length - len(filtered_traj)} 条, {((original_traj_length - len(filtered_traj))/original_traj_length*100):.2f}%)")
                print_green(f"  零动作数量变化: {original_zero_count} → {filtered_zero_count} (减少 {original_zero_count - filtered_zero_count} 条)")
                print_green(f"  零动作比例变化: {original_zero_count/original_traj_length*100:.2f}% → {filtered_zero_count/len(filtered_traj)*100:.2f}%")
                if np.mean(original_action_norms) > 0:
                    print_green(f"  动作范数均值变化: {np.mean(original_action_norms):.6f} → {np.mean(filtered_action_norms):.6f} (提升 {((np.mean(filtered_action_norms) - np.mean(original_action_norms))/np.mean(original_action_norms)*100):.2f}%)")
                else:
                    print_green(f"  动作范数均值变化: {np.mean(original_action_norms):.6f} → {np.mean(filtered_action_norms):.6f}")
                
                # 判断过滤是否成功
                if filtered_zero_count / len(filtered_traj) < original_zero_count / len(selected_traj) * 0.5:
                    print_green(f"  ✓ 过滤成功: 零动作比例显著降低")
                elif filtered_zero_count / len(filtered_traj) < original_zero_count / len(selected_traj):
                    print_yellow(f"  ⚠ 过滤部分成功: 零动作比例有所降低，但可能仍需更激进过滤")
                else:
                    print_yellow(f"  ✗ 过滤效果不明显: 零动作比例未明显降低")
        else:
            print_yellow(f"[BC TRAINING] 零动作过滤已禁用 (--enable_filtering=False 或 --filter_max_consecutive=0)")
            # 统计原始数据
            zero_count = 0
            for transition in selected_traj:
                action = transition.get('actions', np.array([0]))
                action_arr = np.array(action)
                if len(action_arr) >= 6:
                    first_6d = action_arr[:6]
                    if np.all(np.abs(first_6d) < 1e-6):
                        zero_count += 1
            print_yellow(f"[BC TRAINING] 原始数据中前6维为零动作: {zero_count}/{len(selected_traj)} ({zero_count/len(selected_traj)*100:.2f}%)")
        
        # 保存参考轨迹（用于后续比较）- 保存过滤后的轨迹
        reference_trajectory = None
        if FLAGS.save_reference_traj:
            reference_trajectory = selected_traj.copy()
            ref_traj_path = os.path.join(FLAGS.bc_checkpoint_path, "reference_trajectory.pkl")
            os.makedirs(FLAGS.bc_checkpoint_path, exist_ok=True)
            with open(ref_traj_path, "wb") as f:
                pkl.dump(reference_trajectory, f)
            print_green(f"[BC TRAINING] 参考轨迹已保存到: {ref_traj_path} (已过滤)")
        
        # 创建replay buffer并加载单条轨迹
        bc_replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
        )
        
        print_green(f"[BC TRAINING] 加载轨迹到replay buffer...")
        for transition in selected_traj:
            bc_replay_buffer.insert(transition)
        
        print_green(f"[BC TRAINING] Replay buffer大小: {len(bc_replay_buffer)}")
        
        # 统计加载到replay buffer后的数据质量（最终确认）
        if len(bc_replay_buffer) > 0:
            sample_size = min(1000, len(bc_replay_buffer))
            sample_batch = bc_replay_buffer.sample(sample_size)
            
            actions = sample_batch["actions"]
            actions_np = np.asarray(actions)
            
            # 检查前6维
            if actions_np.shape[1] >= 6:
                first_6d = actions_np[:, :6]
                action_norms = np.linalg.norm(first_6d, axis=-1)
                zero_action_count = np.sum(action_norms < 1e-6)
            else:
                action_norms = np.linalg.norm(actions_np, axis=-1)
                zero_action_count = np.sum(action_norms < 1e-6)
            
            print_green(f"\n[BC TRAINING] ========== Replay Buffer数据质量确认 ==========")
            print_green(f"  Replay buffer大小: {len(bc_replay_buffer)} transitions")
            print_green(f"  采样统计 (采样 {sample_size} 条):")
            print_green(f"    前6维为零动作: {zero_action_count} ({zero_action_count/len(actions_np)*100:.2f}%)")
            print_green(f"    前6维动作范数 - Mean: {np.mean(action_norms):.6f}, Std: {np.std(action_norms):.6f}")
            print_green(f"    前6维动作范数 - Min: {np.min(action_norms):.6f}, Max: {np.max(action_norms):.6f}")
            print_green(f"    各维度均值: {np.mean(actions_np, axis=0)}")
            print_green(f"    各维度标准差: {np.std(actions_np, axis=0)}")
            
            # 检查数据质量
            if zero_action_count / len(actions_np) > 0.3:
                print_yellow(f"  [WARNING] Replay buffer中仍有 {zero_action_count/len(actions_np)*100:.2f}% 零动作，可能需要更激进的过滤")
            elif zero_action_count / len(actions_np) > 0.1:
                print_yellow(f"  [INFO] Replay buffer中有 {zero_action_count/len(actions_np)*100:.2f}% 零动作，在可接受范围内")
            else:
                print_green(f"  [OK] Replay buffer中零动作比例较低 ({zero_action_count/len(actions_np)*100:.2f}%)")
                
            if np.mean(action_norms) < 0.05:
                print_yellow(f"  [WARNING] 动作范数均值很小 ({np.mean(action_norms):.6f})，BC可能学习到接近零的动作分布")
            else:
                print_green(f"  [OK] 动作范数均值合理 ({np.mean(action_norms):.6f})，数据质量良好，可以开始训练")
        
        # 设置wandb logger
        wandb_logger = make_wandb_logger(
            project="hil-serl",
            description=f"{FLAGS.exp_name}_bc_single_traj_{FLAGS.traj_index}",
            debug=FLAGS.debug,
        )
        
        # 训练BC
        print_green(f"[BC TRAINING] 开始训练，共 {FLAGS.train_steps} 步...")
        
        # 使用 iterator 来获取批次数据（与 train_bc.py 一致）
        bc_replay_iterator = bc_replay_buffer.get_iterator(
            sample_args={
                "batch_size": config.batch_size,
                "pack_obs_and_next_obs": False,
            },
            device=sharding,
        )
        
        for step in tqdm.tqdm(
            range(FLAGS.train_steps),
            dynamic_ncols=True,
            desc="bc_pretraining",
        ):
            batch = next(bc_replay_iterator)
            bc_agent, bc_update_info = bc_agent.update(batch)
            
            # 记录训练信息
            if step % 100 == 0:
                if wandb_logger:
                    wandb_logger.log({"bc": bc_update_info}, step=step)
            
            if (step + 1) % 1000 == 0:
                print_green(f"[BC TRAINING] Step {step + 1}/{FLAGS.train_steps}, Loss: {bc_update_info['actor_loss']:.6f}")
                
            # 定期保存checkpoint（最后100步每10步保存一次）
            if step > FLAGS.train_steps - 100 and step % 10 == 0:
                checkpoints.save_checkpoint(
                    os.path.abspath(FLAGS.bc_checkpoint_path), bc_agent.state, step=step, keep=5
                )
        
        # 保存最终checkpoint
        print_green("[BC TRAINING] BC pretraining done and saved checkpoint")
        
        # 训练后评估（如果设置了eval_n_trajs）
        if FLAGS.eval_n_trajs > 0:
            print_green(f"\n[BC EVAL] 开始评估...")
            eval_rng = jax.random.PRNGKey(FLAGS.seed)
            eval_rng = jax.device_put(eval_rng, sharding)
            success_count, action_errors = eval(
                env,
                bc_agent,
                eval_rng,
                reference_trajectory=reference_trajectory,
            )
            
            print_green(f"\n[BC EVAL] 最终结果: {success_count}/{FLAGS.eval_n_trajs} 成功")
            if len(action_errors) > 0:
                print_green(f"[BC EVAL] 动作误差统计:")
                print_green(f"  平均: {np.mean(action_errors):.6f}")
                print_green(f"  最大: {np.max(action_errors):.6f}")
                print_green(f"  最小: {np.min(action_errors):.6f}")
                print_green(f"  标准差: {np.std(action_errors):.6f}")
    except Exception as e:
        print_yellow(f"[ERROR] 发生错误: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # 关闭日志文件
        if log_file_obj is not None:
            log_file_obj.close()
            print(f"[LOG] 日志已保存到: {log_file_path}")


if __name__ == "__main__":
    app.run(main)
