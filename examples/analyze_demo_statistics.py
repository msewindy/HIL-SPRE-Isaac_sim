#!/usr/bin/env python3
"""
分析演示数据的统计信息
"""
import os
import sys
import pickle as pkl
import numpy as np


def analyze_demo_statistics(demo_path):
    """分析演示数据的统计信息"""
    print("=" * 80)
    print("演示数据统计分析")
    print("=" * 80)
    
    print(f"\n加载文件: {demo_path}")
    with open(demo_path, "rb") as f:
        transitions = pkl.load(f)
    
    print(f"总transitions数: {len(transitions)}")
    
    # 1. 分析轨迹数量和成功轨迹数量
    print("\n" + "=" * 80)
    print("1. 轨迹统计")
    print("=" * 80)
    
    trajectories = []
    current_traj = []
    total_trajectories = 0
    successful_trajectories = 0
    
    for i, transition in enumerate(transitions):
        current_traj.append(transition)
        
        # 检查是否轨迹结束
        done = transition.get('dones', False)
        if done:
            total_trajectories += 1
            
            # 检查是否成功
            reward = transition.get('rewards', 0.0)
            info = transition.get('infos', {})
            succeed = False
            if isinstance(info, dict):
                succeed = info.get('succeed', False)
            
            # 判断成功：reward > 0.5 或 succeed == True
            if reward > 0.5 or succeed:
                successful_trajectories += 1
                trajectories.append({
                    'transitions': current_traj.copy(),
                    'length': len(current_traj),
                    'success': True,
                    'reward': reward
                })
            else:
                trajectories.append({
                    'transitions': current_traj.copy(),
                    'length': len(current_traj),
                    'success': False,
                    'reward': reward
                })
            
            current_traj = []
    
    # 如果最后一个轨迹没有done标记
    if len(current_traj) > 0:
        total_trajectories += 1
        last_transition = current_traj[-1]
        reward = last_transition.get('rewards', 0.0)
        info = last_transition.get('infos', {})
        succeed = False
        if isinstance(info, dict):
            succeed = info.get('succeed', False)
        if reward > 0.5 or succeed:
            successful_trajectories += 1
            trajectories.append({
                'transitions': current_traj.copy(),
                'length': len(current_traj),
                'success': True,
                'reward': reward
            })
        else:
            trajectories.append({
                'transitions': current_traj.copy(),
                'length': len(current_traj),
                'success': False,
                'reward': reward
            })
    
    print(f"总轨迹数: {total_trajectories}")
    print(f"成功轨迹数: {successful_trajectories}")
    print(f"失败轨迹数: {total_trajectories - successful_trajectories}")
    if total_trajectories > 0:
        print(f"成功率: {successful_trajectories / total_trajectories * 100:.2f}%")
    
    # 2. 分析所有轨迹的数据帧统计
    print("\n" + "=" * 80)
    print("2. 数据帧统计")
    print("=" * 80)
    
    trajectory_lengths = [traj['length'] for traj in trajectories]
    total_frames = sum(trajectory_lengths)
    
    print(f"所有轨迹的总数据帧数: {total_frames}")
    if len(trajectory_lengths) > 0:
        print(f"最多数据帧: {max(trajectory_lengths)}")
        print(f"最少数据帧: {min(trajectory_lengths)}")
        print(f"平均数据帧: {np.mean(trajectory_lengths):.2f}")
        print(f"中位数数据帧: {np.median(trajectory_lengths):.2f}")
        print(f"标准差: {np.std(trajectory_lengths):.2f}")
    
    # 只统计成功轨迹的数据帧
    successful_lengths = [traj['length'] for traj in trajectories if traj['success']]
    if len(successful_lengths) > 0:
        print(f"\n成功轨迹的数据帧统计:")
        print(f"  总数据帧数: {sum(successful_lengths)}")
        print(f"  最多数据帧: {max(successful_lengths)}")
        print(f"  最少数据帧: {min(successful_lengths)}")
        print(f"  平均数据帧: {np.mean(successful_lengths):.2f}")
        print(f"  中位数数据帧: {np.median(successful_lengths):.2f}")
        print(f"  标准差: {np.std(successful_lengths):.2f}")
    
    # 3. 分析action前6维都为0的数据帧
    print("\n" + "=" * 80)
    print("3. Action前6维为零的统计")
    print("=" * 80)
    
    zero_action_6d_count = 0
    total_action_frames = 0
    
    for transition in transitions:
        if 'actions' in transition:
            total_action_frames += 1
            action = np.array(transition['actions'])
            if len(action) >= 6:
                # 检查前6维是否都为0（使用小的阈值）
                if np.all(np.abs(action[:6]) < 1e-6):
                    zero_action_6d_count += 1
    
    print(f"总action数据帧数: {total_action_frames}")
    print(f"前6维都为0的数据帧数: {zero_action_6d_count}")
    if total_action_frames > 0:
        print(f"前6维都为0的占比: {zero_action_6d_count / total_action_frames * 100:.2f}%")
    
    # 4. Action各维度的统计学指标
    print("\n" + "=" * 80)
    print("4. Action各维度统计学指标")
    print("=" * 80)
    
    all_actions = []
    for transition in transitions:
        if 'actions' in transition:
            action = np.array(transition['actions'])
            all_actions.append(action)
    
    if len(all_actions) > 0:
        all_actions = np.array(all_actions)
        action_dim = all_actions.shape[1]
        
        print(f"Action维度: {action_dim}")
        print(f"总样本数: {len(all_actions)}")
        
        # 维度名称
        dim_names = ['x', 'y', 'z', 'rx', 'ry', 'rz', 'gripper']
        if action_dim > len(dim_names):
            dim_names.extend([f'dim_{i}' for i in range(len(dim_names), action_dim)])
        
        print("\n各维度统计:")
        print("-" * 80)
        print(f"{'维度':<10} {'最小值':<12} {'最大值':<12} {'均值':<12} {'标准差':<12} {'中位数':<12} {'绝对值均值':<12}")
        print("-" * 80)
        
        for dim in range(action_dim):
            dim_vals = all_actions[:, dim]
            dim_name = dim_names[dim] if dim < len(dim_names) else f'dim_{dim}'
            print(f"{dim_name:<10} {dim_vals.min():<12.6f} {dim_vals.max():<12.6f} "
                  f"{dim_vals.mean():<12.6f} {dim_vals.std():<12.6f} "
                  f"{np.median(dim_vals):<12.6f} {np.abs(dim_vals).mean():<12.6f}")
        
        # 额外统计：前6维的联合统计
        print("\n前6维联合统计:")
        print("-" * 80)
        first_6d = all_actions[:, :6]
        norms = np.linalg.norm(first_6d, axis=1)
        print(f"前6维范数统计:")
        print(f"  最小范数: {norms.min():.6f}")
        print(f"  最大范数: {norms.max():.6f}")
        print(f"  平均范数: {norms.mean():.6f}")
        print(f"  中位数范数: {np.median(norms):.6f}")
        print(f"  标准差: {norms.std():.6f}")
        
        # 统计前6维全为0的比例
        zero_6d_mask = np.all(np.abs(first_6d) < 1e-6, axis=1)
        print(f"  前6维全为0的比例: {np.sum(zero_6d_mask) / len(norms) * 100:.2f}%")
        
        # 统计各维度为零的比例
        print(f"\n各维度为零的比例（绝对值<1e-6）:")
        for dim in range(min(6, action_dim)):
            dim_vals = all_actions[:, dim]
            zero_count = np.sum(np.abs(dim_vals) < 1e-6)
            dim_name = dim_names[dim] if dim < len(dim_names) else f'dim_{dim}'
            print(f"  {dim_name}: {zero_count / len(dim_vals) * 100:.2f}%")
    
    # 5. 数据有效性评价
    print("\n" + "=" * 80)
    print("5. 数据有效性评价")
    print("=" * 80)
    
    evaluation = []
    
    # 评价1: 轨迹数量
    if successful_trajectories >= 5:
        evaluation.append(("✓", f"成功轨迹数量充足 ({successful_trajectories}条)"))
    elif successful_trajectories >= 3:
        evaluation.append(("⚠", f"成功轨迹数量较少 ({successful_trajectories}条)，建议至少5条"))
    else:
        evaluation.append(("✗", f"成功轨迹数量过少 ({successful_trajectories}条)，建议至少5条"))
    
    # 评价2: 数据帧数量
    if total_frames >= 1000:
        evaluation.append(("✓", f"总数据帧数充足 ({total_frames}帧)"))
    elif total_frames >= 500:
        evaluation.append(("⚠", f"总数据帧数较少 ({total_frames}帧)，建议至少1000帧"))
    else:
        evaluation.append(("✗", f"总数据帧数过少 ({total_frames}帧)，建议至少1000帧"))
    
    # 评价3: 轨迹长度分布
    if len(trajectory_lengths) > 0:
        avg_length = np.mean(trajectory_lengths)
        if avg_length >= 200:
            evaluation.append(("✓", f"平均轨迹长度充足 ({avg_length:.1f}帧)"))
        elif avg_length >= 100:
            evaluation.append(("⚠", f"平均轨迹长度较短 ({avg_length:.1f}帧)，建议至少200帧"))
        else:
            evaluation.append(("✗", f"平均轨迹长度过短 ({avg_length:.1f}帧)，建议至少200帧"))
    
    # 评价4: 零动作比例
    if total_action_frames > 0:
        zero_ratio = zero_action_6d_count / total_action_frames
        if zero_ratio < 0.1:
            evaluation.append(("✓", f"零动作比例合理 ({zero_ratio*100:.2f}%)"))
        elif zero_ratio < 0.3:
            evaluation.append(("⚠", f"零动作比例较高 ({zero_ratio*100:.2f}%)，可能影响学习效果"))
        else:
            evaluation.append(("✗", f"零动作比例过高 ({zero_ratio*100:.2f}%)，严重影响学习效果"))
    
    # 评价5: Action分布
    if len(all_actions) > 0:
        first_6d = all_actions[:, :6]
        norms = np.linalg.norm(first_6d, axis=1)
        non_zero_actions = norms[norms > 1e-6]
        if len(non_zero_actions) > 0:
            avg_norm = np.mean(non_zero_actions)
            if avg_norm >= 0.05:
                evaluation.append(("✓", f"非零动作平均范数合理 ({avg_norm:.4f})"))
            elif avg_norm >= 0.01:
                evaluation.append(("⚠", f"非零动作平均范数较小 ({avg_norm:.4f})，动作幅度可能不足"))
            else:
                evaluation.append(("✗", f"非零动作平均范数过小 ({avg_norm:.4f})，动作幅度严重不足"))
    
    # 评价6: Action各维度变化范围
    if len(all_actions) > 0:
        first_6d = all_actions[:, :6]
        ranges = np.max(first_6d, axis=0) - np.min(first_6d, axis=0)
        min_range = np.min(ranges)
        if min_range >= 0.1:
            evaluation.append(("✓", f"Action各维度变化范围充足 (最小范围: {min_range:.4f})"))
        elif min_range >= 0.05:
            evaluation.append(("⚠", f"Action某些维度变化范围较小 (最小范围: {min_range:.4f})"))
        else:
            evaluation.append(("✗", f"Action某些维度变化范围过小 (最小范围: {min_range:.4f})"))
    
    print("\n评价结果:")
    for status, msg in evaluation:
        print(f"  {status} {msg}")
    
    # 综合评分
    score = 0
    total = len(evaluation)
    for status, _ in evaluation:
        if status == "✓":
            score += 1
        elif status == "⚠":
            score += 0.5
    
    print(f"\n综合评分: {score}/{total} ({score/total*100:.1f}%)")
    
    if score / total >= 0.8:
        print("总体评价: ✓ 数据质量良好，可用于训练")
    elif score / total >= 0.6:
        print("总体评价: ⚠ 数据质量一般，建议补充更多数据或改进采集方式")
    else:
        print("总体评价: ✗ 数据质量较差，需要重新采集或大幅改进")
    
    return {
        'total_trajectories': total_trajectories,
        'successful_trajectories': successful_trajectories,
        'total_frames': total_frames,
        'trajectory_lengths': trajectory_lengths,
        'zero_action_6d_count': zero_action_6d_count,
        'zero_action_6d_ratio': zero_action_6d_count / total_action_frames if total_action_frames > 0 else 0,
        'action_stats': {
            'mean': np.mean(all_actions, axis=0) if len(all_actions) > 0 else None,
            'std': np.std(all_actions, axis=0) if len(all_actions) > 0 else None,
            'min': np.min(all_actions, axis=0) if len(all_actions) > 0 else None,
            'max': np.max(all_actions, axis=0) if len(all_actions) > 0 else None,
        } if len(all_actions) > 0 else None
    }


def main():
    # 默认使用指定的文件
    demo_path = "./demo_data/gear_assembly_5_demos_2026-02-04_09-01-56.pkl"
    
    # 如果提供了命令行参数，使用命令行参数
    if len(sys.argv) > 1:
        demo_path = sys.argv[1]
    
    if not os.path.exists(demo_path):
        print(f"错误: 文件不存在 {demo_path}")
        return
    
    analyze_demo_statistics(demo_path)


if __name__ == "__main__":
    main()
