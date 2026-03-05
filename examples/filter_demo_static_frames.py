#!/usr/bin/env python3
"""
演示数据清理：去除同一条轨迹内连续的静态帧，只保留每段静态段的第一帧。

静态帧定义：
- action 前 6 维全为 0（位姿/旋转无变化）
- action 第 7 维（gripper）与上一帧相同（夹爪无变化）

去重规则：在同一条轨迹内，连续多帧均为静态时，只保留第一帧，后续帧删除。
"""
import os
import sys
import pickle as pkl
import numpy as np
from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_string("input", None, "输入的演示数据 pkl 文件路径。")
flags.DEFINE_string("output", None, "输出的过滤后 pkl 文件路径。不指定时在输入路径上加 _filtered。")
flags.DEFINE_float("zero_tol", 1e-6, "判定前 6 维为 0 的阈值（绝对值小于此视为 0）。")


def _split_into_trajectories(transitions):
    """按 dones=True 将 transitions 拆成多条轨迹（每条轨迹包含结束帧）。"""
    trajectories = []
    current = []
    for t in transitions:
        current.append(t)
        if t.get("dones", False):
            trajectories.append(current)
            current = []
    if current:
        trajectories.append(current)
    return trajectories


def _is_static(action, prev_action, zero_tol):
    """当前帧是否为静态：前 6 维全 0 且第 7 维与上一帧相同。"""
    arr = np.asarray(action, dtype=np.float64)
    if arr.size < 7:
        return False
    first_6_zero = np.all(np.abs(arr[:6]) <= zero_tol)
    if prev_action is None:
        return False  # 第一帧无“上一帧”，不视为静态（用于去重判断）
    prev_arr = np.asarray(prev_action, dtype=np.float64)
    gripper_same = np.abs(arr[6] - prev_arr[6]) <= zero_tol
    return bool(first_6_zero and gripper_same)


def _filter_trajectory_static_frames(transitions, zero_tol):
    """
    同一条轨迹内，连续静态帧只保留第一帧。
    第一帧始终保留（作为参照）。
    """
    if not transitions:
        return []
    kept = []
    prev_action = None
    prev_was_static = False
    for t in transitions:
        action = t.get("actions", None)
        if action is None:
            kept.append(t)
            prev_action = None
            prev_was_static = False
            continue
        is_static = _is_static(action, prev_action, zero_tol)
        # 仅当「当前是静态 且 上一帧也是静态」时才删掉当前帧
        if is_static and prev_was_static:
            pass
        else:
            kept.append(t)
        prev_action = action
        prev_was_static = is_static
    return kept


def filter_demo_static_frames(transitions, zero_tol=1e-6):
    """
    对演示数据做静态帧去重：按轨迹拆分，轨迹内连续静态帧只保留第一帧。

    Args:
        transitions: list of dict (observations, actions, next_observations, rewards, masks, dones, infos)
        zero_tol: 判定前 6 维为 0 的容差

    Returns:
        list of dict: 过滤后的 transitions（顺序与轨迹结构不变）
    """
    trajectories = _split_into_trajectories(transitions)
    filtered = []
    for traj in trajectories:
        filtered.extend(_filter_trajectory_static_frames(traj, zero_tol))
    return filtered


def main(_):
    if not FLAGS.input or not os.path.isfile(FLAGS.input):
        print("请指定有效的 --input 演示数据 pkl 文件。")
        sys.exit(1)
    out_path = FLAGS.output
    if not out_path:
        base, ext = os.path.splitext(FLAGS.input)
        out_path = base + "_filtered" + ext

    print("=" * 60)
    print("演示数据静态帧过滤")
    print("=" * 60)
    print(f"输入: {FLAGS.input}")
    print(f"输出: {out_path}")
    print(f"零值容差: {FLAGS.zero_tol}")

    with open(FLAGS.input, "rb") as f:
        transitions = pkl.load(f)
    original_count = len(transitions)

    filtered = filter_demo_static_frames(transitions, zero_tol=FLAGS.zero_tol)
    new_count = len(filtered)
    removed = original_count - new_count

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        pkl.dump(filtered, f)

    print()
    print(f"原始帧数: {original_count}")
    print(f"过滤后帧数: {new_count}")
    print(f"删除帧数: {removed} ({removed / max(1, original_count) * 100:.1f}%)")
    print(f"已保存: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    app.run(main)
