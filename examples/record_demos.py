import os
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
from absl import app, flags
import time
import glob
import shutil

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 20, "Number of successful demos to collect.")
flags.DEFINE_float("success_sleep_sec", 3.0, "Seconds to wait after a success before reset (0 to disable). Was 10s, reduce to avoid long freeze.")
flags.DEFINE_integer("post_success_steps", 10, "Number of steps to continue recording after success is detected. This ensures the complete installation process is captured.")
flags.DEFINE_boolean("fake_env", False, "Use Isaac Sim simulation environment.")

def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=FLAGS.fake_env, save_video=False, classifier=False)
    
    obs, info = env.reset()
    # print(f"[VERIFY] Post-Reset State: {np.round(obs['state'][0], 4)}")
    print("Reset done")
    
    # 初始化变量
    success_needed = FLAGS.successes_needed
    
    # 创建或查找临时目录存储单个轨迹文件
    if not os.path.exists("./demo_data"):
        os.makedirs("./demo_data")
    
    # 查找是否已有未完成的临时文件夹（匹配实验名称和所需数量）
    temp_dir_pattern = f"./demo_data/{FLAGS.exp_name}_{success_needed}_demos_*_temp"
    existing_temp_dirs = glob.glob(temp_dir_pattern)
    
    if existing_temp_dirs:
        # 找到已有的临时文件夹，使用最新的（按修改时间排序）
        existing_temp_dirs.sort(key=os.path.getmtime, reverse=True)
        temp_dir = existing_temp_dirs[0]
        # 从文件夹名中提取 UUID
        uuid = os.path.basename(temp_dir).replace(f"{FLAGS.exp_name}_{success_needed}_demos_", "").replace("_temp", "")
        
        # 统计已有的轨迹文件数量
        existing_trajectory_files = sorted(glob.glob(os.path.join(temp_dir, "trajectory_*.pkl")))
        success_count = len(existing_trajectory_files)
        
        print(f"[INFO] Found existing temporary directory: {temp_dir}")
        print(f"[INFO] Resuming collection: {success_count}/{success_needed} trajectories already collected")
        print(f"[INFO] Will continue collecting from trajectory #{success_count}")
    else:
        # 创建新的临时文件夹
        uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        temp_dir = f"./demo_data/{FLAGS.exp_name}_{success_needed}_demos_{uuid}_temp"
        os.makedirs(temp_dir)
        success_count = 0
        print(f"[INFO] Created new temporary directory: {temp_dir}")
        print(f"[INFO] Starting fresh collection: 0/{success_needed} trajectories")
    
    pbar = tqdm(total=success_needed, initial=success_count)
    trajectory = []
    returns = 0
    
    try:
        while success_count < success_needed:
            actions = np.zeros(env.action_space.sample().shape) 
            # print(f"[DEBUG] Pre-Step Action (Input to step): {np.round(actions, 4)}")
            next_obs, rew, done, truncated, info = env.step(actions)
            # 手柄 Y 键重置：丢弃当前轨迹（不写入 transitions），重置环境后继续采集
            if info.get("user_reset_scene"):
                trajectory = []
                returns = 0
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]}] [DemoRecorder] [Y-RESET] User reset scene. Discarding current trajectory and resetting.")
                obs, info = env.reset()
                continue
            returns += rew
            if "intervene_action" in info:
                actions = info["intervene_action"]
                # print(f"[DEBUG] Client Gamepad Action: {actions}")
            else:
                # [FIX] If no intervene_action, but we need to preserve gripper state
                # Get current gripper state from environment to avoid recording 0 (half-open)
                # when gripper should maintain its current state
                try:
                    base_env = env.unwrapped
                    if hasattr(base_env, 'curr_gripper_pos') and base_env.curr_gripper_pos is not None:
                        gripper_pos = float(base_env.curr_gripper_pos[0])  # [0, 1] range
                        actions[6] = gripper_pos * 2.0 - 1.0  # Map to [-1, 1]
                except:
                    pass  # If cannot get, keep zero (will be recorded as half-open, which is not ideal but acceptable)
            transition = copy.deepcopy(
                dict(
                    observations=obs,
                    actions=actions,
                    next_observations=next_obs,
                    rewards=rew,
                    masks=1.0 - done,
                    dones=done,
                    infos=info,
                )
            )
            trajectory.append(transition)
            
            pbar.set_description(f"Return: {returns}")

            obs = next_obs
            
            # 检查是否成功（但可能还需要继续记录几个步骤）
            if done and info.get("succeed", False):
                # 成功判定后，继续记录几个步骤以确保记录完整的安装过程
                # 这很重要，因为环境在reward>0时立即设置done=True，
                # 但齿轮可能还没有完全安装到位或稳定
                post_success_steps = FLAGS.post_success_steps
                if post_success_steps > 0:
                    print(f"[{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]}] [DemoRecorder] [SUCCESS DETECTED] Task succeeded, continuing to record {post_success_steps} more steps...")
                
                for i in range(post_success_steps):
                    # 继续执行零动作（保持当前状态）并记录
                    actions = np.zeros(env.action_space.sample().shape)
                    next_obs, rew, done, truncated, info = env.step(actions)
                    
                    transition = copy.deepcopy(
                        dict(
                            observations=obs,
                            actions=actions,
                            next_observations=next_obs,
                            rewards=rew,
                            masks=1.0 - done,
                            dones=done,
                            infos=info,
                        )
                    )
                    trajectory.append(transition)
                    returns += rew
                    obs = next_obs
                    
                    # 如果环境再次设置done（如达到max_episode_length），提前结束
            if done:
                        break
                
                # 保存完整的成功轨迹（包括成功后的步骤）
                    trajectory_copy = copy.deepcopy(trajectory)
                    single_traj_file = os.path.join(temp_dir, f"trajectory_{success_count:04d}.pkl")
                    with open(single_traj_file, "wb") as f:
                        pkl.dump(trajectory_copy, f)
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]}] [DemoRecorder] [SUCCESS #{success_count+1}/{success_needed}] Saved trajectory ({len(trajectory_copy)} steps) to {single_traj_file}")
                    
                    success_count += 1
                    pbar.update(1)
                    
                    if FLAGS.success_sleep_sec > 0:
                        print(f"[INFO] Success! Waiting {FLAGS.success_sleep_sec}s before reset (set --success_sleep_sec=0 to skip).")
                        time.sleep(FLAGS.success_sleep_sec)

                trajectory = []
                returns = 0
                obs, info = env.reset()
            elif done:
                # 失败或超时
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]}] [DemoRecorder] [FAIL] Episode failed or timed out. Discarding trajectory.")
                trajectory = []
                returns = 0
                obs, info = env.reset()
        
        # 合并所有轨迹文件
        print(f"\n[INFO] Collecting all trajectories from {temp_dir}...")
        all_transitions = []
        trajectory_files = sorted(glob.glob(os.path.join(temp_dir, "trajectory_*.pkl")))
        
        if len(trajectory_files) != success_needed:
            print(f"[WARNING] Expected {success_needed} trajectory files, but found {len(trajectory_files)}")
        
        for traj_file in trajectory_files:
            try:
                with open(traj_file, "rb") as f:
                    trajectory = pkl.load(f)
                    all_transitions.extend(trajectory)
            except Exception as e:
                print(f"[ERROR] Failed to load {traj_file}: {e}")
                continue
        
        # 保存合并后的文件
        final_file_name = f"./demo_data/{FLAGS.exp_name}_{success_needed}_demos_{uuid}.pkl"
        with open(final_file_name, "wb") as f:
            pkl.dump(all_transitions, f)
        print(f"[INFO] Merged {len(trajectory_files)} trajectories ({len(all_transitions)} transitions) saved to {final_file_name}")
        
        # 删除临时目录
        try:
            shutil.rmtree(temp_dir)
            print(f"[INFO] Temporary directory {temp_dir} removed")
        except Exception as e:
            print(f"[WARNING] Failed to remove temporary directory {temp_dir}: {e}")
            print(f"[INFO] You can manually remove it later")
            
    except KeyboardInterrupt:
        print(f"\n[WARNING] Interrupted by user. Collected {success_count}/{success_needed} trajectories.")
        print(f"[INFO] Individual trajectory files are saved in: {temp_dir}")
        print(f"[INFO] You can manually merge them later or resume collection.")
        raise
    except Exception as e:
        print(f"\n[ERROR] Exception occurred: {e}")
        print(f"[INFO] Individual trajectory files are saved in: {temp_dir}")
        print(f"[INFO] You can manually merge them later.")
        raise

if __name__ == "__main__":
    app.run(main)