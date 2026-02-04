#!/usr/bin/env python3

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
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
flags.DEFINE_integer("train_steps", 20_000, "Number of pretraining steps.")
flags.DEFINE_bool("save_video", False, "Save video of the evaluation.")
flags.DEFINE_multi_string("demo_path", None, "Path to the demo data. If not set, uses demo_data/*.pkl")
flags.DEFINE_boolean("use_sim", False, "Use Isaac Sim simulation environment for evaluation.")
flags.DEFINE_string("isaac_server_url", None, "Isaac Sim server URL (e.g., http://192.168.1.100:5001/).")
flags.DEFINE_boolean("debug", False, "Debug mode.")  # debug mode will disable wandb logging
flags.DEFINE_integer("filter_max_consecutive", 10, "Maximum consecutive zero actions before filtering. Set to 0 to disable filtering.")
flags.DEFINE_boolean("enable_filtering", True, "Enable consecutive zero action filtering.")
flags.DEFINE_float("mse_weight", 0.1, "Weight for normalized MSE loss in total loss (default: 0.1).")


devices = jax.local_devices()
num_devices = len(devices)
# JAX 0.9.0+ removed PositionalSharding, use NamedSharding instead
if num_devices == 1:
    sharding = jax.sharding.SingleDeviceSharding(devices[0])
else:
    # For multiple devices, use NamedSharding with a mesh
    from jax.sharding import Mesh, NamedSharding, PartitionSpec
    mesh = Mesh(devices, axis_names=('devices',))
    sharding = NamedSharding(mesh, PartitionSpec('devices'))


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


def print_yellow(x):
    return print("\033[93m {}\033[00m".format(x))


##############################################################################

def eval(
    env,
    bc_agent: BCAgent,
    sampling_rng,
):
    """
    Evaluation loop for BC policy.
    """
    success_counter = 0
    time_list = []
    for episode in range(FLAGS.eval_n_trajs):
        print_green(f"[BC EVAL] Starting episode {episode + 1}/{FLAGS.eval_n_trajs}")
        obs, _ = env.reset()
        done = False
        start_time = time.time()
        step_count = 0
        while not done:
            step_count += 1
            if step_count % 100 == 0:
                print(f"[BC EVAL] Episode {episode + 1}, Step {step_count}")
            
            rng, key = jax.random.split(sampling_rng)
            sampling_rng = rng

            try:
                # 使用argmax=True来使用分布的mode（均值）而不是采样
                # 这样可以获得确定性的动作，更接近演示数据
                actions = bc_agent.sample_actions(
                    observations=jax.device_put(obs), 
                    seed=key,
                    argmax=True  # 使用mode（均值）而不是采样
                )
                actions = np.asarray(jax.device_get(actions))
                
                # 调试：打印动作值（每100步打印一次）
                if step_count % 100 == 0:
                    action_norm = np.linalg.norm(actions)
                    print(f"[BC EVAL DEBUG] Step {step_count}: Action norm: {action_norm:.6f}, "
                          f"Action values: {actions}, "
                          f"Scaled X: {actions[0] * 0.01:.6f}m, "
                          f"Scaled Y: {actions[1] * 0.01:.6f}m, "
                          f"Scaled Z: {actions[2] * 0.01:.6f}m")
            except Exception as e:
                print(f"[ERROR] Failed to sample actions: {e}")
                import traceback
                traceback.print_exc()
                break
            
            try:
                next_obs, reward, done, truncated, info = env.step(actions)
            except Exception as e:
                print(f"[ERROR] Failed to step environment: {e}")
                import traceback
                traceback.print_exc()
                break
            
            obs = next_obs
            if done or truncated:
                if reward:
                    dt = time.time() - start_time
                    time_list.append(dt)
                    print(f"[SUCCESS] Episode {episode + 1}: Time = {dt:.2f}s, Steps = {step_count}")
                success_counter += reward
                print(f"[EPISODE {episode + 1}] Reward: {reward}, Success: {success_counter}/{episode + 1}, Steps: {step_count}")
                break

    print_green(f"\n[EVAL RESULTS]")
    print_green(f"  Success rate: {success_counter / FLAGS.eval_n_trajs * 100:.2f}%")
    if time_list:
        print_green(f"  Average time: {np.mean(time_list):.2f}s")
    else:
        print_yellow(f"  No successful episodes, cannot compute average time")


##############################################################################


def train(
    bc_agent: BCAgent,
    bc_replay_buffer,
    config,
    wandb_logger=None,
):

    bc_replay_iterator = bc_replay_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size,
            "pack_obs_and_next_obs": False,
        },
        device=sharding,
    )
    
    # Pretrain BC policy
    print_green(f"[BC TRAINING] Starting BC pretraining for {FLAGS.train_steps} steps")
    print_green(f"[BC TRAINING] Replay buffer size: {len(bc_replay_buffer)}")
    
    for step in tqdm.tqdm(
        range(FLAGS.train_steps),
        dynamic_ncols=True,
        desc="bc_pretraining",
    ):
        batch = next(bc_replay_iterator)
        bc_agent, bc_update_info = bc_agent.update(batch)
        
        # 监控BC输出的动作分布（用于诊断BC是否在学习输出接近零）
        if step % config.log_period == 0:
            # 采样一批观察，检查BC输出
            sample_obs = batch["observations"]
            rng, key = jax.random.split(bc_agent.state.rng)
            bc_agent = bc_agent.replace(state=bc_agent.state.replace(rng=rng))
            
            try:
                bc_actions = bc_agent.sample_actions(
                    observations=sample_obs,
                    seed=key,
                )
                bc_actions_np = np.asarray(jax.device_get(bc_actions))
                
                # 统计BC输出的动作分布
                action_norms = np.linalg.norm(bc_actions_np, axis=-1)
                action_stats = {
                    "bc_action_mean": float(np.mean(bc_actions_np)),
                    "bc_action_std": float(np.std(bc_actions_np)),
                    "bc_action_norm_mean": float(np.mean(action_norms)),
                    "bc_action_norm_std": float(np.std(action_norms)),
                    "bc_action_norm_min": float(np.min(action_norms)),
                    "bc_action_norm_max": float(np.max(action_norms)),
                }
                
                # 统计演示数据的动作分布（用于对比）
                demo_actions = batch["actions"]
                demo_actions_np = np.asarray(jax.device_get(demo_actions))
                demo_action_norms = np.linalg.norm(demo_actions_np, axis=-1)
                demo_action_stats = {
                    "demo_action_mean": float(np.mean(demo_actions_np)),
                    "demo_action_std": float(np.std(demo_actions_np)),
                    "demo_action_norm_mean": float(np.mean(demo_action_norms)),
                    "demo_action_norm_std": float(np.std(demo_action_norms)),
                }
                
                # 计算动作分布差异
                action_diff = {
                    "action_mean_diff": abs(action_stats["bc_action_mean"] - demo_action_stats["demo_action_mean"]),
                    "action_norm_diff": abs(action_stats["bc_action_norm_mean"] - demo_action_stats["demo_action_norm_mean"]),
                }
                
                if wandb_logger:
                    wandb_logger.log({
                        "bc": bc_update_info,
                        "bc_action_stats": action_stats,
                        "demo_action_stats": demo_action_stats,
                        "action_diff": action_diff,
                    }, step=step)
                
                # 打印警告（如果BC输出动作范数太小）
                if action_stats["bc_action_norm_mean"] < 0.05:
                    print_yellow(f"[WARNING] Step {step}: BC output action norm is very small ({action_stats['bc_action_norm_mean']:.6f}), "
                                f"BC may be learning to output near-zero actions!")
                    print_yellow(f"  Demo action norm mean: {demo_action_stats['demo_action_norm_mean']:.6f}")
                    print_yellow(f"  Action norm difference: {action_diff['action_norm_diff']:.6f}")
                    print_yellow(f"  BC action std: {action_stats['bc_action_std']:.6f}")
                    print_yellow(f"  [DIAGNOSIS] BC策略可能学习到了'不动'策略，建议：")
                    print_yellow(f"    1. 使用更激进的过滤: --filter_max_consecutive=5")
                    print_yellow(f"    2. 检查演示数据质量（动作值是否太小）")
                    print_yellow(f"    3. 考虑实现加权训练")
                    
            except Exception as e:
                print_yellow(f"[WARNING] Failed to compute BC action stats: {e}")
                if wandb_logger:
                    wandb_logger.log({"bc": bc_update_info}, step=step)
        elif wandb_logger:
            wandb_logger.log({"bc": bc_update_info}, step=step)
            
        if step > FLAGS.train_steps - 100 and step % 10 == 0:
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.bc_checkpoint_path), bc_agent.state, step=step, keep=5
            )
    print_green("[BC TRAINING] BC pretraining done and saved checkpoint")


##############################################################################


def main(_):
    config = CONFIG_MAPPING[FLAGS.exp_name]()

    assert config.batch_size % num_devices == 0
    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    
    # 打印JAX设备信息
    print_green(f"JAX devices: {jax.devices()}")
    print_green(f"JAX platform: {jax.devices()[0].platform if jax.devices() else 'No devices'}")
    print_green(f"Number of devices: {num_devices}")
    
    eval_mode = FLAGS.eval_n_trajs > 0
    
    # 环境配置
    # 如果提供了 isaac_server_url，自动使用 Isaac Sim 环境
    if eval_mode and FLAGS.isaac_server_url is not None:
        use_fake_env = True  # 使用 Isaac Sim 仿真环境
    else:
        use_fake_env = FLAGS.use_sim if eval_mode else True  # 训练时使用fake_env，评估时根据use_sim决定
    skip_server_connection = not eval_mode  # 训练时跳过服务器连接
    
    env = config.get_environment(
        fake_env=use_fake_env,
        save_video=FLAGS.save_video,
        classifier=not use_fake_env,  # 评估时使用分类器（如果真实环境），训练时不用
        isaac_server_url=FLAGS.isaac_server_url,
        skip_server_connection=skip_server_connection,
    )
    env = RecordEpisodeStatistics(env)

    bc_agent: BCAgent = make_bc_agent(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=config.image_keys,
        encoder_type=config.encoder_type,
        mse_weight=FLAGS.mse_weight,  # Pass mse_weight from command line
    )

    # replicate agent across devices
    # JAX 0.8+ doesn't have sharding.replicate(), use device_put with sharding directly
    bc_agent: BCAgent = jax.device_put(
        jax.tree.map(jnp.array, bc_agent), sharding
    )

    if not eval_mode:
        # 训练模式
        assert FLAGS.bc_checkpoint_path is not None, "Must specify --bc_checkpoint_path for training"
        assert not os.path.exists(
            os.path.join(FLAGS.bc_checkpoint_path, f"checkpoint_{FLAGS.train_steps}")
        ), f"Checkpoint {FLAGS.train_steps} already exists. Remove it or use a different train_steps."

        bc_replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
        )

        # set up wandb and logging
        wandb_logger = make_wandb_logger(
            project="hil-serl",
            description=f"{FLAGS.exp_name}_bc",
            debug=FLAGS.debug,
        )

        # 加载演示数据
        if FLAGS.demo_path:
            demo_paths = FLAGS.demo_path
        else:
            # 默认从 demo_data 目录加载
            demo_paths = glob.glob(os.path.join(os.getcwd(), "demo_data", "*.pkl"))
        
        assert len(demo_paths) > 0, f"No demo files found. Checked: {demo_paths if FLAGS.demo_path else 'demo_data/*.pkl'}"

        print_green(f"[BC TRAINING] Loading demo data from {len(demo_paths)} file(s)...")
        total_transitions = 0
        total_zero_actions = 0
        total_filtered_transitions = 0
        
        def filter_consecutive_zero_actions(transitions, max_consecutive=10):
            """
            过滤连续零动作，保留间隔零动作
            
            Args:
                transitions: 演示数据列表
                max_consecutive: 连续零动作的最大数量，超过此数量只保留1个
            
            Returns:
                filtered: 过滤后的transitions列表
                zero_count: 零动作总数
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
                action_norm = np.linalg.norm(action)
                action_norms.append(action_norm)
                
                if action_norm < 1e-6:  # 零动作
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
        
        all_stats = []
        for path in demo_paths:
            print(f"  Loading: {path}")
            with open(path, "rb") as f:
                transitions = pkl.load(f)
                # 检查数据格式
                if isinstance(transitions, list):
                    # 如果是列表，可能是多个轨迹的列表
                    if len(transitions) > 0 and isinstance(transitions[0], list):
                        # 嵌套列表：每个元素是一个轨迹
                        for traj in transitions:
                            if FLAGS.enable_filtering and FLAGS.filter_max_consecutive > 0:
                                # 对每个轨迹进行过滤
                                filtered_traj, zero_count, filtered_count, stats = filter_consecutive_zero_actions(
                                    traj, max_consecutive=FLAGS.filter_max_consecutive
                                )
                                all_stats.append(stats)
                                total_zero_actions += zero_count
                                total_filtered_transitions += filtered_count
                                for transition in filtered_traj:
                                    bc_replay_buffer.insert(transition)
                                    total_transitions += 1
                            else:
                                # 不过滤，直接加载
                                for transition in traj:
                                    action_norm = np.linalg.norm(transition.get('actions', np.array([0])))
                                    if action_norm < 1e-6:
                                        total_zero_actions += 1
                                    bc_replay_buffer.insert(transition)
                                    total_transitions += 1
                    else:
                        # 扁平列表：直接是transitions
                        if FLAGS.enable_filtering and FLAGS.filter_max_consecutive > 0:
                            # 对transitions进行过滤
                            filtered_transitions, zero_count, filtered_count, stats = filter_consecutive_zero_actions(
                                transitions, max_consecutive=FLAGS.filter_max_consecutive
                            )
                            all_stats.append(stats)
                            total_zero_actions += zero_count
                            total_filtered_transitions += filtered_count
                            for transition in filtered_transitions:
                                bc_replay_buffer.insert(transition)
                                total_transitions += 1
                        else:
                            # 不过滤，直接加载
                            for transition in transitions:
                                action_norm = np.linalg.norm(transition.get('actions', np.array([0])))
                                if action_norm < 1e-6:
                                    total_zero_actions += 1
                                bc_replay_buffer.insert(transition)
                                total_transitions += 1
                else:
                    print_yellow(f"  Warning: Unexpected data format in {path}, skipping...")
        
        # 打印过滤统计信息
        if FLAGS.enable_filtering and FLAGS.filter_max_consecutive > 0:
            print_green(f"\n[BC TRAINING] ========== 数据过滤统计 ==========")
            if all_stats:
                total_before = sum(s["total"] for s in all_stats)
                total_zero_before = sum(s["zero_count"] for s in all_stats)
                total_filtered = sum(s["filtered_count"] for s in all_stats)
                total_after = sum(s["filtered"] for s in all_stats)
                
                print_green(f"  过滤前: {total_before} transitions")
                print_green(f"  零动作: {total_zero_before} ({total_zero_before/total_before*100:.2f}%)")
                print_green(f"  过滤掉: {total_filtered} 零动作")
                print_green(f"  过滤后: {total_after} transitions")
                print_green(f"  零动作比例: {total_zero_before/total_before*100:.2f}% → {(total_zero_before-total_filtered)/total_after*100:.2f}%")
                print_green(f"  数据压缩率: {(1 - total_after/total_before)*100:.2f}%")
            else:
                print_green(f"  过滤前: {total_transitions + total_filtered_transitions} transitions")
                print_green(f"  零动作: {total_zero_actions}")
                print_green(f"  过滤掉: {total_filtered_transitions} 零动作")
                print_green(f"  过滤后: {total_transitions} transitions")
                if total_transitions + total_filtered_transitions > 0:
                    zero_ratio_before = total_zero_actions / (total_transitions + total_filtered_transitions)
                    zero_ratio_after = (total_zero_actions - total_filtered_transitions) / total_transitions if total_transitions > 0 else 0
                    print_green(f"  零动作比例: {zero_ratio_before*100:.2f}% → {zero_ratio_after*100:.2f}%")
                    print_green(f"  数据压缩率: {(total_filtered_transitions / (total_transitions + total_filtered_transitions))*100:.2f}%")
        else:
            print_green(f"\n[BC TRAINING] ========== 数据统计 ==========")
            print_green(f"  总transitions: {total_transitions}")
            print_green(f"  零动作: {total_zero_actions} ({total_zero_actions/total_transitions*100:.2f}%)" if total_transitions > 0 else "  零动作: 0")
            print_yellow(f"  [INFO] 零动作过滤已禁用 (--enable_filtering=False 或 --filter_max_consecutive=0)")
        
        print_green(f"[BC TRAINING] Final replay buffer size: {total_transitions} transitions")
        
        print_green(f"[BC TRAINING] Loaded {total_transitions} transitions into replay buffer")
        print_green(f"[BC TRAINING] Final replay buffer size: {len(bc_replay_buffer)}")
        
        # 统计过滤后数据的动作分布
        if len(bc_replay_buffer) > 0:
            # 采样一些数据来检查
            sample_size = min(1000, len(bc_replay_buffer))
            sample_batch = bc_replay_buffer.sample(sample_size)
            
            # 奖励统计
            rewards = sample_batch["rewards"]
            success_count = np.sum(rewards > 0.5)
            print_green(f"\n[BC TRAINING] ========== 过滤后数据质量统计 ==========")
            print_green(f"  奖励统计 (采样 {sample_size} 条):")
            print_green(f"    Min: {np.min(rewards):.4f}, Max: {np.max(rewards):.4f}, Mean: {np.mean(rewards):.4f}")
            print_green(f"    成功率: {success_count / len(rewards) * 100:.2f}%")
            
            # 动作分布统计
            actions = sample_batch["actions"]
            actions_np = np.asarray(actions)
            action_norms = np.linalg.norm(actions_np, axis=-1)
            zero_action_count = np.sum(action_norms < 1e-6)
            
            print_green(f"  动作统计 (采样 {sample_size} 条):")
            print_green(f"    零动作: {zero_action_count} ({zero_action_count/len(actions_np)*100:.2f}%)")
            print_green(f"    动作范数 - Mean: {np.mean(action_norms):.6f}, Std: {np.std(action_norms):.6f}")
            print_green(f"    动作范数 - Min: {np.min(action_norms):.6f}, Max: {np.max(action_norms):.6f}")
            print_green(f"    各维度均值: {np.mean(actions_np, axis=0)}")
            print_green(f"    各维度标准差: {np.std(actions_np, axis=0)}")
            
            # 检查数据质量
            if zero_action_count / len(actions_np) > 0.3:
                print_yellow(f"  [WARNING] 过滤后仍有 {zero_action_count/len(actions_np)*100:.2f}% 零动作，可能需要更激进的过滤")
            if np.mean(action_norms) < 0.05:
                print_yellow(f"  [WARNING] 动作范数均值很小 ({np.mean(action_norms):.6f})，BC可能学习到接近零的动作分布")
            else:
                print_green(f"  [OK] 动作范数均值合理 ({np.mean(action_norms):.6f})，数据质量良好")

        # 训练
        print_green(f"\n[BC TRAINING] ========== 开始BC训练 ==========")
        print_green(f"  训练步数: {FLAGS.train_steps}")
        print_green(f"  批次大小: {config.batch_size}")
        print_green(f"  数据量: {len(bc_replay_buffer)} transitions")
        if FLAGS.enable_filtering and FLAGS.filter_max_consecutive > 0:
            print_green(f"  过滤配置: 启用 (max_consecutive={FLAGS.filter_max_consecutive})")
        else:
            print_yellow(f"  过滤配置: 禁用")
        
        train(
            bc_agent=bc_agent,
            bc_replay_buffer=bc_replay_buffer,
            config=config,
            wandb_logger=wandb_logger,
        )
        
        # 训练完成后的提示
        print_green(f"\n[BC TRAINING] ========== 训练完成 ==========")
        print_green(f"  Checkpoint已保存到: {FLAGS.bc_checkpoint_path}")
        print_green(f"\n[下一步] 评估BC策略性能:")
        print_green(f"  python examples/train_bc.py \\")
        print_green(f"    --exp_name={FLAGS.exp_name} \\")
        print_green(f"    --bc_checkpoint_path={FLAGS.bc_checkpoint_path} \\")
        print_green(f"    --eval_n_trajs=10 \\")
        if FLAGS.isaac_server_url:
            print_green(f"    --isaac_server_url={FLAGS.isaac_server_url}")
        else:
            print_green(f"    --use_sim")
        print_green(f"\n[判断标准]")
        print_green(f"  - 成功率 > 50%: BC学到了有效策略，可以继续RLPD训练")
        print_green(f"  - 成功率 20-50%: BC学到部分策略，可能需要更多数据或改进数据质量")
        print_green(f"  - 成功率 < 20%: BC无法学到有效策略，需要检查数据质量或重新采集数据")

    else:
        # 评估模式
        assert FLAGS.bc_checkpoint_path is not None, "Must specify --bc_checkpoint_path for evaluation"
        
        rng = jax.random.PRNGKey(FLAGS.seed)
        sampling_rng = jax.device_put(rng, sharding)

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

        print_green(f"[BC EVAL] Starting evaluation with {FLAGS.eval_n_trajs} trajectories...")
        eval(
            env=env,
            bc_agent=bc_agent,
            sampling_rng=sampling_rng,
        )


if __name__ == "__main__":
    app.run(main)
