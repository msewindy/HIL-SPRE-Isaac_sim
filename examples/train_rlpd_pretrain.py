#!/usr/bin/env python3
"""
RLPD with Demo Pretraining & Dynamic Sampling Ratio

针对长程任务（如 gear assembly，~550 步/轨迹）优化的 RLPD 训练脚本。

与原始 train_rlpd.py 的核心区别：
1. Learner 预训练阶段：在 Actor 开始收集数据之前，先用纯 demo 数据训练 N 步，
   让 Critic 建立价值地图、Actor 学会基础策略。
2. 动态采样比例：正式训练初期使用高 demo 比例（如 90%），随训练进行逐渐降低到 50%，
   减少早期脏数据对训练的干扰。

用法与 train_rlpd.py 完全相同，只需替换脚本名即可。
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
import copy
import pickle as pkl
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from natsort import natsorted

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.agents.continuous.sac_hybrid_dual import SACAgentHybridDualArm
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.utils.train_utils import concat_batches

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from serl_launcher.utils.launcher import (
    make_sac_pixel_agent,
    make_sac_pixel_agent_hybrid_single_arm,
    make_sac_pixel_agent_hybrid_dual_arm,
    make_trainer_config,
)
from serl_launcher.common.tensorboard_logger import TensorBoardLogger

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "filter_demo_static_frames",
    os.path.join(os.path.dirname(__file__), "filter_demo_static_frames.py"),
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
is_static_frame = _mod._is_static
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("learner", False, "Whether this is a learner.")
flags.DEFINE_boolean("actor", False, "Whether this is an actor.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_multi_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_integer("eval_checkpoint_step", 0, "Step to evaluate the checkpoint.")
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
flags.DEFINE_boolean("save_video", False, "Save video.")
flags.DEFINE_boolean("use_sim", False, "Use Isaac Sim simulation environment for actor.")
flags.DEFINE_string("isaac_server_url", None, "Isaac Sim server URL.")
flags.DEFINE_boolean("debug", False, "Debug mode.")


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


##############################################################################


def actor(agent, data_store, intvn_data_store, env, sampling_rng):
    """
    Actor loop — identical to train_rlpd.py.
    """
    if FLAGS.eval_checkpoint_step:
        success_counter = 0
        time_list = []

        ckpt = checkpoints.restore_checkpoint(
            os.path.abspath(FLAGS.checkpoint_path),
            agent.state,
            step=FLAGS.eval_checkpoint_step,
        )
        agent = agent.replace(state=ckpt)

        for episode in range(FLAGS.eval_n_trajs):
            obs, _ = env.reset()
            done = False
            start_time = time.time()
            while not done:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    argmax=False,
                    seed=key
                )
                actions = np.asarray(jax.device_get(actions))
                next_obs, reward, done, truncated, info = env.step(actions)
                obs = next_obs
                if done:
                    if reward:
                        dt = time.time() - start_time
                        time_list.append(dt)
                        print(dt)
                    success_counter += reward
                    print(reward)
                    print(f"{success_counter}/{episode + 1}")

        print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
        print(f"average time: {np.mean(time_list)}")
        return

    start_step = (
        int(os.path.basename(natsorted(glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")))[-1])[12:-4]) + 1
        if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path)
        and glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl"))
        else 0
    )

    datastore_dict = {
        "actor_env": data_store,
        "actor_env_intvn": intvn_data_store,
    }

    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_stores=datastore_dict,
        wait_for_server=True,
        timeout_ms=3000,
    )
    print_green(f"[ACTOR] TrainerClient connected to Learner at {FLAGS.ip}")

    def update_params(params):
        nonlocal agent
        if not hasattr(update_params, '_update_count'):
            update_params._update_count = 0
        update_params._update_count += 1
        if update_params._update_count % 100 == 0:
            print_green(f"[ACTOR] Received network parameters update #{update_params._update_count}!")
        agent = agent.replace(state=agent.state.replace(params=params))

    client.recv_network_callback(update_params)

    transitions = []
    demo_transitions = []

    obs, _ = env.reset()
    done = False

    timer = Timer()
    running_return = 0.0
    already_intervened = False
    intervention_count = 0
    intervention_steps = 0
    prev_intvn_action = None
    prev_intvn_was_static = False
    intvn_static_filtered_count = 0

    pbar = tqdm.tqdm(range(start_step, config.max_steps), dynamic_ncols=True)

    base_env = env.unwrapped
    if hasattr(config, 'early_max_episode_length') and hasattr(config, 'early_training_steps'):
        if hasattr(base_env, 'set_max_episode_length'):
            if start_step < config.early_training_steps:
                base_env.set_max_episode_length(config.early_max_episode_length)

    for step in pbar:
        timer.tick("total")

        if hasattr(config, 'early_max_episode_length') and hasattr(config, 'early_training_steps'):
            if hasattr(base_env, 'set_max_episode_length'):
                if step == config.early_training_steps:
                    original_length = base_env._original_max_episode_length if hasattr(base_env, '_original_max_episode_length') else 1800
                    base_env.set_max_episode_length(original_length)

        with timer.context("sample_actions"):
            if step < config.random_steps:
                actions = env.action_space.sample()
                if step == 0 or step % 100 == 0:
                    print(f"[ACTOR] Step {step}: Using random actions (random_steps={config.random_steps})")
            else:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    seed=key,
                    argmax=False,
                )
                actions = np.asarray(jax.device_get(actions))
                if step == config.random_steps or (step > config.random_steps and step % 5000 == 0):
                    print(f"[ACTOR] Step {step}: Using policy network (action norm: {np.linalg.norm(actions):.4f})")

        with timer.context("step_env"):
            next_obs, reward, done, truncated, info = env.step(actions)
            if "left" in info:
                info.pop("left")
            if "right" in info:
                info.pop("right")

            if "intervene_action" in info:
                actions = info.pop("intervene_action")
                intervention_steps += 1
                if not already_intervened:
                    intervention_count += 1
                already_intervened = True
            else:
                already_intervened = False

            running_return += reward
            effective_done = done or truncated or info.get("user_reset_scene", False)
            transition = dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=reward,
                masks=1.0 - effective_done,
                dones=effective_done,
            )
            if 'grasp_penalty' in info:
                transition['grasp_penalty'] = info['grasp_penalty']
            data_store.insert(transition)
            transitions.append(copy.deepcopy(transition))
            if already_intervened:
                cur_is_static = is_static_frame(actions, prev_intvn_action, zero_tol=1e-6)
                if cur_is_static and prev_intvn_was_static:
                    intvn_static_filtered_count += 1
                else:
                    intvn_data_store.insert(transition)
                    demo_transitions.append(copy.deepcopy(transition))
                prev_intvn_action = actions
                prev_intvn_was_static = cur_is_static
            else:
                prev_intvn_action = None
                prev_intvn_was_static = False

            if step % 10 == 0:
                client.update()
                if step % 5000 == 0:
                    print(f"[ACTOR] Sent data to Learner at step {step} (queue size: {len(data_store)})")

            obs = next_obs
            if effective_done:
                if "episode" in info:
                    info["episode"]["intervention_count"] = intervention_count
                    info["episode"]["intervention_steps"] = intervention_steps
                else:
                    info["episode"] = {
                        "intervention_count": intervention_count,
                        "intervention_steps": intervention_steps,
                    }
                stats = {"environment": info}
                client.request("send-stats", stats)
                pbar.set_description(f"last return: {running_return}")
                if intvn_static_filtered_count > 0:
                    print(f"[ACTOR] Episode ended: filtered {intvn_static_filtered_count} static intervention frames")
                running_return = 0.0
                intervention_count = 0
                intervention_steps = 0
                already_intervened = False
                prev_intvn_action = None
                prev_intvn_was_static = False
                intvn_static_filtered_count = 0
                client.update()
                obs, _ = env.reset()

        if step > 0 and config.buffer_period > 0 and step % config.buffer_period == 0:
            buffer_path = os.path.join(FLAGS.checkpoint_path, "buffer")
            demo_buffer_path = os.path.join(FLAGS.checkpoint_path, "demo_buffer")
            if not os.path.exists(buffer_path):
                os.makedirs(buffer_path)
            if not os.path.exists(demo_buffer_path):
                os.makedirs(demo_buffer_path)
            with open(os.path.join(buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                pkl.dump(transitions, f)
                transitions = []
            with open(
                os.path.join(demo_buffer_path, f"transitions_{step}.pkl"), "wb"
            ) as f:
                pkl.dump(demo_transitions, f)
                demo_transitions = []

        timer.tock("total")

        if step % config.log_period == 0:
            stats = {"timer": timer.get_average_times()}
            client.request("send-stats", stats)


##############################################################################


def _compute_demo_ratio(step, pretrain_steps, initial_demo_ratio, final_demo_ratio, anneal_steps):
    """
    计算当前步数对应的 demo 采样比例。
    预训练阶段后，从 initial_demo_ratio 线性退火到 final_demo_ratio。
    """
    effective_step = step - pretrain_steps
    if effective_step <= 0:
        return initial_demo_ratio
    if effective_step >= anneal_steps:
        return final_demo_ratio
    progress = effective_step / anneal_steps
    return initial_demo_ratio + (final_demo_ratio - initial_demo_ratio) * progress


def learner(rng, agent, replay_buffer, demo_buffer, tb_logger=None):
    """
    Learner loop with demo pretraining and dynamic sampling ratio.
    """
    # ===================== 读取预训练配置 =====================
    pretrain_steps = getattr(config, 'pretrain_steps', 5000)
    initial_demo_ratio = getattr(config, 'initial_demo_ratio', 0.9)
    final_demo_ratio = getattr(config, 'final_demo_ratio', 0.5)
    demo_ratio_anneal_steps = getattr(config, 'demo_ratio_anneal_steps', 20000)

    start_step = 0
    if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path):
        latest_ckpt = checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))
        if latest_ckpt is not None:
            start_step = int(os.path.basename(latest_ckpt)[11:]) + 1
    step = start_step

    def stats_callback(type: str, payload: dict) -> dict:
        assert type == "send-stats", f"Invalid request type: {type}"
        if tb_logger is not None:
            tb_logger.log(payload, step=step)
        return {}

    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.register_data_store("actor_env_intvn", demo_buffer)
    server.start(threaded=True)

    if isinstance(agent, SACAgent):
        train_critic_networks_to_update = frozenset({"critic"})
        train_networks_to_update = frozenset({"critic", "actor", "temperature"})
    else:
        train_critic_networks_to_update = frozenset({"critic", "grasp_critic"})
        train_networks_to_update = frozenset({"critic", "grasp_critic", "actor", "temperature"})

    timer = Timer()

    # ===================== 阶段 1: 纯 Demo 预训练 =====================
    if start_step == 0 and pretrain_steps > 0:
        print_green("=" * 70)
        print_green(f"[PRETRAIN] 开始纯 Demo 预训练: {pretrain_steps} 步")
        print_green(f"[PRETRAIN] Demo buffer size: {len(demo_buffer)}")
        print_green(f"[PRETRAIN] 此阶段不启动 Actor，仅使用演示数据训练 Critic + Actor")
        print_green("=" * 70)

        pretrain_demo_iterator = demo_buffer.get_iterator(
            sample_args={
                "batch_size": config.batch_size,
                "pack_obs_and_next_obs": True,
            },
            device=sharding,
        )

        for pt_step in tqdm.tqdm(
            range(pretrain_steps), dynamic_ncols=True, desc="pretrain (demo only)"
        ):
            for critic_step in range(config.cta_ratio - 1):
                with timer.context("pretrain_sample"):
                    batch = next(pretrain_demo_iterator)
                with timer.context("pretrain_critics"):
                    agent, _ = agent.update(
                        batch,
                        networks_to_update=train_critic_networks_to_update,
                    )

            with timer.context("pretrain_joint"):
                batch = next(pretrain_demo_iterator)
                agent, pretrain_info = agent.update(
                    batch,
                    networks_to_update=train_networks_to_update,
                )

            if pt_step % config.log_period == 0 and tb_logger:
                tb_logger.log({"pretrain": pretrain_info}, step=pt_step)

            if pt_step % 500 == 0:
                critic_info = pretrain_info.get("critic", {})
                actor_info = pretrain_info.get("actor", {})
                critic_loss = critic_info.get("critic_loss", "N/A")
                actor_loss = actor_info.get("actor_loss", "N/A")
                print_green(
                    f"[PRETRAIN] step {pt_step}/{pretrain_steps} | "
                    f"critic_loss: {critic_loss} | actor_loss: {actor_loss}"
                )

        # 预训练完成后保存一个 checkpoint
        if FLAGS.checkpoint_path:
            os.makedirs(os.path.abspath(FLAGS.checkpoint_path), exist_ok=True)
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path), agent.state, step=0, keep=100
            )
            print_green(f"[PRETRAIN] Saved pretrain checkpoint to {FLAGS.checkpoint_path}")

        print_green("=" * 70)
        print_green("[PRETRAIN] 预训练完成！发布预训练参数给 Actor")
        print_green("=" * 70)
    else:
        if start_step > 0:
            print_green(f"[INFO] 从 checkpoint step {start_step} 恢复，跳过预训练")
        else:
            print_green(f"[INFO] pretrain_steps=0，跳过预训练阶段")

    # 发布（预训练后的）参数给 Actor
    server.publish_network(agent.state.params)
    print_green("sent initial network to actor (after pretraining)")

    # ===================== 阶段 2: 等待在线数据 =====================
    pbar = tqdm.tqdm(
        total=config.training_starts,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    while len(replay_buffer) < config.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)
    pbar.close()

    # ===================== 阶段 3: 正式训练（动态采样比例）=====================
    print_green("=" * 70)
    print_green(f"[TRAIN] 开始正式训练（动态采样比例）")
    print_green(f"[TRAIN] 初始 demo 比例: {initial_demo_ratio:.0%}")
    print_green(f"[TRAIN] 最终 demo 比例: {final_demo_ratio:.0%}")
    print_green(f"[TRAIN] 退火步数: {demo_ratio_anneal_steps}")
    print_green("=" * 70)

    for step in tqdm.tqdm(
        range(start_step, config.max_steps), dynamic_ncols=True, desc="learner"
    ):
        # 计算当前的 demo 采样比例
        demo_ratio = _compute_demo_ratio(
            step, pretrain_steps, initial_demo_ratio, final_demo_ratio, demo_ratio_anneal_steps
        )
        demo_batch_size = int(config.batch_size * demo_ratio)
        replay_batch_size = config.batch_size - demo_batch_size

        # 每次迭代都创建新的 iterator（batch_size 可能变化）
        # 为效率起见，只在 ratio 变化时重建
        if step == start_step or (step > start_step and step % 100 == 0):
            replay_iterator = replay_buffer.get_iterator(
                sample_args={
                    "batch_size": replay_batch_size,
                    "pack_obs_and_next_obs": True,
                },
                device=sharding,
            )
            demo_iterator = demo_buffer.get_iterator(
                sample_args={
                    "batch_size": demo_batch_size,
                    "pack_obs_and_next_obs": True,
                },
                device=sharding,
            )

        for critic_step in range(config.cta_ratio - 1):
            with timer.context("sample_replay_buffer"):
                batch = next(replay_iterator)
                demo_batch = next(demo_iterator)
                batch = concat_batches(batch, demo_batch, axis=0)

            with timer.context("train_critics"):
                agent, critics_info = agent.update(
                    batch,
                    networks_to_update=train_critic_networks_to_update,
                )

        with timer.context("train"):
            batch = next(replay_iterator)
            demo_batch = next(demo_iterator)
            batch = concat_batches(batch, demo_batch, axis=0)
            agent, update_info = agent.update(
                batch,
                networks_to_update=train_networks_to_update,
            )

        if step > 0 and step % (config.steps_per_update) == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)
            if step % 1000 == 0:
                print_green(
                    f"[LEARNER] step {step} | publishing params | "
                    f"demo_ratio: {demo_ratio:.2%} "
                    f"(demo_bs={demo_batch_size}, replay_bs={replay_batch_size})"
                )

        if step % config.log_period == 0 and tb_logger:
            tb_logger.log(update_info, step=step)
            tb_logger.log({"timer": timer.get_average_times()}, step=step)
            tb_logger.log({"demo_ratio": demo_ratio}, step=step)

        if (
            step > 0
            and config.checkpoint_period
            and step % config.checkpoint_period == 0
        ):
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path), agent.state, step=step, keep=100
            )


##############################################################################


def main(_):
    global config
    config = CONFIG_MAPPING[FLAGS.exp_name]()

    print_green(f"JAX devices: {jax.devices()}")
    print_green(f"JAX platform: {jax.devices()[0].platform if jax.devices() else 'No devices'}")
    print_green(f"Number of devices: {num_devices}")

    assert config.batch_size % num_devices == 0
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    use_fake_env = FLAGS.use_sim if FLAGS.actor else FLAGS.learner
    skip_server_connection = FLAGS.learner
    env = config.get_environment(
        fake_env=use_fake_env,
        save_video=FLAGS.save_video,
        classifier=not use_fake_env,
        isaac_server_url=FLAGS.isaac_server_url,
        skip_server_connection=skip_server_connection,
    )
    env = RecordEpisodeStatistics(env)

    rng, sampling_rng = jax.random.split(rng)

    if config.setup_mode == 'single-arm-fixed-gripper' or config.setup_mode == 'dual-arm-fixed-gripper' or config.setup_mode == 'single-arm-continuous-gripper':
        agent: SACAgent = make_sac_pixel_agent(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = False
    elif config.setup_mode == 'single-arm-learned-gripper':
        agent: SACAgentHybridSingleArm = make_sac_pixel_agent_hybrid_single_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = True
    elif config.setup_mode == 'dual-arm-learned-gripper':
        agent: SACAgentHybridDualArm = make_sac_pixel_agent_hybrid_dual_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = True
    else:
        raise NotImplementedError(f"Unknown setup mode: {config.setup_mode}")

    agent = jax.device_put(
        jax.tree.map(jnp.array, agent), sharding
    )

    if FLAGS.checkpoint_path is not None and os.path.exists(FLAGS.checkpoint_path):
        latest_ckpt = checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))
        if latest_ckpt is not None:
            input("Checkpoint path already exists. Press Enter to resume training.")
            ckpt = checkpoints.restore_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path),
                agent.state,
            )
            agent = agent.replace(state=ckpt)
            ckpt_number = os.path.basename(latest_ckpt)[11:]
            print_green(f"Loaded previous checkpoint at step {ckpt_number}.")
        else:
            print("[INFO] Checkpoint directory exists but no checkpoint files found. Starting fresh training.")

    def create_replay_buffer_and_tb_logger():
        replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
        )
        log_dir = os.path.join(
            FLAGS.checkpoint_path or "runs",
            "tensorboard",
        )
        if FLAGS.debug:
            tb_logger = None
        else:
            tb_logger = TensorBoardLogger(
                log_dir=log_dir,
                description=FLAGS.exp_name,
            )
        return replay_buffer, tb_logger

    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng, device=sharding)
        replay_buffer, tb_logger = create_replay_buffer_and_tb_logger()
        demo_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
        )

        assert FLAGS.demo_path is not None
        for path in FLAGS.demo_path:
            with open(path, "rb") as f:
                transitions = pkl.load(f)
                for transition in transitions:
                    if 'infos' in transition and 'grasp_penalty' in transition['infos']:
                        transition['grasp_penalty'] = transition['infos']['grasp_penalty']
                    demo_buffer.insert(transition)
        print_green(f"demo buffer size: {len(demo_buffer)}")
        print_green(f"online buffer size: {len(replay_buffer)}")

        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "buffer")
        ):
            for file in glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        replay_buffer.insert(transition)
            print_green(
                f"Loaded previous buffer data. Replay buffer size: {len(replay_buffer)}"
            )

        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "demo_buffer")
        ):
            for file in glob.glob(
                os.path.join(FLAGS.checkpoint_path, "demo_buffer/*.pkl")
            ):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        demo_buffer.insert(transition)
            print_green(
                f"Loaded previous demo buffer data. Demo buffer size: {len(demo_buffer)}"
            )

        print_green("starting learner loop (pretrain mode)")
        learner(
            sampling_rng,
            agent,
            replay_buffer,
            demo_buffer=demo_buffer,
            tb_logger=tb_logger,
        )

    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding)
        data_store = QueuedDataStore(50000)
        intvn_data_store = QueuedDataStore(50000)

        print_green("starting actor loop")
        actor(
            agent,
            data_store,
            intvn_data_store,
            env,
            sampling_rng,
        )

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    app.run(main)
