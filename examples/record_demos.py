import os
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
from absl import app, flags
import time

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 20, "Number of successful demos to collect.")
flags.DEFINE_float("success_sleep_sec", 2.0, "Seconds to wait after a success before reset (0 to disable). Was 10s, reduce to avoid long freeze.")
flags.DEFINE_boolean("fake_env", False, "Use Isaac Sim simulation environment.")

def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=FLAGS.fake_env, save_video=False, classifier=False)
    
    obs, info = env.reset()
    # print(f"[VERIFY] Post-Reset State: {np.round(obs['state'][0], 4)}")
    print("Reset done")
    transitions = []
    success_count = 0
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    trajectory = []
    returns = 0
    
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
        if done:
            if info["succeed"]:
                for transition in trajectory:
                    transitions.append(copy.deepcopy(transition))
                success_count += 1
                pbar.update(1)
                
                if FLAGS.success_sleep_sec > 0:
                    print(f"[INFO] Success! Waiting {FLAGS.success_sleep_sec}s before reset (set --success_sleep_sec=0 to skip).")
                    time.sleep(FLAGS.success_sleep_sec)
            else:
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]}] [DemoRecorder] [FAIL] Episode failed or timed out. Discarding trajectory.")

            trajectory = []
            returns = 0
            obs, info = env.reset() 
            
    if not os.path.exists("./demo_data"):
        os.makedirs("./demo_data")
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"./demo_data/{FLAGS.exp_name}_{success_needed}_demos_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(transitions, f)
        print(f"saved {success_needed} demos to {file_name}")

if __name__ == "__main__":
    app.run(main)