#!/usr/bin/env python3
"""
从演示数据中读取一条成功轨迹，并用其action序列驱动Isaac Sim中的机械臂
用于验证演示数据的质量和轨迹的合理性
"""

import os
import pickle as pkl
import numpy as np
import time
from absl import app, flags
import glob
import cv2

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", "gear_assembly", "实验名称")
flags.DEFINE_string("demo_path", None, "演示数据路径（.pkl文件）")
flags.DEFINE_integer("traj_index", 0, "要回放的轨迹索引（0表示第一条成功轨迹）")
flags.DEFINE_string("isaac_server_url", None, "Isaac Sim server URL (e.g., http://192.168.31.198:5001/)")
flags.DEFINE_float("action_delay", 0.1, "每个action之间的延迟（秒），默认0.1秒（10Hz）")
flags.DEFINE_boolean("auto_reset", True, "是否在回放前自动重置环境")
flags.DEFINE_boolean("show_images", True, "是否显示观察图像（从演示数据中读取）")
flags.DEFINE_integer("image_display_size", 256, "图像显示尺寸（像素）")


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


def print_yellow(x):
    return print("\033[93m {}\033[00m".format(x))


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
            elif hasattr(info, 'get'):
                succeed = info.get('succeed', False)
            
            # 如果成功（reward > 0.5 或 succeed == True），保存轨迹
            if reward > 0.5 or succeed:
                trajectories.append(current_traj.copy())
                print(f"[INFO] 找到成功轨迹 #{len(trajectories)}: {len(current_traj)} 步, reward={reward}, succeed={succeed}")
            
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
            print(f"[INFO] 找到成功轨迹 #{len(trajectories)} (最后一条): {len(current_traj)} 步, reward={reward}, succeed={succeed}")
    
    return trajectories


def extract_actions_from_trajectory(trajectory):
    """
    从轨迹中提取action序列
    
    Args:
        trajectory: 轨迹列表，每个元素是一个transition字典
    
    Returns:
        list: action序列
    """
    actions = []
    for transition in trajectory:
        if 'actions' in transition:
            actions.append(np.array(transition['actions']))
    return actions


def extract_observations_from_trajectory(trajectory):
    """
    从轨迹中提取观察序列（包括图像）
    
    Args:
        trajectory: 轨迹列表，每个元素是一个transition字典
    
    Returns:
        list: 观察序列
    """
    observations = []
    for transition in trajectory:
        if 'observations' in transition:
            observations.append(transition['observations'])
    return observations


def display_demo_images(obs, display_size=256, step=None, debug=False):
    """
    显示演示数据中的图像观察
    
    支持两种观察格式：
    1. 嵌套格式：obs['images']['wrist_1'], obs['images']['wrist_2']
    2. 扁平格式：obs['wrist_1'], obs['wrist_2']（图像直接作为顶层键）
    
    Args:
        obs: 观察字典
        display_size: 图像显示尺寸
        step: 当前步数（用于在图像上显示）
        debug: 是否打印调试信息
    """
    # 尝试两种格式：先检查嵌套格式，再检查扁平格式
    images = None
    if 'images' in obs:
        # 嵌套格式：obs['images'] 是一个字典
        images_dict = obs['images']
        if isinstance(images_dict, dict):
            images = images_dict
            if debug:
                print(f"[DEBUG] 使用嵌套格式，找到'images'键")
    else:
        # 扁平格式：图像直接作为顶层键（如 'wrist_1', 'wrist_2'）
        # 识别图像键：排除 'state' 等非图像键
        image_keys_candidates = [k for k in obs.keys() if k not in ['state', 'observations', 'next_observations']]
        if len(image_keys_candidates) > 0:
            # 检查这些键是否包含numpy数组（可能是图像）
            images = {}
            for key in image_keys_candidates:
                val = obs[key]
                if isinstance(val, np.ndarray):
                    # 处理不同维度的图像数组
                    original_shape = val.shape
                    
                    # 如果是4维数组 (1, H, W, C)，去掉batch维度
                    if len(val.shape) == 4 and val.shape[0] == 1:
                        val = np.squeeze(val, axis=0)  # 去掉第一个维度
                        if debug:
                            print(f"[DEBUG] 图像 {key} 从4维 {original_shape} 压缩为3维 {val.shape}")
                    
                    # 检查是否是有效的图像格式
                    if len(val.shape) >= 2:
                        # 进一步检查：如果是3维，最后一维应该是通道数（通常是3）
                        # 如果是2维，可能是灰度图，我们也接受
                        if len(val.shape) == 3 and val.shape[-1] in [1, 3, 4]:
                            images[key] = val
                        elif len(val.shape) == 2:
                            # 灰度图，转换为3通道
                            images[key] = val
                        elif len(val.shape) == 3:
                            # 可能是其他格式，也接受
                            images[key] = val
                        elif len(val.shape) == 4:
                            # 如果仍然是4维，可能是 (B, H, W, C)，取第一个
                            if val.shape[0] > 0:
                                images[key] = val[0]
            if len(images) > 0:
                if debug:
                    print(f"[DEBUG] 使用扁平格式，找到图像键: {list(images.keys())}")
                    for key in images.keys():
                        print(f"[DEBUG]   图像 {key}: shape={images[key].shape}, dtype={images[key].dtype}")
    
    if images is None or len(images) == 0:
        if debug:
            print(f"[DEBUG] 未找到图像数据，观察的键: {list(obs.keys())}")
        return False
    
    # 收集所有图像
    display_images = []
    image_keys = []
    
    for key, img in images.items():
        if not isinstance(img, np.ndarray):
            if debug:
                print(f"[DEBUG] 图像 {key} 不是numpy数组，类型: {type(img)}")
            continue
        
        if debug and step == 1:
            print(f"[DEBUG] 图像 {key}: shape={img.shape}, dtype={img.dtype}, min={img.min()}, max={img.max()}")
        
        # 确保图像是uint8格式
        if img.dtype != np.uint8:
            # 如果是浮点数，假设范围是[0, 1]
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        
        # 处理不同形状的图像
        if len(img.shape) == 3:
            # 3维图像 (H, W, C)
            if img.shape[-1] == 3:
                # RGB图像
                resized = cv2.resize(img, (display_size, display_size))
                # RGB转BGR用于cv2显示
                img_bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
            elif img.shape[-1] == 1:
                # 单通道图像，转换为3通道
                resized = cv2.resize(img, (display_size, display_size))
                img_bgr = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
            elif img.shape[-1] == 4:
                # RGBA图像，转换为RGB再转BGR
                resized = cv2.resize(img, (display_size, display_size))
                rgb = cv2.cvtColor(resized, cv2.COLOR_RGBA2RGB)
                img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            else:
                if debug:
                    print(f"[DEBUG] 图像 {key} 通道数不支持: {img.shape[-1]}, shape: {img.shape}")
                continue
        elif len(img.shape) == 2:
            # 2维灰度图，转换为3通道
            resized = cv2.resize(img, (display_size, display_size))
            img_bgr = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        else:
            if debug:
                print(f"[DEBUG] 图像 {key} 维度不支持: {len(img.shape)}, shape: {img.shape}")
            continue
        
        # 添加文本标签（相机名称）
        cv2.putText(
            img_bgr, 
            key, 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 0), 
            2
        )
        
        # 如果指定了步数，也显示步数
        if step is not None:
            cv2.putText(
                img_bgr, 
                f"Step: {step}", 
                (10, display_size - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (255, 255, 255), 
                2
            )
        
        display_images.append(img_bgr)
        image_keys.append(key)
    
    if len(display_images) > 0:
        # 水平拼接所有图像
        if len(display_images) == 1:
            combined = display_images[0]
        else:
            combined = np.concatenate(display_images, axis=1)
        
        # 显示图像（已经是BGR格式）
        window_name = "Demo Images (Press 'q' to quit, Space to pause)"
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # 创建窗口（如果不存在）
            cv2.imshow(window_name, combined)
            cv2.waitKey(1)  # 非阻塞，允许其他处理继续
            
            if debug:
                print_green(f"[DEBUG] 成功显示 {len(display_images)} 张图像: {image_keys}")
                print_green(f"[DEBUG] 窗口名称: {window_name}")
                print_green(f"[DEBUG] 图像尺寸: {combined.shape}")
                print_green(f"[DEBUG] 图像数据类型: {combined.dtype}")
                print_green(f"[DEBUG] 图像值范围: [{combined.min()}, {combined.max()}]")
            
            return True  # 返回True表示成功显示
        except Exception as e:
            if debug:
                print_yellow(f"[DEBUG] 显示图像时出错: {e}")
                import traceback
                traceback.print_exc()
            return False
    else:
        if debug:
            print_yellow(f"[DEBUG] 没有可显示的图像，display_images长度: {len(display_images)}")
        
        return False  # 返回False表示没有显示


def replay_trajectory(env, trajectory, action_delay=0.1, show_images=True, image_display_size=256):
    """
    在环境中回放轨迹，显示演示数据中的观察图像
    
    Args:
        env: Gym环境
        trajectory: 轨迹列表，包含observations和actions
        action_delay: 每个action之间的延迟（秒）
        show_images: 是否显示图像
        image_display_size: 图像显示尺寸
    """
    print_green(f"[REPLAY] 开始回放轨迹，共 {len(trajectory)} 个transitions")
    
    # 提取actions和observations
    actions = extract_actions_from_trajectory(trajectory)
    observations = extract_observations_from_trajectory(trajectory)
    
    obs, info = env.reset()
    print_green(f"[REPLAY] 环境已重置")
    
    # 检查初始状态差异
    if len(observations) > 0:
        initial_demo_obs = observations[0]
        if 'state' in initial_demo_obs and 'state' in obs:
            demo_initial_state = initial_demo_obs['state']
            env_initial_state = obs['state']
            if isinstance(demo_initial_state, np.ndarray) and isinstance(env_initial_state, np.ndarray):
                if demo_initial_state.shape == env_initial_state.shape:
                    initial_diff = np.linalg.norm(demo_initial_state - env_initial_state)
                    print_yellow(f"[REPLAY] 初始状态差异: {initial_diff:.6f}")
                    if initial_diff > 0.1:
                        print_yellow(f"[WARNING] 初始状态差异较大（>{initial_diff:.3f}），可能影响后续回放准确性")
    
    if show_images:
        print_green(f"[REPLAY] 图像显示已启用（窗口将显示演示数据中的图像）")
        print_yellow(f"[REPLAY] 提示: 按'q'键退出，按空格键暂停/继续")
        print_yellow(f"[REPLAY] 注意: 演示图像窗口名称: 'Demo Images (Press 'q' to quit, Space to pause)'")
        print_yellow(f"[REPLAY] 如果看不到窗口，请检查是否被其他窗口遮挡")
    
    paused = False
    demo_window_created = False
    
    for step, transition in enumerate(trajectory):
        # 检查暂停
        if show_images:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print_yellow(f"[REPLAY] 用户按'q'键退出")
                break
            elif key == ord(' '):
                paused = not paused
                if paused:
                    print_yellow(f"[REPLAY] 已暂停，按空格键继续")
                else:
                    print_green(f"[REPLAY] 继续回放")
        
        if paused:
            time.sleep(0.1)
            continue
        
        action = transition.get('actions', None)
        if action is None:
            continue
        
        action = np.array(action)
        
        # 显示演示数据中的观察图像（在发送action之前）
        if show_images and step < len(observations):
            demo_obs = observations[step]
            # 在前几步打印调试信息
            debug = (step < 3 and not demo_window_created)
            result = display_demo_images(demo_obs, display_size=image_display_size, step=step + 1, debug=debug)
            if result:
                demo_window_created = True
            elif step < 3:
                print_yellow(f"[REPLAY] Step {step + 1}: 未能显示演示图像，请检查观察数据格式")
        
        print(f"[REPLAY] Step {step + 1}/{len(trajectory)}: Action = {np.round(action, 4)}")
        
        try:
            # 1. 比较执行action之前的状态（正确的时序对齐）
            if step < len(observations):
                demo_obs = observations[step]
                if 'state' in demo_obs and 'state' in obs:
                    demo_state = demo_obs['state']
                    env_state = obs['state']
                    
                    # 处理不同的state格式
                    if isinstance(demo_state, np.ndarray) and isinstance(env_state, np.ndarray):
                        if demo_state.shape == env_state.shape:
                            state_diff_before = np.linalg.norm(demo_state - env_state)
                            mean_abs_diff = np.mean(np.abs(demo_state - env_state))
                            max_abs_diff = np.max(np.abs(demo_state - env_state))
                            if step % 50 == 0:  # 每50步打印一次
                                print(f"  State差异 (执行前): norm={state_diff_before:.6f}, mean_abs={mean_abs_diff:.6f}, max_abs={max_abs_diff:.6f}")
            
            # 2. 发送action到环境
            next_obs, reward, done, truncated, info = env.step(action)
            
            # 3. 打印一些状态信息
            if 'state' in next_obs:
                state = next_obs['state']
                # 处理不同的观察结构：可能是字典或numpy数组
                if isinstance(state, dict):
                    tcp_pose = state.get('tcp_pose', None)
                    if tcp_pose is not None:
                        tcp_pose = np.array(tcp_pose)
                        print(f"  TCP位置: {np.round(tcp_pose[:3], 3)}")
                elif isinstance(state, np.ndarray):
                    # 如果state是numpy数组，可能是经过包装器处理后的结构
                    # 尝试直接使用state的前3个元素作为位置
                    if len(state) >= 3:
                        print(f"  TCP位置: {np.round(state[:3], 3)}")
            
            # 4. 比较执行action之后的状态（正确的时序对齐）
            if 'next_observations' in transition:
                demo_next_obs = transition['next_observations']
                if 'state' in demo_next_obs and 'state' in next_obs:
                    demo_next_state = demo_next_obs['state']
                    env_next_state = next_obs['state']
                    
                    # 处理不同的state格式
                    if isinstance(demo_next_state, np.ndarray) and isinstance(env_next_state, np.ndarray):
                        if demo_next_state.shape == env_next_state.shape:
                            state_diff_after = np.linalg.norm(demo_next_state - env_next_state)
                            mean_abs_diff = np.mean(np.abs(demo_next_state - env_next_state))
                            max_abs_diff = np.max(np.abs(demo_next_state - env_next_state))
                            if step % 50 == 0:  # 每50步打印一次
                                print(f"  State差异 (执行后): norm={state_diff_after:.6f}, mean_abs={mean_abs_diff:.6f}, max_abs={max_abs_diff:.6f}")
            
            # 5. 更新obs为next_obs，准备下一次循环
            obs = next_obs
            
            if reward > 0:
                print_green(f"  [SUCCESS] 任务完成！Reward = {reward}")
            
            if done or truncated:
                print_yellow(f"  [DONE] Episode结束 (done={done}, truncated={truncated})")
                break
            
            # 延迟
            time.sleep(action_delay)
            
        except Exception as e:
            print(f"[ERROR] Step {step + 1} 执行失败: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # 关闭图像窗口
    if show_images:
        cv2.destroyAllWindows()
    
    print_green(f"[REPLAY] 轨迹回放完成")


def main(_):
    # 1. 检查实验配置
    assert FLAGS.exp_name in CONFIG_MAPPING, f"实验 '{FLAGS.exp_name}' 未找到"
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    
    # 2. 加载演示数据
    if FLAGS.demo_path:
        demo_path = FLAGS.demo_path
    else:
        # 默认从 demo_data 目录加载
        demo_paths = glob.glob(os.path.join(os.getcwd(), "demo_data", "*.pkl"))
        if len(demo_paths) == 0:
            print_yellow(f"[ERROR] 未找到演示数据文件")
            return
        demo_path = demo_paths[0]
        print_green(f"[INFO] 使用默认演示数据: {demo_path}")
    
    print_green(f"[INFO] 加载演示数据: {demo_path}")
    with open(demo_path, "rb") as f:
        transitions = pkl.load(f)
    
    print_green(f"[INFO] 演示数据包含 {len(transitions)} 个transitions")
    
    # 3. 找到成功轨迹
    print_green(f"[INFO] 正在查找成功轨迹...")
    trajectories = find_successful_trajectories(transitions)
    
    if len(trajectories) == 0:
        print_yellow(f"[ERROR] 未找到成功轨迹！")
        print_yellow(f"[INFO] 请检查演示数据中是否有 reward > 0.5 或 succeed == True 的轨迹")
        return
    
    print_green(f"[INFO] 找到 {len(trajectories)} 条成功轨迹")
    
    # 4. 选择要回放的轨迹
    if FLAGS.traj_index >= len(trajectories):
        print_yellow(f"[WARNING] 轨迹索引 {FLAGS.traj_index} 超出范围，使用索引 0")
        FLAGS.traj_index = 0
    
    selected_traj = trajectories[FLAGS.traj_index]
    print_green(f"[INFO] 选择轨迹 {FLAGS.traj_index}，包含 {len(selected_traj)} 个transitions")
    
    # 5. 提取action和observation序列
    actions = extract_actions_from_trajectory(selected_traj)
    observations = extract_observations_from_trajectory(selected_traj)
    print_green(f"[INFO] 提取了 {len(actions)} 个actions, {len(observations)} 个observations")
    
    # 打印action统计信息
    if len(actions) > 0:
        actions_array = np.array(actions)
        print(f"[INFO] Action统计:")
        print(f"  形状: {actions_array.shape}")
        print(f"  范围: [{actions_array.min():.4f}, {actions_array.max():.4f}]")
        print(f"  均值: {actions_array.mean(axis=0)}")
        print(f"  范数均值: {np.linalg.norm(actions_array, axis=1).mean():.4f}")
    
    # 检查观察中是否包含图像（支持嵌套和扁平两种格式）
    has_images = False
    image_keys_found = []
    if len(observations) > 0:
        first_obs = observations[0]
        print_green(f"[INFO] 第一个观察的键: {list(first_obs.keys())}")
        
        # 检查嵌套格式
        if 'images' in first_obs:
            images = first_obs['images']
            print_green(f"[INFO] 找到'images'键（嵌套格式），类型: {type(images)}")
            if isinstance(images, dict):
                image_keys_found = list(images.keys())
                has_images = len(image_keys_found) > 0
                print_green(f"[INFO] 观察中包含图像（嵌套格式）: {image_keys_found}")
        else:
            # 检查扁平格式：查找可能是图像的键（排除'state'等）
            candidate_keys = [k for k in first_obs.keys() if k not in ['state', 'observations', 'next_observations']]
            for key in candidate_keys:
                val = first_obs[key]
                if isinstance(val, np.ndarray) and len(val.shape) >= 2:
                    # 可能是图像
                    image_keys_found.append(key)
            if len(image_keys_found) > 0:
                has_images = True
                print_green(f"[INFO] 观察中包含图像（扁平格式）: {image_keys_found}")
        
        # 检查第一张图像的详细信息
        if has_images and len(image_keys_found) > 0:
            first_key = image_keys_found[0]
            if 'images' in first_obs and isinstance(first_obs['images'], dict):
                first_img = first_obs['images'][first_key]
            else:
                first_img = first_obs[first_key]
            
            print_green(f"[INFO] 图像 '{first_key}' 详细信息:")
            print_green(f"  类型: {type(first_img)}")
            if isinstance(first_img, np.ndarray):
                print_green(f"  Shape: {first_img.shape}")
                print_green(f"  Dtype: {first_img.dtype}")
                print_green(f"  Min: {first_img.min()}, Max: {first_img.max()}")
    
    if FLAGS.show_images and not has_images:
        print_yellow(f"[WARNING] --show_images=True 但观察中不包含图像，将禁用图像显示")
        FLAGS.show_images = False
    elif has_images:
        print_green(f"[INFO] 图像显示已启用，将显示 {len(image_keys_found)} 个相机图像: {image_keys_found}")
    
    # 6. 初始化环境
    print_green(f"[INFO] 初始化Isaac Sim环境...")
    
    # 确定Isaac Sim服务器URL
    if FLAGS.isaac_server_url is None:
        # 从config中获取
        try:
            from examples.experiments.gear_assembly.config import IsaacSimEnvConfig
            isaac_config = IsaacSimEnvConfig()
            isaac_server_url = isaac_config.SERVER_URL
        except:
            print_yellow(f"[ERROR] 未提供Isaac Sim服务器URL，且无法从config获取")
            print_yellow(f"[INFO] 请使用 --isaac_server_url 参数指定服务器URL")
            print_yellow(f"[INFO] 例如: --isaac_server_url=http://192.168.31.198:5001/")
            return
    else:
        isaac_server_url = FLAGS.isaac_server_url
    
    # 确保URL以/结尾
    if not isaac_server_url.endswith('/'):
        isaac_server_url += '/'
    
    print_green(f"[INFO] Isaac Sim服务器URL: {isaac_server_url}")
    
    # 创建环境
    env = config.get_environment(
        fake_env=True,  # 使用仿真环境
        save_video=False,
        classifier=False,
        isaac_server_url=isaac_server_url,
        skip_server_connection=False,  # 需要连接服务器
    )
    
    print_green(f"[INFO] 环境初始化完成")
    
    # 7. 回放轨迹
    try:
        replay_trajectory(
            env, 
            selected_traj,  # 传递完整轨迹，包含observations和actions
            action_delay=FLAGS.action_delay,
            show_images=FLAGS.show_images,
            image_display_size=FLAGS.image_display_size
        )
    except KeyboardInterrupt:
        print_yellow(f"\n[INFO] 用户中断回放")
    except Exception as e:
        print_yellow(f"\n[ERROR] 回放过程中出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if FLAGS.show_images:
            cv2.destroyAllWindows()
        env.close()
        print_green(f"[INFO] 环境已关闭")


if __name__ == "__main__":
    app.run(main)
