"""
Isaac Sim Gear 组装任务环境（高保真度版本）

继承自 IsaacSimFrankaEnv，添加 Gear 组装任务的特定逻辑和高保真度功能
包括：
- 重新抓取逻辑（regrasp）
- 任务特定的重置逻辑
- 域随机化支持（如果启用）
- 任务对象管理（对象应在 USD 场景文件中定义）

注意：任务对象（gear_medium、gear_base、gear_large）应该在 USD 场景文件中定义，
由 isaac_sim_server 加载，而不是在代码中创建。
"""

import copy
import numpy as np
import time
from typing import Dict, Tuple, Optional
from pynput import keyboard

from franka_env.envs.isaac_sim_env import IsaacSimFrankaEnv
from franka_env.envs.franka_env import DefaultEnvConfig
from franka_env.utils.rotations import euler_2_quat


class IsaacSimGearAssemblyEnvEnhanced(IsaacSimFrankaEnv):
    """
    Gear 组装任务的高保真度仿真环境
    
    继承自 IsaacSimFrankaEnv，添加：
    - 重新抓取逻辑（regrasp）
    - 任务特定的重置逻辑
    - Gear 对象位置管理
    - 域随机化支持（可选）
    
    注意：任务对象（gear_medium、gear_base、gear_large）应该在 USD 场景文件中定义，
    由 isaac_sim_server 加载，而不是在代码中创建。
    """
    
    def __init__(self, enable_domain_randomization: bool = False, skip_server_connection: bool = False, **kwargs):
        """
        初始化 Gear 组装任务环境
        
        Args:
            enable_domain_randomization: 是否启用域随机化（需要服务器端支持）
            **kwargs: 传递给基类的参数
        """
        # 任务对象引用（对象应在 USD 场景中定义）
        self.gear_medium = None
        self.gear_base = None
        self.gear_large = None
        
        # 重新抓取标志
        self.should_regrasp = False
        
        # 域随机化配置
        self.enable_domain_randomization = enable_domain_randomization
        self.domain_randomization_params = {}
        
        # Gear 与夹爪的约束状态
        self.gear_grasp_constraint = None

        # 成功持续状态跟踪（用于持续状态判断）
        # 需要连续成功保持2秒以上才判定为最终成功
        self.success_hold_duration_sec = 2.0  # 需要保持2秒
        self.success_hold_steps = None  # 将在初始化后根据控制频率计算
        self.consecutive_success_steps = 0  # 当前连续成功的步数
        self.success_confirmed = False  # 是否已确认成功（达到持续要求）

        # 调用基类初始化
        super().__init__(skip_server_connection=skip_server_connection, **kwargs)
        
        # 根据控制频率计算需要连续成功的步数
        # 默认 hz=10，即每步0.1秒，2秒需要20步
        if self.success_hold_steps is None:
            self.success_hold_steps = int(self.success_hold_duration_sec * self.hz)
            print(f"[INFO] Success hold requirement: {self.success_hold_steps} steps ({self.success_hold_duration_sec}s at {self.hz}Hz)")
        
        # 设置键盘监听（F1 键触发重新抓取）
        def on_press(key):
            if str(key) == "Key.f1":
                self.should_regrasp = True
        
        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        self.keyboard_listener = listener
    
    def _add_task_objects(self):
        """
        添加任务相关对象（gear_medium、gear_base、gear_large）
        
        注意：任务对象应该在 USD 场景文件中定义，由 isaac_sim_server 加载
        此方法保留为空，仅作为占位符（如果需要在环境初始化时执行某些操作）
        """
        # 任务对象（gear_medium、gear_base、gear_large）应该在 USD 场景文件中定义
        # 由 isaac_sim_server.py 在加载 USD 场景时自动创建
        # 这里不需要直接操作 Isaac Sim 对象
        
        # Gear 与夹爪的约束状态（用于跟踪，实际约束在服务器端管理）
        self.gear_grasp_constraint = None
        print("[INFO] Task objects should be defined in USD scene file")
        
        # 域随机化功能已关闭（根据项目需求）
        # 如果将来需要启用，可以取消下面的注释
        # if self.enable_domain_randomization:
        #     self._apply_domain_randomization()
    
    def _attach_gear_to_gripper(self):
        """
        建立 gear_medium 与夹爪的约束
        
        注意：约束管理应该在服务器端实现，或通过 HTTP 接口
        这里仅更新本地状态标记
        """
        # TODO: 如果服务器端提供了约束管理接口，可以通过 HTTP 调用
        # 例如：self.session.post(self.url + "attach_gear", ...)
        self.gear_grasp_constraint = True  # 标记为已连接
        print("[INFO] Gear medium attached to gripper (constraint managed by server)")
    
    def _detach_gear_from_gripper(self):
        """
        解除 gear_medium 与夹爪的约束
        
        注意：约束管理应该在服务器端实现，或通过 HTTP 接口
        这里仅更新本地状态标记
        """
        # TODO: 如果服务器端提供了约束管理接口，可以通过 HTTP 调用
        # 例如：self.session.post(self.url + "detach_gear", ...)
        self.gear_grasp_constraint = None
        print("[INFO] Gear medium detached from gripper (constraint managed by server)")
    
    def reset(self, **kwargs) -> Tuple[Dict, Dict]:
        """
        重置环境
        
        对于 Gear 组装任务：
        1. 重置机器人到 RESET_POSE
        2. 重置 gear_medium 位置（如果不在夹爪中）
        3. 重置场景（如果支持）
        """
        # 保存视频（如果需要）
        if self.save_video:
            self.save_video_recording()
        
        # 检查是否需要重新抓取
        if self.should_regrasp:
            self.regrasp()
            self.should_regrasp = False
        
        # 移动到重置位置
        self.go_to_reset(joint_reset=False)
        
        # [DRIFT FIX] Initialize last_commanded_pose to the commanded reset pose
        # Since we just moved to reset (using go_to_reset), we should anchor the control loop here.
        # Note: go_to_reset uses _get_reset_pose internally.
        self.last_commanded_pose = self._get_reset_pose()
        # DEBUG日志已关闭，如需调试请取消注释
        # print(f"[DEBUG-RESET] Config Pose: {self.config.RESET_POSE}")
        # print(f"[DEBUG-RESET] Calculated Reset Pose: {np.round(self.last_commanded_pose, 3)}")
        
        # 重置路径长度
        self.curr_path_length = 0
        
        # [CRITICAL] 重置成功持续状态（确保不影响下一轮）
        self.consecutive_success_steps = 0
        self.success_confirmed = False
        
        # [MODIFIED] Explicitly trigger server-side scene reset to apply domain randomization
        print("[INFO] GearEnv: Triggering Server Scene Reset (Randomized)...")
        self.reset_scene()
        
        # Call super().reset() to ensure robot moves to reset pose and returns observation
        obs, _ = super().reset(**kwargs)
        
        # Get observation again just in case (though super().reset() returns it)
        # obs = self._get_obs()
        
        # Reset terminate flag
        self.terminate = False
        
        return obs, {}
    
    def regrasp(self):
        """
        重新抓取 gear_medium（仿真版本）
        
        步骤：
        1. 移动到安全位置
        2. 解除 gear_medium 与夹爪的约束（如果存在）
        3. 重置 gear_medium 位置
        4. 移动到抓取位置上方
        5. 下降到抓取位置
        6. 关闭夹爪
        7. 建立 gear_medium 与夹爪的约束（如果夹爪已关闭）
        8. 移动到重置位置
        """
        # 1. 移动到安全位置（使用基类方法）
        self.go_to_reset(joint_reset=False)
        
        # 2. 解除 gear_medium 与夹爪的约束（如果存在）
        self._detach_gear_from_gripper()
        
        # 3. 重置 gear_medium 位置（通过 HTTP 接口，如果服务器支持）
        self._reset_gear_medium_to_holder()
        
        # 4. 移动到抓取位置上方
        grasp_pose = self.config.GRASP_POSE.copy()
        grasp_pose[2] += 0.05
        grasp_pose[0] += np.random.uniform(-0.005, 0.005)  # 添加随机化
        target_pose = np.concatenate([
            grasp_pose[:3],
            euler_2_quat(grasp_pose[3:])
        ])
        self._send_pos_command(target_pose)
        time.sleep(0.5)
        
        # 5. 下降到抓取位置
        grasp_pose[2] -= 0.05
        target_pose = np.concatenate([
            grasp_pose[:3],
            euler_2_quat(grasp_pose[3:])
        ])
        self._send_pos_command(target_pose)
        time.sleep(0.5)
        
        # 6. 关闭夹爪
        self._send_gripper_command(0.0)  # 关闭夹爪
        time.sleep(2.0)
        
        # 6.5. 建立 gear_medium 与夹爪的约束（如果夹爪已关闭）
        self._attach_gear_to_gripper()
        
        # 7. 移动到抓取位置上方
        grasp_pose[2] += 0.05
        target_pose = np.concatenate([
            grasp_pose[:3],
            euler_2_quat(grasp_pose[3:])
        ])
        self._send_pos_command(target_pose)
        time.sleep(0.2)
        
        # 8. 移动到重置位置
        reset_pose = self._get_reset_pose()
        # DEBUG日志已关闭，如需调试请取消注释
        # print(f"[VERIFY] Client Reset Pose: {np.round(reset_pose, 4)}")
        self._send_pos_command(reset_pose)
        time.sleep(0.5)
    
    def _reset_gear_medium_to_holder(self):
        """
        将 gear_medium 重置到支架位置
        
        注意：对象重置可以通过 `/reset_scene` 接口重置整个场景实现
        这里仅作为占位符
        """
        # TODO: 如果服务器端提供了对象重置接口，可以通过 HTTP 调用
        # 例如：
        # holder_position = self.config.GRASP_POSE[:3].copy()
        # try:
        #     self.session.post(
        #         self.url + "reset_object",
        #         json={"object_path": "/World/factory_gear_medium", "position": holder_position.tolist()},
        #         timeout=1.0
        #     )
        # except:
        #     pass
        
        # 或者使用场景重置接口
        # try:
        #     self.session.post(self.url + "reset_scene", timeout=1.0)
        # except:
        #     pass
        
        print("[INFO] Gear medium reset to holder (should be managed by server)")
    
    def go_to_reset(self, joint_reset=False):
        """
        移动到重置位置（重写基类方法，添加任务特定逻辑）
        
        Args:
            joint_reset: 是否执行关节重置（Isaac Sim 中可能不需要）
        """
        # 调用基类方法
        super().go_to_reset(joint_reset=joint_reset)
        
        # 可以在这里添加任务特定的重置逻辑
        # 例如：确保 gear_medium 在正确位置等
    
    def _apply_domain_randomization(self):
        """
        应用域随机化（如果启用）
        
        注意：域随机化功能已关闭（根据项目需求）
        此方法保留为空，仅作为占位符
        """
        if not self.enable_domain_randomization:
            return
        
        # 域随机化功能已关闭（根据项目需求）
        # 如果将来需要启用，可以取消下面的注释
        # 
        # self.domain_randomization_params = {
        #     'gear_mass': None,
        #     'gear_friction': None,
        #     'gear_scale': None,
        #     # ... 其他随机化参数
        # }
        # 
        # # 通过 HTTP 接口应用随机化（如果服务器支持）
        # try:
        #     self.session.post(
        #         self.url + "apply_domain_randomization",
        #         json=self.domain_randomization_params,
        #         timeout=1.0
        #     )
        # except:
        #     pass
        
        pass
    
    def _get_domain_randomization_ranges(self) -> Dict:
        """
        获取域随机化参数范围（如果启用）
        
        注意：域随机化功能已关闭（根据项目需求）
        此方法保留为空，仅作为占位符
        """
        # 域随机化功能已关闭（根据项目需求）
        # 如果将来需要启用，可以取消下面的注释
        # 
        # return {
        #     "gear_mass_range": (0.03, 0.05),  # 30-50g
        #     "gear_friction_range": (0.3, 0.5),
        #     "gear_size_variation": 0.001,  # ±1mm
        #     # ... 其他随机化范围
        # }
        
        return {}

    def compute_reward(self, obs: Dict) -> float:
        """
        计算奖励 (Ground Truth Reward) - Geometric Logic
        
        Constraints:
        1. Alignment: Gear Z-axis parallel to Base Z-axis (cos_theta > 0.97)
        2. Centering: Gear Origin XY close to Base Origin XY (dist_xy < 1cm)
           (Both Pin and Hole are at local (0.02, 0), so Origins should align)
        3. Insertion: Gear Z < 0.42 (World Frame)
        """
        try:
            # Check if we have object states from server
            # [DEBUG] Force print periodically to check data presence
            if hasattr(self, "_rt_debug_counter"):
                 self._rt_debug_counter += 1
            else:
                 self._rt_debug_counter = 0
            
            # DEBUG日志已关闭，如需调试请取消注释
            # if self._rt_debug_counter % 60 == 0:
            #      has_states = hasattr(self, "object_states") and bool(self.object_states)
            #      print(f"[REWARD-DEBUG] Step {self._rt_debug_counter}: HasStates={has_states}")
            #      if has_states:
            #          print(f" -> Keys: {list(self.object_states.keys())}")
            
            if hasattr(self, "object_states") and self.object_states:
                gear_state = self.object_states.get("gear_medium")
                base_state = self.object_states.get("gear_base")
                
                if gear_state is not None and base_state is not None:
                    # 1. Extract Poses
                    gear_pos = np.array(gear_state[:3])
                    gear_quat = np.array(gear_state[3:]) # xyzw
                    
                    base_pos = np.array(base_state[:3])
                    base_quat = np.array(base_state[3:]) # xyzw
                    
                    # 2. Check Z-Axis Alignment
                    # Quat to Rotation Matrix Z-vector
                    # R * [0,0,1]^T is the third column of R
                    from scipy.spatial.transform import Rotation as R
                    
                    # Scipy uses (x, y, z, w), my tracking sends (x, y, z, w)
                    r_gear = R.from_quat(gear_quat).as_matrix()
                    r_base = R.from_quat(base_quat).as_matrix()
                    
                    gear_z = r_gear[:, 2] # 3rd column
                    base_z = r_base[:, 2]
                    
                    # Dot product for alignment
                    dot_z = np.dot(gear_z, base_z)
                    # Threshold: < 5 degrees
                    # cos(5) ~= 0.99619
                    alignment_ok = dot_z > 0.996 
                    
                    # 3. Check XY Centering (Relative to Base Frame)
                    from scipy.spatial.transform import Rotation as R
                    
                    # A. Get Gear's Geometric Center (Hole) in World Frame
                    # Offset (0.02, 0, 0) in Gear Frame -> World Frame
                    gear_center_offset_local = np.array([0.02, 0.0, 0.0])
                    r_gear = R.from_quat(gear_quat)
                    gear_hole_world = gear_pos + r_gear.apply(gear_center_offset_local)
                    
                    # B. Transform Gear Hole into Base Local Frame
                    # P_local = R_base_inv * (P_world - P_base_origin)
                    r_base = R.from_quat(base_quat)
                    r_base_inv = r_base.inv()
                    
                    rel_pos = gear_hole_world - base_pos
                    hole_in_base_frame = r_base_inv.apply(rel_pos)
                    
                    # C. Check Proximity to Pin Location (0.02, 0.0, 0.0)
                    # User requirement: x in [0.018, 0.022]
                    # We also implictly check Y is close to 0 to ensure it's actually on the pin
                    
                    x_error = abs(hole_in_base_frame[0] - 0.02)
                    y_error = abs(hole_in_base_frame[1])
                    
                    # Threshold: 2mm tolerance on X (0.018-0.022) and Y
                    centering_ok = (x_error < 0.002) and (y_error < 0.002)
                    
                    # 4. Check Z Insertion
                    insertion_ok = gear_pos[2] < 0.402
                    
                    # 检查当前步是否满足所有成功条件
                    current_step_success = alignment_ok and centering_ok and insertion_ok
                    
                    # [持续状态判断] 跟踪连续成功的步数
                    if current_step_success:
                        self.consecutive_success_steps += 1
                        
                        # 检查是否达到持续成功要求（2秒 = success_hold_steps 步）
                        if self.consecutive_success_steps >= self.success_hold_steps:
                            # 首次达到持续要求时，确认成功并返回奖励
                            if not self.success_confirmed:
                                self.success_confirmed = True
                                print(f"\n[REWARD-SUCCESS] Success Confirmed (Held for {self.consecutive_success_steps} steps / {self.consecutive_success_steps / self.hz:.2f}s)!")
                        print(f"  -> Gear Z-Height: {gear_pos[2]:.4f} (Thresh: < 0.402)")
                        print(f"  -> Z-Alignment: {dot_z:.4f} (Thresh: > 0.996)")
                        print(f"  -> Hole in Base Frame: {hole_in_base_frame} (Target: [0.02, 0, 0])")
                        print(f"  -> Errors: X_err={x_error:.4f}, Y_err={y_error:.4f} (Thresh: 0.002)")
                        print(f"  -> Raw Gear Pos: {gear_pos}")
                        print(f"  -> Raw Base Pos: {base_pos}")
                        return 1.0
                        else:
                            # 正在累积成功步数，但尚未达到要求
                            # 可选：打印进度信息（每10步打印一次，避免刷屏）
                            if self.consecutive_success_steps % 10 == 0:
                                print(f"[REWARD-PROGRESS] Success holding: {self.consecutive_success_steps}/{self.success_hold_steps} steps ({self.consecutive_success_steps / self.hz:.2f}s / {self.success_hold_duration_sec:.2f}s)")
                            return 0.0
                    else:
                        # 当前步不满足成功条件，重置连续成功计数
                        if self.consecutive_success_steps > 0:
                            print(f"[REWARD-RESET] Success condition lost. Reset counter from {self.consecutive_success_steps} steps.")
                        self.consecutive_success_steps = 0
                        self.success_confirmed = False
                    return 0.0
                    
        except Exception as e:
            # print(f"[WARNING] GT Reward calculation failed: {e}")
            pass
            
        # Fallback to TCP logic if GT fails (e.g. at start)
        return super().compute_reward(obs)

