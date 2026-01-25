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
    
    def __init__(self, enable_domain_randomization: bool = False, **kwargs):
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
        
        # 调用基类初始化
        super().__init__(**kwargs)
        
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
        
        # 重置路径长度
        self.curr_path_length = 0
        
        # 更新当前位置
        self._update_currpos()
        
        # 重置 gear_medium 位置（如果不在夹爪中）
        # 注意：如果 gear_medium 被夹爪抓住，不应该重置位置
        if self.gear_grasp_constraint is None:
            self._reset_gear_medium_to_holder()
        
        # 获取观察
        obs = self._get_obs()
        
        # 重置终止标志
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
