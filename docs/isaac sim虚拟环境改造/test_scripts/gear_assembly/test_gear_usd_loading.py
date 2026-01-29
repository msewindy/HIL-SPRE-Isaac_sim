try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp

import os

# 启动 SimulationApp
config = {"headless": True}
simulation_app = SimulationApp(config)

# 必须在启动后导入
import omni.usd
from omni.isaac.core.utils.stage import get_current_stage
from pxr import UsdGeom

# 加载 USD 场景（修改为 gear_assembly 的 USD 文件）
usd_path = "examples/experiments/gear_assembly/HIL_franka_gear.usda"
usd_path = os.path.abspath(usd_path)
print(f"[INFO] Loading USD scene from: {usd_path}")

try:
    usd_context = omni.usd.get_context()
    usd_context.open_stage(usd_path)
    print("[INFO] USD scene loaded successfully")
    
    # 验证场景内容
    stage = get_current_stage()
    
    # 检查机器人
    robot_prim = stage.GetPrimAtPath("/World/franka")
    if robot_prim.IsValid():
        print("[INFO] Robot prim found at /World/franka")
    else:
        print("[ERROR] Robot prim not found!")
    
    # 检查相机
    camera1_prim = stage.GetPrimAtPath("/World/franka/panda_hand/wrist_1")
    camera2_prim = stage.GetPrimAtPath("/World/franka/panda_hand/wrist_2")
    if camera1_prim.IsValid() and camera2_prim.IsValid():
        print("[INFO] Cameras found")
    else:
        print("[ERROR] Cameras not found!")
    
    # 检查任务对象（修改为 gear 相关对象）
    gear_medium_prim = stage.GetPrimAtPath("/World/factory_gear_medium")
    gear_base_prim = stage.GetPrimAtPath("/World/factory_gear_base")
    gear_large_prim = stage.GetPrimAtPath("/World/factory_gear_large")
    
    if gear_medium_prim.IsValid():
        print("[INFO] Gear medium found")
    else:
        print("[ERROR] Gear medium not found!")
    
    if gear_base_prim.IsValid():
        print("[INFO] Gear base found")
    else:
        print("[ERROR] Gear base not found!")
    
    if gear_large_prim.IsValid():
        print("[INFO] Gear large found")
    else:
        print("[ERROR] Gear large not found!")
        
except Exception as e:
    print(f"[ERROR] Failed to load USD scene: {e}")
    raise

# 关闭应用
simulation_app.close()
print("[INFO] Test completed")
