try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp

import os

# 启动 SimulationApp
config = {"headless": True}
print("Starting SimulationApp...")
simulation_app = SimulationApp(config)
print("SimulationApp started.")

# 必须在启动后导入
import omni.usd
from omni.isaac.core.utils.stage import get_current_stage
from pxr import UsdGeom

# 加载 USD 场景
usd_path = "examples/experiments/gear_assembly/HIL_franka_gear.usda"
# Ensure absolute path or correct relative path
usd_path = os.path.abspath(usd_path)
print(f"[INFO] Loading USD scene from: {usd_path}")

try:
    if not os.path.exists(usd_path):
        print(f"[ERROR] USD file does not exist: {usd_path}")
        exit(1)

    usd_context = omni.usd.get_context()
    usd_context.open_stage(usd_path)
    print("[INFO] USD scene loaded successfully")
    
    # 获取 stage
    stage = get_current_stage()
    
    # 检查机器人
    robot_prim = stage.GetPrimAtPath("/World/franka")
    if robot_prim.IsValid():
        print("[INFO] ✅ Robot prim found at /World/franka")
    else:
        print("[ERROR] ❌ Robot prim not found at /World/franka")
    
    # 检查相机
    camera1_prim = stage.GetPrimAtPath("/World/franka/panda_hand/wrist_1")
    camera2_prim = stage.GetPrimAtPath("/World/franka/panda_hand/wrist_2")
    if camera1_prim.IsValid():
        print("[INFO] ✅ Camera 'wrist_1' found at /World/franka/panda_hand/wrist_1")
    else:
        print("[ERROR] ❌ Camera 'wrist_1' not found")
    if camera2_prim.IsValid():
        print("[INFO] ✅ Camera 'wrist_2' found at /World/franka/panda_hand/wrist_2")
    else:
        print("[ERROR] ❌ Camera 'wrist_2' not found")
    
    # 检查任务对象
    gear_medium_prim = stage.GetPrimAtPath("/World/factory_gear_medium")
    gear_base_prim = stage.GetPrimAtPath("/World/factory_gear_base")
    gear_large_prim = stage.GetPrimAtPath("/World/factory_gear_base") # Checking base again if large not found or typo? Wait, code says large.
    
    # Correcting logic based on my memory of previous file content.
    # Checking lines 126-143 of Step 11
    gear_medium_prim = stage.GetPrimAtPath("/World/factory_gear_medium")
    gear_base_prim = stage.GetPrimAtPath("/World/factory_gear_base")
    gear_large_prim = stage.GetPrimAtPath("/World/factory_gear_large")
    
    if gear_medium_prim.IsValid():
        print("[INFO] ✅ Gear medium found at /World/factory_gear_medium")
    else:
        print("[ERROR] ❌ Gear medium not found")
    
    if gear_base_prim.IsValid():
        print("[INFO] ✅ Gear base found at /World/factory_gear_base")
    else:
        print("[ERROR] ❌ Gear base not found")
    
    if gear_large_prim.IsValid():
        print("[INFO] ✅ Gear large found at /World/factory_gear_large")
    else:
        print("[ERROR] ❌ Gear large not found")
    
    print("[INFO] USD scene validation completed")
        
except Exception as e:
    print(f"[ERROR] Failed to load USD scene: {e}")
    import traceback
    traceback.print_exc()
    raise

# 关闭应用
simulation_app.close()
print("[INFO] Test completed")
