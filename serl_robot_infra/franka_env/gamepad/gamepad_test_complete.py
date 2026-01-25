"""
游戏手柄全面测试脚本
根据新映射方案测试所有6DOF控制和按钮功能

测试内容：
1. 初始值验证（不操作时所有输出应为0.0）
2. 位置控制测试（x, y, z）
3. 旋转控制测试（roll, pitch, yaw）
4. 组合控制测试（LT/LB组合控制z，RT/RB组合控制roll）
5. 按钮测试（A/B键用于夹爪）
6. 实时监控模式
"""
import sys
import os

# Add serl_robot_infra to path
script_dir = os.path.dirname(os.path.abspath(__file__))
serl_robot_infra_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
if os.path.exists(serl_robot_infra_dir) and serl_robot_infra_dir not in sys.path:
    sys.path.insert(0, serl_robot_infra_dir)

import time
import numpy as np
from franka_env.gamepad.gamepad_expert import GamepadExpert


def print_separator():
    """打印分隔线"""
    print("\n" + "=" * 80 + "\n")


def test_initial_values(gamepad):
    """测试1：验证初始值（不操作时所有输出应为0.0）"""
    print_separator()
    print("【测试1：初始值验证】")
    print("请确保手柄处于初始状态（不操作任何输入）")
    input("准备好后按 Enter 继续...")
    
    action, buttons = gamepad.get_action()
    
    print(f"\n当前输出值：")
    print(f"  x:     {action[0]:7.3f}  (预期: 0.000)")
    print(f"  y:     {action[1]:7.3f}  (预期: 0.000)")
    print(f"  z:     {action[2]:7.3f}  (预期: 0.000)")
    print(f"  roll:  {action[3]:7.3f}  (预期: 0.000)")
    print(f"  pitch: {action[4]:7.3f}  (预期: 0.000)")
    print(f"  yaw:   {action[5]:7.3f}  (预期: 0.000)")
    print(f"  按钮:  {buttons}")
    
    # 验证
    tolerance = 0.01
    all_zero = all(abs(a) < tolerance for a in action)
    
    if all_zero:
        print("\n✅ 通过：所有初始值接近0.0")
    else:
        print("\n❌ 失败：部分初始值不为0.0")
        for i, name in enumerate(['x', 'y', 'z', 'roll', 'pitch', 'yaw']):
            if abs(action[i]) >= tolerance:
                print(f"  ⚠️  {name} = {action[i]:.3f} (超出容差 {tolerance})")
    
    return all_zero


def test_position_control(gamepad):
    """测试2：位置控制（x, y, z）"""
    print_separator()
    print("【测试2：位置控制】")
    
    results = {}
    
    # 测试 x 平移
    print("\n【x 平移测试】")
    print("请将左摇杆向右推到底，然后按 Enter...")
    input()
    action, _ = gamepad.get_action()
    results['x_right'] = action[0]
    print(f"  输出: x = {action[0]:7.3f}  (预期: 接近 1.000)")
    
    print("请将左摇杆向左推到底，然后按 Enter...")
    input()
    action, _ = gamepad.get_action()
    results['x_left'] = action[0]
    print(f"  输出: x = {action[0]:7.3f}  (预期: 接近 -1.000)")
    
    # 测试 y 平移
    print("\n【y 平移测试】")
    print("请将左摇杆向前推到底，然后按 Enter...")
    input()
    action, _ = gamepad.get_action()
    results['y_forward'] = action[1]
    print(f"  输出: y = {action[1]:7.3f}  (预期: 接近 1.000，注意取反)")
    
    print("请将左摇杆向后拉到底，然后按 Enter...")
    input()
    action, _ = gamepad.get_action()
    results['y_backward'] = action[1]
    print(f"  输出: y = {action[1]:7.3f}  (预期: 接近 -1.000，注意取反)")
    
    # 测试 z 平移（LT + LB 组合）
    print("\n【z 平移测试 - LT + LB 组合】")
    print("请按下 LT（不按LB），然后按 Enter...")
    input()
    action, buttons = gamepad.get_action()
    results['z_down'] = action[2]
    print(f"  输出: z = {action[2]:7.3f}  (预期: 负值，向下)")
    print(f"  LT状态: 按下, LB状态: {buttons[4] if len(buttons) > 4 else 'N/A'}")
    
    print("请同时按下 LT 和 LB，然后按 Enter...")
    input()
    action, buttons = gamepad.get_action()
    results['z_up'] = action[2]
    print(f"  输出: z = {action[2]:7.3f}  (预期: 正值，向上)")
    print(f"  LT状态: 按下, LB状态: {buttons[4] if len(buttons) > 4 else 'N/A'}")
    
    # 验证结果
    print("\n【验证结果】")
    passed = 0
    total = 6
    
    if abs(results['x_right']) > 0.8:
        print("  ✅ x 向右: 通过")
        passed += 1
    else:
        print(f"  ❌ x 向右: 失败 (值: {results['x_right']:.3f})")
    
    if abs(results['x_left']) > 0.8 and results['x_left'] < 0:
        print("  ✅ x 向左: 通过")
        passed += 1
    else:
        print(f"  ❌ x 向左: 失败 (值: {results['x_left']:.3f})")
    
    if abs(results['y_forward']) > 0.8 and results['y_forward'] > 0:
        print("  ✅ y 向前: 通过")
        passed += 1
    else:
        print(f"  ❌ y 向前: 失败 (值: {results['y_forward']:.3f})")
    
    if abs(results['y_backward']) > 0.8 and results['y_backward'] < 0:
        print("  ✅ y 向后: 通过")
        passed += 1
    else:
        print(f"  ❌ y 向后: 失败 (值: {results['y_backward']:.3f})")
    
    if results['z_down'] < -0.5:
        print("  ✅ z 向下: 通过")
        passed += 1
    else:
        print(f"  ❌ z 向下: 失败 (值: {results['z_down']:.3f})")
    
    if results['z_up'] > 0.5:
        print("  ✅ z 向上: 通过")
        passed += 1
    else:
        print(f"  ❌ z 向上: 失败 (值: {results['z_up']:.3f})")
    
    print(f"\n通过: {passed}/{total}")
    return passed == total


def test_rotation_control(gamepad):
    """测试3：旋转控制（roll, pitch, yaw）"""
    print_separator()
    print("【测试3：旋转控制】")
    
    results = {}
    
    # 测试 yaw 旋转
    print("\n【yaw 旋转测试】")
    print("请将右摇杆向右推到底，然后按 Enter...")
    input()
    action, _ = gamepad.get_action()
    results['yaw_right'] = action[5]
    print(f"  输出: yaw = {action[5]:7.3f}  (预期: 接近 1.000)")
    
    print("请将右摇杆向左推到底，然后按 Enter...")
    input()
    action, _ = gamepad.get_action()
    results['yaw_left'] = action[5]
    print(f"  输出: yaw = {action[5]:7.3f}  (预期: 接近 -1.000)")
    
    # 测试 pitch 旋转
    print("\n【pitch 旋转测试】")
    print("请将右摇杆向上推到底，然后按 Enter...")
    input()
    action, _ = gamepad.get_action()
    results['pitch_up'] = action[4]
    print(f"  输出: pitch = {action[4]:7.3f}  (预期: 接近 1.000，注意取反)")
    
    print("请将右摇杆向下拉到底，然后按 Enter...")
    input()
    action, _ = gamepad.get_action()
    results['pitch_down'] = action[4]
    print(f"  输出: pitch = {action[4]:7.3f}  (预期: 接近 -1.000，注意取反)")
    
    # 测试 roll 旋转（RT + RB 组合）
    print("\n【roll 旋转测试 - RT + RB 组合】")
    print("请按下 RT（不按RB），然后按 Enter...")
    input()
    action, buttons = gamepad.get_action()
    results['roll_left'] = action[3]
    print(f"  输出: roll = {action[3]:7.3f}  (预期: 负值，向左)")
    print(f"  RT状态: 按下, RB状态: {buttons[5] if len(buttons) > 5 else 'N/A'}")
    
    print("请同时按下 RT 和 RB，然后按 Enter...")
    input()
    action, buttons = gamepad.get_action()
    results['roll_right'] = action[3]
    print(f"  输出: roll = {action[3]:7.3f}  (预期: 正值，向右)")
    print(f"  RT状态: 按下, RB状态: {buttons[5] if len(buttons) > 5 else 'N/A'}")
    
    # 验证结果
    print("\n【验证结果】")
    passed = 0
    total = 6
    
    if abs(results['yaw_right']) > 0.8:
        print("  ✅ yaw 向右: 通过")
        passed += 1
    else:
        print(f"  ❌ yaw 向右: 失败 (值: {results['yaw_right']:.3f})")
    
    if abs(results['yaw_left']) > 0.8 and results['yaw_left'] < 0:
        print("  ✅ yaw 向左: 通过")
        passed += 1
    else:
        print(f"  ❌ yaw 向左: 失败 (值: {results['yaw_left']:.3f})")
    
    if abs(results['pitch_up']) > 0.8 and results['pitch_up'] > 0:
        print("  ✅ pitch 向上: 通过")
        passed += 1
    else:
        print(f"  ❌ pitch 向上: 失败 (值: {results['pitch_up']:.3f})")
    
    if abs(results['pitch_down']) > 0.8 and results['pitch_down'] < 0:
        print("  ✅ pitch 向下: 通过")
        passed += 1
    else:
        print(f"  ❌ pitch 向下: 失败 (值: {results['pitch_down']:.3f})")
    
    if results['roll_left'] < -0.5:
        print("  ✅ roll 向左: 通过")
        passed += 1
    else:
        print(f"  ❌ roll 向左: 失败 (值: {results['roll_left']:.3f})")
    
    if results['roll_right'] > 0.5:
        print("  ✅ roll 向右: 通过")
        passed += 1
    else:
        print(f"  ❌ roll 向右: 失败 (值: {results['roll_right']:.3f})")
    
    print(f"\n通过: {passed}/{total}")
    return passed == total


def test_buttons(gamepad):
    """测试4：按钮控制（A/B键用于夹爪）"""
    print_separator()
    print("【测试4：按钮控制】")
    
    print("\n【A 键测试（关闭夹爪）】")
    print("请按下 A 键，然后按 Enter...")
    input()
    _, buttons = gamepad.get_action()
    a_pressed = buttons[0] if len(buttons) > 0 else 0
    print(f"  按钮状态: buttons[0] = {a_pressed}  (预期: 1)")
    
    print("\n【B 键测试（打开夹爪）】")
    print("请按下 B 键，然后按 Enter...")
    input()
    _, buttons = gamepad.get_action()
    b_pressed = buttons[1] if len(buttons) > 1 else 0
    print(f"  按钮状态: buttons[1] = {b_pressed}  (预期: 1)")
    
    # 验证
    passed = 0
    total = 2
    
    if a_pressed == 1:
        print("\n  ✅ A 键: 通过")
        passed += 1
    else:
        print(f"\n  ❌ A 键: 失败 (值: {a_pressed})")
    
    if b_pressed == 1:
        print("  ✅ B 键: 通过")
        passed += 1
    else:
        print(f"  ❌ B 键: 失败 (值: {b_pressed})")
    
    print(f"\n通过: {passed}/{total}")
    return passed == total


def realtime_monitor(gamepad):
    """实时监控模式：持续显示所有输入和输出值"""
    print_separator()
    print("【实时监控模式】")
    print("实时显示所有输入和输出值")
    print("按 Ctrl+C 退出监控\n")
    
    try:
        while True:
            action, buttons = gamepad.get_action()
            
            # 清屏
            print("\033[2J\033[H", end="")
            print("=" * 80)
            print("游戏手柄实时监控")
            print("=" * 80)
            print("\n【6DOF 输出值】")
            print(f"  x:     {action[0]:7.3f}")
            print(f"  y:     {action[1]:7.3f}")
            print(f"  z:     {action[2]:7.3f}")
            print(f"  roll:  {action[3]:7.3f}")
            print(f"  pitch: {action[4]:7.3f}")
            print(f"  yaw:   {action[5]:7.3f}")
            print(f"\n【按钮状态】")
            print(f"  A 键 (关闭夹爪): {buttons[0] if len(buttons) > 0 else 'N/A'}")
            print(f"  B 键 (打开夹爪): {buttons[1] if len(buttons) > 1 else 'N/A'}")
            print(f"  LB: {buttons[4] if len(buttons) > 4 else 'N/A'}")
            print(f"  RB: {buttons[5] if len(buttons) > 5 else 'N/A'}")
            print("\n按 Ctrl+C 退出监控")
            
            time.sleep(0.1)  # 100ms 更新间隔
            
    except KeyboardInterrupt:
        print("\n\n退出实时监控模式")


def main():
    """主测试流程"""
    print("=" * 80)
    print("游戏手柄全面测试")
    print("=" * 80)
    print("\n本测试将验证新映射方案的所有功能：")
    print("1. 初始值验证（不操作时输出应为0.0）")
    print("2. 位置控制测试（x, y, z）")
    print("3. 旋转控制测试（roll, pitch, yaw）")
    print("4. 按钮测试（A/B键）")
    print("5. 实时监控模式")
    print("\n请确保手柄已连接并处于 X 模式（XInput）")
    input("\n按 Enter 键开始测试...")
    
    try:
        # 初始化手柄
        print("\n正在初始化手柄...")
        gamepad = GamepadExpert(deadzone=0.0, sensitivity=1.0)
        print("✅ 手柄初始化成功！")
        
        # 运行测试
        test_results = {}
        
        # 测试1：初始值
        test_results['initial'] = test_initial_values(gamepad)
        
        # 测试2：位置控制
        test_results['position'] = test_position_control(gamepad)
        
        # 测试3：旋转控制
        test_results['rotation'] = test_rotation_control(gamepad)
        
        # 测试4：按钮
        test_results['buttons'] = test_buttons(gamepad)
        
        # 测试总结
        print_separator()
        print("【测试总结】")
        print(f"初始值验证:     {'✅ 通过' if test_results['initial'] else '❌ 失败'}")
        print(f"位置控制测试:   {'✅ 通过' if test_results['position'] else '❌ 失败'}")
        print(f"旋转控制测试:   {'✅ 通过' if test_results['rotation'] else '❌ 失败'}")
        print(f"按钮测试:       {'✅ 通过' if test_results['buttons'] else '❌ 失败'}")
        
        total_passed = sum(test_results.values())
        total_tests = len(test_results)
        print(f"\n总通过率: {total_passed}/{total_tests}")
        
        if total_passed == total_tests:
            print("\n🎉 所有测试通过！")
        else:
            print("\n⚠️  部分测试失败，请检查手柄映射或代码实现")
        
        # 询问是否进入实时监控模式
        print_separator()
        choice = input("是否进入实时监控模式？(y/n): ").strip().lower()
        if choice == 'y':
            realtime_monitor(gamepad)
        
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            gamepad.close()
        except:
            pass
        print("\n测试完成")


if __name__ == "__main__":
    main()