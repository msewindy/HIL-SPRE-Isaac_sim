#!/usr/bin/env python3
"""
完整的手柄按钮映射测试脚本

使用方法：
1. 连接手柄
2. 运行脚本：python test_gamepad_buttons.py
3. 依次按下各个按钮，查看映射关系
4. 特别关注 Back/Select 键的索引
"""

import pygame
import sys
import time
import os

# Xbox 360/One 标准按钮映射（参考）
XBOX_BUTTON_MAP = {
    0: "A (关闭夹爪)",
    1: "B (打开夹爪)",
    2: "X",
    3: "Y (场景重置)",
    4: "LB (左肩键)",
    5: "RB (右肩键)",
    6: "Back/Select (精准模式)",
    7: "Start",
    8: "Left Stick Press (左摇杆按下)",
    9: "Right Stick Press (右摇杆按下)",
}

def main():
    # 初始化 pygame
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    pygame.joystick.init()
    
    # 检查是否有手柄连接
    joystick_count = pygame.joystick.get_count()
    if joystick_count == 0:
        print("[ERROR] 未检测到手柄！请连接手柄后重试。")
        return
    
    # 使用第一个手柄
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    
    print("=" * 70)
    print(f"手柄信息:")
    print(f"  名称: {joystick.get_name()}")
    print(f"  轴数量: {joystick.get_numaxes()}")
    print(f"  按钮数量: {joystick.get_numbuttons()}")
    print("=" * 70)
    print("\n[测试说明]")
    print("1. 按下各个按钮，查看其索引和功能")
    print("2. 特别关注 Back/Select 键的索引")
    print("3. 按 Ctrl+C 退出")
    print("=" * 70)
    
    # 显示已知的按钮映射
    print("\n[已知的 Xbox 按钮映射（仅供参考）]")
    for btn_id, btn_name in XBOX_BUTTON_MAP.items():
        if btn_id < joystick.get_numbuttons():
            print(f"  按钮 {btn_id}: {btn_name}")
    
    print("\n" + "=" * 70)
    print("开始检测... (按下按钮或移动摇杆查看实时状态)\n")
    print("[提示] 移动左摇杆和右摇杆，查看轴的变化")
    print("[提示] 按下左摇杆和右摇杆，查看按钮索引\n")
    
    try:
        last_button_states = [0] * joystick.get_numbuttons()
        last_axis_values = [0.0] * joystick.get_numaxes()
        button_press_count = {}  # 记录每个按钮被按下的次数
        axis_movement_count = {}  # 记录每个轴的运动次数
        
        while True:
            pygame.event.pump()
            
            # 读取所有按钮状态
            current_states = []
            for i in range(joystick.get_numbuttons()):
                state = joystick.get_button(i)
                current_states.append(state)
                
                # 检测按钮按下（边缘触发）
                if state == 1 and last_button_states[i] == 0:
                    button_press_count[i] = button_press_count.get(i, 0) + 1
                    
                    # 显示按钮信息
                    print(f"\n[按钮按下] 索引: {i}")
                    if i in XBOX_BUTTON_MAP:
                        print(f"  功能: {XBOX_BUTTON_MAP[i]}")
                    else:
                        print(f"  功能: 未知（可能是其他按钮）")
                    print(f"  按下次数: {button_press_count[i]}")
                    
                    # 特别提示
                    if i == 6:
                        print("  ⭐ 这可能是 Back/Select 键（精准模式切换）")
                    elif i == 7:
                        print("  ⭐ 这可能是 Start 键")
                    elif i == 8:
                        print("  ⚠️  这是按钮 8（标准映射：左摇杆按下）")
                        print("     如果这是您按下的右摇杆，则映射可能相反")
                    elif i == 9:
                        print("  ⚠️  这是按钮 9（标准映射：右摇杆按下）")
                        print("     如果这是您按下的左摇杆，则映射可能相反")
            
            # 读取所有轴的值
            axis_values = []
            for i in range(joystick.get_numaxes()):
                value = joystick.get_axis(i)
                axis_values.append(value)
                
                # 检测轴值变化（超过阈值）
                if abs(value - last_axis_values[i]) > 0.2:
                    axis_movement_count[i] = axis_movement_count.get(i, 0) + 1
                    print(f"\n[轴变化] 轴 {i}: {value:7.3f}")
                    if i == 0:
                        print("  → 这可能是左摇杆 X 轴（标准映射）")
                    elif i == 1:
                        print("  → 这可能是左摇杆 Y 轴（标准映射）")
                    elif i == 2:
                        print("  → 这可能是左扳机 (LT)")
                    elif i == 3:
                        print("  → 这可能是右摇杆 X 轴（标准映射）")
                    elif i == 4:
                        print("  → 这可能是右摇杆 Y 轴（标准映射）")
                    elif i == 5:
                        print("  → 这可能是右扳机 (RT)")
            
            # 显示当前按下的按钮和活动的轴（实时更新）
            pressed_buttons = [i for i, state in enumerate(current_states) if state == 1]
            active_axes = [i for i, value in enumerate(axis_values) if abs(value) > 0.01]
            
            status_parts = []
            if pressed_buttons:
                button_info = []
                for btn_id in pressed_buttons:
                    if btn_id in XBOX_BUTTON_MAP:
                        button_info.append(f"按钮{btn_id}({XBOX_BUTTON_MAP[btn_id].split()[0]})")
                    else:
                        button_info.append(f"按钮{btn_id}(?)")
                status_parts.append("按钮: " + " | ".join(button_info))
            if active_axes:
                axis_info = [f"轴{i}:{axis_values[i]:5.2f}" for i in active_axes]
                status_parts.append("轴: " + " | ".join(axis_info))
            
            if status_parts:
                print(f"\r当前状态: {' | '.join(status_parts)}", end="", flush=True)
            else:
                print(f"\r当前状态: 无输入", end="", flush=True)
            
            last_button_states = current_states
            last_axis_values = axis_values
            time.sleep(0.05)  # 50ms 采样间隔
            
    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print("[测试总结]")
        print(f"检测到的按钮总数: {joystick.get_numbuttons()}")
        if button_press_count:
            print("\n按钮按下统计:")
            for btn_id in sorted(button_press_count.keys()):
                count = button_press_count[btn_id]
                name = XBOX_BUTTON_MAP.get(btn_id, "未知")
                print(f"  按钮 {btn_id}: {name} - 按下 {count} 次")
        
        # 特别提示 Back 键
        if 6 in button_press_count:
            print("\n✅ 按钮 6 被检测到，这很可能是 Back/Select 键")
            print("   如果这是您按下的 Back 键，则代码中的按钮索引 6 是正确的")
        else:
            print("\n⚠️  按钮 6 未被按下")
            print("   请确认 Back 键的实际索引，并更新代码中的按钮索引")
        
        print("=" * 70)
    finally:
        pygame.quit()

if __name__ == "__main__":
    main()

