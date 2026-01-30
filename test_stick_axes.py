#!/usr/bin/env python3
"""
测试脚本：检测摇杆轴映射

使用方法：
1. 连接手柄
2. 运行脚本：python test_stick_axes.py
3. 依次移动左摇杆和右摇杆，查看轴索引
"""

import pygame
import sys
import time
import os

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
    print(f"手柄名称: {joystick.get_name()}")
    print(f"轴数量: {joystick.get_numaxes()}")
    print(f"按钮数量: {joystick.get_numbuttons()}")
    print("=" * 70)
    print("\n[测试说明]")
    print("1. 移动左摇杆，查看哪个轴在变化")
    print("2. 移动右摇杆，查看哪个轴在变化")
    print("3. 按下左摇杆，查看按钮索引")
    print("4. 按下右摇杆，查看按钮索引")
    print("5. 按 Ctrl+C 退出")
    print("=" * 70)
    print("\n实时轴状态（移动摇杆查看变化）：\n")
    
    try:
        last_axis_values = [0.0] * joystick.get_numaxes()
        last_button_states = [0] * joystick.get_numbuttons()
        
        while True:
            pygame.event.pump()
            
            # 读取所有轴的值
            axis_values = []
            for i in range(joystick.get_numaxes()):
                value = joystick.get_axis(i)
                axis_values.append(value)
                
                # 检测轴值变化（超过阈值）
                if abs(value - last_axis_values[i]) > 0.1:
                    print(f"\n[轴变化] 轴 {i}: {value:7.3f}")
                    if i == 0:
                        print("  → 这可能是左摇杆 X 轴")
                    elif i == 1:
                        print("  → 这可能是左摇杆 Y 轴")
                    elif i == 2:
                        print("  → 这可能是左扳机 (LT)")
                    elif i == 3:
                        print("  → 这可能是右摇杆 X 轴")
                    elif i == 4:
                        print("  → 这可能是右摇杆 Y 轴")
                    elif i == 5:
                        print("  → 这可能是右扳机 (RT)")
            
            # 读取所有按钮状态
            for i in range(joystick.get_numbuttons()):
                state = joystick.get_button(i)
                # 检测按钮按下（边缘触发）
                if state == 1 and last_button_states[i] == 0:
                    print(f"\n[按钮按下] 按钮 {i}")
                    if i == 8:
                        print("  → 这可能是左摇杆按下")
                    elif i == 9:
                        print("  → 这可能是右摇杆按下")
            
            # 显示当前轴值（实时更新，只显示非零的轴）
            active_axes = []
            for i, value in enumerate(axis_values):
                if abs(value) > 0.01:  # 只显示有值的轴
                    active_axes.append(f"轴{i}:{value:6.3f}")
            
            if active_axes:
                status = " | ".join(active_axes)
                print(f"\r当前活动轴: {status}", end="", flush=True)
            else:
                print(f"\r当前活动轴: 无", end="", flush=True)
            
            last_axis_values = axis_values
            last_button_states = [joystick.get_button(i) for i in range(joystick.get_numbuttons())]
            time.sleep(0.05)  # 50ms 采样间隔
            
    except KeyboardInterrupt:
        print("\n\n[退出] 测试结束")
        print("\n[总结]")
        print("根据测试结果，确认以下映射：")
        print("  - 左摇杆 X 轴应该是轴 0")
        print("  - 左摇杆 Y 轴应该是轴 1")
        print("  - 右摇杆 X 轴应该是轴 3")
        print("  - 右摇杆 Y 轴应该是轴 4")
        print("  - 左摇杆按下应该是按钮 8")
        print("  - 右摇杆按下应该是按钮 9")
    finally:
        pygame.quit()

if __name__ == "__main__":
    main()

