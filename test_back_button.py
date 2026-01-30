#!/usr/bin/env python3
"""
测试脚本：检测手柄 Back 键的按钮索引

使用方法：
1. 连接手柄
2. 运行脚本：python test_back_button.py
3. 按下 Back 键，查看输出的按钮索引
"""

import pygame
import sys
import time

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
    
    print("=" * 60)
    print(f"手柄名称: {joystick.get_name()}")
    print(f"轴数量: {joystick.get_numaxes()}")
    print(f"按钮数量: {joystick.get_numbuttons()}")
    print("=" * 60)
    print("\n[说明]")
    print("1. 按下 Back 键（或 Select 键）")
    print("2. 查看下方显示的按钮索引")
    print("3. 按 Ctrl+C 退出")
    print("=" * 60)
    print("\n实时按钮状态（按下的按钮会高亮显示）：\n")
    
    try:
        last_button_states = [0] * joystick.get_numbuttons()
        while True:
            pygame.event.pump()
            
            # 读取所有按钮状态
            current_states = []
            for i in range(joystick.get_numbuttons()):
                state = joystick.get_button(i)
                current_states.append(state)
                
                # 检测按钮按下（边缘触发）
                if state == 1 and last_button_states[i] == 0:
                    print(f"\n[检测到按钮按下] 按钮索引: {i}")
                    if i == 6:
                        print("  → 这是按钮 6（可能是 Back 键）")
                    elif i == 7:
                        print("  → 这是按钮 7（可能是 Start 键）")
                    elif i == 8:
                        print("  → 这是按钮 8（可能是左摇杆按下）")
                    elif i == 9:
                        print("  → 这是按钮 9（可能是右摇杆按下）")
                    else:
                        print(f"  → 按钮 {i} 的功能未知")
            
            # 显示所有按钮状态（每 0.5 秒更新一次）
            if int(time.time() * 2) % 2 == 0:  # 每 0.5 秒刷新
                pressed_buttons = [i for i, state in enumerate(current_states) if state == 1]
                if pressed_buttons:
                    button_str = ", ".join([f"按钮{i}" for i in pressed_buttons])
                    print(f"\r当前按下的按钮: {button_str}", end="", flush=True)
                else:
                    print(f"\r当前按下的按钮: 无", end="", flush=True)
            
            last_button_states = current_states
            time.sleep(0.05)  # 50ms 采样间隔
            
    except KeyboardInterrupt:
        print("\n\n[退出] 测试结束")
    finally:
        pygame.quit()

if __name__ == "__main__":
    import os
    main()

