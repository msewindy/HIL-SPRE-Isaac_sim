
import pygame
import time
import sys
import os

def apply_deadzone(value, deadzone=0.1):
    if abs(value) < deadzone:
        return 0.0
    return value

def main():
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("[ERROR] No joystick found!")
        return

    # Use the first joystick
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    print(f"[INFO] Initialized Joystick: {joystick.get_name()}")
    print(f"[INFO] Axes: {joystick.get_numaxes()}")
    print(f"[INFO] Buttons: {joystick.get_numbuttons()}")
    print("="*50)
    print("Press Ctrl+C to exit")
    print("="*50)

    try:
        while True:
            pygame.event.pump()
            
            # --- Read Inputs ---
            # Axes
            axis_0 = joystick.get_axis(0) # Left Stick X
            axis_1 = joystick.get_axis(1) # Left Stick Y
            axis_2 = joystick.get_axis(2) # LT (Left Trigger)
            axis_3 = joystick.get_axis(3) # Right Stick X
            axis_4 = joystick.get_axis(4) # Right Stick Y
            axis_5 = joystick.get_axis(5) # RT (Right Trigger)
            
            # Buttons
            btn_a = joystick.get_button(0) # A
            btn_b = joystick.get_button(1) # B
            btn_y = joystick.get_button(3) # Y (Reset)
            btn_lb = joystick.get_button(4) # LB
            btn_rb = joystick.get_button(5) # RB

            # --- Apply Mapping Logic (from usb_gamepad_mapping.md) ---
            deadzone = 0.1
            sensitivity = 1.0

            # 1. Position Control (Left Hand)
            # x: Left Stick X
            x = apply_deadzone(axis_0, deadzone) * sensitivity
            
            # y: Left Stick Y (Inverted)
            y = -apply_deadzone(axis_1, deadzone) * sensitivity

            # z: LT + LB Combo
            # Normalize LT from [-1, 1] to [0, 1]
            lt_normalized = (axis_2 + 1.0) / 2.0
            if btn_lb == 0:
                z = -lt_normalized # Down (Negative)
            else:
                z = lt_normalized  # Up (Positive)
            z = apply_deadzone(z, deadzone) * sensitivity

            # 2. Rotation Control (Right Hand)
            # yaw: Right Stick X
            yaw = apply_deadzone(axis_3, deadzone) * sensitivity

            # pitch: Right Stick Y (Inverted)
            pitch = -apply_deadzone(axis_4, deadzone) * sensitivity

            # roll: RT + RB Combo
            # Normalize RT from [-1, 1] to [0, 1]
            rt_normalized = (axis_5 + 1.0) / 2.0
            if btn_rb == 0:
                roll = -rt_normalized # Left (Negative)
            else:
                roll = rt_normalized  # Right (Positive)
            roll = apply_deadzone(roll, deadzone) * sensitivity

            # 3. Gripper Logic Simulation (Continuous with Latch)
            if 'gripper_val' not in locals():
                gripper_val = 0.0 # Init Closed (Simulating sync)
                gripper_latched = False
            
            step_size = 0.05
            
            # Button Logic
            if btn_a: # Close
                gripper_val = max(0.0, gripper_val - step_size)
                gripper_latched = True
            elif btn_b: # Open
                gripper_val = min(1.0, gripper_val + step_size)
                gripper_latched = True
            
            # X Button: Release Latch
            btn_x = joystick.get_button(2)
            if btn_x:
                gripper_latched = False
            
            reset_scene = btn_y

            # --- Display Output ---
            os.system('clear' if os.name == 'posix' else 'cls')
            print("="*20 + " Gamepad Mapping Test " + "="*20)
            print(f"Raw Axes:")
            print(f"  L_Stick: ({axis_0:.2f}, {axis_1:.2f}) | LT: {axis_2:.2f}")
            print(f"  R_Stick: ({axis_3:.2f}, {axis_4:.2f}) | RT: {axis_5:.2f}")
            print("-" * 50)
            print(f"Mapped Control Outputs (Target 6DOF):")
            print(f"  Pos (X, Y, Z)    : ({x:.2f}, {y:.2f}, {z:.2f})")
            print(f"  Rot (R, P, Y)    : ({roll:.2f}, {pitch:.2f}, {yaw:.2f})")
            print("-" * 50)
            print(f"Gripper State (Continuous):")
            print(f"  Button A (Close) : {'PRESSED' if btn_a else '---'}")
            print(f"  Button B (Open)  : {'PRESSED' if btn_b else '---'}")
            print(f"  Button X (Resume): {'PRESSED' if btn_x else '---'}")
            print(f"  Latch Active     : {'YES (AI Locked)' if gripper_latched else 'NO (AI Control)'}")
            print(f"  Gripper Value    : {gripper_val:.2f}  [0.0 (Closed) <--> 1.0 (Open)]")
            print("-" * 50)
            print(f"Actions:")
            print(f"  SCENE RESET   (Y): {'TRIGGERED' if reset_scene else '---'}")
            
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nExiting...")
        pygame.quit()

if __name__ == "__main__":
    main()
