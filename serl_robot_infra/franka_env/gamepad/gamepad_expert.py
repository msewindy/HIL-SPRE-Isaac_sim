"""
Gamepad Expert for controlling robot with gamepad/joystick.
Provides the same interface as SpaceMouseExpert.
"""
import multiprocessing
import numpy as np
from typing import Tuple
import time
import os

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Warning: pygame not installed. Gamepad support will not work.")
    print("Install with: pip install pygame")


def apply_deadzone(value: float, deadzone: float) -> float:
    """Apply deadzone to joystick value.
    
    Args:
        value: Raw joystick value in range [-1, 1]
        deadzone: Deadzone threshold in range [0, 1]
                  When deadzone=0.0, returns value unchanged (matches SpaceMouse behavior)
    
    Returns:
        Processed value with deadzone applied, range [-1, 1]
        Note: Maximum values (1.0 or -1.0) are preserved after deadzone removal
        When deadzone=0.0, output equals input (no processing)
    """
    # Special case: deadzone=0.0 means no processing (matches SpaceMouse)
    if deadzone == 0.0:
        return value
    
    if abs(value) < deadzone:
        return 0.0
    # Remap: from [deadzone, 1.0] to [0, 1.0]
    # This preserves the maximum value (1.0) after remapping
    sign = 1.0 if value >= 0 else -1.0
    abs_value = abs(value)
    # Remap from [deadzone, 1.0] to [0, 1.0]
    remapped = (abs_value - deadzone) / (1.0 - deadzone)
    # Clamp to ensure we don't exceed 1.0
    remapped = min(1.0, remapped)
    return sign * remapped


class GamepadExpert:
    """
    Gamepad interface class that provides the same interface as SpaceMouseExpert.
    
    Maps gamepad inputs to 6DOF actions [x, y, z, roll, pitch, yaw]:
    - Position control (left hand): left stick (x, y) + left trigger/bumper (z)
    - Rotation control (right hand): right stick (yaw, pitch) + right trigger/bumper (roll)
    
    Interface compatibility:
    - get_action() -> Tuple[np.ndarray, list]
      - Returns [x, y, z, roll, pitch, yaw] and button list
    """
    
    def __init__(self, deadzone=0.0, sensitivity=1.0, joystick_id=0):
        """
        Initialize gamepad expert.
        
        Args:
            deadzone: Deadzone threshold for joysticks (0.0-1.0)
                     Default 0.0 to match SpaceMouse output range [-1.0, 1.0]
                     User can increase to filter joystick noise if needed
            sensitivity: Sensitivity scaling factor (0.0-2.0)
                        Default 1.0 to match SpaceMouse output range
                        User can adjust for finer/coarser control
            joystick_id: Joystick device ID if multiple gamepads are connected
        """
        if not PYGAME_AVAILABLE:
            raise ImportError("pygame is required for gamepad support. Install with: pip install pygame")
        
        # Initialize pygame
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        pygame.joystick.init()
        
        # Check if gamepad is available
        joystick_count = pygame.joystick.get_count()
        if joystick_count == 0:
            raise RuntimeError("No gamepad detected. Please connect a gamepad and try again.")
        
        if joystick_id >= joystick_count:
            print(f"Warning: Joystick ID {joystick_id} not available. Using joystick 0.")
            joystick_id = 0
        
        self.joystick_id = joystick_id
        self.deadzone = deadzone
        self.sensitivity = sensitivity
        
        # Manager to handle shared state between processes
        self.manager = multiprocessing.Manager()
        self.latest_data = self.manager.dict()
        self.latest_data["action"] = [0.0] * 6  # [x, y, z, roll, pitch, yaw]
        self.latest_data["buttons"] = [0, 0, 0, 0]  # [A, B, X, Y] for gripper control
        
        # Start a process to continuously read the gamepad state
        self.process = multiprocessing.Process(target=self._read_gamepad)
        self.process.daemon = True
        self.process.start()
        
        # Give the process time to initialize
        time.sleep(0.1)
    
    def _read_gamepad(self):
        """Continuously read gamepad state (runs in separate process)."""
        # Initialize pygame in this process
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        pygame.joystick.init()
        
        # Get joystick
        joystick = pygame.joystick.Joystick(self.joystick_id)
        joystick.init()
        
        print(f"Gamepad initialized: {joystick.get_name()}")
        print(f"  Axes: {joystick.get_numaxes()}")
        print(f"  Buttons: {joystick.get_numbuttons()}")
        
        clock = pygame.time.Clock()
        
        while True:
            # Process pygame events
            pygame.event.pump()
            
            # Initialize action array [x, y, z, roll, pitch, yaw]
            action = [0.0] * 6
            
            try:
                # Read joystick axes (normalized to [-1, 1])
                # Xbox 360/One controller mapping (verified):
                # Axis 0: Left stick X
                # Axis 1: Left stick Y
                # Axis 2: Left trigger (LT) - initial -1.0, fully pressed 1.0
                # Axis 3: Right stick X
                # Axis 4: Right stick Y
                # Axis 5: Right trigger (RT) - initial -1.0, fully pressed 1.0
                
                num_axes = joystick.get_numaxes()
                
                # Left stick: position control (x, y)
                if num_axes > 0:
                    left_x = joystick.get_axis(0)  # Left stick X → x translation
                    action[0] = apply_deadzone(left_x, self.deadzone) * self.sensitivity
                
                if num_axes > 1:
                    left_y = joystick.get_axis(1)  # Left stick Y → y translation (inverted)
                    action[1] = -apply_deadzone(left_y, self.deadzone) * self.sensitivity
                
                # Right stick: rotation control (yaw, pitch)
                # Xbox 360/One mapping (verified):
                # Axis 0: Left stick X
                # Axis 1: Left stick Y
                # Axis 2: Left trigger (LT) - initial value -1.0, fully pressed 1.0
                # Axis 3: Right stick X
                # Axis 4: Right stick Y
                # Axis 5: Right trigger (RT) - initial value -1.0, fully pressed 1.0
                
                # Right stick: yaw and pitch
                # Note: action format is [x, y, z, roll, pitch, yaw]
                if num_axes > 3:
                    right_x = joystick.get_axis(3)  # Right stick X (axis 3) → yaw
                    action[5] = apply_deadzone(right_x, self.deadzone) * self.sensitivity
                else:
                    action[5] = 0.0
                
                if num_axes > 4:
                    right_y = joystick.get_axis(4)  # Right stick Y (axis 4) → pitch (inverted)
                    action[4] = -apply_deadzone(right_y, self.deadzone) * self.sensitivity
                else:
                    action[4] = 0.0
                
                # Z translation: LT (axis 2) + LB (button 4) combination control
                num_buttons = joystick.get_numbuttons()
                left_bumper = joystick.get_button(4) if num_buttons > 4 else 0  # LB
                
                if num_axes > 2:
                    left_trigger = joystick.get_axis(2)  # LT (axis 2): initial -1.0, fully pressed 1.0
                    
                    # Normalize LT from [-1, 1] to [0, 1]
                    # Unpressed (-1.0) → 0.0, fully pressed (1.0) → 1.0
                    lt_normalized = (left_trigger + 1.0) / 2.0  # Range [0.0, 1.0]
                    
                    # Combine with LB to determine direction
                    if left_bumper == 0:
                        # LB not pressed: z downward (negative)
                        # lt_normalized [0, 1] → z [-1, 0]
                        z_value = -lt_normalized
                    else:
                        # LB pressed: z upward (positive)
                        # lt_normalized [0, 1] → z [0, 1]
                        z_value = lt_normalized
                    
                    # Apply deadzone and sensitivity
                    action[2] = apply_deadzone(z_value, self.deadzone) * self.sensitivity
                else:
                    action[2] = 0.0
                
                # Roll rotation: RT (axis 5) + RB (button 5) combination control
                right_bumper = joystick.get_button(5) if num_buttons > 5 else 0  # RB
                
                if num_axes > 5:
                    right_trigger = joystick.get_axis(5)  # RT (axis 5): initial -1.0, fully pressed 1.0
                    
                    # Normalize RT from [-1, 1] to [0, 1]
                    # Unpressed (-1.0) → 0.0, fully pressed (1.0) → 1.0
                    rt_normalized = (right_trigger + 1.0) / 2.0  # Range [0.0, 1.0]
                    
                    # Combine with RB to determine direction
                    if right_bumper == 0:
                        # RB not pressed: roll left (negative)
                        # rt_normalized [0, 1] → roll [-1, 0]
                        roll_value = -rt_normalized
                    else:
                        # RB pressed: roll right (positive)
                        # rt_normalized [0, 1] → roll [0, 1]
                        roll_value = rt_normalized
                    
                    # Apply deadzone and sensitivity
                    action[3] = apply_deadzone(roll_value, self.deadzone) * self.sensitivity
                else:
                    action[3] = 0.0
                
                # Read buttons for gripper control
                buttons = [0, 0, 0, 0]
                if num_buttons > 0:
                    buttons[0] = joystick.get_button(0)  # A button - close gripper
                if num_buttons > 1:
                    buttons[1] = joystick.get_button(1)  # B button - open gripper
                if num_buttons > 2:
                    buttons[2] = joystick.get_button(2)  # X button (optional)
                if num_buttons > 3:
                    buttons[3] = joystick.get_button(3)  # Y button (optional)
                
            except Exception as e:
                # If gamepad disconnects, set all actions to zero
                print(f"Warning: Error reading gamepad: {e}")
                action = [0.0] * 6
                buttons = [0, 0, 0, 0]
            
            # Debug: Print raw values occasionally
            if hasattr(self, '_debug_counter'):
                self._debug_counter += 1
            else:
                self._debug_counter = 0
            
            if self._debug_counter % 100 == 0:  # Print every 100 ticks (approx 1 sec)
                if any(abs(a) > 0.01 for a in action):
                    print(f"[DEBUG] Gamepad Raw Action: {action}")
            
            # Update shared state
            try:
                self.latest_data["action"] = action
                self.latest_data["buttons"] = buttons
            except (ConnectionResetError, FileNotFoundError, BrokenPipeError, EOFError):
                 # Main process probably died (Manager shutdown)
                 # Exit gracefully without crashing
                 break
            except Exception as e:
                 # Unexpected error
                 print(f"[ERROR] Shared memory write error: {e}")
                 break
            
            # Control update rate (similar to SpaceMouse)
            clock.tick(100)  # 100 Hz update rate
    
    def get_action(self) -> Tuple[np.ndarray, list]:
        """
        Returns the latest action and button state of the gamepad.
        
        Returns:
            Tuple of (action_array, button_list):
            - action_array: np.ndarray of shape (6,) [x, y, z, roll, pitch, yaw]
            - button_list: list of button states [A, B, X, Y]
        """
        action = self.latest_data["action"]
        buttons = self.latest_data["buttons"]
        return np.array(action), buttons
    
    def close(self):
        """Close gamepad connection and terminate reading process."""
        if hasattr(self, 'process') and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1.0)
        pygame.joystick.quit()
        pygame.quit()
