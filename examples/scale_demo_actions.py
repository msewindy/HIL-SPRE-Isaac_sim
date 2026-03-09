import pickle
import numpy as np
import argparse
import sys
import copy

def scale_actions(input_file, output_file, scale_factor=5.0):
    try:
        with open(input_file, 'rb') as f:
            data = pickle.load(f)
            
        print(f"Loading '{input_file}' with {len(data)} trajectories.")
        
        modified_data = copy.deepcopy(data)
        
        # Action space in RLPD is usually saved in the 'actions' key of the transition
        for i, traj in enumerate(modified_data):
            # check the shape of actions. Usually it's an array of arrays e.g (N, 7)
            # if it's just (7,) then it is only a single step. We check for a 2D matrix
            if len(traj['actions'].shape) == 1:
                # Single state action trajectory
                action = traj['actions']
                if len(action) >= 6:
                    action[:6] = action[:6] * scale_factor
                    action[:6] = np.clip(action[:6], -1.0, 1.0)
                traj['actions'] = action
            else:
                for step in range(len(traj['actions'])):
                    action = traj['actions'][step]
                    # Scale only the first 6 dims (translation + rotation)
                    if len(action) >= 6:
                        # scale by 5 to offset previous sensitivity=0.2
                        action[:6] = action[:6] * scale_factor
                        
                        # Ensure it doesn't overshoot [-1, 1] bounds due to precision issues
                        action[:6] = np.clip(action[:6], -1.0, 1.0)
                    
                    traj['actions'][step] = action
                
        with open(output_file, 'wb') as f:
            pickle.dump(modified_data, f)
            
        print(f"Successfully scaled actions by {scale_factor} and saved to '{output_file}'.")

    except FileNotFoundError:
         print(f"Error: Required file {input_file} not found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scale existing action bounds on a demo pickle file by a given factor.")
    parser.add_argument("input_file", type=str, help="Path to input .pkl dataset")
    parser.add_argument("output_file", type=str, help="Path to output .pkl dataset")
    parser.add_argument("--scale", type=float, default=5.0, help="Factor to scale translation and rotation action dims by")

    args = parser.parse_args()
    scale_actions(args.input_file, args.output_file, args.scale)
