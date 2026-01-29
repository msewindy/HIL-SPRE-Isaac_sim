
import requests
import time
import numpy as np
import socketio
import threading
import cv2
import base64

SERVER_URL = "http://127.0.0.1:5001"
WS_URL = "http://127.0.0.1:5001"

def test_http_api():
    print("\n[TEST] Testing HTTP API...")
    
    # 1. Health Check
    try:
        resp = requests.get(f"{SERVER_URL}/health")
        print(f"Health Check: {resp.status_code} - {resp.json()}")
        if resp.status_code != 200:
            print("FAILED: Health check failed")
            return
    except Exception as e:
        print(f"FAILED: Connection refused: {e}")
        return

    # 2. Get State
    try:
        resp = requests.post(f"{SERVER_URL}/getstate")
        if resp.status_code == 200:
            state = resp.json()
            print(f"Get State: Success. Keys: {list(state.keys())}")
            print(f"Pose: {state['pose']}")
            print(f"Joints: {state['q']}")
        else:
            print(f"Get State Failed: {resp.status_code}")
    except Exception as e:
        print(f"FAILED: Get state failed: {e}")

    # 3. Control Gripper
    print("Testing Gripper Close...")
    requests.post(f"{SERVER_URL}/close_gripper")
    time.sleep(1)
    
    print("Testing Gripper Open...")
    requests.post(f"{SERVER_URL}/open_gripper")
    time.sleep(1)

    # 4. Pose Control (Small movement)
    print("Testing Pose Control (Small movement)...")
    # Get current pose first
    current_pose = requests.post(f"{SERVER_URL}/getstate").json()['pose']
    target_pose = np.array(current_pose)
    target_pose[2] += 0.05 # Move up 5cm
    
    resp = requests.post(f"{SERVER_URL}/pose", json={"arr": target_pose.tolist()})
    print(f"Pose Command Sent: {resp.text}")
    
    time.sleep(2) # Wait for movement
    
    new_pose = requests.post(f"{SERVER_URL}/getstate").json()['pose']
    print(f"New Pose Z: {new_pose[2]}")

def test_websocket():
    print("\n[TEST] Testing WebSocket Image Stream...")
    sio = socketio.Client()
    
    image_received = {"wrist_1": False, "wrist_2": False}
    
    @sio.on('connect')
    def on_connect():
        print("WebSocket Connected!")

    @sio.on('image')
    def on_image(data):
        try:
            # print(f"DEBUG: Received data type: {type(data)}")
            if isinstance(data, bytes):
                # Format: <len><key><data>
                key_len = data[0]
                key = data[1:1+key_len].decode('utf-8')
                img_data = data[1+key_len:]
                # print(f"DEBUG: Parsed key: {key}, image size: {len(img_data)}")
                image_received[key] = True
            else:
                print(f"DEBUG: Unexpected data format: {data}")
        except Exception as e:
            print(f"DEBUG: Error in on_image: {e}")

    try:
        sio.connect(WS_URL)
        time.sleep(5) # Listen for 5 seconds (increased to allow renderer to warm up)
        sio.disconnect()
        
        if image_received["wrist_1"] and image_received["wrist_2"]:
            print("SUCCESS: Received images from both cameras.")
        else:
            print(f"PARTIAL/FAILED: Received - {image_received}")
            
    except Exception as e:
        print(f"WebSocket Verification Failed: {e}")

if __name__ == "__main__":
    test_http_api()
    test_websocket()
