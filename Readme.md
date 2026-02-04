# isaac sim server段启动
./run_isaac.sh serl_robot_infra/robot_servers/isaac_sim_server.py \
    --flask_url=0.0.0.0 \
    --flask_port=5001 \
    --headless=False \
    --sim_width=1280 \
    --sim_height=720 \
    --sim_hz=60.0 \
    --usd_path=examples/experiments/gear_assembly/HIL_franka_gear.usda \
    --robot_prim_path=/World/franka \
    --camera_prim_paths=/World/franka/panda_hand/wrist_1,/World/franka/panda_hand/wrist_2 \
    --config_module=examples.experiments.gear_assembly.config

# 演示行为收集record demos运行
python examples/record_demos.py --exp_name=gear_assembly --successes_needed=25 --fake_env

