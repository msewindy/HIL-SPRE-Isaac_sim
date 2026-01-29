
# Gear Assembly 集成测试启动指南 (Launch Guide)

本文档列出了进行 **Gear Assembly** 任务集成测试所需的完整启动命令和步骤。

## 0. 核心依赖检查

在开始之前，请确保已安装以下关键库：
```bash
pip install pygame flask flask-socketio requests numpy scipy absl-py tqdm tqdm
```
> **注意**: `pygame` 是手柄输入的关键依赖。

---

## 1. 启动 Isaac Sim Server (服务端)

这是测试的第一步，启动仿真环境并等待客户端连接。

**命令**:
```bash
# 在项目根目录下执行
# 替换 /path/to/isaac-sim 为实际的 Isaac Sim 根目录
/path/to/isaac-sim/python.sh serl_robot_infra/robot_servers/isaac_sim_server.py \
    --flask_url=127.0.0.1 \
    --flask_port=5001 \
    --headless=False \
    --sim_width=1280 \
    --sim_height=720 \
    --sim_hz=60.0 \
    --usd_path=examples/experiments/gear_assembly/HIL_franka_gear.usda \
    --robot_prim_path=/World/franka \
    --camera_prim_paths=/World/franka/panda_hand/wrist_1,/World/franka/panda_hand/wrist_2 \
    --config_module=experiments.gear_assembly.config
```

**参数说明**:
- `--headless=False`: 第一次测试建议开启 GUI (设为 False) 以观察机械臂动作。
- `--usd_path`: 指向 Gear Assembly 的 USD 场景文件。
- `--flask_port=5001`: 改为 5001 避免与可能运行的其他服务冲突 (原项目默认 5000)。
- `--sim_hz=60.0`: 设定物理仿真频率。

**期望输出**:
- 终端显示 `[INFO] Isaac Sim Server initialized`。
- 终端显示 `* Running on http://127.0.0.1:5001`。
- 如果 GUI 开启，应该能看到 Isaac Sim 窗口和机械臂场景。

---

## 2. 启动数据采集客户端 (客户端)

这是测试的核心，通过手柄控制机械臂，验证通信链路和控制逻辑。

**重要前置修改**:
由于 `examples/experiments/gear_assembly/config.py` 中默认使用 `SpacemouseIntervention`，我们需要确保它能使用手柄 (`GamepadIntervention`)。

**修改 `examples/experiments/gear_assembly/config.py` (第 214 行左右)**:
```python
        # 3. SpaceMouse 干预（真实环境必需，仿真环境可选）
        if not fake_env:
            # 真实环境：必需 SpaceMouse 进行干预
            env = SpacemouseIntervention(env)
        else:
            # [新增] 仿真环境：使用手柄控制
            try:
                from franka_env.envs.wrappers import GamepadIntervention
                env = GamepadIntervention(env, joystick_id=0)
                print("[INFO] Using Gamepad for intervention in Simulation")
            except ImportError:
                print("[WARNING] Gamepad wrapper not found, falling back to SpaceMouse or No-Intervention")
                # env = SpacemouseIntervention(env) # 如果想回退到 SpaceMouse
```
*(如果没有 `GamepadIntervention` wrapper，可以暂时用 `test_scripts/test_gamepad_mapping.py` 先验证硬件，或者在此步骤直接运行修改后的 `record_demos.py`)*

**启动命令**:
```bash
# 在项目根目录下执行
# --fake_env 必须添加，以指示使用 Isaac Sim 模拟环境配置
python examples/record_demos.py \
    --exp_name=gear_assembly \
    --successes_needed=5 \
    --fake_env
```

**期望行为**:
1. 脚本启动，连接到 `http://127.0.0.1:5001`。
2. 终端显示进度条 `0/5`。
3. **操作手柄**:
   - 推 **左摇杆**: 机械臂应在 XY 平面移动。
   - 按 **LT/LB**: 机械臂应上下移动 (Z轴)。
   - 按 **A/B**: 夹爪应闭合/打开。
   - 按 **RT/RB**: 机械臂末端应旋转 (Roll)。

---

## 3. (可选) 独立手柄测试脚本

如果在步骤 2 发现控制不对，先单独测试手柄映射：

**命令**:
```bash
python docs/isaac\ sim虚拟环境改造/test_scripts/test_gamepad_mapping.py
```

**期望行为**:
- 终端刷新显示轴的数值和计算后的动作值。
- 确保所有轴都能归零，且方向正确。

---

## 4. 故障排除 (Troubleshooting)

- **Connection Refused**: 检查 Server 是否已启动，端口是否为 5001。客户端配置中的 URL 端口也必须是 5001。
- **机械臂不因手柄动作而移动**:
    - 检查 Server 终端是否有报错 (如 IK Solver 失败)。
    - 检查 Client 终端是否显示 `Intervention Active` (干预需要激活才能控制)。
- **图像全黑**: 检查 Server 启动参数中 `camera_prim_paths` 是否正确，或 `--headless` 模式下的渲染设置。

