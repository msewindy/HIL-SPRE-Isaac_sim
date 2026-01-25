# Gear Assembly Task

Gear 组装任务：使用 Franka 机械臂将 `gear_medium` 安装到 `gear_base` 上。

## 文件说明

- `config.py` - 任务配置（位姿、相机、动作缩放等）
- `wrapper.py` - 任务特定环境包装器（真实环境）
- `HIL_franka_gear.usda` - Isaac Sim USD 场景文件
- `run_actor.sh` - Actor 节点启动脚本
- `run_learner.sh` - Learner 节点启动脚本

## 配置说明

### 位姿配置（需要根据实际测量值更新）

在 `config.py` 中需要更新以下位姿：

- `TARGET_POSE`: gear_medium 安装到 gear_base 的目标位姿
- `GRASP_POSE`: 抓取 gear_medium 的位姿
- `RESET_POSE`: 重置位姿
- `ABS_POSE_LIMIT_LOW/HIGH`: 探索边界框

**获取位姿的方法**：
1. 在 Isaac Sim 中手动移动机器人到目标位置
2. 使用 `curl -X POST http://127.0.0.1:5001/getstate` 获取当前位姿
3. 或通过 `isaac_sim_server.py` 的 `/getstate` 接口

### Isaac Sim 相机配置

对于 Isaac Sim 环境，`REALSENSE_CAMERAS` 配置只需要键名：

```python
REALSENSE_CAMERAS = {
    "wrist_1": {},  # 只需要键名，字段值不使用
    "wrist_2": {},
}
```

相机通过 USD 文件中的 prim 路径加载，不依赖配置中的字段值。

## 使用方法

### 1. 启动 Isaac Sim 服务器

```bash
python serl_robot_infra/robot_servers/isaac_sim_server.py \
    --usd_path=/path/to/examples/experiments/gear_assembly/HIL_franka_gear.usda \
    --flask_url=0.0.0.0 \
    --flask_port=5001 \
    --headless=True
```

### 2. 训练

#### 启动 Actor
```bash
cd examples/experiments/gear_assembly
bash run_actor.sh
```

#### 启动 Learner
```bash
cd examples/experiments/gear_assembly
bash run_learner.sh
```

## 注意事项

1. **位姿配置**：必须根据实际测量值设置，不能直接使用 RAM 插入任务的位姿
2. **USD 文件**：确保 USD 文件中的相机路径为 `/World/franka/panda_hand/wrist_1` 和 `/World/franka/panda_hand/wrist_2`
3. **相机配置**：Isaac Sim 环境中，相机配置只需要键名，字段值不使用
