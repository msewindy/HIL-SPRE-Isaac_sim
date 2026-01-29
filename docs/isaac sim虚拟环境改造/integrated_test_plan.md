# Gear Assembly 任务集成测试规划 (Integrated Test Plan)

本文档基于 `isaac_sim_interface_design.md` (架构设计) 和 `usb手柄代替spacemouse_新映射方案.md` (手柄控制)，针对 **Gear Assembly** 任务的后续集成测试进行详细规划。

文档旨在细化用户提出的四大测试方向，确保从硬件控制到算法训练的全链路打通。

## 测试阶段概览

| 阶段 | 测试任务 | 关键目标 | 依赖文档 |
|------|---------|---------|---------|
| **阶段一** | **手柄遥操作与映射测试** | 验证手柄 6DOF 映射准确性、场景重置功能、延迟可控性 | `usb手柄代替spacemouse_新映射方案.md` |
| **阶段二** | **数据采集与干预测试** | 验证 `record_demos.py` 采集链路、图像传输/裁剪、人工干预信号记录 | `isaac_sim_interface_design.md` |
| **阶段三** | **数据传输与存储验证** | 验证 `ReplayBuffer` 数据格式、Action/Observation 空间匹配、磁盘存储完整性 | `isaac_sim_interface_design.md` |
| **阶段四** | **RL 训练闭环测试** | 验证 `train_rlpd.py` 完整跑通、Actor-Learner 通信、Loss 下降 | 原项目 `train_rlpd.py` |

---

## 阶段一：手柄遥操作与映射测试 (Gamepad Teleoperation)

**目标**：替代原 SpaceMouse，验证手柄对 Isaac Sim 机械臂的精确控制及辅助功能。

### 1.1 手柄轴映射与组合键验证 (单元测试)
- **测试内容**：编写/运行脚本读取手柄输入，验证新方案的映射逻辑。
- **验证点**：
    - [ ] **静止零值**：未操作时所有轴输出严格为 0.0。
    - [ ] **XYZ 平移**：
        - 左摇杆 X/Y 控制平面移动。
        - **LT + LB 组合**：验证 LT 单按向下，LT+LB 组合向上。
    - [ ] **RPY 旋转**：
        - 右摇杆 X/Y 控制 Yaw/Pitch。
        - **RT + RB 组合**：验证 RT 单按向左 Roll，RT+RB 组合向右 Roll。
    - [ ] **数值范围**：确认所有模拟量输出范围在 `[-1.0, 1.0]`。

### 1.2 机械臂实时控制测试 (集成测试)
- **测试内容**：启动 Isaac Sim Server，通过手柄控制机械臂末端移动。
- **验证点**：
    - [ ] **坐标系一致性**：手柄推前，机械臂是否向前 (验证 World/Robot 坐标系转换)。
    - [ ] **夹爪控制**：A 键闭合，B 键打开。
    - [ ] **场景重置**：按下 **Y 键**，验证场景 (World) 及机器人关节是否瞬间复位。
    - [ ] **控制延迟**：主观感受延迟是否在可接受范围内 (<100ms)。

---

## 阶段二：数据采集与干预测试 (Data Collection & Intervention)

**目标**：验证 `record_demos.py` 脚本在 Isaac Sim 架构下的适配性，确保"专家演示"和"人工干预"数据能被正确记录。

### 2.1 图像传输与质量验证
- **测试内容**：在采集过程中监控图像数据流。
- **验证点**：
    - [ ] **服务器端裁剪**：验证收到的图像是否已按 `config.py` 中的 `IMAGE_CROP` 裁切。
    - [ ] **WebSocket 传输**：验证图像是否通过 WebSocket 通道实时到达。
    - [ ] **分辨率/格式**：验证图像是否为 RGB 格式，尺寸符合 Network 输入要求 (如 128x128)。

### 2.2 干预信号 (Intervention Signal) 验证
- **测试内容**：运行 `record_demos.py`，进行“自动策略(如有) -> 人工接管 -> 自动策略”的切换模拟。
- **验证点**：
    - [ ] **接管触发**：手柄动作是否立即触发 `intervention` 标志位翻转。
    - [ ] **数据标记**：录制的数据中，人工接管段的 trajectories 是否包含正确的 `intervention_label` 或类似标记。

---

## 阶段三：数据传输与存储验证 (Data Pipeline)

**目标**：确保数据从 Actor (Gym Env) 到 Learner (Buffer) 的整个链路数据完整、格式正确。

### 3.1 观察空间 (Observation Space) 对齐
- **测试内容**：检查 Server 返回的 State 字典键值。
- **验证点**：
    - [ ] **Key 匹配**：`state`, `images`, `wrist_1`, `wrist_2` 等键名是否与 `IsaacSimFrankaEnv` 定义一致。
    - [ ] **维度检查**：
        - `state`: (TCP Pose, Vel, Gripper) 维度是否正确 (e.g., 14维或更多)。
        - `images`: (Basis, Height, Width, Channel) 维度是否正确。

### 3.2 序列化与磁盘存储
- **测试内容**：运行采集脚本保存少量 demo 数据到磁盘，手动读取 `.pkl` 文件。
- **验证点**：
    - [ ] **文件可读性**：使用 `pickle` 或 `numpy` 加载保存的文件不报错。
    - [ ] **数据完整性**：随机抽查一帧，确保图像和状态数据非空且数值合理。

---

## 阶段四：RL 训练闭环测试 (RL Training Loop)

**目标**：跑通 `train_rlpd.py`，验证算法在仿真环境下的训练稳定性。

### 4.1 环境实例化与 Reset
- **测试内容**：在训练脚本中初始化 `IsaacSimFrankaEnv`。
- **验证点**：
    - [ ] **环境启动**：Gym 环境能成功连接到已启动的 Isaac Sim Server。
    - [ ] **Reset 耗时**：`env.reset()` 调用耗时正常，且场景正确复位。

### 4.2 Actor-Learner 通信 (AgentLace)
- **测试内容**：启动 Training Server 和 Actor Client。
- **验证点**：
    - [ ] **数据发送**：Actor 采集的 steps 能成功推送到 Training Server。
    - [ ] **参数同步**：Learner 更新的 Policy 参数能回传给 Actor (Actor 网络的权重发生变化)。

### 4.3 训练稳定性验证
- **测试内容**：运行训练 500-1000 steps。
- **验证点**：
    - [ ] **无崩溃**：训练循环无内存泄漏、无连接超时崩溃。
    - [ ] **Loss 变化**：Critic/Actor Loss 有数值输出且非 NaN。
    - [ ] **FPS 监控**：记录 `Samples/sec`，评估仿真训练效率。

---

## 建议测试执行顺序

1. **环境准备**：
   - 启动 Isaac Sim Server (`isaac_sim_server.py`)。
   - 确认 Flask 端口 (5001) 和 WebSocket 端口可用。

2. **执行阶段一 (手柄)**：
   - 运行独立的手柄测试脚本 (需编写 `test_scripts/test_gamepad_mapping.py`)。
   - 确认控制顺滑。

3. **执行阶段二 & 三 (数据)**：
   - 运行 `examples/record_demos.py` (需配置使用 `IsaacSimFrankaEnv`)。
   - 录制 5-10 条轨迹，检查生成的 `.pkl` 文件。

4. **执行阶段四 (训练)**：
   - 运行 `examples/train_rlpd.py`。
   - 观察 TensorBoard 或 Log 输出。
