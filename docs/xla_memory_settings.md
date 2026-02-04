# XLA 内存设置说明

## 一、环境变量说明

### 1.1 `XLA_PYTHON_CLIENT_PREALLOCATE`

**作用**：控制 XLA 是否在启动时预分配 GPU 内存。

**取值**：
- `true`（默认）：预分配所有可用 GPU 内存
- `false`：按需分配内存，不预分配

**为什么设置为 `false`**：
1. **避免内存浪费**：预分配会占用所有 GPU 内存，即使实际不需要那么多
2. **多进程共享**：当多个进程（Learner、Actor、Isaac Sim）共享 GPU 时，预分配会导致冲突
3. **灵活分配**：按需分配可以根据实际使用情况动态调整

**示例**：
```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

---

### 1.2 `XLA_PYTHON_CLIENT_MEM_FRACTION`

**作用**：控制 JAX 使用的 GPU 内存比例。

**取值**：
- `0.0` 到 `1.0` 之间的浮点数
- 例如：`.3` 表示使用 30% 的 GPU 内存

**为什么设置为 `.3`（30%）**：
1. **预留内存**：为其他进程（Isaac Sim、系统）预留内存
2. **避免 OOM**：防止 JAX 占用过多内存导致其他进程崩溃
3. **多进程协调**：当多个进程共享 GPU 时，需要合理分配

**不同进程的推荐值**：
- **Learner**：`.3`（30%）- 需要较多内存用于训练
- **Actor**：`.1`（10%）- 只需要推理，内存需求较小
- **Isaac Sim**：剩余内存（约 60%）

**示例**：
```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3
```

---

## 二、使用场景

### 2.1 单 GPU 多进程场景

当你在同一台机器上运行多个进程时：

```
GPU 内存分配（假设 24GB）：
├─ Isaac Sim: ~14GB (60%)
├─ Learner:   ~7GB  (30%)
└─ Actor:     ~2GB  (10%)
```

**配置**：
```bash
# Learner
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3

# Actor
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1
```

---

### 2.2 单进程场景

如果只有一个进程使用 GPU（例如只有 Learner）：

```bash
# 可以使用更多内存
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.8  # 80%
```

---

## 三、常见问题

### 3.1 OOM (Out of Memory) 错误

**症状**：
```
RuntimeError: RESOURCE_EXHAUSTED: Out of memory
```

**解决方案**：
```bash
# 降低内存使用比例
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2  # 从 .3 降到 .2
# 或
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1  # 进一步降低
```

---

### 3.2 内存分配冲突

**症状**：
```
RuntimeError: Failed to allocate memory
```

**解决方案**：
```bash
# 确保所有进程都设置 PREALLOCATE=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# 合理分配内存比例
# Learner: .3
# Actor: .1
# Isaac Sim: 剩余（不设置此变量）
```

---

### 3.3 内存使用不足

**症状**：GPU 利用率低，训练速度慢

**解决方案**：
```bash
# 如果只有 Learner 使用 GPU，可以增加内存比例
export XLA_PYTHON_CLIENT_MEM_FRACTION=.6  # 增加到 60%
```

---

## 四、最佳实践

### 4.1 推荐配置

**多进程场景**（Learner + Actor + Isaac Sim）：
```bash
# Learner
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3

# Actor
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1
```

**单进程场景**（只有 Learner）：
```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.8
```

---

### 4.2 内存监控

**查看 GPU 内存使用**：
```bash
nvidia-smi
```

**查看 JAX 内存使用**：
```python
import jax
print(jax.devices())  # 查看可用设备
print(jax.local_devices())  # 查看本地设备
```

---

## 五、技术细节

### 5.1 XLA 内存管理

XLA (Accelerated Linear Algebra) 是 JAX 的编译后端：
- 负责将 Python 代码编译为 GPU 可执行代码
- 管理 GPU 内存的分配和释放
- 优化计算图的执行

### 5.2 内存分配策略

**预分配（PREALLOCATE=true）**：
- 启动时分配所有可用内存
- 优点：避免运行时分配延迟
- 缺点：占用所有内存，多进程冲突

**按需分配（PREALLOCATE=false）**：
- 运行时按需分配内存
- 优点：灵活，多进程友好
- 缺点：可能有轻微分配延迟

---

## 六、总结

### 6.1 关键要点

1. **`PREALLOCATE=false`**：多进程场景必需，避免内存冲突
2. **`MEM_FRACTION`**：根据进程类型和 GPU 容量调整
3. **监控内存**：使用 `nvidia-smi` 监控实际使用情况

### 6.2 推荐值

| 进程类型 | PREALLOCATE | MEM_FRACTION | 说明 |
|---------|-------------|--------------|------|
| Learner | false | 0.3 | 训练需要较多内存 |
| Actor | false | 0.1 | 推理只需少量内存 |
| Isaac Sim | - | - | 不设置，使用剩余内存 |

---

**参考**：
- [JAX 文档 - GPU 内存管理](https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html)
- [XLA 文档](https://www.tensorflow.org/xla)

