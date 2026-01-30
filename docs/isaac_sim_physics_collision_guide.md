# Isaac Sim 物理碰撞设置详解

本文档介绍 Isaac Sim / PhysX 5 中与碰撞相关的核心设置：**Collision System（PCM / SAT）**、**Solver Type（TGS / PGS）**、**Enable CCD**，以及 **Collision Approximation（SDF MESH / Convex Hull）**。并结合本项目中 gear、gear base、Franka 机械臂的配置说明差异与选型建议。

---

## 一、Physics Scene 中的 Collision System：PCM 与 SAT

### 1.1 概述

PhysX 提供两种**窄相位（Narrow Phase）碰撞检测**方式：

- **SAT（Separating Axis Theorem，分离轴定理）**：传统/默认方式（在部分版本中作为“Legacy”选项）。
- **PCM（Persistent Contact Manifold，持久接触流形）**：PhysX 5 中已成为**默认**的碰撞检测方式。

### 1.2 SAT（分离轴定理）

- **原理**：混合使用 SAT 与基于距离的方法，在**一帧内**生成**完整接触流形**，即一次性给出该帧所有潜在接触点。
- **优点**：
  - 接触点数量多、接触面表达完整，**堆叠（stacking）更稳定**。
  - 在 **Contact Offset / Rest Offset 较小** 时表现稳定。
- **缺点**：
  - 当 **Contact Offset、Rest Offset 较大** 时，接触点通过“平面平移”近似，**可能不够准确**。
  - 每帧都重新生成完整流形，内存与计算相对更重。

### 1.3 PCM（持久接触流形）

- **原理**：完全基于**距离**的碰撞检测（GJK + EPA 等），首帧生成完整接触流形，之后**复用并更新上一帧的接触点**，仅在相对运动超过阈值或接触丢失时补充/重新生成接触。
- **优点**：
  - **性能与内存更优**，适合实时仿真。
  - 基于真实几何距离，**任意 Contact Offset / Rest Offset 下接触点更正确**。
- **缺点**：
  - 同一时刻接触点数量可能少于 SAT，在**大时间步、高堆叠、小物体**场景下，**堆叠稳定性可能不如 SAT**。
  - 官方文档曾注明 PCM 仍在持续改进，成熟度略逊于传统 SAT。

### 1.4 如何选择（本项目：精密装配）

- **精密装配（小 Contact/Rest Offset、少量物体、重视接触精度）**：更适合 **PCM**，与项目里 `_optimize_physics_precision()` 将 ContactOffset 设为 0.0001、RestOffset 设为 0 的用法一致。
- **大量堆叠、大偏移、优先堆叠稳定**：可考虑 **SAT**（若场景提供该选项）。

---

## 二、Solver Type：TGS 与 PGS

### 2.1 概述

- **PGS（Projected Gauss-Seidel）**：PhysX 传统的约束求解器，基于高斯-赛德尔迭代求解接触/关节约束。
- **TGS（Temporal Gauss-Seidel）**：PhysX 4 引入的时序版求解器，收敛特性更好，对**大质量比**、复杂约束更友好。

### 2.2 主要区别

| 方面 | PGS | TGS |
|------|-----|-----|
| 收敛与稳定性 | 传统、稳定、行为可预期 | 收敛更快，但**行为与 PGS 不同** |
| 关节/约束刚度 | 相同参数下关节更“硬” | 相同参数下关节更“软”，需**大幅提高刚度**（文献中有约 5 个数量级）才能接近 PGS 手感 |
| 堆叠与 Rest Offset | 对 Rest Offset 非零的堆叠更稳定 | 部分场景下（如非零 Rest Offset 堆叠）可能出现不稳定 |
| 适用场景 | 机械臂、精密接触、需可重复行为 | 复杂多体、大质量比、可接受重新调参 |

### 2.3 使用注意

- TGS 与 PGS **不能简单互换**：相同参数下仿真结果会明显不同。
- **机械臂 + 精密装配**：若当前用 PGS 已调好（接触、夹取、装配稳定），建议保持 **PGS**；若改用 TGS，需要重新调刚度、迭代次数等。

---

## 三、Enable CCD（Continuous Collision Detection）

### 3.1 作用

- **离散碰撞检测**：只在每帧的**当前位形**做碰撞检测。物体若在一帧内移动过快，可能从另一物体**穿过去**（tunnelling）。
- **CCD（连续碰撞检测）**：在物体**本帧运动路径**上做检测，在**发生碰撞的时刻**停下来并响应，从而**避免高速穿模**。

### 3.2 开启 CCD 的影响

| 项目 | 说明 |
|------|------|
| 效果 | 高速运动物体不会轻易“穿透”其他物体，碰撞更可信。 |
| 性能 | 有额外开销；物体越快、越密集，开销越大。默认有速度阈值，只有超过阈值才做 CCD。 |
| 限制 | 一般**不能用于 Kinematic 物体**（设为 Kinematic 会禁用 CCD）；需在 Scene、Pair、Body 三个层面都开启/配置。 |

### 3.3 与本项目的关系

- **齿轮装配、机械臂抓取**：多为**低速、受控**运动，穿模风险小，通常**不必开 CCD**。
- **若有抛射、高速碰撞**（如零件被甩飞）：可对相关刚体开启 CCD，或使用 **Speculative CCD**（基于放大 Contact Offset 的近似方式，可与 sweep-based CCD 同时开）。

---

## 四、Collision Approximation：SDF MESH 与 Convex Hull

碰撞体在 Isaac Sim 中可由**网格（Mesh）** 生成，并选择不同的**近似方式**，用于实际碰撞计算。本项目中：

- **Gear、Gear Base**：Physics Collider 的 **Approximation = SDF MESH**。
- **Franka 机械臂组件**：**Approximation = Convex Hull**。

### 4.1 Convex Hull（凸包）

- **含义**：用物体表面的**凸包**作为碰撞体，即“把模型塞进一个凸多面体里”。
- **特点**：
  - **计算快**：凸包面数有上限（如 64 面），碰撞检测简单。
  - **形状偏差**：凹的部分会被“填满”，**不能表达孔洞、内凹**。
- **适用**：机械臂连杆、夹爪等**外形较整、偏凸**的部件，在保证性能的同时足够用。

### 4.2 SDF MESH（Signed Distance Field Mesh）

- **含义**：对三角网格做**有符号距离场（SDF）**，用网格 + SDF 做精确的三角网格碰撞（包括凹、孔洞）。
- **特点**：
  - **形状准**：可忠实表达齿轮、底座等**复杂、带孔、凹槽**的几何。
  - **性能贵**：同一物体下，SDF 碰撞通常比 Convex Hull **慢很多**（文献中有约 20 倍量级）；主要瓶颈往往是**三角形数量**，SDF 分辨率主要影响内存。
- **适用**：齿轮、齿轮底座等**需要精确啮合/装配**的零件，用 SDF MESH 更合理。

### 4.3 本项目中的分工

| 物体 | Approximation | 原因 |
|------|----------------|------|
| **Gear / Gear Base** | **SDF MESH** | 不规则、有孔/齿，需要**精确接触与插入**，用 SDF 保证几何一致。 |
| **Franka 机械臂** | **Convex Hull** | 多为凸形部件，**性能优先**；凸包足以满足夹取、避障等需求。 |

### 4.4 使用 SDF 时的注意点（来自官方/社区）

- **三角形数**：碰撞用 mesh 尽量**简化**（如用 Meshlab 减面），渲染可用高模；只给带 Collider 的 mesh 做物理。
- **SDF 分辨率**：过高主要增加内存，对速度影响相对小；优先减面再考虑分辨率。
- **已知限制**：SDF 三角网格与传送带、高速 Kinematic 等组合时，可能出现接触质量或异常；对 GPU RL 管线，更推荐 Convex Hull、SDF tri-mesh、基础几何（球/盒/胶囊）等“原生”近似。

---

## 五、小结与对应关系

| 设置项 | 本项目推荐/说明 |
|--------|------------------|
| **Collision System** | **PCM**：小 offset、精密接触，接触点更准确。 |
| **Solver Type** | **PGS**：机械臂与装配已按此调参则保持；若用 TGS 需重调刚度与迭代。 |
| **Enable CCD** | **关闭**即可：装配与抓取多为低速；若有高速抛射再对个别物体开启。 |
| **Gear / Gear Base** | **SDF MESH**：保证装配几何精度。 |
| **Franka 机械臂** | **Convex Hull**：性能与稳定性平衡。 |

以上对应关系与项目里 `_optimize_physics_precision()` 对 ContactOffset/RestOffset 和 Solver 迭代的优化是一致的：在**精密装配**场景下优先**接触精度**（PCM + 小 offset + SDF 对齿轮），机械臂侧用**凸包**控制成本。

---

## 参考资料

- [PhysX 5.4.1 – Rigid Body Collision](https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/RigidBodyCollision.html)
- [PhysX 5.4.1 – Advanced Collision Detection（PCM / CCD）](https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/AdvancedCollisionDetection.html)
- [Isaac Sim – Physics Simulation Fundamentals](https://docs.omniverse.nvidia.com/isaacsim/latest/simulation_fundamentals.html)
- [Omniverse Physics Resources and Limitations](https://docs.isaacsim.omniverse.nvidia.com/latest/physics/physics_resources.html)
- NVIDIA Developer Forums: SDF Mesh Collision Performance, Convex decomposition
