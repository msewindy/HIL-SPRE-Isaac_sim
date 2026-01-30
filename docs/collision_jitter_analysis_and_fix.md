# 机械臂-齿轮碰撞与抓取后抖动 — 分析与修复

## 一、你的两个问题（理解确认）

1. **靠近时齿轮被推走/抖动**：机械臂靠近齿轮后，齿轮会抖动或慢慢被挤走；代码里设置的 ContactOffset/RestOffset 似乎没有对“机械臂–齿轮”这对碰撞生效。
2. **抓取后夹爪和齿轮一起抖**：夹爪抓取齿轮后建立了 FixedJoint 并“关闭碰撞”，但夹爪和齿轮仍然会抖动；你怀疑是求解器在抖。

下面结合代码和物理设置说明原因和改法。

---

## 二、问题 1：靠近时齿轮被推走/抖动，Offset 像没生效

### 2.1 可能原因

1. **机械臂是 Articulation，碰撞体未必被遍历到**  
   `_optimize_physics_precision()` 里用 `stage.Traverse()` 只给带 **UsdPhysics.CollisionAPI** 的 prim 设 ContactOffset/RestOffset。Franka 若是 **Articulation**，其碰撞体可能在关节/连杆子 prim 上，由运行时创建或使用别的 Schema，不一定被这段遍历改到，所以**机械臂侧的 offset 可能仍是默认值**（如 ContactOffset≈0.02）。这样会出现“力场盾”：臂还没真碰到齿轮，就已在 2cm 量级开始产生接触力，把齿轮推走或顶得抖动。

2. **ContactOffset 过小反而容易抖**  
   当前设成 **0.0001 m（0.1 mm）**。接触在“非常近”时才生成，求解器一步就要用很大力把穿透解开，容易造成**一帧内大力、下一帧又微穿**的振荡，看起来像抖动或慢慢挤走齿轮。适当**加大 ContactOffset**（例如 0.001–0.002 m），让接触“早一点”参与求解，往往更稳。

3. **求解迭代不够**  
   位置迭代只有 8 次时，刚性接触+小 offset 容易收敛不好，也会表现为抖动或缓慢漂移。**提高 SolverPositionIterations**（例如 16）有助于稳定。

4. **齿轮用 SDF、臂用 Convex**  
   SDF 接触的法向/接触点不如 Convex 稳定，被推时更容易出现滑动、微抖。这是几何类型带来的，只能通过 offset、迭代和下面说的“抓取后关碰撞”减轻。

### 2.2 建议修改（保持精密感，又减少推走/抖）

- 在 **保持 RestOffset=0** 的前提下，把 **ContactOffset 从 0.0001 提到 0.001（1 mm）**，让机械臂–齿轮这对更早、更平滑地进入接触。
- 将 **SolverPositionIterations 从 8 提到 16**，改善接触收敛。
- 若仍发现“只有齿轮/底座被改了、机械臂没改到”，需要针对 **Articulation 的 collision shape prim** 再查一次并显式设 offset（本方案先做全局+迭代，多数情况已够用）。

---

## 三、问题 2：抓取后夹爪和齿轮仍抖 — 碰撞其实没关

### 3.1 根本原因：当前并没有“关掉”手–齿轮碰撞

代码里写的是：

```python
# [CRITICAL] Disable collision between Hand and Gear
joint.CreateExcludeFromArticulationAttr().Set(True)
```

**ExcludeFromArticulation** 的含义是：该关节**不参与 Articulation 的约束求解**（例如单独用 PGS 解），**和“是否生成接触”无关**。  
因此，夹爪和齿轮之间**仍然在做碰撞检测并生成接触**。结果是：

- **FixedJoint** 要求：手和齿轮 rigid 在一起；
- **接触约束** 要求：把两个碰撞体推开。

两者在数值误差下会轻微冲突，每帧接触点/法向又可能略有变化，求解器就会在“贴紧”和“推开”之间来回微调，表现为**夹爪和齿轮一起抖**。所以你的理解对：**确实是求解器在“抖”，根源是抓取后这对碰撞没有被关掉。**

### 3.2 正确做法：用 FilteredPairsAPI 真正关掉手–齿轮碰撞

在 USD/Isaac Sim 里，要禁用**指定两体之间的碰撞**，应用 **UsdPhysics.FilteredPairsAPI**：

- 在**手**（或齿轮）的 prim 上 **Apply FilteredPairsAPI**；
- 在 **FilteredPairsRel** 里 **AddTarget** 另一个 prim（齿轮或手）；
- 这样这一对就不会再产生接触。

抓取时：创建 FixedJoint 的**同时**，给 hand 加上对 gear 的 FilteredPairs（或反过来）；  
释放时：删掉 FixedJoint 的**同时**，从 FilteredPairsRel 里 **RemoveTarget** 掉对方，恢复碰撞。

这样抓取后就不会再有两套约束（关节 + 接触）打架，抖动会明显减轻或消失。

---

## 四、代码修改摘要

| 位置 | 修改内容 |
|------|----------|
| `_optimize_physics_precision()` | ContactOffset 从 0.0001 改为 0.001；SolverPositionIterations 从 8 改为 16。 |
| 创建 FixedJoint 处（两处：旧逻辑 + v2） | 创建关节后，对 **wrist_prim（panda_hand）** 应用 **UsdPhysics.FilteredPairsAPI**，并在 **FilteredPairsRel** 中 **AddTarget(gear_prim)**；可选对 gear_prim 也做反向，保证双向过滤。 |
| 释放关节处（RemovePrim grasp_joint 的两处） | 在 RemovePrim 之前，从 **wrist_prim** 的 FilteredPairsRel 中 **RemoveTarget(gear_prim)**（若之前加了 gear→hand 也一并移除）。 |

说明：

- 若 Franka 的碰撞体不在 `panda_hand` 而在其子 prim（如某 link），FilteredPairsAPI 应加在**带 CollisionAPI 的 prim** 上，或对 **articulation root** 应用（视 Isaac Sim 文档）；若当前场景里 hand 即代表夹爪刚体，则先对手 prim 做即可。
- 释放时 gear_prim 路径要与抓取时一致（例如都用 `gear_rb_path` 或 `grasp_obj_path`），避免删错 target。

按上述修改后，预期效果是：

1. 靠近时齿轮不再被“提前”推走、抖动减轻（Offset + 迭代）。
2. 抓取后夹爪和齿轮不再因手–齿轮接触与 FixedJoint 冲突而抖动（FilteredPairs 真正关掉这对碰撞）。
