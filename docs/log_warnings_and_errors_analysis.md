# 运行日志报错与警告分析

本文档对应终端日志（如 `terminals/6.txt` 753–876 行）中的报错与警告，说明原因及已做/可选处理。

---

## 修改后运行状态小结（参考 754–821 行日志）

- **PhysxSceneAPI 求解器**：已修复，日志出现 `[INFO] Enhanced physics scene solver iterations for /World/PhysicsScene`，无 “Failed to set solver iterations” WARN。
- **碰撞 offset 显式设置**：`factory_gear_medium`、`factory_gear_base_loose`、`GroundPlane` 已出现 `[INFO] Explicit offset applied to scene collision`；`factory_gear_base_large` 因其 `collisions` 来自 reference（instance proxy），Python 无法编辑，已在场景 USD（`HIL_franka_gear.usda`）中对该 prim 的 override 里增加 `physx:restOffset = 0`、`physx:contactOffset = 0.001`。
- **Error 出现时机**：PhysX 在首次校验场景时（约 15,458ms）会打印 “rest offset must be lesser then contact offset” 等；我们的 Python 显式 offset 在此之后执行，因此首帧仍可能看到这些 Error，后续步进时数值已正确。若需完全消除首帧报错，需在加载场景后、创建/步进物理世界前完成 offset 设置（当前顺序下可接受）。

---

## 1. PhysxSceneAPI 求解器迭代次数

**日志：**
```text
[WARN] Failed to set solver iterations: 'PhysxSceneAPI' object has no attribute 'CreateSolverPositionIterationsAttr'
```

**原因：** Isaac Sim 5.x 中 PhysxSceneAPI 的接口变更，不再使用 `CreateSolverPositionIterationsAttr()`，改为 `CreateMaxPositionIterationCountAttr()` / `CreateMinPositionIterationCountAttr()`。

**处理：** 在 `_optimize_physics_precision()` 中已改为兼容写法：优先使用 `CreateMaxPositionIterationCountAttr` / `CreateMinPositionIterationCountAttr`，若不存在再回退到 `CreateSolverPositionIterationsAttr`。重新运行后该 WARN 应消失（若当前版本支持新 API）。

---

## 2. 碰撞体 Rest/Contact Offset 报错

**日志：**
```text
[Error] Collision rest offset must be lesser then contact offset, prim: /World/factory_gear_medium/factory_gear_medium/collisions
[Error] Collision contact offset must be positive and greater then restOffset, prim: ...
```
涉及 prim：
- `/World/factory_gear_medium/factory_gear_medium/collisions`
- `/World/factory_gear_base/factory_gear_base_loose/collisions`
- `/World/factory_gear_base/factory_gear_base_large/factory_gear_large/collisions`
- `/World/GroundPlane/CollisionPlane`

**原因：** 这些碰撞体要么在 **reference** 中（Traverse 时编辑目标未写到对应层），要么是 **instance proxy** 被跳过，导致仍为默认值。若默认是 `restOffset > contactOffset`（例如 rest=0.02、contact=0.001），PhysX 会报上述错误。

**处理：** 在 **`HIL_franka_gear.usda`** 中为所有报错路径增加 **override**，使加载时即有合法 offset，PhysX 校验前不再报错：
- **factory_gear_medium**：在 `over "factory_gear_medium"` 下增加 `over "collisions"`（PhysxCollisionAPI，`physx:restOffset = 0`，`physx:contactOffset = 0.001`）。
- **factory_gear_base_loose**：在 `over "factory_gear_base_loose"` 下增加 `over "collisions"`（同上）。
- **factory_gear_base_large / factory_gear_large**：在 `over "collisions"` 下已有 PhysxCollisionAPI 及 offset（此前已加）。
- **GroundPlane/CollisionPlane**：在 `def Plane "CollisionPlane"` 上增加 PhysxCollisionAPI 及 `physx:restOffset`、`physx:contactOffset`。

Python 中 `_optimize_physics_precision()` 仍对可编辑 prim 做显式 offset 设置，作为补充；reference 下的 prim 由上述 USD override 覆盖。

---

## 3. 夹爪碰撞体为 Instance Proxy（INFO）

**日志：**
```text
[INFO] Gripper collision /World/franka/panda_leftfinger/collisions/mesh_0 is instance proxy, skip explicit offset (inherits from prototype).
[INFO] Gripper collision /World/franka/panda_rightfinger/collisions/mesh_0 is instance proxy, skip explicit offset (inherits from prototype).
```

**原因：** Franka 以 **instance** 形式加载，夹爪下的碰撞体路径是 instance proxy，不能在其上直接编辑（否则会触发 “authoring to an instance proxy is not allowed”）。

**处理：** 显式夹爪 offset 逻辑已对 instance proxy **跳过**，避免崩溃。offset 由 **prototype** 继承；若需调整夹爪碰撞 offset，需在 Franka 的 prototype 资源上改。

---

## 4. 手-齿轮碰撞过滤（抓取时）与 physics:filteredPairs 警告

**设计：** 抓取时用「刚体对」方式在**齿轮 prim** 上添加 FilteredPairsAPI（target=左右指），关闭手-齿轮碰撞以减轻 FixedJoint+接触冲突抖动。不在夹爪 prim 上添加，避免破坏 instance 合成导致夹爪消失。

**逻辑：**
- 若**任一手指为 instance proxy**（Franka 以 instance 加载时常见）：**不添加** FilteredPairs，仅创建 FixedJoint，夹爪外观保持，抓取时可能有轻微抖动。
- 若左右指均**非** instance proxy：在齿轮 prim 上添加 FilteredPairs、左右指为 target，由代码完成碰撞过滤；释放时从齿轮的 FilteredPairsRel 移除左右指。

**日志：** 若曾对手指做过 FilteredPairs 或引擎在 pair 两端查找属性，可能出现 `[Warning] attribute physics:filteredPairs not found for path /World/franka/panda_leftfinger`，可忽略；当前实现已不在手指上添加 FilteredPairs。

---

## 5. 其他警告（可暂不处理）

| 日志 | 含义 | 建议 |
|------|------|------|
| `rendervar copy from texture directly to host buffer is counter-performant` | 渲染变量从纹理直接拷到 host 的性能提示 | 可忽略，或按 Isaac 文档改用 device buffer 再拷 host |
| `The rigid body at /World/franka/panda_fingertip_centered has a possibly invalid inertia tensor...` | TCP 刚体惯性/质量为占位值 | 若仅作逻辑用、不参与重力/碰撞，可忽略；否则在 USD 中为 TCP 设置合理 mass/inertia |
| `No adjacent samples found for interpolation at time 3907/30` | 重置后某时刻插值无相邻采样 | 多为重置时序导致，一般不影响逻辑 |
| `Annotator rgb already attached to ...` | 相机 annotator 重复挂载 | 多为扩展内部行为，可忽略 |

---

## 6. 修改小结

- **PhysxSceneAPI：** 使用 `CreateMaxPositionIterationCountAttr` / `CreateMinPositionIterationCountAttr`（兼容旧版）。
- **齿轮/地面碰撞 offset：** 对常报错的四条碰撞路径做显式 RestOffset/ContactOffset 设置（非 proxy 且带 CollisionAPI 才写）。
- **手-齿轮碰撞过滤：** 仅在齿轮 prim 上做刚体对过滤（FilteredPairsAPI，target=左右指）；若任一手指为 instance proxy 则跳过过滤以保持夹爪 visuals，释放时从齿轮 FilteredPairsRel 移除左右指。
