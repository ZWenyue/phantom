# Stage B 进展与验证 — Contact-Grounded Retargeting

> 配套设计文档：`contact_grounded_retargeting.md`（§2.3/§2.6 后端优化、§3 Stage B）
> 前置：`stage_a_progress.md`（Stage A 产出 `intent.npz`）
> Status: **已实现并端到端验证**（Zed RGB-D pick_and_place，184 帧，单臂）。

---

## 0. TL;DR

用**整段关节空间优化（FK 在环）**替换原 pipeline 的「逐帧解析 IK + GP/SLERP 事后平滑 + 超阈丢帧」：

```
minimize over q_{1:T}
    Σ_t  w_p(t)·‖FK_pos(q_t) − p_t*‖²          位置保真
       + w_r(t)·d_SO3(FK_ori(q_t), R_t*)²      姿态保真
       + w_s ·‖q_{t+1} − 2q_t + q_{t−1}‖²       二阶平滑（替代 GP/SLERP）
       + w_reg·‖q_t − q_neutral‖²               姿态正则
s.t. 关节限位（box bounds）（+ 可选软速度限）
```

**三个结构性病一次性解决**（设计 §2.5）：
- **无 IK 无解**：直接优化 `q`、FK 在环 → 可行即构造，最坏是残差增大，不会返回 NaN/无解。
- **无丢帧**：每帧都有 `q_t`，`TRACKING_ERROR` 逐帧丢帧逻辑退休（改为轨迹级质量剪枝，留待 Stage C）。
- **obs-action 一致**：标签存 `q_t` / `FK(q_t)`，与将来渲染的机器人像素同源。

---

## 1. 实现

### 1.1 新增/改动文件

| 文件 | 改动 | 说明 |
|---|---|---|
| `phantom/traj_opt.py` | **新增** | 优化器核心 `TrajectoryOptimizer` + `ArmKinematics` 接口 + `TrajOptConfig`。**不依赖 MuJoCo**，纯 numpy+scipy，可单测 |
| `phantom/processors/stageb_processor.py` | **新增** | `StageBProcessor(BaseProcessor)` + `MujocoPandaArm`（MuJoCo FK/Jacobian provider）+ 无头环境工厂 |
| `phantom/process_data.py` | 改 | 注册 `stageb` mode（enum + 两个处理顺序表 + registry） |
| `phantom/processors/paths.py` | 改 | 新增 `stageb_processor/`：`q_trajectory.npz`、`stageb_diagnostic.png` |
| `b/configs/pickplace_intent.yaml` | 改 | 新增 `stageb_*` 配置 |
| `b/tests/test_traj_opt.py` | **新增** | 合成 kinematics 单测（位置/姿态/平滑/边界） |

### 1.2 优化器核心（`traj_opt.py`）
- **决策变量**：整段单臂关节 `q_{1:T}`（Panda 7 DoF）。
- **求解器**：`scipy.optimize.least_squares(method='trf')`——`trf` 支持 box bounds，关节限位作为**硬约束**直接进 bounds。
- **解析稀疏 Jacobian**（`scipy.sparse`）：
  - 位置块 = `w_p · jacp`（世界系位置 Jacobian，精确）；
  - 姿态块 = `−w_r · jacr`（世界系角 Jacobian，SO(3) log-map 残差的一阶 GN 近似——单测证明符号/约定正确，偏航精确恢复）；
  - 平滑块 = 常量三对角（`+1,−2,+1`）；正则块 = 常量对角。
- **姿态残差**：`rotvec(R_t* · FK_ori(q_t)^T)`（世界系）。
- **依赖注入**：优化器只认 `ArmKinematics.fk(q)->(pos,R)` 与 `.jac(q)->(jacp,jacr)`，因此可用合成臂单测、用 MuJoCo 跑真实。

### 1.3 MuJoCo Panda provider（`stageb_processor.py`）— **与 Stage C 渲染环境同源**
- 无头 robosuite **`Phantom` 单臂环境**（`env_name="Phantom"`, `robots=["Panda"]`, `render_offscreen=False` → FK 不需要 GL）。**这与 Stage C 单臂渲染用的 `TwinRobot` 是同一套环境**（同 base、同 `gripper0_grip_site`），保证「FK 在环 == 渲染像素」。
  - ⚠️ **重要修正**：早期版本误用 `shoulders` **bimanual** 环境（含 `BASE_T_1` 旋转）做 FK，与单臂渲染环境坐标系不一致——这会让优化出的 `q` 在渲染时错位。已改到单臂环境后重跑，关键帧误差从厘米级降到 **0.04cm/0.2°**（见 §3.2）。
- FK：`sim.data.qpos[joint_indexes]=q; sim.forward()` → `site_xpos`（世界位置）+ `site_xmat`（世界朝向），site = `gripper0_grip_site`。
- Jacobian：`mujoco.mj_jacSite(...)` 取 `jacp`/`jacr`，切出 7 个臂关节列。
- 关节限位/中性位：`PANDA_JOINT_LIMITS` + `robot_model.init_qpos`。
- **base 位姿从 env 读取**（`body robot0_base` 的 `body_xpos/body_xmat`），不再硬编码变换：`Phantom` 环境 base = 纯平移 `[-0.56, 0, 0.912]`、旋转为单位阵。

### 1.4 帧变换 + warm-start
- **意图目标 robot 系 → 世界系**（用 §1.3 读到的 base）：`p_world = base_R·p_robot + base_t`；`R_world = base_R·(R_target · Rz(offset))`。单臂 `Phantom` 的 `base_R=I`、`base_t=[-0.56,0,0.912]`，即纯平移。`Rz(offset=90°)` 对齐 pipeline 夹爪约定（`HandModel` 用 `grasp_ori = gripper_ori @ Rz(90)`）；**§3.2 grasp 姿态误差 0.2° 证实该约定正确**。
- **warm-start**：轻量**逐帧位置-only DLS**（用同一 MuJoCo Jacobian，`J^T(JJ^T+λI)^{-1}·e`，时序播种，每帧 ≤20 迭代）给 `q_init`。不再依赖 `frantik`（frantik 走 bimanual `BASE_T_1`，与单臂环境不兼容）。可用 `stageb_warm_start=false` 关闭退回中性位。

### 1.5 关键调参：位置(米)/姿态(弧度)量纲重平衡
- **现象**：关键帧 `w_p=w_r=5` 时，姿态残差以弧度计（~0.5rad），位置以米计（~0.05m），平方和里姿态压过位置约 100×，导致 grasp/release **位置误差反而变大**（首跑 grasp 10.6cm / release 17.5cm）。
- **对策**：`stageb_ori_scale`（默认 `0.2`）统一乘到 `w_r`，让姿态不至于在关键帧压垮位置。这也符合设计取向——平行夹爪有旋转对称余量，抓取时**位置必须准**（要够到物体），姿态可在余量内让路（见 `stage_a_progress.md` §5.2 讨论）。

### 1.6 配置项（`stageb_*`）
```yaml
stageb_warm_start: true              # 逐帧 frantik/DLS 播种 q_{1:T}
stageb_grip_rot_offset_deg: 90.0     # 对齐 pipeline 夹爪约定（HandModel Rz90）
stageb_ori_scale: 0.2                # rad/m 量纲重平衡（姿态权重）
stageb_w_smooth: 1.0                 # 二阶平滑权重（替代 GP/SLERP）
stageb_w_reg: 0.01                   # 姿态正则（趋向中性位）
stageb_w_vel: 0.0                    # 软速度限惩罚（0 关闭；靠平滑项即可）
stageb_dq_max: 0.3                   # 软速度限阈（rad/帧）
stageb_max_nfev: 200                 # least_squares 最大评估次数
```

### 1.7 输出（`{demo}/stageb_processor/q_trajectory.npz`）

| 字段 | 形状 | 含义 |
|---|---|---|
| `q` | `(T,7)` | 优化后关节轨迹（**主产物**，Stage C 标签） |
| `pos_err` / `ori_err` | `(T,)` | 逐帧位置(米)/姿态(弧度)残差 |
| `ee_pos_world` / `ee_R_world` | `(T,3)`/`(T,3,3)` | `FK(q_t)` 世界系位姿 |
| `ee_pos_robot` / `ee_R_robot` | 同上 | `FK(q_t)` robot 系位姿（用 env base 逆变换，供 Stage C 标签与 intent 同系对齐） |
| `gripper_width` | `(T,)` | 夹爪宽度指令（承自 intent） |
| `phase` / `valid` | `(T,)` | 相位 / 全 True（无丢帧） |
| 其余 | — | `arm_side`/`robot_idx`/`joint_limits`/`cost_*`/`jerk_rms`/`nfev`/`success`/`grip_rot_offset_deg` |
| `stageb_diagnostic.png` | — | pos_err/ori_err 逐帧曲线 + 相位阴影 + 旧 5cm 丢帧阈参考线 |

---

## 2. 如何运行

```bash
cd phantom
# 需先有 Stage A 产出的 intent.npz（mode=intent）
python process_data.py --config-path=../b/configs \
    --config-name=pickplace_intent mode=stageb demo_num=0
```

优化器单测（无需 MuJoCo）：
```bash
python b/tests/test_traj_opt.py
```

---

## 3. 验证结果

### 3.1 优化器核心单测（`b/tests/test_traj_opt.py`，合成 7-DoF 臂，pos=q[:3]、R=Rz(q3)）
- **位置跟踪**：可达目标 → `pos_err` max `1e-5`，无限位违反。
- **姿态跟踪**：目标 `Rz(θ_t)` → `ori_err ≈ 0`，偏航 `q3` 精确恢复 → 证明 `jacr` 符号/约定 + rotvec 残差正确。
- **平滑去噪**：抖动 warm-start → jerk `0.080 → 0.0004`，位置仍 `~1e-4`。

### 3.2 端到端（`data/raw/pick_and_place`，Zed RGB-D，184 帧，左臂）— **单臂环境（修正后）**
用 Stage A 的真实 `intent.npz` 跑通 `mode=stageb`（单臂 `Phantom` 环境 + DLS warm-start + MuJoCo FK 整段优化，`ori_scale=0.2`）：

| 指标 | 结果 |
|---|---|
| 帧覆盖 | **184/184 全部有解、`valid` 全 True** —— 零丢帧 |
| 关节限位违反 | **0** |
| 目标 cost | `20.6 → 0.069`（下降 ~300×，`nfev=117`） |
| 位置误差 | mean **0.48cm**，max 10.9cm（尖峰在 free/transport，非关键帧） |
| 轨迹 jerk_rms | **0.0036**（二阶差分，平滑） |

分相位（体现设计 §2.4 的相位保真度预算）：

| 相位 | pos_err | ori_err | 帧数 |
|---|---|---|---|
| free | 0.35cm | 24.5° | 67 |
| **grasp** | **0.04cm** | **0.2°** | 3 |
| transport | 0.58cm | 28.2° | 111 |
| **release** | **0.49cm** | **0.4°** | 3 |

- **关键帧位置+姿态几乎完美**（grasp 0.04cm/0.2°、release 0.49cm/0.4°）→ 物体锚定抓取位姿在正确的单臂坐标系里**完全可达**，且 `Rz(90)` 夹爪约定正确（否则 grasp 姿态不会是 0.2°）。
- transport 姿态放任到 ~28°（`w_r` 很小，设计如此）；位置全程基本 <1cm，仅 free/transport 两处小尖峰（≤11cm）——这些帧在旧 pipeline 会因 >5cm 阈值被**整帧丢弃**，现在全部保留有解，交由 Stage C 轨迹级质量门判定。
- **对比早期 bimanual 环境误跑**（grasp 10.6cm/12.7°、release 17.5cm/20.8°）：修正环境一致性后关键帧误差降 **2~3 个数量级**，印证「FK 环境必须与渲染环境同源」。

### 3.3 EgoDex `basic_pick_place` demo 0（头戴 + `T_camera` 冻世界 + `T_place`）

肩部标定把 `G*` 丢到 Panda 底座后方。改成 `T_cam2robot(t)=T_place @ T_c2w(t)`：抓取手中点放到 `intent_place_target=[0.50,0,0.05]`（Mujoco 约 `[-0.06,0,0.96]`，底座前方）。接触相位不变（`grasp_kf=32` / `release_kf=88`）。`intent_grasp_offset=hand`。126/126 有解。

| 指标 | 固定外参 | 冻世界（肩部标定） | **冻世界 + T_place + 手 offset** |
|---|---|---|---|
| `p_target` Δ mean/max | 2.1 / 72.5 cm | 1.4 / 21.1 cm | **1.3 / 15.7 cm** |
| pos_err mean/max | 2.1 / 27.6 cm | 1.36 / 13.0 cm | **0.62 / 8.55 cm** |
| **grasp** kf t=32 | 5.8 cm / 12.6° | 9.9 cm / 27° | **0.22 cm / 0.2°** |
| **release** kf t=88 | 2.2 cm / 6.5° | 1.4 cm / 6.6° | **1.36 cm / 0.9°** |
| cost | 39.9 → 0.57 | 29.7 → 0.95 | **14.0 → 0.10** |
| 关键帧 max（grasp+release） | >3 cm | 10 cm | **1.36 cm（过 `retarget_key_pos_thresh=3cm`）** |

Grasp 与 Zed RGB-D（0.04 cm / 0.2°）同一量级。剩余尖峰在 free t=89（放完切回手跟随，8.6 cm，低 `w_p`）。Stage C 可以跑。

---

## 4. 已知遗留 / 后续

1. ~~**姿态约定需可视校验**~~ — ✅ **已解决**：Stage C 渲染后目视确认夹爪在 grasp 帧精确贴合物体，且 grasp 姿态误差 0.2° 数值印证 `Rz(90)` 约定正确。
2. ~~**环境一致性**~~ — ✅ **已解决**：Stage B FK 已改用与 Stage C 渲染**同一套单臂 `Phantom` 环境**（base 从 sim 读取），关键帧误差随之降 2~3 个数量级（§3.2）。
3. **姿态跟物体旋转**：当前 `R_target` 为常量（Stage A v1 假设），见 `stage_a_progress.md` §5.2 门控待办。
4. **收敛**：Zed RGB-D 上 `nfev=117 < 200` 即收敛。EgoDex demo 0 在 `T_place` 之后 cost 14→0.10，关键帧已是毫米/亚度；仍撞 `max_nfev=200` 是 transport/free 的平滑项，不影响 Stage C 门。
5. ~~**轨迹级质量剪枝**~~ — ✅ **已在 Stage C 实现**（见 `stage_c_progress.md`）：`TRACKING_ERROR_THRESHOLD` 逐帧丢帧退休，改为按关键帧残差/速度/jerk/限位**保留或丢弃整条 demo**。
6. **双臂**：当前单臂（`target_hand`）；双臂需扩展决策变量与（可选）双臂耦合项。

---

## 5. 里程碑对齐（设计文档 §7）
- **M1（RGB-D 主线打通）**：Stage A ✅ + **Stage B ✅** + **Stage C ✅**（三段全部跑通，见 `stage_c_progress.md`）。
- 关键结论：修正环境一致性后，物体锚定抓取位姿在单臂坐标系里**完全可达**，整段优化在关键帧达到 0.04cm/0.2°、全程无 IK 失败、无丢帧、平滑（jerk 0.0036）。
