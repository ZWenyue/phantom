# Contact-Grounded, Feasibility-Aware Retargeting

> 面向"人类视频 → 机器人训练数据"的重定向重构方案
> Status: proposal / design draft
> Scope: 替换现有 `action → smoothing → robot_inpaint` 中的动作抽取与 IK 环节

---

## 0. TL;DR

把现在这条 **"逐帧手腕映射 → 事后平滑 → 逐帧硬 IK → 贴图"** 的开环解耦级联，
换成一条 **"接触事件锚定意图 → 一次整段轨迹优化"** 的主线。

核心 claim：

> **为训练造数据时，(观测, 动作) 对的"可行性"和"一致性"是可以在构造上被保证的；
> 并且保真度预算应集中在接触事件（抓取/释放）上，而非逐帧复刻人手腕。**

一次性解决当前三个结构性病：

1. **IK 无解**（`frantik.ik` 要求完整 6D 位姿精确可达，腕部朝向超限即全 NaN）；
2. **丢帧**（`pos_err > TRACKING_ERROR_THRESHOLD` 的帧被整帧丢弃，轨迹出现时序空洞）；
3. **观测-动作标签错位**（EE 标签存的是人手目标 `left_state.pos`，画面渲染的是 IK 执行位姿 `FK(q)`，二者在 IK 有残差处系统性不一致）。

---

## 1. Motivation

### 1.1 现状回顾

当前 pipeline（`phantom/processors/`）：

| Stage | 文件 | 做什么 |
|---|---|---|
| action | `action_processor.py` | 手 21 关键点 → 机器人坐标系 → `HandModel` → EE 位姿(pos, ori) + 夹爪宽度，**逐帧独立** |
| smoothing | `smoothing_processor.py` | 位置 GP、姿态 SLERP、宽度 GP，**事后平滑** |
| robot_inpaint | `robotinpaint_processor.py` | frantik + MuJoCo DLS **逐帧 IK** → q，渲染叠加，保存训练数据 |

### 1.2 三个结构性病根（附代码证据）

**(a) IK 无解** — `phantom/panda_frantik_ik.py`

```python
q = self._frantik_seed(T_be, q_seed)
if q is None:
    return None, float("inf")     # 解析 IK 全 NaN → 直接"无解"
```

`frantik.ik` 是 Franka 解析 IK，要求 **位置 + 姿态整个 6D 精确可达**。人手腕的
roll/pitch 常超出 Panda 腕部关节限位 → 所有 q7 采样返回 NaN → 无解。

**(b) 丢帧** — `phantom/processors/robotinpaint_processor.py`

```python
if robot_results['pos_err'] > self.TRACKING_ERROR_THRESHOLD:
    return None            # 该帧被丢弃 → 轨迹出现时序空洞
```

**(c) 观测-动作标签错位** — `phantom/processors/robotinpaint_processor.py`

```python
sequence.add_frame(TrainingData(
    action_pos_left=left_state.pos,        # ← 人手推出的"目标"位姿
    action_orixyzw_left=left_state.ori_xyzw,
    ...
    joint_pos_left=jp_left,                # ← IK 解出的"执行"关节（画面像素来自它）
))
```

EE 动作标签 = 人的目标；画面里的机器人像素 = `FK(jp)` 执行位姿。IK 有残差时二者
系统性偏差 → 若在 EE 空间训策略（Phantom 原文用 Diffusion Policy on EE），
这些帧提供的是**轻微错误的监督**。

### 1.3 为什么不直接搬 OKAMI

- OKAMI 的 **object-aware warping** 是**部署时**对真实可移动物体做的几何适配；
  在**离线编辑固定真实视频**的范式下，warp 轨迹会让机器人与画面里的物体错位，
  产出无效 (观测, 动作) 对。→ 不适用。
- OKAMI 的 **加权软 IK**（Pink，位置权重 1.0 / 姿态 0.08）只是"逐帧"层面的工程解，
  能治"无解"，但治不了丢帧与标签错位。本方案是它的"整段 + 一致性"原理版。

---

## 2. Formulation

### 2.1 两半其实是一件事

- **接触/抓取感知的动作抽取（前端）** 定义"意图" = 轨迹优化要逼近的 cost；
- **可行性感知的整段轨迹优化（后端）** 求解这个 cost，产出可行、平滑、一致的 `q_{1:T}`。

不是两个模块并列，而是 **"前端定义 cost，后端求解 cost"** 的一条自洽主线。

### 2.2 意图表示

对每个 demo，抽取一串任务空间目标：

- 逐帧 EE 位置目标 `p_t*`（接触段用**物体相对**坐标系表达）；
- 逐帧 EE 朝向目标 `R_t*`；
- 接触/释放关键帧的**由物体几何锚定的抓取位姿** `G*`；
- 离散抓合状态 `g_t ∈ {open, closed}`；
- 相位标签 `phase_t ∈ {free, grasp, transport, release}`。

### 2.3 后端优化目标

决策变量：整段关节轨迹 `q_{1:T}`（每臂），夹爪指令由 `g_t` + 物体宽度导出。

```
minimize over q_{1:T}:
   Σ_t [  w_p(t) · || FK_pos(q_t) − p_t* ||²                 # 位置保真
        + w_r(t) · d_SO3( FK_ori(q_t), R_t* )²               # 姿态保真
        + w_s   · || q_{t+1} − 2 q_t + q_{t−1} ||²           # 二阶平滑（替代 GP/SLERP）
        + w_reg · || q_t − q_neutral ||²          ]          # 姿态正则（自然姿态）

subject to:
   关节限位 q_min ≤ q_t ≤ q_max
   速度限位 || q_{t+1} − q_t || ≤ Δq_max
   （可选）自碰撞 / 桌面碰撞惩罚
```

> **先验背书（HOP, Singh et al. 2024）**：本目标与 HOP 的 *simulator-in-the-loop
> retargeting*（其 Eq. 2）本质一致——都是**在关节空间优化、FK 在环内**：
>
> ```
> min_{a[k]}  ½ · || x_h[k] − f(a[k]) ||²  +  λ · || a[k] − φ[k−1] ||²   s.t. a[k] ∈ A
> ```
>
> 其中 `f` = 机器人 FK，`x_h` = 人手关键点，第二项 = 能量正则（贴近上一帧配置）。
> 这印证了"直接优化 `q` → 可行即构造 → 无 IK 无解"的核心思路。
> 本方案相对 HOP 的**增量**是：(a) **相位相关权重**（见 §2.4）；(b) **整段二阶平滑**
> 而非仅逐帧对上一帧正则；(c) 面向**夹爪**的关键点匹配（见下）。

> **可借的 cost 细节（来自 HOP）**：把位置项换成/补充为**关键点匹配**——
> HOP 匹配一组手指尖关键点而非单一 6D EE pose，天然处理形态学映射。
> 迁到夹爪：用 `thumb-tip / index-tip → 夹爪两指位置` 的匹配来表达抓取意图，
> 比"合成一个离散 6D 抓取位姿 `G*` 再匹配"更平滑、更易调。二者可并存：
> 自由段用 EE 位置项，接触段用指尖关键点项锁定抓取。

### 2.4 关键设计：相位相关的权重

保真度预算集中在真正重要的接触时刻：

| 相位 | `w_p` | `w_r` | 说明 |
|---|---|---|---|
| free / transport | 中 | **很低** | 只防姿态翻转，给可行性与平滑让路 |
| grasp / release | **高** | **高** | 锁定到合成抓取位姿 `G*`，位置+姿态都精确 |

### 2.5 两个"免费"红利

- 直接优化 `q` 且 FK 在环内 → **可行即构造**，永远有最优努力解 →
  **不再有"无解"，不再丢帧**。
- 渲染的机器人 = `FK(q_t)` = 保存的标签 → **观测-动作一致性同时被解决**
  （标签直接存 `q_t`，或存 `FK(q_t)` 作为 EE 标签，二者与像素一致）。

### 2.6 求解方法

- 整段 Gauss-Newton / Levenberg-Marquardt；MuJoCo 已提供 FK + Jacobian
  （现有 `_refine_mujoco` 的 DLS 就是"单帧位置版"，扩成"整段 + 含姿态项 + 时序耦合"）。
- 用现有逐帧解（frantik/DLS 结果）作 **warm-start**，加速收敛、稳住局部最优。

---

## 3. 三阶段实现

### Stage A — 感知 + 意图抽取（改造 `action_processor`）

1. **手/身体重建**：沿用 HaMeR，得到手腕轨迹 + 手指关键点 + 接近方向。
2. **物体点云**：分割任务物体、反投影深度得逐帧点云。
   - ⚠️ 新增输入：现在 action 阶段**完全没有物体信息**（repo 有 SAM2/DINO 检测器可复用，但未接线）。
   - 遮挡时的选项：可用 **MCC-HO**（Wu et al. 2024，HOP 采用）——以手为锚**联合推理手-物点云**，
     在部分遮挡下也能补出物体几何（HOP 为通用性砍掉 CAD 拟合，代价是精度/时序一致性下降，但对本用途够用）。
3. **接触检测**：指尖-物体表面距离（或轻量接触分类器）判定接触起止，
   切分 `free → grasp → transport → release`，输出 `g_t` 与抓/放关键帧。
4. **手→夹爪抓取合成**：抓取起始帧，不复刻五指位姿，
   而是从物体点云找 **antipodal 抓取**：闭合轴对齐 thumb-index 轴、
   接近方向对齐人手接近方向 → 输出物体几何锚定的 `G*`。

### Stage B — 可行性感知轨迹优化（吸收 `smoothing` + 替换逐帧 IK）

按 §2 求解 `q_{1:T}`。相位相关权重、关节/速度约束、warm-start。

### Stage C — 渲染 + 标签 + 质量剪枝（`robot_inpaint`）

- 渲染 `FK(q_t)`；标签 = `q_t`（+ 由 `g_t` 得到的夹爪指令）。
- `TRACKING_ERROR` 逐帧丢帧逻辑退休（所有帧可行），改为**轨迹级质量剪枝**。
- **质量剪枝（借鉴 HOP）**：HOP 只保留"重定向残差 < 3cm 且不与桌/地碰撞"的轨迹。
  本方案统一到轨迹级：按关键点匹配残差 + 碰撞 + 平滑度阈值**保留/丢弃整条 demo**，
  而不是在时间轴上戳洞。与现有 `TRACKING_ERROR_THRESHOLD` 同源，可合并。

### 关于"数据多样性 / 物体位置泛化"（合法做法）

- 你最初想要的"按物体位置适配轨迹"，**正确路线是在仿真里随机化 + 多次优化**，
  而不是在真实视频上 warp（会让机器人与画面物体错位，产出无效样本）。
- HOP 的做法可参考：对每条视频**跑多次优化**（HOP 用 700 次），随机化桌位/机器人初始
  关节，靠冗余自由度产出多样关节轨迹。这条**仅在仿真支路成立**；真实视频编辑支路
  仍靠"多样化真实视频 + 下游策略泛化"。

---

## 4. 与现有 stage 的替换点

| 现有 stage | 新方案 | 备注 |
|---|---|---|
| `action_processor`（逐帧手腕→EE + thumb-middle 宽度） | 变成"意图抽取"：加物体点云 + 接触检测 + 抓取合成 | `HandModel` 复用做接近方向 |
| `smoothing_processor`（GP+SLERP 事后平滑） | 被优化里的二阶平滑项吸收 | 可删或仅作初值 |
| `robotinpaint` frantik/DLS 逐帧 IK + 丢帧 | 换成整段轨迹优化 | 复用 MuJoCo FK/Jacobian |
| 训练标签 `action_pos = human target` | 标签 = `q_t` 或 `FK(q_t)` | 与像素一致 |

---

## 5. 实验与消融

基线：现有 Phantom pipeline（逐帧 IK + GP/SLERP + human-target 标签）。

指标：

- 下游策略成功率（主指标）；
- IK 失败率 / 丢帧数（可行性）；
- 轨迹 jerk / 二阶差分（平滑度）；
- 观测-动作偏差幅度 `|| FK(q_t) − action_pos_t ||`（一致性）。

消融：

1. **抓取合成 vs 原始手腕映射** → 抓取成功率、对重建噪声/快速运动的鲁棒性
   （正面回应 OKAMI 报告的"demonstrator 动作快 → 重建退化"痛点）；
2. **相位相关权重 vs 均匀权重**；
3. **整段优化 vs 逐帧 IK + 平滑**；
4. **标签 = human-target vs FK(executed)** → 量化一致性对策略成功率的价值；
5. **关键点匹配（指尖）cost vs 单一 6D EE pose cost**（HOP 式 vs 现有）→
   对可行性、抓取自然度的影响。

---

## 6. 风险与对策

| # | 风险 | 对策 |
|---|---|---|
| 1 | **抓取时物体被手遮挡**，点云缺失 → antipodal 抓取估计噪声大 | 用**接触前（未遮挡）**点云估好 `G*` 再锁定；或把物体当刚体从遮挡前**跟踪**位姿带到接触帧；或用 **MCC-HO** 式手-物联合重建在遮挡下补几何（见 Stage A）；必要时加类别级**形状补全 / amodal**。**注意：这是遮挡问题，单目深度估计模型解决不了**（它只能给可见像素估深度，补不出手背后的表面）。 |
| 2 | **单目接触起止检测噪声大**，误报抓取事件 = 坏样本 | 鲁棒事件检测 / 轻量学习模型；结合手速、指-物距离、光流多线索；对关键帧做时序一致性约束 |
| 3 | **依赖目标物体识别**（意图/物体相对系需知道"哪个是目标"） | 检测/分割用现成 Grounded-DINO / SAM2；**目标选择**用 VLM（如 GPT-4V，OKAMI 做法）或"接触/在动的物体 = 目标"启发式（与 #2 耦合互助）；需处理**跨帧关联 + 穿遮挡跟踪**（Cutie/SAM2 video） |
| 4 | **RGB-only（Masquerade）3-4cm 深度误差** 让接触检测 + 点云抓取更难 | **先在 EgoDex/Phantom 的 RGB-D 上跑通主线**；in-the-wild 再上**视频级、时序一致 + 度量**的深度模型（逐帧图像深度模型会 flicker/尺度漂移，不可用）。EgoDex 本身也无场景深度：DA3 + 手 GT scale-lock 后仍有 ~5–7 cm 残差，补偿须加在深度上（见 `egodex_depth_residual.md`），不要放宽 3 cm 接触阈 |

### 关于"先跑深度估计模型"的澄清

- 对 **#4（RGB-only 缺深度）**：能**大幅缓解**，但须用视频级时序一致 + 度量深度，
  且仅对 RGB-only 支路有用（Phantom 用 Zed 本就有 RGB-D）。仍受尺度/一致性误差限制。
- 对 **#1（抓取遮挡）**：**无效**。这是"根本看不见"的遮挡问题，不是"看得见但没深度"，
  深度模型补不出手背后的物体表面。
- 对 **#3（目标识别）**：检测/分割现成；但"目标选择 + 穿遮挡跟踪 + 接入 action 阶段"仍需自己做。

---

## 7. 里程碑

- **M1（RGB-D 主线打通）**：在 EgoDex/Phantom RGB-D 上实现 Stage A（物体点云 + 接触检测 + 抓取合成）与 Stage B（整段优化），跑通单臂单任务，验证"无 IK 失败、无丢帧、obs-action 一致"。
- **M2（质量量化）**：完成 §5 四项消融，给出相对基线的可行性/一致性/平滑度指标改善。
- **M3（策略验证）**：用新数据训下游策略，报告成功率提升；重点验证抓取合成对鲁棒性的贡献。
- **M4（in-the-wild 扩展，可选）**：接视频级深度模型，扩到 Masquerade/Epic RGB-only；评估遮挡与深度误差下的退化。

---

## 8. 与已有工作的关系（related work 抓手）

- **OKAMI**：部署时逐步加权 IK + 物体 warp；无整段可行性优化、无面向夹爪的接触锚定抓取合成、不关心训练数据一致性。
- **HOP（Hand-Object Interaction Pretraining, Singh et al. 2024）**：与本方案最接近的先验工作，
  提供了 *simulator-in-the-loop retargeting*（关节空间 FK-in-loop 优化 + 能量正则 + 残差<3cm 剪枝）、
  MCC-HO 手-物联合 3D lifting、以及仿真随机化增广。**但范式相反**（见下），且目标是"任务无关预训练 prior + RL/BC 微调"、
  面向灵巧手；本方案借其**动作侧**配方，服务于 Phantom 的数据编辑范式。
- **Phantom / Masquerade**：逐帧解耦映射 + 事后平滑；本方案换成联合可行性 + 一致性表述。
- **人手→夹爪抓取合成**：连接抓取检测文献，但**以人类演示为条件**（接近方向对齐），是新意所在。

### 8.1 关键范式差异与综合点（HOP vs Phantom）

| | 观测来源 | 动作来源 | 优点 | 代价 |
|---|---|---|---|---|
| **HOP** | **仿真**渲染（sim depth/pointcloud） | sim-in-the-loop 优化 | 物理有依据、动作可行、**obs-action 天然一致**、可无限增广 | **sim-real gap**（合成观测） |
| **Phantom（本仓库）** | **真实视频**合成机器人 | 逐帧重定向 | **真实像素**（真背景/光照/物体） | 动作来自不完美重定向（obs-action 错位）+ 无物理 |

**综合点（本方案的定位）**：取两者之长——

> 用 **HOP 式 sim-in-the-loop 优化拿到可行的 `q`（动作侧）**，
> 再保留 **Phantom 把 `FK(q)` 合成到真实视频（观测侧）**。
> 结果 = 真实观测 + 可行动作 + **构造上一致**（渲染的就是 `FK(q)`）。

注意：本方案**不需要** HOP 的 sim 观测（我们有真实视频），只借它的重定向优化与质量剪枝。