# SLAM 约束的多帧手部 3D 姿态优化

## 1. 问题

Phantom pipeline 在 EgoDex/Epic 等自我中心（ego）视角下，手部 3D 姿态估计完全依赖 HaMeR 单帧单目预测（`Hand2DProcessor`），存在两个核心缺陷：

1. **帧间抖动**：HaMeR 逐帧独立推理，无时序约束，输出的 `kpts_3d` 帧间不一致
2. **相机运动被忽略**：pipeline 使用固定 `T_cam2robot`（`camera_extrinsics.json`），但 ego 视角下相机在持续运动，导致相机系下的 3D 关键点混入了相机自身运动的分量

后续的 GP 平滑（`SmoothingProcessor`）在 action 空间做事后插值，无法区分 "相机运动导致的表观位移" 与 "手部真实运动"，效果有限。

## 2. 核心思路

引入视觉 SLAM 估计逐帧相机位姿 `T_world_cam[t]`，将手部姿态投影到世界坐标系后做多帧一致性优化：

```
hand_world[t] = T_world_cam[t] · hand_cam[t]
```

在世界系下，手的运动应该是物理上平滑连续的。SLAM 剥离了相机运动分量，使得残差中仅剩 HaMeR 估计噪声，可以用时序约束高效消除。

## 3. 方法

### 3.1 SLAM 前端：逐帧相机位姿估计

对输入 RGB 视频运行 SLAM，输出全局相机轨迹：

```
输入: RGB 视频 (N 帧)
输出: T_world_cam[0..N-1]  (4x4 刚体变换)
```

SLAM 选型：

| 方法 | 类型 | 优势 | 劣势 |
|------|------|------|------|
| DROID-SLAM | 学习+优化 | 精度最高，单目/RGB-D 均可 | 需 GPU，~2GB 显存 |
| DPVO | 学习 VO | 专为 ego 设计，轻量快速 | 仅单目，无回环 |
| ORB-SLAM3 | 传统 | 成熟，CPU 可跑 | ego 场景下容易跟丢 |

**首选 DROID-SLAM**：pipeline 已有 GPU 需求（HaMeR、SAM2），精度对实验结论可信度至关重要。

### 3.2 单目尺度恢复（Scale Recovery）

单目 SLAM 输出的相机轨迹 `T_world_cam[t]` 缺少绝对尺度（up to scale）。若直接用于世界系投影，手部在世界系下会忽大忽小。EgoDex 为纯 RGB 无深度，必须显式恢复 metric scale。

**核心问题**：SLAM 的平移 `t_w_c[t]` 的单位是任意的，需要找到一个缩放因子 `s` 使得：

```
T_world_cam_metric[t] = [ R_w_c[t],  s · t_w_c[t] ]
                        [    0    ,       1        ]
```

三种互补方案，按推荐优先级排列：

#### 方案 S1：Metric 单目深度对齐（首选）

运行 Depth Anything v2（metric mode）获取每帧的 metric 深度图 `D_metric[t]`，与 SLAM 内部维护的逆深度做比值对齐：

```
s = median_t( median_pixels( D_metric[t] / D_slam[t] ) )
```

- **优势**：不依赖手部检测质量，全图像信息参与，最鲁棒
- **劣势**：需要额外跑一个深度模型（但 Depth Anything v2 很快，ViT-S 在 1080p 下 ~30ms/帧）
- **参考**：HaWoR (CVPR 2025) 采用此方案成功恢复 ego 视角下的手部 metric 尺度

#### 方案 S2：HaMeR 深度共识（轻量备选）

HaMeR 的 `T_cam_pred` 包含相机系下手部的 metric 3D 位置（Z 分量 = 手到相机的 metric 深度），可直接用于标定 SLAM 尺度：

```
对所有手部被检测的帧 t:
  Z_hamer[t] = T_cam_pred[t][2]                         # HaMeR 给出的 metric 深度 (m)
  Z_slam[t]  = 从 SLAM 稀疏地图中查询手部 2D 位置对应的深度   # SLAM 的 arbitrary 深度

s = median( Z_hamer[t] / Z_slam[t] )                    # 多帧中值，抑制 HaMeR 单帧噪声
```

- **优势**：不需要额外模型，直接利用 pipeline 已有输出
- **劣势**：HaMeR 的深度估计本身有噪声，且仅在手部区域有信号；SLAM 稀疏地图不一定在手部位置有点
- **适用条件**：手部在多数帧可见且 HaMeR 检测稳定

#### 方案 S3：初始标定外参反推（兜底）

利用已有的标定文件 `T_cam2robot_init`（首帧相机到机器人/身体的 metric 变换）反推绝对尺度：

```
# SLAM 将第 0 帧设为世界原点: T_w_c[0] = I
# 取 SLAM 第 k 帧 (k > 0)，此时相机相对首帧有可观测的位移
Δt_slam = t_w_c[k]                                       # SLAM 的 arbitrary 位移
Δt_metric = T_cam2robot_init⁻¹ @ T_cam2robot[k]          # 如果有多帧标定

# 若只有单帧标定，则需要一个已知 metric 参考：
#   - 手部首帧 HaMeR 深度 Z_hamer[0]
#   - 或场景中已知尺寸的物体（如操作台面高度）
s = ‖Δt_metric‖ / ‖Δt_slam‖
```

- **优势**：完全不需要额外模型
- **劣势**：仅靠单帧或少数帧标定，尺度估计不够鲁棒；ego 场景下 `T_cam2robot_init` 定义的是相机到身体的关系，不直接提供世界系位移
- **适用条件**：有多帧标定数据，或场景中有已知 metric 尺寸的参照物

#### 实验策略

| 阶段 | 尺度方案 | 理由 |
|------|---------|------|
| 快速验证 | S2 (HaMeR 深度共识) | 零额外依赖，先跑通 pipeline |
| 正式实验 | S1 (Depth Anything v2) | 最鲁棒，论文结果需要最强 baseline |
| 消融对比 | S1 vs S2 vs S3 | 展示尺度恢复方案对最终精度的敏感度 |

### 3.3 世界系投影与多帧优化

对 HaMeR 输出的相机系关键点 `hamer_cam[t]` (shape: `[21, 3]`)，求解优化后的 `hand_cam*[t]`。

世界系投影使用尺度恢复后的 metric 相机位姿（3.2 节）：

```
hand_world*[t] = R_w_c[t] · hand_cam*[t] + s · t_w_c[t]
```

优化目标函数：

```
min_{hand_cam*[t], L_b}
    Σ_t  ‖hand_cam*[t] - hamer_cam[t]‖²                                    (数据项)
  + λ₁ Σ_t ‖hand_world*[t+1] - hand_world*[t]‖²                            (世界系速度平滑)
  + λ₂ Σ_t ‖hand_world*[t+1] - 2·hand_world*[t] + hand_world*[t-1]‖²      (世界系加速度正则)
  + λ₃ Σ_t Σ_b (‖kpt*[t,j_b] - kpt*[t,k_b]‖ - L_b)²                      (骨骼长度约束)
```

其中：
- 数据项：优化结果不能偏离 HaMeR 观测太远
- 速度项（λ₁）：世界系下相邻帧手部位移应连续
- 加速度项（λ₂）：抑制高频抖动，等价于假设手运动近似匀速
- **骨骼长度约束（λ₃）**：手部骨段 b 连接关键点 j_b → k_b，其 3D 距离应在所有帧中保持恒定值 L_b

骨骼长度约束的关键细节：

- MANO 手部模型有 20 段骨骼（21 个关节），每段对应一对 (j_b, k_b)：
  ```
  手腕→掌根(×5), 掌根→近指节(×5), 近指节→中指节(×5), 中指节→远指节(×4), 远指节→指尖(×4)
  拇指: 掌根→近节, 近节→远节, 远节→指尖  (3段)
  共 20 段骨骼
  ```
- L_b 作为优化变量联合求解（不需要预设），优化器自动找到与所有帧最一致的骨长
- 这比用中值估计 L_b 更鲁棒：中值估计在遮挡严重时本身就不准
- 该约束在遮挡帧尤其关键：即使 HaMeR 预测大拇指从 5cm 跳到 3cm，优化器会通过 L_b 把它拉回真实长度

为什么这项约束特别强：
1. **物理硬约束**：骨骼长度是人体结构不变量，不像平滑性可以被违反
2. **跨帧耦合**：一帧的骨长观测影响所有帧的估计，单帧遮挡的损坏被多帧分摊
3. **互补性**：速度/加速度约束管时序平滑，骨骼约束管空间结构一致性，两者正交

优化方法：21 个关键点 × 3 坐标 × N 帧 + 20 个骨长 = 63N + 20 个变量，稀疏带状结构，可用 `scipy.optimize.minimize` (L-BFGS-B) 或 Ceres/GTSAM factor graph。

### 3.3 Pipeline 集成

```
现有 (epic):  bbox → hand2d → arm_seg → action → smoothing → inpaint → robot

改后:         bbox → hand2d → arm_seg → slam → hand3d_slam → action → inpaint → robot
                                         ↑新增   ↑新增（替代 smoothing）
```

新增两个 processor：
- `SLAMProcessor`：运行 SLAM，输出相机轨迹 `camera_poses.npz`
- `SLAMHand3DProcessor`：加载 HaMeR 结果 + 相机轨迹，运行多帧优化，输出 refined `kpts_3d`

优化后的轨迹已在世界系下平滑，**不再需要** `SmoothingProcessor` 的 GP 平滑。

### 3.4 Per-frame T_cam2robot 替代固定外参

`ActionProcessor._convert_pts_to_robot_frame` 当前使用固定 `T_cam2robot`：

```python
# 现有: 所有帧用同一个变换
skeleton_poses_rf = T_cam2robot @ skeleton_poses_cf

# 改为: 每帧用 SLAM 提供的相机位姿
T_robot_world = T_cam2robot_init @ inv(T_world_cam[0])  # 首帧对齐
T_cam2robot[t] = T_robot_world @ T_world_cam[t]         # per-frame
skeleton_poses_rf[t] = T_cam2robot[t] @ skeleton_poses_cf[t]
```

首帧的 `T_cam2robot_init` 仍来自标定文件，后续帧通过 SLAM 的相对运动推算。

## 4. 实验设计

### 4.1 数据集

| 数据集 | 相机类型 | 深度 | 场景 | 用途 |
|--------|---------|------|------|------|
| EgoDex | ego (移动) | 无 | 桌面操作 | 主实验 |
| Phantom (原版 Zed2) | 固定三脚架 | 有 | 桌面操作 | 对照组 |

### 4.2 Baselines

| 编号 | 方法 | 描述 |
|------|------|------|
| B0 | HaMeR only | 现有 `Hand2DProcessor`，无任何后处理 |
| B1 | HaMeR + GP smooth | 现有完整 pipeline（`Hand2DProcessor` → `SmoothingProcessor`） |
| B2 | HaMeR + world proj | 仅用 SLAM 投影到世界系，无优化（验证坐标变换本身的价值） |
| **Ours** | HaMeR + SLAM optim | 完整方案：SLAM + 多帧优化 |

### 4.3 评估指标

#### 4.3.1 手部姿态质量（无需 ground truth）

| 指标 | 定义 | 衡量什么 |
|------|------|---------|
| **Jerk (世界系)** | `mean(‖d³pos/dt³‖)` 世界系下关键点加加速度 | 轨迹平滑度 |
| **帧间一致性** | 世界系下相邻帧同一关键点的位移标准差 | 时序稳定性 |
| **骨骼长度方差** | 同一手指骨段长度在序列内的 std/mean | 3D 结构一致性（骨长应恒定） |

#### 4.3.2 手部姿态精度（如有 ground truth）

| 指标 | 定义 |
|------|------|
| **MPJPE** | Mean Per-Joint Position Error (mm) |
| **PA-MPJPE** | Procrustes-aligned MPJPE |
| **PCK@k** | Percentage of Correct Keypoints within k mm |

#### 4.3.3 下游任务（端到端验证）

| 指标 | 定义 | 衡量什么 |
|------|------|---------|
| **Action MSE** | 与 GT action 的均方误差 | action 提取质量 |
| **Policy success rate** | Diffusion Policy 训练后在仿真环境中的任务成功率 | 最终目标 |
| **训练数据利用率** | 达到相同 success rate 所需的 demo 数量 | 数据效率 |

### 4.4 消融实验

| 消融 | 修改 | 验证什么 |
|------|------|---------|
| w/o 骨骼约束 | λ₃ = 0 | 骨骼长度约束对 3D 结构一致性的贡献 |
| w/o 加速度项 | λ₂ = 0 | 加速度正则的必要性 |
| w/o 数据项 | λ_data → 0 | 过度平滑的风险 |
| 窗口大小 | W = {5, 10, 20, 全序列} | 局部 vs 全局优化的 tradeoff |
| SLAM 选型 | DROID vs DPVO vs ORB-SLAM3 | SLAM 精度对最终结果的敏感度 |
| 尺度恢复方案 | S1 (Depth Anything) vs S2 (HaMeR depth) vs S3 (标定外参) | 尺度精度对世界系优化的影响 |
| SLAM 退化 | 人为在 ego 快速旋转片段测试 | 方法的鲁棒性边界 |

### 4.5 分析实验

1. **Ego vs. Static 对比**：在 EgoDex（ego）和 Phantom（static）上分别跑全部方法，预期 ego 场景收益显著更大
2. **SLAM 置信度相关性**：分析 SLAM tracking 质量（协方差/重投影误差）与手部估计改善程度的关系
3. **遮挡鲁棒性**：手部被遮挡帧，SLAM + 运动模型能否比 carry-forward 插值更好地预测手部位置
4. **计算开销分析**：SLAM 前端 + 优化器的额外时间 vs. 总 pipeline 时间占比

### 4.6 可视化方案

论文/报告中的关键图表，按说服力排序：

#### Fig 1. 3D 轨迹对比图（Trajectory Plot）

食指尖（index fingertip, keypoint #8）在某段操作中的世界系 3D 轨迹。三列子图分别为 X/Y/Z 轴随时间的变化，叠加 B0、B1、Ours 三条曲线。

- **预期效果**：B0 锯齿严重（HaMeR 逐帧抖动 + 相机运动混入）；B1 经 GP 平滑后稍好但仍有低频漂移；Ours 是一条平滑的人手运动弧线
- **变体**：同一数据也可画成 3D 空间曲线（三轴合一），颜色编码时间，更直观展示运动路径的几何形状

#### Fig 2. 骨骼长度稳定性图（Bone Length vs. Time）

横轴帧数，纵轴手掌到中指尖（palm → middle fingertip）的 3D 欧氏距离（mm）。每个方法一条线。

- **预期效果**：B0/B1 上下波动大（骨长不应变化，波动 = 3D 估计不一致）；Ours 接近水平直线
- **标注**：在图上标出各方法的 CV（变异系数 = std/mean），数字越小越好

#### Fig 3. 关键点重投影一致性（Reprojection Overlay）

选取 4-6 个关键帧，将 Ours 优化后的世界系 3D 关键点通过 SLAM 相机位姿反投影到图像上，叠加在原始 RGB 帧上。左右对比：左列 HaMeR 原始投影，右列 Ours 投影。

- **预期效果**：HaMeR 原始投影在部分帧会偏离手部轮廓（特别是遮挡/运动模糊帧）；Ours 投影持续贴合

#### Fig 4. Jerk 分布直方图

所有关键点在整段序列上的 jerk（三阶导数范数）分布直方图，四个方法叠加。

- **预期效果**：B0 分布右偏（大量高 jerk 值）；Ours 集中在低 jerk 区域
- **补充**：可加一个 box plot 版本，更适合多序列聚合比较

#### Fig 5. SLAM 相机轨迹 + 手部轨迹联合可视化

在同一个 3D 场景中渲染：(1) SLAM 估计的相机运动轨迹（蓝色），(2) 优化后的手部关键点轨迹（红色），(3) 关键帧的相机视锥。

- **用途**：直观展示 "相机在运动" 这一前提，以及手部轨迹与相机轨迹的解耦效果
- **工具**：Open3D 或 Plotly 3D

#### Fig 6. 下游 Policy 学习曲线

横轴训练 step，纵轴仿真环境中的 task success rate。四条线对应四种方法生成的训练数据。

- **预期效果**：Ours 训练数据对应的 policy 收敛更快、最终成功率更高
- **补充**：可加 "数据效率" 子图 — 横轴 demo 数量，纵轴达到 80% 成功率所需的 demo 数

#### Fig 7. 失败案例与 SLAM 退化分析

选取 SLAM 跟踪质量下降的片段（快速旋转、低纹理），展示该段的手部估计质量对比。

- **用途**：诚实展示方法边界，reviewer 会更信任有 failure analysis 的论文

## 5. 实现计划

### Phase 1: SLAM 集成（~1 周）

- [ ] 封装 DROID-SLAM 为 `SLAMProcessor`，输入 RGB 视频，输出 `camera_poses.npz`
- [ ] 在 EgoDex 数据上验证 SLAM 跟踪质量（可视化轨迹、检查跟丢率）
- [ ] 实现尺度恢复：先用 S2 (HaMeR depth consensus) 快速验证，再接入 S1 (Depth Anything v2)
- [ ] 处理 SLAM 失败的 fallback（退化为固定外参）

### Phase 2: 多帧优化器（~1 周）

- [ ] 实现 `SLAMHand3DProcessor`：加载 HaMeR `kpts_3d` + `camera_poses`，构建优化问题
- [ ] 用 `scipy.optimize.minimize` (L-BFGS-B) 实现求解器
- [ ] 调参 λ₁, λ₂（在小规模数据上 grid search）

### Phase 3: Pipeline 集成（~3 天）

- [ ] 注册新的 processing mode 到 `process_data.py`
- [ ] 修改 `ActionProcessor` 支持 per-frame `T_cam2robot`
- [ ] 添加 config 项（`use_slam`, `slam_model`, `optim_window_size`, `lambda_*`）

### Phase 4: 实验与分析（~2 周）

- [ ] 跑 4.2 中所有 baselines 和消融
- [ ] 生成可视化（轨迹对比图、骨骼长度随时间变化、世界系 3D 轨迹）
- [ ] 跑下游 policy 训练实验
- [ ] 整理结果、写分析

## 6. 风险与缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| SLAM 在 ego 手部遮挡严重时跟丢 | 无法获得有效相机位姿 | 检测 SLAM 置信度，低置信段退化为帧间插值 |
| HaMeR 3D 估计本身误差大于 SLAM 能纠正的范围 | 优化收益有限 | 数据项权重自适应（HaMeR 置信度低时降低权重） |
| 优化问题非凸，可能陷入局部最优 | 输出质量不稳定 | HaMeR 原始估计作为初始值，通常已接近全局最优 |
| DROID-SLAM 环境安装复杂 | 拖慢开发进度 | 先用 DPVO 快速验证方案可行性，再切换 |
