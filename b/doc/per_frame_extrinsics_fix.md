# Per-frame 相机外参修复

`ActionProcessor` 把手部关键点从相机系转到机器人系时，整段序列使用同一个固定 `T_cam2robot`。这个假设只在相机相对场景刚性固定时成立（原版 Phantom 的 Zed2 三脚架），在 EgoDex 等第一人称数据上不成立 —— 相机跟着头动，相机自身的运动被当成了手的运动写进 action label。

本文档记录该 bug 的量化诊断、修复方案和验证方法。

## 1. 问题

```python
# phantom/processors/action_processor.py 原代码
skeleton_poses_rf = T_cam2robot @ skeleton_poses_cf   # 所有帧同一个变换
```

`T_cam2robot` 来自标定文件 `camera/camera_extrinsics_ego_bimanual_shoulders.json`，是一个 4x4 常量矩阵。对 EgoDex：

- 相机（Vision Pro 头显）在录制过程中持续小幅运动
- 手在世界系里的真实运动 = 相机系观测 - 相机自身运动
- 用固定外参 = 假设相机自身运动为 0 → 相机运动被完整地混入手部轨迹

后续 `SmoothingProcessor` 的 GP 平滑在 action 空间做事后插值，无法区分"相机运动导致的表观位移"和"手部真实运动"。

## 2. 诊断结论

诊断脚本 `b/diagnose_ego_camera_motion.py`（block 1-3），EgoDex `basic_pick_place`，20 个 episode。所有指标对照 EgoDex 自带的 GT（`transforms/camera` 逐帧头显位姿 + `transforms/<joint>` 世界系手部关节）。

### 2.1 相机运动特征：原地抖动，不是持续位移

| 量 | 值 |
|---|---|
| 平移速度（均值） | 0.036 m/s |
| 角速度（均值 / p95） | 9.05 / 35 °/s |
| 路径长度 | 0.05 – 0.36 m |
| **净位移（首尾）** | **0.004 – 0.02 m** |
| **总旋转（首尾）** | **1.4 – 3.4°** |
| Episode 长度 | 47 – 307 帧（1.6 – 10.2 s） |

净位移只有 1 cm 而路径长度有 12–36 cm：头部在原地来回晃动并回到起点，不是走动。**这一点很重要** —— 它同时说明了两件事：固定外参的误差来自累积抖动而非大幅移动；以及单目 SLAM 在这个数据上处于退化配置（基线 ≈ 1–2 cm，深度 ≈ 0.5 m，旋转主导），不适合用来估相机位姿。EgoDex 本来就提供 GT 位姿，无需 SLAM。

### 2.2 核心指标：假设手部估计完美时的不可避免误差

`ActionProcessor` 用一个固定变换把相机系映射到机器人系，所以它的输出就是 `hand_cam[t]` 差一个刚体变换。因此把"最佳拟合刚体变换后的残差"作为该假设的不可避免误差 —— 相机静止时该残差恒为 0。这个量与相机系约定、MANO/ARKit 关节定义偏移、以及未知的 `T_cam2robot` 都无关。

| | 中位数 | 跨 episode 范围 |
|---|---|---|
| 全 21 关节 | **25.1 mm** | 6.2 – 66.2 mm |
| 仅腕部 | 19.8 mm | — |

按固定外参保持的窗口长度分解（误差随时长累积）：

| 窗口 | 0.5 s | 1 s | 2 s | 5 s |
|---|---|---|---|---|
| 残差 | ~15 mm | ~30 mm | ~51 mm | ~79 mm |

参照：并联夹爪抓取容差约 5–10 mm；Diffusion Policy 的 action chunk 通常 0.5–2 s，对应 15–50 mm。**这个误差量级足以导致抓取失败。**

### 2.3 误差是速度依赖的，且集中在操作关键相位

按手部真实速度分层后，相机污染的程度完全不同：

| 手的状态 | 世界系真实速度 | 相机系表观速度 | 表观/真实 | 相机主导的帧占比 |
|---|---|---|---|---|
| **闲置手** | 0.008 – 0.014 m/s | 0.037 – 0.144 m/s | **4 – 12×** | **79 – 99%** |
| 工作手 | 0.13 – 0.36 m/s | 0.12 – 0.38 m/s | 0.91 – 1.02 | 2 – 27% |

（`basic_pick_place` 是单手任务：一只手世界系位移 0.22–0.57 m，另一只只有 0.003–0.03 m。20 个 episode 的 36 个手-序列里约 17 个是闲置手。）

两个后果：

1. **bimanual 配置受害最重**（`egodex_r1pro_bimanual.yaml`）：那只静止的手，action label 里 79–99% 的"运动"是相机抖动。它明明不动，pipeline 认为它在以 0.12 m/s 移动 —— 等于往训练数据里注入纯噪声。
2. **快速搬运阶段几乎不受影响，慢速阶段被彻底污染**。而慢速阶段正是接近、抓取、释放、精细对齐 —— 操作精度真正起作用的时刻。误差精确地集中在伤害最大的地方。

> 注意：不要用聚合后的表观/真实比值中位数（在这批数据上是 1.02）—— 该分布是双峰的，中位数没有意义。诊断脚本已按手速和闲置/工作手分层输出。

## 3. 修复方案

EgoDex 的 HDF5 里 `transforms/camera` 就是逐帧的 camera-to-world 位姿（metric，来自 Vision Pro）。把它导出给 pipeline，按首帧锚定生成逐帧 `T_cam2robot`：

```
T_robot_world  = T_cam2robot_init @ inv(T_world_cam[0])
T_cam2robot[t] = T_robot_world @ T_world_cam[t]
```

- 首帧代入可得 `T_cam2robot[0] == T_cam2robot_init`，标定文件定义的机器人摆放位置完全不变，只修正后续帧
- 语义上：机器人在世界系里保持静止，相机相对它运动 —— 这正是真实机器人工作站的物理情形
- 不需要知道 EgoDex 世界系的朝向或原点，只用到相机的相对运动

相机静止时 `T_world_cam[t]` 恒定，逐帧变换退化为 `T_cam2robot_init`，与修复前逐位相同（见 §5 验证），所以原版 Zed2 数据不受影响。

## 4. 改动一览

### 4.1 `phantom/processors/action_processor.py`

新增 `_get_cam2robot(paths, n_frames, data_sub_folder)`，解析该 demo 应使用的外参：

```python
if not self.use_per_frame_extrinsics:
    return self.T_cam2robot            # 消融基线：强制旧行为
# 找 camera_poses.npz（processed 目录 → raw 目录）
# 读 T_world_cam，校验 shape/finite
T_robot_world = self.T_cam2robot @ np.linalg.inv(T_world_cam[0])
return T_robot_world @ T_world_cam     # (N, 4, 4)
```

以下任一情况都安全退回固定外参，并打 log 说明原因：

| 情况 | 行为 |
|---|---|
| `use_per_frame_extrinsics: false` | 返回固定外参 |
| `camera_poses.npz` 不存在 | 返回固定外参（info 级 log） |
| 文件读不出 / 缺 `T_world_cam` key | 返回固定外参（warning） |
| 数组不是 `(N,4,4)` 或为空 | 返回固定外参（warning） |
| **首帧**位姿含 NaN/Inf（无法锚定） | 返回固定外参（warning） |
| 位姿数 > 帧数 | 截断 |
| 位姿数 < 帧数 | 用最后一个位姿前向填充（warning） |
| **中间帧**位姿含 NaN/Inf | 用前一个有效位姿填充，不污染整段（warning） |

新增 `_resize_transform_stack()` 静态方法做截断/前向填充。

`_convert_pts_to_robot_frame` 现在同时接受 `(4,4)` 和 `(N,4,4)`：

```python
if T_cam2robot.ndim == 2:
    skeleton_poses_rf_h0 = np.einsum('ij,bpj->bpi', T_cam2robot, skeleton_poses_cf_h)
else:
    if len(T_cam2robot) != len(skeleton_poses_cf):
        raise ValueError(...)          # 逐帧数量不匹配是数据问题，不静默容忍
    skeleton_poses_rf_h0 = np.einsum('bij,bpj->bpi', T_cam2robot, skeleton_poses_cf_h)
```

`process_one_demo` 解析一次外参后传给 `_process_single_arm` / `_process_bimanual`（两者签名各加一个 `T_cam2robot` 参数），不再直接读 `self.T_cam2robot`。`_process_hand_sequence` 在 ndim==3 时按该手序列长度再对齐一次，防止左右手长度不一致时抛异常。

### 4.2 `b/convert_egodex.py`

新增 `export_camera_poses(hdf5_path, output_dir)`，写出 `camera_poses.npz`：

```python
np.savez_compressed(output_dir / "camera_poses.npz",
                    T_world_cam=...,   # (N, 4, 4) camera-to-world
                    intrinsic=...)     # (3, 3)
```

`convert_one_episode` 里调用。**并且 `main()` 在跳过已存在的 episode 时也会补写这个文件** —— 已经转换过的数据不需要 `--overwrite` 重新转换：

```bash
python b/convert_egodex.py --task basic_pick_place --egodex-root /mnt/r/DATA/EgoDex/test
#   [0] Skipping 0: already exists (backfilled 126 camera poses)
```

### 4.3 `phantom/processors/paths.py`

新增 `paths.camera_poses` → `<demo>/camera_poses.npz`。

### 4.4 配置项

`configs/default.yaml`、`configs/epic.yaml`、`b/configs/egodex*.yaml`（4 个）新增：

```yaml
use_per_frame_extrinsics: true
```

代码里用 `getattr(args, "use_per_frame_extrinsics", True)` 读取，所以旧 config 缺这一项也能正常工作（默认开启）。

## 5. 验证

```bash
python b/test_per_frame_extrinsics.py     # 需在 phantom 环境中运行（经 phantom.hand 引入 torch）
```

17 项检查，绕过 `__init__` 构造裸实例，不需要 config / 数据集 / 标定文件。其中三项是关键：

| 检查 | 断言 |
|---|---|
| 静止相机与固定外参逐位一致 | 最大逐元素差 `< 1e-12` —— 原版 Zed2 数据零回归 |
| 运动相机下机器人系轨迹恰为世界系真值的一个刚体变换 | 误差 `< 1e-9 m`（固定外参做不到） |
| 静止的手在逐帧外参下位移严格为 0 | 逐帧 `0.000 mm` vs 固定外参 `~100 mm` |

其余覆盖：首帧等于标定外参、变换保持刚性、各类退回路径、截断/填充、NaN 前向填充、长度不匹配抛 `ValueError`。

跑 pipeline 时确认日志中出现：

```
Using 126 per-frame camera poses from .../camera_poses.npz
```

若出现 `No per-frame camera poses at ...`，说明 npz 未生成，重跑一次 `convert_egodex.py` 补写即可。

## 6. 消融用法

修复前后的对比就是 baseline vs. ours 的一组消融：

```bash
# 固定外参（旧行为）
python process_data.py --config-path=../b/configs --config-name=egodex_r1pro_bimanual \
    demo_name=egodex_basic_pick_place mode=action use_per_frame_extrinsics=false

# 逐帧外参
python process_data.py --config-path=../b/configs --config-name=egodex_r1pro_bimanual \
    demo_name=egodex_basic_pick_place mode=action use_per_frame_extrinsics=true
```

## 7. 已知遗留问题

**`robotinpaint_processor.py` 仍假设相机静止。** [`_get_mujoco_camera_params`](../../phantom/processors/robotinpaint_processor.py) 用 `self.extrinsics[0]` 建了一个静态 MuJoCo 相机，把机器人渲染进画面。现在 action 是对的，但渲染时相机不动，叠加出的机器人与背景存在同量级（~25 mm）的不一致。

- 只评估 action 精度 → 可以先放着
- 要生成带图像观测的训练数据 → 必须一起修（把 MuJoCo 相机改成逐帧）

**HaMeR 自身误差未包含在本次修复中。** 本修复消除的是 §2.2 那 25 mm 的相机项。HaMeR 单目 3D 估计自身的误差（尤其是深度方向和全局平移）是另一个独立且可能更大的误差源，需要跑 `b/diagnose_ego_camera_motion.py` 的 block 4-6 才能量化。

**Epic-Kitchens 无逐帧位姿。** `configs/epic.yaml` 里的开关目前是 no-op —— Epic 不提供相机位姿，需要 SLAM 才能获得。与 EgoDex 不同，Epic 里人在厨房走动，有真实平移基线，SLAM 在那里既必要也可行（见 §2.1）。

## 8. 复现诊断

```bash
# 只用 GT，不需要跑 pipeline，几秒一个 episode
python b/diagnose_ego_camera_motion.py \
    --egodex-root /mnt/r/DATA/EgoDex/test --task basic_pick_place \
    --max-episodes 20 --out b/out/diag --plot

# 先自检（合成数据，已知答案，只需 numpy）
python b/diagnose_ego_camera_motion.py --selftest
```

诊断的其余发现，与本次修复无直接关系但影响后续实验设计：

- **EgoDex GT 骨长 CV 精确为 0** —— GT 是刚性骨架模型拟合的结果。所以骨长稳定性类指标对照 GT 只能证明"与 ARKit 手模型一致"，说服力有限；对 EgoDex 算 MPJPE 衡量的也是与该手模型的一致性，而非绝对真值。
- **19% 的 GT 手部关节投影在画面外**（`in_image_frac = 0.81`）—— 部分可见/不可见的情况很常见，缺失帧补全的重要性高于预期。
