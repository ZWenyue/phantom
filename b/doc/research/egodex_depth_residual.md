# EgoDex 度量深度与残差补偿

> 配套：`contact_grounded_retargeting.md` Risk #4 / M4；`stage_a_progress.md` §5.3。
> Status: **已落地**（`egodex_da3_depth.py --compensate-only`）。Demo 0 结论：有符号手部残差在 affine lock 后已是 ~0；接触 4.8 cm 是**横向（X）缝**，不是深度偏置——再补 \(z\) 也进不了 3 cm 带。
> Scope: 把 DA3 深度对齐到 EgoDex 米制手。接触阈值保持 `dist_in=0.03`。

---

## 0. TL;DR

EgoDex 有 RGB + 米制 3D 手 + `T_camera`，**无场景深度**。Stage A 物体点云需要 `mask × metric depth`，深度用 Depth Anything 3（视频窗口、pose-conditioned），再用 GT 手做

\[
z_\text{GT} \approx a \cdot z_\text{DA3} + b.
\]

这步只消掉全局尺度；demo 0 上 lock 后手部像素残差中位数仍约 **6.7 cm**，指–物欧氏最近距离最低 **4.8 cm**，进不了 3 cm 接触带。

补偿加在 **深度（反投影之前）**，不要从 `d_finger_obj` 减 6.7 cm。入口：`Depth-Anything-3/b/egodex_da3_depth.py --compensate-only`（从 `depth_locked.npy` 接着做，不重跑 DA3）。

`--residual auto` 实际顺序：dump 有符号残差 → \(|\mathrm{median\ signed}| > 1.5\,\mathrm{cm}\) 才做 \(b_t\) → **跳过**用手拟合的残差场 → 始终用闲置手拟的 GT 桌面对齐 DA3 桌面像素（stapler 跟着桌面走）。

完整 DA3 仍先写 `depth_locked.npy`（affine lock），再写补偿后的 `depth.npy`。

**Demo 0 已经表明主缺口不在 \(z\)。** 最近一帧（t=79，右拇指）Δ = `(-4.7, -1.0, +0.3) cm`：深度已对齐（指尖处 DA3 \(z\) 差 5 mm），欧氏 4.8 cm 几乎全是相机 X。这是 mask 可见表面 vs 包握接触面（Risk #1），不是 scale-lock 残差。

---

## 1. 为什么要补、补的是哪条缝

接触主线索（`IntentProcessor._detect_contacts`）是 **GT 指尖**到 **物体点云表面**的最近欧氏距离。两边不对称：

| 量 | 来源 | 坐标系 |
|---|---|---|
| 指尖 | EgoDex HDF5 GT 关节 4/8/12，经 `T_cam2robot` | 米制，可信 |
| 物体点云 | YOLO-World/SAM2 mask × `depth.npy` 反投影 | 深度来自 DA3 |

Zed RGB-D 上 `dist_in=0.03` 是因为指尖真能贴到表面（合成手单测最近约 2 mm）。EgoDex 上这条 3 cm 带仍然表示「真接触」；现在进不去，是 **stapler 像素上的 DA3 \(z\) 与手 GT 不共尺**，不是人手离 stapler 还有 5 cm。

HaMeR **不能**产出物体深度；EgoDex GT 手优于 HaMeR，只用来 lock / 估残差，手轨迹本身仍走 GT。

---

## 2. 已做：DA3 + 手部 affine lock

脚本：`/home/a26160/SRC/Depth-Anything-3/b/egodex_da3_depth.py`  
环境：`conda activate da`（勿与 phantom 的 torch 2.1 混用）  
权重默认：`checkpoints/DA3NESTED-GIANT-LARGE-1.1`

约定：DA3 外参必须是 OpenCV **world-to-camera**。EgoDex `transforms/camera` 是 camera-to-world；脚本按 in-image 自检选了 `opencv`。

`collect_hand_z` 把 GT 关节 `HAND_LOCK_IDXS=(0,4,8,12,16,20)` 投影到图像，在 \(3\times3\) 窗里采样 DA3 \(z\)，再 `fit_scale_shift`：中位比 \(a\) + 残差截距 \(b\)，MAD inlier；\(|b|>0.15\,\text{m}\) 时退回 scale-only。然后整图 `depth = a * depth + b`。

### 2.1 demo 0 数字（`da3_depth_report.json`）

`basic_pick_place` / demo 0，126 帧，1920×1080。

| 模型 | lock 前 median \|z\| | \(a\) | lock 后 median \|z\| | p90 |
|---|---|---|---|---|
| DA3-LARGE-1.1 | 14.3 cm | 0.765 | 5.8 cm | 18.7 cm |
| Nested Giant-1.1 | 7.8 cm | **0.940** | **6.7 cm** | 23.0 cm |

Nested Giant：`a=0.9395`，`b≈0`，`n=1251/1263` inlier。深度中位约 0.45 m。尺度够用来 smoke 点云，**不够**支撑 3 cm 指尖接触带。

当前产物：

- `{processed}/egodex_basic_pick_place/0/depth.npy` — `(126, 1080, 1920)` float32 米
- `da3_depth_report.json`、`da3_depth_preview.png`

重跑：

```bash
conda activate da
python /home/a26160/SRC/Depth-Anything-3/b/egodex_da3_depth.py \
    --task basic_pick_place --demo 0 --overwrite
```

### 2.2 和接触失败的关系

Intent smoke（YOLO-World `yolov8x-worldv2` + SAM2，GT 右手）：

- 种子框 stapler `[749,657,909,744]`；126/126 点云有效；种子尺度约 `5.2×1.7×3.5 cm`，中后段约 `12×5×8 cm`。
- `source=fingertip`，`d_finger_obj` min/p50/max = **4.8 / 9.1 / 27.3 cm**，从未低于 `dist_in=0.03`。
- 诊断上大部分时间停在 8–15 cm，只在约第 78 帧掉到 ~4.8 cm（越过 `dist_out=0.05` 一线、进不了 3 cm）。
- `grasp_valid=False`，无 `grasp.npz`。

`T_cam2robot` 是固定肩部外参。头动时 robot 系物速会抖，但手和物乘同一 \(T\)，**指–物距离仍是相机系几何**，不能用外参解释这 4.8 cm。

---

## 3. 明确不做的两件事

**不要** `d_finger_obj -= 0.067`。

- 6.7 cm 是手部像素上的 **无符号** \(|z|\) 中位数，不是 stapler 表面误差，也不是欧氏 \(d\)。
- 抓取时指尖像素是皮肤，物体点云来自 SAM2 stapler mask，不是同一片表面。
- 减完会把 8–15 cm 悬停平台吞进接触带。

**不要**把 `intent_contact_dist_in` 提到 5.5–6 cm 当成主修复。那只改「何时贴标签」，不改正表面几何；grasp 合成仍对着偏深的点云闭合。小幅放宽最多当 pipeline smoke 的权宜之计（见 `stage_a_progress.md` 接触配置）。

---

## 4. 方案：残差补进 `depth.npy`

原则：**改深度，再反投影，再跑接触**。`dist_in=0.03` 保持「真接触」的物理含义。grasp / Stage B 吃同一团点云，深度对上了接触和抓取几何一起好。

### 4.0 先 dump 有符号残差

当前 `hand_lock.after` 只有 `median_abs_m`。在现有 `collect_hand_z` 配对上，lock 之后算

\[
r = z_\text{GT} - (a\, z_\text{DA3} + b).
\]

按帧、按关节（腕 vs 指尖）、按投影是否落在物体 mask 内拆开，写入 `da3_depth_report.json`（至少 `median_signed_m`、`p10/p90`、分箱）。没有符号就不知道 stapler 被推远还是拉近；demo 0 的 4.8 cm 低谷更像 DA3 把桌面物体估深了，必须用有符号量确认。

### 4.1 逐帧偏置 \(b_t\)（先做）

全局 \(a,b\) 之后，对每一帧用手部像素：

\[
b_t = \mathrm{median}_j \bigl( z_{\text{GT},t,j} - z_{\text{DA3,lock},t,j} \bigr),
\]

`depth[t] += b_t`。实现就是 `fit_scale_shift` 的 per-frame 截距。吃 clip 内尺度漂移；**补不了皮肤 vs stapler 的材质差**。

### 4.2 工作空间残差场

若 \(b_t\) 之后手部 \(|r|\) 仍明显高于接触带，在手部对应点上拟合低阶场，而不是第二个全局 \(a,b\)：

\[
r(u,v,z) \approx c_0 + c_z(z-\bar z) + c_u u + c_v v,
\]

再加到**整张**深度（含 stapler 像素）。**实现已去掉 \(c_z\)**：demo 0 上 \(|c_z|\approx 0.72\) 会把整场深度压向 \(\bar z\)。auto 默认也不跑 field（手部空间残差搬不到 stapler）。 stapler 黑体若系统比皮肤更深 → 4.3。

### 4.3 桌面约束（pick-and-place 最对症）

前几帧 / 闲置手贴桌：GT 指尖拟一张桌面，DA3 再拟一张，沿法向把 DA3 对齐到 GT。stapler 在桌上，物体 \(z\) 跟着桌面走。这比把 6.7 cm 手部残差直接搬到 stapler 像素更贴这个任务。

### 4.4 接触判定仍用 3 cm

流程：残差补偿 → 写 `depth.npy` → `IntentProcessor._build_object_pointclouds` → `_detect_contacts`。不要在 `_detect_contacts` 里减一个全局 \(\delta\)。

### 4.5 2D overlap 只验证，不当训练集标签

指尖投影落在物体 mask 内时，比较 \(z_\text{GT}\) 与该像素（或最近 mask 像素）的 DA3 \(z\)：接近段应 \(z_\text{finger}<z_\text{obj}\)，真接触应接近 0。序列上 `min(Δz)` 用来看 stapler 有没有被拉回「手够得到」的地方。

**不要**把 `min_t d_finger_obj(t)` 本身当 \(\delta\) 去减——那等于默认这次 demo 一定抓到了，会把每条序列的最低点都标定成 0（相对局部最低，不是深度残差补偿）。

---

## 5. 补偿后仍可能剩下的 1–2 cm

深度对上之后，若最近距离仍在 1–2 cm，多半不再是 DA3：

- `intent_mask_erode=2` 把表面往里收（1080p、约 0.4 m 时大约 1 cm 量级）；
- 用的是关节 4/8/12，不是指腹接触点；
- 包握时最近的**可见** stapler 点在侧面，接触面被手挡住（设计 Risk #1：深度模型补不出手背后的表面）。

这些用略减 erode / 很小的表面偏置处理，不要再加大深度补偿。

头戴相机 + 固定 `T_cam2robot` 会让 robot 系 `obj_speed` 抖动；有手时运动回退不会触发。残差补偿不解决这件事。**已接**：`intent_use_T_camera=auto` 用 HDF5 `T_camera` 把点和手变到相对 t0 的稳定 robot 系（见 `stage_a_progress.md`）。

---

## 6. 落地位置

| 位置 | 做什么 |
|---|---|
| `Depth-Anything-3/b/egodex_da3_depth.py` | affine lock 后写 `depth_locked.npy`；`--compensate-only` / `--residual auto` 接 4.0 dump、条件 \(b_t\)、桌面；field 仅显式 `--residual field` |
| `da3_depth_report.json` | `residual.before/after`（有符号）、`table.offset_m`、`b_t_m` |
| `phantom/phantom/processors/intent_processor.py` | **不改阈值公式**；深度补偿不够时不要用它当接触开关 |
| `b/configs/egodex_panda_intent.yaml` | `intent_contact_dist_in/out` 保持 0.03 / 0.05 |

建议：dump 有符号残差 → 看 Δxyz 而不只看 \(d\)。Δz≈0 且 Δxy 大则停补深度。

`--residual`：`auto`（默认）| `bt` | `field` | `table` | `none`。

```bash
conda activate da
python /home/a26160/SRC/Depth-Anything-3/b/egodex_da3_depth.py \
    --task basic_pick_place --demo 0 --compensate-only --residual auto --no-preview
```

---

## 7. Demo 0 实验结果

Affine lock 快照：`depth_locked.npy`。补偿写回 `depth.npy`；细节在 `da3_depth_report.json` 的 `residual`。

### 7.1 有符号残差（4.0）

| | n | median signed | p10 / p90 signed | median \|r\| |
|---|---|---|---|---|
| lock 后全部关节 | 1263 | **−0.1 cm** | −15.3 / +15.3 | 6.9 cm |
| 仅指尖 | 1073 | −1.4 cm | −19.7 / +15.9 | 6.7 cm |

先前 6.7 cm 是**无符号**中位数。有符号几乎为 0：全局 \(a,b\) 已经把偏置吃掉，剩下的是散射。auto 因此 **跳过 \(b_t\)**。

### 7.2 不要用带 \(c_z\) 的残差场

第一次拟合 \(r \approx c_0 + c_z(z-\bar z)+c_u u'+c_v v'\) 得到 **\(c_z \approx -0.72\)**，整幅深度被压向 \(\bar z\)（范围 0.31–0.65 m），接触 min 从 4.8 cm **恶化到 6.3 cm**。代码已去掉 \(c_z\)；auto 不再跑 field。

### 7.3 桌面约束（4.3，desk 像素 vs GT 平面）

左手前 20 帧指尖拟 GT 桌面；桌面网格采 DA3（排除手 40 px 与物体 mask），\(n_\text{desk}=194665\)，**desk offset = +3.8 cm**。沿法向拉回后：

| | min \(d\) | @frame | 物体 median \(z\) |
|---|---|---|---|
| `depth_locked.npy` | **4.79 cm** | 79 | 48.4 cm |
| table 对齐后 | 4.88 cm | 79 | 47.3 cm |

物体 \(z\) 只动了 ~1 cm（法向 ≠ 相机 \(z\)），接触距离基本不变。

### 7.4 最近接触帧的真正缺口（t=79，右手拇指）

`d=4.79 cm`，最近 stapler 点 − 拇指 = **(−4.68, −0.96, +0.30) cm**。拇指像素 `z_gt=48.9 cm`、`z_DA3=49.5 cm`（Δz = 0.5 cm）。拇指 uv **不在** stapler mask 内——点云是顶面，拇指在侧面包握。

3 cm 带进不去，不是再补深度能解决的（设计 Risk #1）。**已改 `_detect_contacts`**：对可见点云做相机 XY 膨胀 `intent_contact_xy_inflate=0.06`，分数为 `hypot(max(dxy−inflate,0), |Δz|)`。Demo 0 离线：`source=fingertip_wrap`，`grasp_kf=68` / `release_kf=83`，`d_wrap min=0.06 cm`（`d_eucl min` 仍 4.8 cm），抓取合成 `hand_anchored`、宽 3.2 cm。运输段只有 ~10 帧，是因为 SAM2 点云在抬起后仍停在桌上（t=90 拇指离质心 33 cm），不是接触公式的问题。

---

## 8. 相关路径（demo 0）

| 路径 | 说明 |
|---|---|
| `/home/a26160/DATA/test/basic_pick_place/{id}.hdf5\|.mp4` | EgoDex 原始 |
| `/home/a26160/DATA/test_phantom_processed/egodex_basic_pick_place/0/` | 处理后 demo（`depth.npy`、GT 手、intent） |
| `phantom/b/export_egodex_hand_gt.py` | 写 `hand_processor/hand_data_{left,right}.npz`（相机系 `kpts_3d`，OpenCV） |
| `phantom/b/export_egodex_objects.py` | `objects.json` |
| `phantom/b/configs/egodex_panda_intent.yaml` | EgoDex intent 配置（`square: false`，`epic: true`） |
| `phantom/phantom/detectors/detector_yolo_world.py` | 本机 H200 上 Grounding-DINO kernel 无效后的替换检测器 |

Intent 重跑（phantom env，cwd=`phantom/phantom`）：

```bash
conda activate phantom
cd /home/a26160/SRC/phantom/phantom
python process_data.py --config-path=../b/configs \
    --config-name=egodex_panda_intent mode=intent demo_num=0
```
