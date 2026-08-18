# 手锚定物体跟踪（类别无关）— 实施方案

> 配套：`stage_a_progress.md`（§1 物体分割/点云）、`contact_grounded_retargeting.md`（Risk #3 目标物体识别）。
> Status: **已实施（implemented）**。`IntentProcessor` 默认 `intent_track_backend=hand_sam2`：v4 开放词表锚定 seed（`intent_seed_grounding=auto`，检测器默认 `yolo`，可切 `owlv2`）+ 手附近过滤 + 几何回退；附录 B 的手区域 / overlap 否决与 §2.3–2.5 传播 / 闸门 / 重抓保留。YOLO 逐帧跟踪后端保留（`intent_track_backend=yolo`）。v4 细节见**附录 C**。
> Scope: 只改 `_resolve_masks` / `_track_masks` 这一层的**目标选择与跟踪**；输出仍是 `object_masks.npy`，下游点云 / 接触 / G* / Stage B **一行不改**。

---

## 0. TL;DR

现状：目标物体靠 **YOLO-World 文本 prompt**（`objects.json` 或 config `object_prompt`）逐帧检测 + 单帧 SAM，miss 就 `hold` 上一帧。

问题：**换数据集 / 换物体就要调**。
- prompt 错 → 整段错（demo 1 落到 `black stapler`，seed 0.012）。
- prompt 对但遮挡 → 检不到 → `hold` 桌面残留（demo 1 iphone：`hold=125/160`，mask 锁死 3569 px，抓取段被误切到 46–54 而不是真实的 ~105–140）。

结论：不该按物体名调参。真正**跨数据集不变的信号**是——

> **目标物体 = 被目标手抓住、并跟手一起运动的那团点。**

EgoDex 已经有逐帧 GT 手（3D + 2D 投影）+ DA3 深度 + 手 mask（`segmentation_processor`）+ SAM2 image/video（`DetectorSam2` 已实现 `segment_video_from_masks`）。用这些就能做到**零文本 prompt**的通用跟踪。

---

## 1. 为什么现在换数据集就要调

`IntentProcessor` 现有链路（`phantom/processors/intent_processor.py`）：

- `_detect_seed`（L734）：YOLO-World 扫帧选最早置信帧，dark-rerank 躲灰色干扰。
- `_track_masks`（L973）：seed 帧单帧 SAM2；之后每 `track_stride` 帧再 YOLO，`_associate_yolo_box`（L835）关联，miss → `hold`。
- `sam2.segment_box`（L905）只用单帧 image predictor。

`DetectorSam2` 里其实**早就有** video 跟踪，但没接线：
- `segment_video_from_masks`（`detector_sam2.py` L199）：多帧 mask 条件 + 正/反向 `propagate_in_video`。
- `add_new_points_or_box`（L102）：点/框提示的 video 跟踪。

`stage_a_progress.md` §1.2 记录当初「**不再做 SAM2 长程 video propagate**」——原因是从 YOLO 大框（含桌面）seed 会漂到桌面残留。**根因是 seed 选错，不是 SAM2 video 不行。** 用手锚定 seed 就能把它开回来。

失败模式对照：

| | demo 1 (stapler 误检) | demo 1 (iphone, 正确 prompt) |
|---|---|---|
| seed 分数 | 0.012 | 0.095 |
| `hold` 帧 | 90/160 | 125/160 |
| 抓取段 | 105–140（凑巧对） | **46–54（错，真实 ~105–140）** |
| mask | 全程有面积 | t≥46 锁死 3569 px |

---

## 2. 目标架构：手锚定 seed + SAM2 video 双向传播

三条原则：**seed 用手不用类别；跟踪用 SAM2 memory 不用逐帧 YOLO；用手/运动一致性做闸门和自检。**

### 2.1 Seed 帧选择（纯手运动学，无物体）

不再用 YOLO 扫帧。用手自身信号定一个**候选抓取帧** `t_seed`：

- 开口 `aperture`（拇指–食指距离，`_load_hand_fingertips` 已算，L1522）出现局部极小；
- 腕/指尖速度由高转低（approach → 稳定持握）；
- `hand_detected=True`。

取 target hand（`self.target_hand`）在候选窗内 aperture 最小且随后一段速度低的帧。**这一步只碰手，不碰物体**，因此不与「接触检测需要物体点云」循环依赖。

> 现有 `_detect_contacts`（L1399）是「指尖–物体点云距离」，需要先有物体 → 作为**第二遍精修**，不要用来定 seed。

### 2.2 Seed 帧「手条件」分割（SAM2 image，负点排除手）

在 `t_seed` 用 SAM2 image predictor 出物体 mask：

- **正点**：抓取中心 UV（拇指尖与食指尖 2D 中点）、必要时加指尖连线中段几个点；
- **负点（关键）**：手骨架 2D 点（`kpts_2d`，21 点投影）或手 mask（`paths.masks_hand_{side}`，`segmentation_processor` 产出）。ego 视角下手会盖住/挨着物体，负点把手本身从 mask 里排除；
- **可选负点**：桌面大面积区域、其它候选物体框中心。

`SAM2ImagePredictor.predict` 支持 `point_coords` + `point_labels`（1 正 0 负）；`DetectorSam2.segment_box` 需扩一个 `segment_points(image, pos_uv, neg_uv)` 薄封装。

产出 `seed_mask`（手里那团），而不是含桌面的大框——这是不漂的前提。

### 2.3 SAM2 video 双向传播（类别无关，靠 memory 扛遮挡）

用 `seed_mask` 从 `t_seed` 调 `segment_video_from_masks` **正向 + 反向**各一遍：

- 反向 → 抓取前物体静止在桌上的样子（供 approach 段点云）；
- 正向 → 抓起 / 搬运 / 放下 / 松手；
- 遮挡由 SAM2 memory bank 处理，**取代「抄上一帧」**。

从「物体最孤立、最好分」的抓取帧 seed，双向覆盖率最高。`video_dir` 用已存在的 `paths.original_images_folder`（`_load_frames` 已抽帧）。

### 2.4 运动一致性闸门（防漂 + 精修接触）

传播得到逐帧 mask 后，用手 3D + 深度做刚性校验：

- 每帧 mask 反投影质心（`_build_object_pointclouds` 已有）；
- **抓取段内**：物体质心速度应与指尖速度一致（刚性附着）→ 不一致的连通域丢弃；
- **抓取段外**：物体应基本静止（速度 < 阈值），否则说明跟丢/漂移。

这一步把「漂到桌面残留」从几何上堵死，且完全无类别依赖。

### 2.5 跟丢重抓（fallback）

若某段 SAM2 输出为空或被闸门判失败：回到该帧**指尖邻域**再做一次 2.2 的手条件分割，作为新 conditioning 帧重新传播（`segment_video_from_masks` 支持多 `(mask, frame_idx)`）。不回退到 YOLO top-1。

---

## 3. 分层与后端切换

`_track_masks` 收成可切后端（config `intent_track_backend`）：

| 后端 | 方法 | 何时用 |
|---|---|---|
| `hand_sam2`（**新默认**） | §2：手 seed + SAM2 video + 手负点 + 运动闸门 | 全数据集通用，零 prompt |
| `yolo`（保留） | 现状 YOLO + 单帧 SAM + `_associate_yolo_box` | 有干净类别名、遮挡少的 clip |

弱先验（可选）：`objects.json` / YOLO 只在「桌上多个可动物体、seed 平局」时打破歧义，**缺了也能跑**。

---

## 4. 自动 QA（跨数据集 scale 的关键）

每条 demo 自动判定，不靠人工看 overlay：

- 抓取段内 `corr(object_centroid_vel, fingertip_vel)` 高（刚性附着）；
- 抓取段外物体位移小；
- mask 面积不出现「锁死常数」（当前失败特征：t≥seed 恒为 3569 px）。

不通过 → 在 `intent_preview` / 日志标记，或直接从数据集剔除。可复用 Stage C `_quality_gate` 的思路，但放在 Stage A 出口。

---

## 5. 改动清单（落地时）

| 文件 | 改动 |
|---|---|
| `phantom/detectors/detector_sam2.py` | 新增 `segment_points(image, pos_uv, neg_uv)` |
| `phantom/processors/intent_processor.py` | `_select_seed_frame_from_hand`、`_seed_mask_hand_conditioned`、`_track_masks_hand_sam2`、`_track_quality_gate`；`_resolve_masks` 按 backend 分派 |
| `phantom/processors/paths.py` | `track_quality.npz` |
| `b/configs/egodex_panda_intent.yaml` | `intent_track_backend: hand_sam2` + 闸门阈值；`object_prompt` 降为可选 |
| `b/configs/pickplace_intent.yaml` | 显式 `intent_track_backend: yolo`（Zed 样本保持原路径） |
| `b/tests/test_hand_anchored_tracking.py` | seed / in-hand / corr 单测（无 SAM2） |
| 下游 | **不改**：`object_masks.npy` 契约不变 |

### 复用点（已有，不用重写）
- 手指尖 / 开口：`_load_hand_fingertips`（L1504）、`FINGERTIP_IDXS=[4,8,12]`（L57）、`THUMB_TIP_IDX/INDEX_TIP_IDX`。
- 手 2D 关键点：`hand_data_{side}.npz` 的 `kpts_2d`（`_load_target_hand_uv` 已在读，L949）。
- 手 mask：`paths.masks_hand_{side}`（`segmentation_processor`）。
- SAM2 video：`DetectorSam2.segment_video_from_masks`（L199）/ `add_new_points_or_box`（L102）。
- 点云反投影 / 质心：`_build_object_pointclouds`。

---

## 6. 验证计划

1. **demo 1（iphone）**：期望抓取段回到 ~105–140、transport 帧数显著 >3、mask 面积不再锁死常数、overlay 里机器人跟着手机进盒子。
2. **demo 0（回归）**：`hand_sam2` 结果不应比现状差（stapler，接触 105–140）。
3. **跨物体**：至少再跑 1–2 个不同 task（不同物体、无 `objects.json`）验证零 prompt 能跑通。
4. **自动 QA**：§4 指标全绿；对故意跟丢的 clip 应判红。

---

## 7. 明确不做

- 每任务手写 `object_prompt`（现在 yaml 兜底炸掉的原因）。
- 每帧 YOLO + miss 就 `hold`（demo 1 失败模式）。
- 纯光流 / 全图运动分割：ego 相机自身在动，桌面整片飘。
- VLM（GPT-4V/Molmo）指物体：更通用但要额外 infra；有 GT 手时「判哪个被抓」更准，非必要。

---

# 附录 A — v2 修订：seed 从「按开口挑帧」改为「候选 + 运动一致性验证」

> Status: **v2 待验证**。select-by-verify 已落地（`_select_seed_by_verify`）；§2.2–2.5 不变。demo 1 尚未用 v2 重跑。

## A.0 目标物体确认（EgoDex GT）

`basic_pick_place/1.hdf5` 根属性（不是猜的）：

```
description  = pick up a grey iPhone from the table and place it on the box lid.
environment  = ... from:table, to:box lid, hand:right
object       = object:iphone, color:grey
llm_verbs    = ['pick' 'place']
llm_objects  = ['iphone' 'table' 'box lid']
```

结论：**被操作物体 = 灰色 iPhone**；`box lid`（盒盖）是**放置目的地（to）**，`table` 是**来源（from）**，都不是要跟踪的物体。
→ 附带强不变量：目标物体 `from table → to box lid`，被拿起后**一定跟手位移**，正好能被运动一致性选中。

## A.1 v1 为什么 seed 选错（实跑结果）

`bash b/run_contact_retarget.sh --demo-num 1 --no-reuse-masks`（backend=hand_sam2）：

| 指标 | 值 | 判读 |
|---|---|---|
| seed | t=**35**，SAM score 0.934，area 59178 | 伸手掠过 **CONNECT 4 盒盖** 时的帧 |
| 有效 mask | 146/160，147 个不同面积，lock=0.09 | mask **不再锁死**（比 v1-YOLO 好） |
| 抓取段 | phase grasp/transport = **0–33** | **错**：真抓 iPhone 在 ~105–130 |
| track QA | **PASS**（corr=0.67） | **误报**：在错误的 0–33 段上算的相关 |
| Stage B | success，pos mean 6mm/max 6.8cm | 数值 OK 但目标是盒盖 |
| Retarget | **PRUNED**（key_pos 3.6cm>3cm） | 夹爪悬在盒盖上方，不跟手机 |

根因：v1 seed 用「开口 `aperture` 局部极小 + 速度转低」。但——

- 真抓 iPhone 时开口 **8.7 cm**（手掌摊开压手机），**不是**捏合；
- 全程开口最小在 t=0（2.8cm）、t=35（伸手掠过盒盖）出现局部极小。

即「开口小」**不是跨物体不变量**，对"平放物体手掌抓取"直接指错。运动一致性（§2.4）v1 只用作**事后闸门**，seed 已经锚死在盒盖上，闸门只能清漂移帧、救不回。

## A.2 修正：select-by-verify（替换 `_select_seed_frame_from_hand`）

把运动一致性从「事后过滤」提到「**选 seed**」。**先找搬运窗口，再在窗口里验证候选**：

1. **搬运窗口**（纯手 3D）：找 target hand 指尖发生**持续大位移**的一段（pick→lift→move）。物体静止在桌上时无法靠运动区分，但被拿起后必跟手走。
   - 指尖速度 `|Δtip|` 平滑后 > `intent_seed_move_vel`（默认 0.01 m/f）连续 ≥ `intent_seed_move_run`（默认 4）帧；取最长一段作为窗口 `[w0, w1]`。
   - 若无明显搬运（推 / 原地操作），回退到 §2.1 老启发式（开口最小）并把 QA 标记为低置信。
2. **候选帧**：在 `[w0, w1]` 里等距采 `intent_seed_cand_k`（默认 5）帧。
3. **各候选轻量验证**：每帧 `_seed_mask_hand_conditioned`（§2.2）→ 只 `_propagate_video_masks` **前后各 `intent_seed_probe_win`（默认 8）帧**（便宜）→ `_mask_centroid_robot` 求质心速度 → 打分：

   ```
   score = cos(obj_centroid_vel, fingertip_vel)  over the probe window
           − area_penalty(过大/过小)
           − nan_penalty(质心缺失帧占比)
   ```
   - t=35 盒盖：手掠过后盒盖**不跟手** → cos 低 → 淘汰。
   - t≈110 iPhone：拿起后**与指尖同向平移** → cos 高 → 当选。
4. **选分最高的候选做锚点**，再走 §2.3 全程双向传播 + §2.4 全局闸门 + §2.5 重抓（都不变）。
5. **QA 收紧**：`inhand_corr` 改成在「验证选中的锚点 + 全局传播后的 grasp/transport 段」上算；`_track_quality_gate` 增补一条——**抓取段物体必须发生净位移**（`from→to`），位移 < `intent_track_min_disp`（默认 0.05 m）判 FAIL，堵住「锚在静止 destination（盒盖）上 corr 仍偶然高」的误报。

## A.3 复用与改动

**复用（已存在，不重写）**：`_vel_corr`、`_mask_centroid_robot`、`_motion_gate_failed`、`_seed_mask_hand_conditioned`、`_propagate_video_masks`、`_inhand_from_aperture`。

| 文件 | 改动 |
|---|---|
| `intent_processor.py` | 新增 `_find_transport_window`、`_score_seed_candidate`、`_select_seed_by_verify`；`_track_masks_hand_sam2` 用它替换 `_select_seed_frame_from_hand`；`_track_quality_gate` 加抓取段净位移检查 |
| `egodex_panda_intent.yaml` | `intent_seed_move_vel`、`intent_seed_move_run`、`intent_seed_cand_k`、`intent_seed_probe_win`、`intent_track_min_disp` |
| `test_hand_anchored_tracking.py` | 加：合成「手拿物体平移 vs 手掠过静止物体」，验证 verify 选中前者；净位移 QA 单测 |
| 下游 / §2.2–2.5 | **不改** |

`_select_seed_frame_from_hand`（v1）保留为无搬运时的 fallback。

## A.4 验收（demo 1）

- seed 落在 ~105–130（拿起 iPhone），**不再是 35**；
- phase grasp/transport 覆盖真实搬运段，物体质心从桌面移向盒盖，净位移 > 5 cm；
- `track_quality.npz` `accept=True` 且 `inhand_corr` 在正确段为高；
- retarget **不再 PRUNED**，overlay 里夹爪夹住手机从桌面移到盒盖。

---

# 附录 B — v3 修订：全局候选 + 硬性排除手（面向多任务数据集）

> Status: **已实施（implemented）**。取代 A.2 的「先锁窗口」。§2.2–2.5、A.2 的「候选轻量传播打分」框架不变，改的是**候选从哪来**和**打分怎么排除手**。手区域用 GT 关键点 **掌骨凸包 + 骨骼折线膨胀**（不用 21 点满凸包，避免盖住 pinch 里的物体）。

## B.0 v2 实跑结果（select-by-verify 首版）

`bash b/run_contact_retarget.sh --demo-num 1 --no-reuse-masks`：

| 指标 | 值 | 判读 |
|---|---|---|
| 搬运窗口 | **26–46**（手划过桌面，净位移 0.37 m） | **错**：真搬运在 105–130（净位移 0.22 m） |
| 候选打分 | t=26/31 corr≈0.83；t≥36 掉到 0.16→0.02 | 选出 **t=31** |
| seed mask | t=31，area 48364 | **盖在右手上，不是 iPhone** |
| track QA | **FAIL** `inhand_corr 0.26 < 0.30` | 净位移 QA 生效，不再误报 PASS |
| G_width | **0.175 m** | 手/前臂宽度，佐证锚在手上 |
| retarget | accepted（key_pos 1.6cm） | 数值过门，但夹爪跟的是「手团」 |

两个根因：

1. **窗口选错**：用「手净位移最大的一段」定窗口 → 空手 reach（26–46）打败了真载物搬运（105–130）。一旦窗口错，真搬运段**根本没被探测**。
2. **候选是手本身**：这条 demo **只有 `masks_arm.npy`，没有 `masks_hand_*`**（已核实），`_load_hand_mask` 返回 None，负点只剩 6 个掌骨关键点 → SAM 把整只手当成 in-hand blob。手与指尖天然共运动 → cos 虚高 0.83 → 必被选中。

## B.1 为什么「post-lift 窗口 + 软惩罚」不够通用

- 「抬起后半段位移最大」是 **pick-place 专用假设**。数据集含 pour / push / open / wipe / handover：
  - push：手不抬，物体贴桌滑；
  - open：铰链弧线，物体净位移可能很小；
  - pour：抬起后是旋转不是平移。
- 真正跨任务的不变量是三条**同时**成立：
  1. **不是手**（mask 不与手区域重叠）；
  2. **跟手一起动**（co-motion，抓取/接触段）；
  3. **平时基本静止**（是可动物体，不是恒动的手）。

## B.2 修正

### B.2.1 手区域用 GT 关键点构造（不依赖 `masks_hand_*`）

每帧用 target hand 的 **21 个 GT 2D 关键点**构造 `hand_region_t`：**掌骨/腕凸包** + **骨骼折线**，再膨胀（`intent_hand_dilate_px`，默认 25px）。不用 21 点满凸包——pinch 抓取时满凸包会盖住物体。GT 手恒在 → 跨所有 demo 可用。`masks_hand_*`（若有）作为并集补充，并在 pinch 处挖洞，避免粗 mask 把物体标成手；**不用** `masks_arm.npy`（人体 inpaint 掩码常把被抓物体算进人体）。缺了 `masks_hand_*` 不影响。

### B.2.2 候选 mask 硬性排除手（第 2 条，加强为否决）

`_seed_mask_hand_conditioned` 产出候选后：

- `hand_region_t` 作为 SAM **负点密集来源**（不只 6 个掌骨点，采样手区域内多点）；
- 得到 mask 后计算 `overlap = |mask ∩ hand_region| / |mask|`；
- `overlap > intent_track_hand_overlap_max`（默认 0.5）→ **直接淘汰该候选**（不是扣分）；
- 若淘汰后 mask 仍够大，再尝试 `mask − hand_region`（去掉与手重叠的部分）保留露出的物体像素。

### B.2.3 全局候选，窗口由验证产出（第 1 条，重构）

不再用 `_find_transport_window` 预锁单窗口。改为：

1. **候选来源**：全序列按粗 stride（`intent_seed_global_stride`，默认 5）撒帧，或限定在「手在动」的所有帧（不止最长一段）；
2. 每个候选：B.2.2 分割（重叠手→淘汰）→ ±`seed_probe_win` 短传播；
3. **打分**（非手已保证）：

   ```
   score = cos(obj_vel, tip_vel)               # 跟手动
           + w_static · static_bonus            # 该 blob 在其运动段之外静止
           − area_penalty − nan_penalty
   ```
   `static_bonus`：候选物体在 probe 窗外若干帧速度 < `intent_track_static_vel_max` 则加分（把「恒动的手/手臂残片」和「平时静止的可动物体」再分开一层）。
4. **全局取最高分**做锚点，其 probe 段即隐含搬运窗口；再走 §2.3 全程双向传播 + §2.4 闸门 + §2.5 重抓。

预期：reach 段（26–46）里唯一跟手走的是手本身 → 被 B.2.2 淘汰 → 该段无候选胜出；真搬运段（~105–130）iPhone 跟手走、不是手、之前静置在桌 → 胜出。

### B.2.4 QA 补强

- 保留 A.2 的抓取段净位移检查（`intent_track_min_disp`）；
- 新增：**seed mask 与手区域重叠比**写入 `track_quality.npz`，overlap 高即使 corr 高也判 FAIL（堵「锚在手上 corr 虚高」）。

## B.3 改动清单（已实施）

| 文件 | 改动 |
|---|---|
| `intent_processor.py` | 新增 `_hand_region_mask`（GT 关键点凸包+膨胀）；`_seed_mask_hand_conditioned` 加手区域负点 + overlap 否决；`_select_seed_by_verify` 改全局候选 + static_bonus 打分；`_track_quality_gate` 加 seed-hand-overlap 判据；`_find_transport_window` 降级为可选 |
| `egodex_panda_intent.yaml` | `intent_hand_dilate_px`、`intent_track_hand_overlap_max`、`intent_seed_global_stride`、static_bonus 权重 |
| `test_hand_anchored_tracking.py` | 加：手 mask 应被 overlap 否决；「空手 reach vs 载物搬运」全局候选应选后者；static_bonus 区分恒动 vs 静置 |
| 下游 / §2.2–2.5 | **不改** |

## B.4 元判断（记录，防止无限调参）

「非手 + 跟手动 + 平时静止」是我认为真正跨任务的三不变量，v3 一次性做对（尤其**硬性排除手**）。若 v3 在 pour/push/open 等仍反复失败，则停止堆几何启发式，转向**手-物接触检测 + 手排除分割**或**轻量 VLM 兜底**，而不是再加第 4、5 条规则。

## B.5 验收（demo 1，v3）

- seed 落在 ~105–130，mask 是 **iPhone 不是手**，`seed_hand_overlap` 低；
- `G_width` 回落到手机量级（数 cm，不再 0.175 m）；
- `track_quality.npz` `accept=True`；
- overlay 里夹爪夹住手机从桌面移到盒盖。
- 回归：demo 0（stapler）不劣于现状。

---

# 附录 C — v4：轻量 VLM（开放词表）锚定 seed（面向多任务数据集）

> Status: **已实施（implemented）**。触发条件见 C.0：v3 在 6 条 demo 上只有 2 条真对。按 B.4 的止损线，**停止堆几何启发式**，改用「提示词 + 手附近 → 选被拿的那个」做 seed。§2.3–2.5 传播 / 闸门 / QA、附录 B 的手区域与 overlap 否决**全部保留**。验收（C.5）待跑 demo 0–5。

## C.0 v3 广度验证结果（demo 0–5，`--no-reuse-masks`）

| demo | 物体(objects[0]) | v3 seed 跟的是什么 | track QA | retarget | 判定 |
|---|---|---|---|---|---|
| 0 | stapler | **指尖**（真抓取帧被 overlap 误杀） | PASS(虚) | accepted 0.8cm | ✗ 错 |
| 1 | iphone | **iPhone** | PASS corr0.69 | accepted 2.1cm | ✓ 对 |
| 2 | banana | **指尖**（香蕉在左，没跟上） | PASS(虚,corr0.43) | **PRUNED 16.4cm** | ✗ 错 |
| 3 | macaron | 空 mask (area 0) | FAIL corr0.18 | **PRUNED 8.2cm** | ✗ 错 |
| 4 | bread | **bread** | FAIL corr0.07 | accepted 2.7cm | △ seed 对，相位错位 |
| 5 | block | 退回 yolo，抓到右边红块/布 | FAIL overlap1.0 | accepted 0.9cm | ✗ 错 |

**清晰跟对：2/6（demo 1、4）。** 叠图证据：demo 2 红 mask 明确在伸手指尖上、香蕉没被选；demo 4 正确盖在面包上；demo 5 抓到右边缘一块红色物。

跨 demo 复现的三个失败模式：

1. **reach 阶段指尖假阳性**（demo 0、2）：手从画面边缘伸入时 GT 关键点越界/缺失 → 手区域算出来很小/空 → 指尖小团 `hand_ov=0.00` 逃过否决；而指尖团质心≡指尖本身，co-motion 天然满分 → 抢走 seed。**这是 v3 硬伤，纯几何补不干净。**
2. **小/软物体 SAM 分 0**（banana、macaron 走 box fallback，seed 质量差）。
3. **contact 相位与真实抓取错位**（demo 4 seed 对，却因 grasp 段无位移 QA FAIL）——属另一子系统（`_detect_contacts`），本附录不动。

## C.1 为什么用开放词表检测器（而非纯几何 / 而非重型 VLM）

- 环境里可用的「VLM」就是 **开放词表检测器**：直接出框、无需推理链、比通用 VLM 轻。
- **文字锚定天然杀掉指尖假阳性**：`"banana"` / `"iphone"` 不会在裸手指上触发 → 直接消灭失败模式 #1。
- `objects.json` 已含被操作物体名（`objects[0]`：demo1=iphone、demo2=banana、demo5=block），**零额外标注**。
- 之前「靠 YOLO 文本 prompt 就要逐数据集调」的老问题，这次靠**手附近过滤 + 真实抓取帧采样 + SAM2 传播**补上——不再逐帧全图检测取最早置信框。
- **为什么不是 LLaVA**：LLaVA-1.5 这类通用 VLM **不原生出框**（无 grounding 头），要吐坐标再接 SAM 既绕又不准，7B 级每帧秒级、需另开环境。真正"小而能出框"的是 Florence-2 / OWLv2 / Qwen2-VL，才是 LLaVA 的正确替代位。

### C.1.1 检测器选择（可插拔，`intent_seed_detector`）

架构与检测器**解耦**——seed 只需要「给定名词 → 出框 + 分数」这一个接口（`get_bboxes(frame, noun, threshold)`）。因此检测器做成可插拔，按可用性与强度排序：

| 后端 | 依赖 | 强度 | 结论 |
|---|---|---|---|
| `yolo`（`DetectorYoloWorld`, worldv2） | **已加载**（ultralytics 8.4.120），零新增 | 常见名词 OK；生僻/细粒度（macaron、pop-it）弱、重遮挡掉框 | **零风险基线，先用它验证架构** |
| `owlv2`（`transformers.Owlv2ForObjectDetection`） | **现有 transformers 4.42.4 原生**，**无自定义 CUDA 算子**，只需下权重 `google/owlv2-base-patch16-ensemble` | 生僻名词/开放词表通常明显强于 YOLO-World，直接出框 | **强项兜底，无环境冲突** |
| ~~`dino`（Grounding-DINO）~~ | **剔除** | — | 其 C++/CUDA 自定义算子与现有 torch 2.1.0+cu121 **冲突，用不了** |
| Florence-2 | `trust_remote_code` + timm/einops（已在），需关 flash-attn | 强 | 备选，比 OWLv2 麻烦（下远程建模代码），暂不列首选 |

DINO 因算子冲突剔除；OWLv2 补上「更强且零冲突」的位置。默认 `yolo`，`intent_seed_detector=owlv2` 一键切换。

## C.2 设计：VLM 锚定候选（替换 B.2.3 的候选来源）

候选帧集合仍是附录 B 的**全局运动段采样**（`_sample_global_candidates`）。改的是「每个候选帧怎么产出 mask」：

对每个候选帧 `t`：

1. **开放词表检测**：`detector.get_bboxes(frame_t, noun, threshold=intent_dino_threshold)`，`detector` 由 `intent_seed_detector`（`yolo`/`owlv2`）选定（同一接口，见 C.1.1）；`noun` 依次取 `objects.json` 的 `objects`（先 `objects[0]` 被操作物，检不到再退下一个）。
2. **手附近过滤（选"被拿的那个"实例）**：只保留框心到**该帧 pinch UV**（拇指-食指中点，退化用指尖均值）距离 ≤ `intent_track_hand_px`（默认 200px）、且面积 ≤ `intent_track_max_area_frac·H·W` 的框（挡掉整桌/容器大框）。取最近的（并列比分数）。
3. **SAM2 image**：`segment_box` 得 mask。
4. **手 overlap 否决**（沿用附录 B.2.2）：与 GT 手区域重叠 > `intent_track_hand_overlap_max` 则减去手区域；剩太小则弃。
5. **co-motion 打分**（沿用 B.2.3）：±`seed_probe_win` 短传播，`score = cos(obj_vel,tip_vel) + w_static·static_bonus − area/nan 惩罚`。
6. `intent_seed_grounding=auto` 时：VLM 在该帧无近手框 → 回退到几何 `_try_hand_seed(t)`（现 v3 逻辑），保证不劣化 iphone 这类几何已对的情形。

**全局取最高分**做锚点 → 再走 §2.3 全程双向传播 + §2.4 闸门 + §2.5 重抓（重抓也优先用 VLM 锚定，检不到再几何）。

预期修复：demo 2 香蕉——reach 帧检不到 "banana" 附近框（指尖不触发）→ 指尖候选出局；真正握香蕉的帧才有近手 "banana" 框 → 胜出。demo 0 stapler——握持帧即便 overlap 高，改由 VLM 框给出物体主体、再减手区域，不再只剩指尖。

## C.3 改动清单（已实施）

| 文件 | 改动 |
|---|---|
| `detector_owlv2.py`（**新增**） | `DetectorOwlv2`：`transformers` 原生 `Owlv2ForObjectDetection` + `AutoProcessor`，实现与 `DetectorYoloWorld` 相同的 `get_bboxes(frame, noun, threshold)`；`google/owlv2-base-patch16-ensemble`，`attn_implementation="eager"`（无 flash-attn） |
| `intent_processor.py` | 新增 `detector` 按 `intent_seed_detector`（`yolo`/`owlv2`）惰性构造；`_get_object_nouns(paths)`（读 objects 列表）、`_pinch_uv(kpts_2d_t)`、`_pick_inhand_box(bboxes,scores,hand_uv,hand_px,max_area)`（静态、可单测）、`_vlm_grounded_mask(frame,nouns,hand_uv,hw)`、`_candidate_seed_mask(...)`（vlm→geom + overlap 否决，返回 mask/score/overlap/source）；`_select_seed_by_verify` 与重抓改调 `_candidate_seed_mask`；日志/`track_quality.npz` 记 `seed_source`（`vlm:noun`/`geom`） |
| `egodex_panda_intent.yaml` | `intent_seed_grounding: auto`、`intent_seed_detector: yolo`（复用 `intent_track_hand_px`、`intent_dino_threshold`、`intent_track_max_area_frac`） |
| `test_hand_anchored_tracking.py` | 加：`_pick_inhand_box` 选近手框、拒整桌大框、无近手框返回 None；`_get_object_nouns` 解析 objects.json |
| 下游 / §2.3–2.5 / 附录 B 手区域与 overlap | **不改** |

## C.4 元判断（延续 B.4）

VLM 锚定用「物体名 + 在手附近」两个强先验替换脆弱的几何 seed，是 B.4 里明确许可的转向，**不算堆第 4、5 条几何规则**。验收线：demo 0–5 里清晰跟对 ≥ 5/6，且 demo 1/4 不劣化。升级顺序（检测器可插拔，见 C.1.1）：`yolo`（零风险基线）→ 若 macaron/block 这类**词表外/小物**仍检不到 → 切 `owlv2`（同环境、无冲突、只多一次权重下载）→ 仍不行再考虑 Florence-2，或接 §B.4 提到的**手-物接触检测 + 手排除分割**。**Grounding-DINO 因算子与现有 torch 冲突，已从路线中剔除。** 无论哪个检测器，都不在 seed 打分上继续加项。

## C.5 验收（demo 0–5，v4）

- demo 2 banana、demo 0 stapler、demo 5 block 的 seed mask 落在**物体本体**（非指尖/非容器/非背景红块）；
- `track_quality.npz` `seed_source` 多为 `vlm:<物体名>`，`inhand_corr` 明显高于 v3；
- retarget 不再 PRUNED（demo 2/3）；
- 回归：demo 1（iphone）、demo 4（bread）不劣于 v3。

## C.6 v4 实测结果（已跑，owlv2 为默认）

除 seed 选择外，还补了一个**来源优先级**修正（`_select_seed_by_verify`）：`auto` 模式下只要**任一候选拿到 VLM 近手框**，最终只在 VLM 候选里取最高分；几何候选仅当**全程零 VLM 命中**才用（几何兜底本是"检测器全灭"的最后手段，不应逐帧和 VLM 抢分——否则 reach 指尖团 corr 高会盖过真实物体框）。

全量 `--no-reuse-masks` 扫 0–5（`intent_seed_detector: owlv2`，默认已切）：

| demo | 目标 | seed | 身份 | track QA | retarget |
|---|---|---|---|---|---|
| 0 | stapler | `vlm:stapler` t=63 | ✅ | PASS corr 0.61 | PRUNED 9.7cm |
| 1 | iphone | `vlm:iphone` t=95 | ✅ | PASS corr 0.63 | ✅ accepted 0.8cm |
| 2 | banana | `vlm:banana` t=192 | ✅ | PASS corr 0.62 | PRUNED 17.6cm |
| 3 | macaron | `vlm:macaron` t=76 | ✅ | FAIL corr −0.14 | PRUNED 8.0cm |
| 4 | bread | `vlm:bread` t=87 | ✅ | FAIL corr −0.05 | ✅ accepted 2.7cm |
| 5 | block | legacy(moving=0) | ❌ | FAIL overlap 1.0 | accepted 0.5cm |

- **物体身份正确率 5/6**（v3 几何 2/6 → yolo 4/6 → owlv2 5/6），无回归；demo 3 macaron 靠 owlv2 从"盘子"修到正确物体（YOLO-World 检不到该冷门词）。
- 唯一未跟对的 demo 5：`moving_frames=0/47`（47 帧短夹、手位移不过 `seed_move_vel/move_run`）整条 hand-anchored 被跳过、回退老 assoc/reid 跟踪器，**与检测器无关**（独立旋钮，暂搁置）。
- **新暴露的下游问题**：demo 0/2/3 retarget 被 PRUNED，见附录 D。

---

# 附录 D：retarget `key_pos` 闸门与 release 相朝向缺陷（S1/S2）

> 背景：v4 让物体身份跟对后，demo 0/2/3 在 retarget 的 `retarget_key_pos_thresh=0.03`（3cm）闸门被 PRUNED。诊断后确认**闸门没冤枉**——它抓到的是一个真实缺陷，位于 Stage A→B 的朝向目标构造，而非阈值过严。本附录记录诊断与两个修法（S1 根因、S2 补充）。**当前状态：S1/S2 均已实施；全量 demo 0–5 验收（2026-08-18）：4/6 PASS（demo 1/2/4/5），demo 0/3 仍 FAIL（`key_pos` 3.35/3.15 cm，边缘超 3 cm）；demo 2 从 17.1 cm 降至 1.6 cm。**

## D.1 诊断：尖峰集中在 release 相，且与 ori_err 强相关

逐帧拆 `stageb_processor/q_trajectory.npz` 的 `pos_err`/`ori_err`（`phase` 1=grasp、3=release，`key_pos`=grasp+release 相 `pos_err` 的 max）：

| demo | 全程 pos_err mean | grasp 相 pos_err max | release 相 pos_err max | release ori_err |
|---|---|---|---|---|
| 0 stapler | 0.85cm | **0.4cm**（完美） | **9.7cm** | **1.3 rad（75°）** |
| 2 banana | 0.67cm | 8.8cm | **17.6cm** | 1.2 rad（69°） |
| 3 macaron | 0.78cm | 3.2cm | **8.0cm** | 0.4 rad |
| 1 iphone（对照） | 0.76cm | — | 5.9cm | 0.36 rad → accepted |
| 4 bread（对照） | 0.45cm | — | 4.3cm | 0.19 rad → accepted |

关键观察：

1. 全程 `pos_err` mean 仅 0.5–0.8cm，**只有 release 那几帧炸到 8–17cm**；grasp 相几乎完美（demo 0 仅 0.4cm）。
2. release 位置尖峰**同时**伴随巨大 `ori_err`（0.8–1.3 rad）；被 accept 的 demo 1/4 则 ori_err 小。
3. **不是单帧 outlier**：demo 0 release 三帧（t=86/87/88）全 ~9cm、p90=9.3cm → 用百分位/放宽阈值都救不了，是**整段 release 系统性偏**。

**为什么"跟对物体反而被 prune"**：v3 跟指尖时，`centroids`≈手点，`p_target = centroid + (hand−centroid)` 塌缩成手轨迹本身，目标自洽且平滑、轻松够到（demo 0 假象 0.8cm）；跟真实物体后，release 的**真实朝向**暴露，而目标朝向还冻结在抓取姿态 → 露馅。

## D.2 根因：朝向目标在 release 处是错的（v1 translation-only 假设）

两处坐实：

1. `intent_processor._integrate_intent`：`R_target` **整段冻结成抓取姿态 `G_rot`**（`R_target = np.tile(G_rot, (n,1,1))`），docstring 自认 *"object motion is tracked translation-only — a v1 assumption (§5.2)"*。物体在搬运→放下过程中在手里转了向，目标朝向却停在抓取时刻。
2. `traj_opt.py`（residual）朝向残差是**完整 SO(3) log-map**：`R_err = R_target[t] @ Rs[t].T; rv = as_rotvec(R_err)`，**没有平口夹旋转对称松弛**（绕接近轴自旋 + 180° 翻转本应是自由 DOF）。

于是 release 帧被要求"既到真实放置点、又保持抓取朝向"——冲突下优化器各让一步：朝向差 1.3 rad **且**位置差 17cm。闸门（评 grasp+release 相 `pos_err` max）如实报出这一冲突。

## D.3 S1（根因，推荐）：让 `R_target` 与 offset 跟随物体/手的旋转

**思路**：手部 GT 存的是完整 `kpts_3d`（21 点 3D，已 `_load_hand` 在用），可逐帧构出手姿态 `R_hand[t]`（掌法向 × 拇指-食指轴，与 grasp 合成同一套坐标构造）。以抓取时刻为参考，用**相对旋转**驱动目标：

- 朝向：`R_target[t] = R_rel[t] · G_rot`，其中 `R_rel[t] = R_hand[t] · R_hand[grasp_kf]⁻¹`（grasp 处 `R_rel=I`，退化回原行为；release 处带上真实转向）。
- 位置（object-relative 段）：`p_target[t] = centroids[t] + R_rel[t] · offset`（offset 仍在 grasp 处锁定，但跟着物体一起转，而非刚性平移）。

**改动位置（仅 `intent_processor._integrate_intent`）**：
- 新增 `_hand_rot_from_kpts(k3_rf_t) -> R(3×3)`：由 `kpts_3d`（robot 帧）构手/夹爪坐标系（approach=掌法向、closing=thumb→index、third=叉乘，Gram–Schmidt 正交化），返回旋转矩阵；缺关键点则 `None`。
- 缓存 `R_hand[grasp_kf]`；对 `t` 有效手姿态时算 `R_rel[t]` 并如上改写 `R_target[t]`、object-relative 段的 `p_target[t]`。
- `R_rel` 对缺帧/跳变做时间平滑或最近有效保持（沿用 `_fill_targets` 思路），避免把手姿态噪声灌进目标。
- 退化：无有效手姿态 → `R_rel=I`，完全回到当前 translation-only 行为（demo 1/4 不劣化）。

**预期**：release 的朝向需求从"冻结 G_rot"变为"真实放置朝向" → ori_err 大幅下降、位置不再被朝向拖偏 → demo 0/2/3 的 release `pos_err` 回落。**数据现成（GT `kpts_3d`）、中等工作量、直击根因。**

**风险**：`kpts_3d` 手姿态在快速运动/遮挡帧有噪声，须平滑；放置瞬间手可能松开导致 `R_hand` 抖动 → 可在 release 窗口内对 `R_rel` 做保持/低通。

## D.4 S2（补充）：traj_opt 平口夹对称约减 + 降 `wr_grasp`

即便 S1 给对了标称朝向，完整 SO(3) 残差仍会**过约束**平口夹本有的自由度。两点补充（互补，非替代 S1）：

1. **对称约减朝向残差**（`traj_opt.py` residual/jac，~143–170）：
   - **自旋自由**：把 `R_err` 的旋转向量 `rv` 沿**夹爪接近轴**的分量置零（该轴自旋不改变平口夹抓取），只惩罚离轴分量。
   - **180° 翻转**：在 `R_target[t]` 与其绕接近轴转 180° 的版本里，取使 `‖rv‖` 更小的一侧再算残差。
   - jac 需相应改（去掉被约减方向的行，或按投影后重算），保持 GN 近似一致。
2. **降 `wr_grasp`**：关键帧朝向权重当前会让朝向压过位置（代码注释本就担心）。适度调低 `wr_grasp`，让位置在冲突时优先，配合 S1 后 ori 本就变小，副作用可控。

**改动位置**：`traj_opt.py`（朝向残差与其 jacobian）+ 配置 `wr_grasp`。**解决"过约束"部分**，与 S1 叠加。

## D.5 不推荐单独做的（治标、会掩盖真实缺陷）

- **放宽阈值到 5cm / 改百分位**：这里是**系统性 release 偏差**（整段 8–17cm、p90≈9cm），band-aid 一个也救不回，还会把真实的放置朝向错误藏起来。
- **只评精确关键帧不评整段**：grasp/release 相各仅 3 帧，尖峰跨满整段，无济于事。

## D.6 实施与验收建议

1. 先做 **S1**，仅在 demo 2（最严重）`--no-reuse-masks` 跑 intent+stageb，比对 `q_trajectory.npz` 的 release 相 `pos_err`/`ori_err` 是否回落（目标：release `pos_err` max < 3cm、ori_err 明显下降）。
2. 若仍有过约束残留 → 叠加 **S2** 的对称约减 + 调 `wr_grasp`，再验。
3. 全量扫 0–5，验收线：
   - demo 0/2/3 retarget 由 PRUNED 转 **accepted**（`key_pos < 3cm`）；
   - 回归：demo 1/4 `key_pos` 不劣化（S1 在无有效手姿态时退化为原行为）；
   - `ori_err` mean 在 pick-place demo 上整体下降。

## D.7 S1/S2 实测 + 后续两项改动（已实施）

### D.7.1 S1/S2 实测（用户跑）

S1（`R_target`/offset 随手相对旋转）+ S2（平口夹对称约减 `_project_parallel_jaw_rotvec` + 降 `wr_grasp`）后，release 位置误差整体塌约 5×（banana 17.6→1.61、stapler 9.7→3.35、macaron release 8.0→1.88）。**4/6 PASS**，剩 demo 0（release 3.35cm）、demo 3（失败点从 release 移到 **grasp** 3.15cm）两个临界项。

同时发现诊断 `ori_err` 冲到 ~2.6–2.8 rad（≈π）：S2 只在**残差**里忽略对称，但 `traj_opt.py` 的**报告** `ori_err` 仍用完整 SO(3)，把合法自旋算成误差 → 失真（非夹爪反向）。

### D.7.2 改动 A：质心时间平滑（治 demo 3 grasp 小物体深度抖动）

- 位置：`intent_processor._build_object_pointclouds` 末尾调用新增 `_smooth_centroids(centroids_robot, valid)`。
- **教训**：先试**中心化移动平均**——**回退了 demo 2/3**（demo 2 1.61→3.1 PRUNED、demo 3 3.15→8.6+vmax 尖峰）。因为 grasp/release 恰是运动**起止拐点**，均值在拐点滞后/过冲，且会跨遮挡 gap 混合。
- **最终实现**：**分段(不跨 valid gap) + 中值滤波**（边缘保持，专杀单帧深度尖峰、保住拐点）。窗宽 `intent_centroid_smooth_win=5`；记录偏移 > `intent_centroid_reject_m=0.05m` 的修正帧数。invalid 帧保持 NaN。

### D.7.3 改动 B：`ori_err` 报告改 reduced

- 位置：`traj_opt.py` optimize() 末尾报告块，`ori_err` 由完整 SO(3) 改为 `‖_project_parallel_jaw_rotvec(R_target, FK_ori)‖`，与优化器实际惩罚一致（自旋/翻转不计）。纯报告改动，不影响优化。

### D.7.4 结果（median 平滑，`--no-reuse-masks` 扫 0–5）

| demo | 目标 | key_pos | grasp/release max | reduced-ori 关键帧 max | retarget |
|---|---|---|---|---|---|
| 0 | stapler | 3.14cm | 0.39 / **3.14** | 37° | PRUNED（key_pos 3.1>3，临界） |
| 1 | iphone | 0.86cm | 0.08 / 0.86 | 16° | ✅ accepted |
| 2 | banana | **1.18cm** | 0.22 / 1.18 | 4° | ✅ accepted（S1/S2 时 1.61 → 更优） |
| 3 | macaron | **2.95cm** | **2.95** / 1.94 | 82° | PRUNED（**key_pos 已过**，仅 vmax 0.86>0.5 rad/帧） |
| 4 | bread | 2.45cm | 0.17 / 2.45 | 8° | ✅ accepted |
| 5 | block | 0.48cm | 0.48 / 0.16 | 1° | ✅ accepted |

- **净改善、零回退**：demo 2 key_pos 1.61→1.18；demo 3 key_pos 3.15→**2.95（过 3cm 门）**——质心平滑修好了小物体 grasp 抖动。
- **reduced-ori 报告可用了**：通过的 demo 关键帧朝向误差仅 1–16°（此前完整 SO(3) 报 ~π 系自旋失真）。
- **仍剩两项（本次不动，属独立旋钮）**：
  1. **demo 0** key_pos 3.14cm（仅超 0.14cm，release，reduced-ori 37°）：S1 未完全压平 stapler 放置朝向；候选 = 收紧 release 窗手姿态平滑，或按 D.5 权衡后微调阈值到 3.5cm。
  2. **demo 3** vmax 0.86 rad/帧（macaron 快速再定向的**关节速度**尖峰，非位置）：属平滑度/速度闸门，候选 = 提高 `w_smooth`/`w_vel` 或松 `vmax` 阈值。

配置新增：`intent_centroid_smooth_win: 5`、`intent_centroid_reject_m: 0.05`（`egodex_panda_intent.yaml`）。
