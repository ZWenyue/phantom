# Stage A 进展与待办 — Contact-Grounded Retargeting

> 配套设计文档：`contact_grounded_retargeting.md`（§3 Stage A）
> 本文件记录 Stage A（感知 + 意图抽取）的实现进度、验证结果与后续待办。
> Status: Stage A 全部 4 个子模块（物体点云 / 接触检测 / 抓取合成 / 意图整合）已实现并验证。

---

## 0. TL;DR

Stage A 拆成 4 个子模块（见设计文档 §3）：

1. **物体点云**（分割 + 反投影深度）— ✅ **已完成并验证**
2. **接触检测**（指尖-物体距离切相位）— ✅ **已完成并验证**
3. **手→夹爪抓取合成**（antipodal `G*`）— ✅ **已完成并验证**
4. **意图整合**（`p_t*`, `R_t*`, `G*`, `g_t`, `phase_t` → `intent.npz`）— ✅ **已完成并验证**

**Stage A 已收尾**，产出 Stage B 轨迹优化直接消费的统一 `intent.npz`。

落地方式：**新增独立 `IntentProcessor`**，不改动现有 `action_processor`，便于 A/B 对比与回滚。

---

## 1. 已完成：物体分割 + 物体点云 + 接触检测

### 1.1 新增/改动的文件

| 文件 | 改动 | 说明 |
|---|---|---|
| `phantom/processors/intent_processor.py` | **新增** | `IntentProcessor(BaseProcessor)`，Stage A 前端主体 |
| `phantom/processors/paths.py` | 改 | 新增 `intent_processor/` 输出路径（见 §1.3） |
| `phantom/process_data.py` | 改 | 注册 `intent` mode（enum + 处理顺序 + registry）；`get_processor_classes` 改为**按需惰性导入** |
| `b/configs/pickplace_intent.yaml` | **新增** | Zed RGB-D 样例（`data/raw/pick_and_place`）的 intent 冒烟配置 |
| `b/export_egodex_objects.py` | **新增** | 从 EgoDex HDF5 的 `llm_objects` 导出每 demo `objects.json`（物体 prompt 来源） |

### 1.2 IntentProcessor 处理流程（本块）

1. **物体 prompt 解析** `_get_object_prompt`：优先读每 demo 的 `objects.json`（EgoDex `llm_objects`），否则用配置 `object_prompt`。
2. **帧加载** `_load_frames` / `_extract_frames_cv2`：纯 cv2 抽帧到 `original_images/`，与 pipeline 一致的 `square` 裁剪（不依赖 ffmpeg）。
3. **物体检测 + mask 传播**：
   - `_detect_seed`：Grounding-DINO 扫帧，选**最早的置信帧**（≥ `intent_seed_min_score`）而非全局最高分——早期帧物体孤立，避开操作/遮挡导致的框漂移（曾观察到全局最高分帧把框漂到旁边的木筐）。
   - `_propagate_mask`：SAM2 以 seed bbox+中心点为条件，前向 + 反向传播全片，得逐帧 mask。
4. **点云反投影** `_build_object_pointclouds`：
   - mask 腐蚀 `intent_mask_erode`（去边缘)；`depth ∈ (0, intent_depth_max)` 有效性过滤。
   - `get_point_cloud_of_segmask` 反投影 → 相机系点云；`remove_statistical_outlier` 去稀疏噪点；`transform_pts(T_cam2robot)` → robot 系点云。
5. **保存 + 可视化** `_save_results`：见 §1.3。

预留下一块的接口 stub：`_synthesize_grasp`。

### 1.3 输出（`{demo}/intent_processor/`）

| 文件 | 内容 |
|---|---|
| `object_masks.npy` | `(T,H,W) uint8` 逐帧物体 mask |
| `object_pcd.npz` | `points_cam` / `points_robot` / `colors`（object 数组，逐帧变长）、`centroids_robot (T,3)`、`valid (T,)`、`object_prompt`、`seed_idx`、`seed_score`、`intrinsics`、`T_cam2robot` |
| `contact_events.npz` | `phase (T,)`、`phase_names (T,)`、`g_closed (T,)`、`contact (T,)`、`d_finger_obj (T,)`、`aperture (T,)`、`obj_speed (T,)`、`grasp_keyframe`、`release_keyframe`、`source` |
| `video_object_mask.mp4` | mask 叠加原图的可视化视频 |
| `object_pcd_preview.png` | seed 帧点云的 3D 散点预览（robot 系） |
| `contact_diagnostic.png` | 接触信号诊断图（指-物距离/开口/物速 + 相位阴影 + 抓/放关键帧竖线） |

### 1.4 配置项（`b/configs/pickplace_intent.yaml`）

```yaml
object_prompt: "blue book"      # 无 objects.json 时的兜底 prompt
intent_dino_threshold: 0.25     # DINO 置信阈值
intent_seed_stride: 3           # seed 搜索抽帧步长
intent_seed_min_score: 0.35     # “最早置信帧”的分数下限
intent_depth_max: 3.0           # 有效深度上限（米）
intent_min_object_pts: 50       # 一帧点云有效的最少点数
intent_mask_erode: 2            # mask 腐蚀像素（去边缘噪点）
intent_outlier_nb: 20           # 统计离群点去除：邻居数
intent_outlier_std: 2.0         # 统计离群点去除：std 比例

# 接触检测
intent_contact_dist_in: 0.03        # 指-物距离进入接触阈（米）
intent_contact_dist_out: 0.05       # 指-物距离退出接触阈（迟滞，米）
intent_contact_min_valid: 5         # 有效指-物距离帧数下限（低于则回退物体运动线索）
intent_contact_min_run: 3           # 丢弃短于此的接触段（帧）
intent_contact_motion_thresh: 0.004 # 物体运动回退阈（米/帧）
intent_grasp_window: 3              # 接触起始的 grasp 相位长度（帧）
intent_release_window: 3           # 接触结束的 release 相位长度（帧）
```

### 1.5 接触检测流程（`_detect_contacts`，第 2 块）

1. **手指尖轨迹** `_load_hand_fingertips`：读 `hand_processor/hand_data_{left,right}.npz` 的 `kpts_3d`（相机系），经 `_to_robot_frame(T_cam2robot)` → robot 系；取 thumb/index/middle 指尖（idx 4/8/12），并算抓握开口 `aperture = ‖thumb_tip − index_tip‖`。
2. **逐帧信号**：
   - `d_finger_obj[t]`：各指尖到该帧物体点云的**最近表面距离**取最小（主线索，符合设计）。
   - `obj_speed[t]`：物体质心逐帧位移（米/帧，辅助/回退线索——物体只在被抓时移动）。
3. **接触判定**：
   - 有手时用**指-物距离双阈迟滞** `_hysteresis_contact`（进 `dist_in`、出 `dist_out`），`source=fingertip`。
   - 无手时（如当前 pick_and_place 尚未跑 HaMeR）自动**回退物体运动** `_motion_contact`（阈 `motion_thresh`，桥接运输中的瞬时静止），`source=object_motion`。
   - `_filter_min_run` 去除过短接触段。
4. **相位切分** `_segment_phases`：取最长接触段为操作段，输出 `phase ∈ {free, grasp, transport, release}`、夹爪状态 `g_closed`、`grasp_keyframe` / `release_keyframe`。
5. **保存 + 诊断图**：`contact_events.npz` + `contact_diagnostic.png`（见 §1.3）。

### 1.6 设计取舍
- **多线索 + 优雅降级**：主线索是设计要求的「指尖-物体表面 3D 距离」；物体运动作辅助/回退，使得在 EgoDex（有手无深度）/ pick_and_place（有深度暂无手）任一半信息缺失时都能产出可用相位（对齐设计 Risk #2 的多线索思想）。
- 手/物帧按视频帧序对齐，长度不一致时取 `min` 并告警。

---

## 1bis. 已完成：手→夹爪抓取合成（第 3 块）

### 1bis.1 流程（`_synthesize_grasp` 及其 helper）
不复刻五指，而是把平行夹爪抓取**锚定到物体点云几何**（设计 §3.4 / 本文 §5.1）：

1. **选帧** `_select_grasp_frame`：在 `[grasp_kf − precontact_window, grasp_kf]` 里挑**点数最多（最少遮挡）的接触前帧**估几何（对策 Risk #1：手遮挡前锁定），无候选时回退到 `grasp_kf` 或全局最稠密有效帧。
2. **闭合轴 + 接近向**：
   - **有手** `_grasp_axes_from_hand`：闭合轴 = `thumb_tip − index_tip`；接近向 = 抓取前若干帧手指质心的**运动方向**（回退为「物体 − 手」方向），`source=hand_anchored`。
   - **无手** `_grasp_axes_from_pca`：接近向取自顶向下（robot −Z）；在与接近向正交的两主轴里，闭合轴取**物体更窄**的那条（平行夹爪跨窄边闭合），`source=object_pca`。
3. **antipodal 接触** `_antipodal_from_axis`：沿闭合轴取投影的 `pct / 100−pct` 分位（默认 2%，抗离群点）对应的**真实物体点**为两指接触点 → 抓取中心 + 开口宽度均由真实几何得出。
4. **夹爪位姿** `_build_grasp_rotation`：右手系 `x=闭合/开合方向`、`z=接近方向`（对接近向做正交化，退化时兜底），并额外输出对齐现有 pipeline 约定的 `G_rot_pipeline = R @ Rz(90°)`（`HandModel` 夹爪朝向约定）。
5. 宽度超 `gripper_max_width` 时告警（物体沿闭合轴过宽）。

### 1bis.2 输出（`grasp.npz` + `grasp_preview.png`，见 §1.3 补充）

| 文件 | 内容 |
|---|---|
| `grasp.npz` | `grasp_frame`（实际估计帧）、`grasp_keyframe`、`G_center (3,)`、`G_rot (3,3)`、`G_rot_pipeline (3,3)`、`G_width`、`closing_axis (3,)`、`approach_axis (3,)`、`contact_points (2,3)`、`source` |
| `grasp_preview.png` | 物体点云 + 抓取坐标系 + 两指接触点的 3D 预览 |

### 1bis.3 配置项（`intent_grasp_*`）
```yaml
intent_grasp_precontact_window: 8   # 接触前选帧窗口（帧）
intent_grasp_approach_frames: 4     # 估接近向所用的抓取前帧数
intent_gripper_max_width: 0.08      # 夹爪最大开口（米），超出告警
intent_grasp_antipodal_pct: 2.0     # antipodal 分位（抗离群点，%）
```

### 1bis.4 验证（`b/tests/test_grasp_synthesis.py`，复用已存产物免跑 DINO/SAM2）
- **PCA 回退**（pick_and_place 真实点云，无手）：`frame=54`，中心 `[0.541,0.180,0.083]`，宽 **3.8cm**，接近向自顶向下 `[0,0,−1]`，`det(R)=1.000`、`RᵀR=I` → 旋转合法、抓取中心 = 两接触点中点。
- **合成手主分支**（注入指尖沿 +x 夹住、从 +z 下压的手）：闭合轴解析出 `≈[1,0,0]`、接近向 `≈[0,0,−1]`、宽 6.2cm、`source=hand_anchored` → 手锚定分支正确。
- 真实手（HaMeR）端到端验证同 §5.3（待手管线环境修复）。

---

## 1ter. 已完成：意图整合（第 4 块，Stage A 总输出）

### 1ter.1 流程（`_integrate_intent`）
把前三块的感知结果融成 Stage B 直接消费的**逐帧任务空间意图**（设计 §2.2/§2.3）：

1. **EE 位置目标 `p_target`**：
   - 抓取段 `[grasp_kf, release_kf]` 用**物体相对**表达——抓取起始锁定偏移 `offset = G_center − centroid[grasp_frame]`，逐帧 `p_t* = centroid_t + offset`，使 EE 跟着物体走（即便手被遮挡也不丢目标，对齐设计"接触段物体相对系"）。
   - 段外优先跟随人手（thumb-index 中点，`_ee_target_from_hand` 选开口更小=正在捏的手）；无手时 hold 抓取位姿；再退化到物体质心。
   - `_fill_targets` 用时序最近有效值补空，保证轨迹**无空洞**（不凭空造运动：前导 hold 首个有效、其余 hold 上一个）。
2. **EE 朝向目标 `R_target`**：全程锁定物体锚定的 `G_rot`（刚体平行夹爪抓取、物体按平移跟踪的 **v1 假设**，见 §5.2）。
3. **夹爪指令**：`g_closed` 来自接触 FSM，`gripper_width` = 闭合时 `G_width` / 张开时 `gripper_max_width`。
4. **相位相关代价权重 `w_p` / `w_r`**（设计 §2.4）：保真度预算集中在 grasp/release（高权重），free/transport 低权重——Stage B 直接读取即可。
5. 无有效抓取时**优雅降级**：手跟随/质心目标 + 单位阵朝向 + 均匀（free）权重。

### 1ter.2 输出（`intent.npz` + `intent_preview.png`）
统一 schema（Stage B 轨迹优化的输入）：

| 字段 | 形状 | 含义 |
|---|---|---|
| `p_target` | `(T,3)` | robot 系 EE 位置目标 |
| `R_target` | `(T,3,3)` | robot 系 EE 朝向目标 |
| `p_valid` | `(T,)` | 目标有效位（填补后全 True） |
| `p_source` | `(T,)` | 每帧目标来源：`object_relative`/`hand`/`hold_grasp`/`object_centroid`/`none` |
| `phase` / `phase_names` | `(T,)` | 相位码 / 名 |
| `g_closed` | `(T,)` | 夹爪开合状态 |
| `gripper_width` | `(T,)` | 夹爪宽度指令（米） |
| `w_p` / `w_r` | `(T,)` | 相位相关位置/姿态代价权重 |
| `object_centroid` | `(T,3)` | 物体质心轨迹 |
| `G_center`/`G_rot`/`G_width` | — | 抓取位姿 + 宽度 |
| `grasp_frame`/`grasp_keyframe`/`release_keyframe` | — | 关键帧 |
| `gripper_open_width`/`grasp_valid` | — | 张开宽度 / 抓取是否有效 |

配置项（`intent_w{p,r}_{free,grasp}`）：`intent_wp_free=1.0`、`intent_wr_free=0.1`、`intent_wp_grasp=5.0`、`intent_wr_grasp=5.0`。

### 1ter.3 验证（`b/tests/test_intent_integration.py`，复用已存产物）
- **PCA/物体相对路径**（pick_and_place，184 帧）：抓取段 `[59,175]` **117/117 帧物体相对**目标；`w_p ∈ {1.0, 5.0}`（free/关键相位分层正确）；`gripper_width` 在闭合帧=3.8cm、张开帧=8cm；`R_target` 全程正交。`intent_preview.png` 的 EE 目标轨迹呈 pick→lift(z 升)→transport(y 入筐)→place(z 降)→release，与真实操作吻合。
- **手跟随路径**（注入抓取前 6 帧合成手）：段外 6/6 帧 `source=hand`，`p_target` 精确等于 thumb-index 中点 → 手跟随分支正确。

---

## 2. 如何运行

```bash
cd phantom
python process_data.py --config-path=../b/configs \
    --config-name=pickplace_intent mode=intent demo_num=0
```

EgoDex 物体 prompt 导出（切回 EgoDex 时用）：

```bash
python b/export_egodex_objects.py --task basic_pick_place   # 写每 demo objects.json
```

---

## 3. 验证结果（`data/raw/pick_and_place`，Zed RGB-D，184 帧）

- 数据：1080×1080 RGB + 度量深度 `depth.npy (184,1080,1080)`；`square` 校正后内参 `cx=cy≈552.5`，与深度对齐。
- DINO：`blue book` 在第 0 帧检出书（score 0.742，bbox `[646,641,730,857]`）。
- SAM2：前向 + 反向传播，**184/184 帧点云有效**。
- Mask 全程正确跟随被操作的蓝书（桌上 → 手中提起 → 放入木筐），不误跟木筐。
- 点云：book 尺寸约 `7.5×15.6×10.3 cm`；质心轨迹呈现 pick→lift（z 升到 ~0.15m）→place 的真实运动。
- 正式 CLI 入口全新跑通，**零 workaround**。

### 3.1 接触检测验证

**物体运动回退分支**（pick_and_place 真实点云；该 demo 暂无手数据）：
- 物体在 frame ~59 前静止、59–173 运动（frame~100 抬举/放置有明显速度峰）、~175 后静止 → FSM 切出 `free[0:59) → grasp[59:62) → transport[62:173) → release[173:176) → free[176:184)`，`grasp_kf=59` / `release_kf=175`。
- 诊断图 `contact_diagnostic.png`：`gripper closed` 黑线与 transport 阴影一致，抓/放竖线落在物速起止处 —— 与真实操作吻合。

**指尖-物体距离主分支**（合成手轨迹单测：指尖 frame 20→30 逼近物体、55→65 撤离）：
- 结果 `source=fingertip`，指-物最近距离 `0.002 .. 0.253 m`，切出 `free → grasp@28 → transport → release@58 → free`，与逼近/撤离时刻一致 → 主分支逻辑正确。
- 真实手（HaMeR）在 RGB-D 上的端到端验证待手管线环境修复（见 §5.4）。

---

## 4. 环境排障记录（历史 torch 2.13 残留污染，非代码问题）

跑通前遇到并解决的环境问题（供复现/排查参考）：

1. **torch/torchvision onnx 冲突**（`cannot import name 'ExportOptions'`）：旧 torch 2.13 卸载不干净，残留新版文件遮蔽 2.1.0 模块 → 彻底删 `torch/`、`torchvision/` 及 dist-info 后重装 `torch==2.1.0+cu121` / `torchvision==0.16.0`。
2. **numpy 双 dist-info 污染**：磁盘文件是 2.2.6、pip 元数据记成 1.26.4（`already satisfied` 但 `import` 出 2.2.6）→ 删 `numpy/`、`numpy-*.dist-info`、`numpy.libs` 后 `pip install --no-cache-dir numpy==1.26.4`。项目要求 `numpy==1.26.4`（`install.sh` / `install_robust.sh` 明确钉死）。
3. **入口一次性导入所有 processor**：`get_processor_classes` 原先 eager import 全部 processor，把 hamer/mmpose/detectron2/xtcocotools 等 intent 用不到的重依赖拖进来，任一坏掉即整体起不来 → 已改为**按需惰性导入**（坏的 mode 仅在实际实例化时报错，不阻塞其它 mode）。

> 关键 pin（来自 `install.sh`）：`torch==2.1.0+cu121`、`torchvision==0.16.0`、`numpy==1.26.4`、`transformers==4.42.4`。

---

## 5. 待办（Stage A 剩余子模块）

### 5.1 手→夹爪抓取合成（`_synthesize_grasp`）— ✅ 已完成（见 §1bis）

### 5.2 意图整合（Stage A 总输出）— ✅ 已完成（见 §1ter）

**⬜ 待办：接触段 `R_target` 跟随物体旋转（带旋转幅度门控）**

- **现状**：`R_target` 全程锁定 `G_rot`，等价于假设物体只平移不旋转（v1）。
- **动机**：本 pipeline 里画面物体是**真实**的、机械臂是渲染的 `FK(q)`；物体明显旋转（倒/插/翻转/拧）时，夹爪不转会导致渲染穿帮 + obs-action 不一致。
- **做法**：加逐帧物体刚体 6DoF 位姿跟踪 `R_obj(t)`，接触段把姿态目标改为
  `R_t* = R_obj(t) · R_obj(grasp)^{-1} · G_rot`；free 段可接入人手腕朝向（当前 `IntentProcessor` 只载入指尖 `FINGERTIP_IDXS`，未载腕部朝向）。
- **⚠️ 门控（重要）**：**默认关闭**，仅当测得物体累计旋转超阈值（如 >15~20°）才启用。原因：
  (a) 平移主导任务（如当前 pick-place）常量假设误差极小，强开会把物体位姿估计的抖动灌进高 `w_r` 的姿态项；
  (b) 抓取时物体正被遮挡（Risk #1），`R_obj(t)` 估计本身噪声大，代价主要在**感知**而非 Stage B 求解（Stage B 无硬 IK，姿态达不到只是残差增大，由 `w_r` + 软优化 + 夹爪对称性吸收，不会"无解"）。
- **优先级**：非阻塞。先在平移任务上跑通 Stage B 主线，遇到真正需重定向的任务再回来加。

### 5.3 已知遗留 / 依赖决策
- **接触检测的真实手端到端验证**：pick_and_place（有深度）暂未跑 HaMeR（当前环境 `mmpose`/`mmcv` import `EOFError`、`_DATA` 手权重目录为空），故 RGB-D 上接触暂走物体运动回退分支；指尖分支已用合成手单测验证。修复手管线（或补 EgoDex 深度）后即可端到端跑指尖分支。
- **EgoDex 无深度**：EgoDex HDF5 只有相机内参 + 3D 手关节，**无场景深度**，故点云链路目前在 Phantom Zed RGB-D 上跑通。切回 EgoDex 需先定深度方案（视频级时序一致 + 度量深度模型，见设计文档 Risk #4 / M4）。`objects.json` 导出脚本已就绪。
- **点云残留噪点**：seed 帧仍有少量书页/筐边稀疏点；如需更干净可加最大连通簇/DBSCAN 提取（当前统计离群点去除已够用）。
- **多 demo / 多物体**：当前单目标物体；多物体、跨帧关联、穿遮挡跟踪（Cutie/SAM2 video）待扩展（Risk #3）。

---

## 6. 里程碑对齐（设计文档 §7）

- **M1（RGB-D 主线打通）**：Stage A 全部 ✅（物体点云 / 接触检测 / 抓取合成 / 意图整合，产出 `intent.npz`）；**Stage B 整段轨迹优化 ✅**（见 `stage_b_progress.md`）；**Stage C 渲染+标签+轨迹级剪枝 ✅**（见 `stage_c_progress.md`）。contact-grounded 三段管线 `intent → stageb → retarget_inpaint` 已端到端跑通并验证。
