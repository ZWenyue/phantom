# Contact-Grounded Retarget 两个问题的改动方案

> 状态：方案 A **代码已接入**；问题二方案一（深度）与方案二（`shoulders_ego` + 躯干工作原点）已落地。本 task 左手常闲置，会 park 在 init_qpos。
> 已确认：**任务是双手（bimanual）**，因此问题一在范围内、且是主线工作。
>
> 针对 `b/run_contact_retarget.sh` 跑出的这批数据的两个问题：
> 1. 只有右手被换成机械臂（左手没有被替换）——需做双臂。
> 2. 机械臂在第一视角里遮挡了大片视野。
>
> 抽检样本：`egodex_basic_pick_place` 的 demo `0 / 1 / 10`，看
> `retarget_processor/retarget_diagnostic.png`（左列=去手背景，右列=贴臂 overlay）。

---

## 0. 现状与证据

- 背景 `inpaint_processor/video_human_inpaint.mkv` 已把**双手**都抹掉，左列诊断图里看不到人手。
- GT 手部数据 `hand_data_left.npz` / `hand_data_right.npz` **两只手都导出了**
  （`b/export_egodex_hand_gt.py` 对 `("left","right")` 都写盘）。
- 但 contact-grounded 的 intent → stageb → retarget 三段全程只处理 `target_hand`（=`right`），
  最终只渲染一条 Panda 臂。
- overlay 默认是「整臂无深度硬贴」，导致机械臂盖住本应在它前面的物体/桌面。

相关配置（`b/configs/egodex_panda_intent.yaml`）：

```yaml
bimanual_setup: "single_arm"
target_hand: "right"
retarget_use_depth: false
camera_extrinsics: "camera/camera_extrinsics_ego_bimanual_shoulders.json"
```

---

## ★ 关键发现：双臂基础设施已存在，只是没接到 contact-grounded 管线

这是本次方案最重要的前提——**双臂不用从零搭**，旧的 inpaint 路径早就跑双臂：

- `PhantomBimanual` 双臂环境 + `TwinBimanualRobot` 已实现
  （`phantom/twin_bimanual_robot.py`）；`RobotInpaintProcessor._initialize_robot`
  在 `bimanual_setup != "single_arm"` 时就用它（`robotinpaint_processor.py:87-149`）。
- `TwinBimanualRobot.move_to_joint_positions(q_right, q_left)` **直接注入双臂关节角**
  （`twin_bimanual_robot.py:520-535`）——正是 retarget 需要的（注入 `q` 而非跑 OSC）。
- 分臂 instance seg mask 提取已有（`twin_bimanual_robot.py:684-707`）。
- Stage B 的 `MujocoPandaArm(env, robot_idx)` **已按 `robot_idx` 读各自 base**
  （`stageb_processor.py:83-101`：`body_name2id(f"robot{idx}_base")`），
  只是 `StageBProcessor` 把 `robot_idx` 写死为 0（`stageb_processor.py:146-148`）。
- 外参 `camera/camera_extrinsics_ego_bimanual_shoulders.json` 本就是**双臂肩装**标定。

所以双臂改动的本质是：**把现成的双臂积木（PhantomBimanual / move_to_joint_positions /
分臂 mask / robot_idx-aware FK）接到 contact-grounded 的三段上**，而不是重写。

---

## 问题一：左手也要换成机械臂（做双臂）

### 根因

contact-grounded 三段目前是单臂假设：

- `RetargetInpaintProcessor.__init__` 明确警告 “only single_arm is implemented”
  （`retarget_inpaint_processor.py:47-51`）。
- `StageBProcessor` 写死 `robot_idx = 0`、`_make_single_arm_env` 只建单个 `Panda`
  （`stageb_processor.py:34-72, 146-148`）。
- `_render_trajectory` 只把 `target_hand` 那侧填 label，另一侧全写零
  （`retarget_inpaint_processor.py:136-161`）。
- IntentProcessor 全程按 `target_hand` 取手（`intent_processor.py:1149/1197/2900`）。

### 一致性红线（方案设计的核心约束）

retarget 的卖点是「渲染像素 = FK(q)」，所以 **Stage B 优化 q 时用的 base，必须等于
retarget 渲染 q 时那条臂的 base**。否则机械臂会渲染在错误的世界位置、对不上要替换的人手。

- 现状：Stage B 用单臂 base `[-0.56,0,0.912]`；`PhantomBimanual` 两臂各有自己的 base
  （`twin_bimanual_robot.py:145-148,218`，robot0=右 / robot1=左）。
- 这条红线决定了两种可行设计：

### 方案 A（推荐）：接入 PhantomBimanual，两臂各自 base、单次渲染

物理正确的双肩机器人，复用现成双臂环境。

1. **Intent：对左右手各跑一次，左右各一套独立目标。**
   - **决策（已定）**：原视频里两只手各自操作各自的物体，所以每只手的抓取/放置意图
     必须**独立地从它自己在视频里的真实运动**推出来——不是共享一套、也不是把同一套目标
     按 `robot_idx` 变换。左右是两条互不相关的 intent。
   - **好消息（已查证）：物体/接触/抓取归属本来就自动按手走。**
     - `_hands_for_contact` 只保留 `target_hand`，避免闲置手抢接触（`intent_processor.py:2899-2904`）。
     - 种子 mask 锚在 `target_hand` 的指尖 `hand_uv`（`intent_processor.py:679-682`），
       `intent_track_backend: hand_sam2` 下 SAM2 直接跟"手里那个物体"。
     - 所以只要把 `target_hand` 设成 `left`/`right` 各跑一遍，各自会锁到各自手里的物体，
       `p_target/R_target/grasp` 天然分手——**这部分几乎零改动**。
   - **`objects.json` 是单一全局 prompt** 的注意点：`{"objects":[...],"prompt":objects[0]}`，
     无左右之分（`export_egodex_objects.py:112`，`intent_processor.py:346-362`）。
     在 `hand_sam2` 下 prompt 只是 YOLO 兜底，主锚是手；但若左右手抓**不同物体**且需要 YOLO
     兜底，单 prompt 会偏置。→ 建议按手拆 `objects.json`（或干脆依赖手锚定、prompt 置空）。
   - 产物：`intent_processor/intent_left.npz` / `intent_right.npz`（`paths.py` 加路径）。

   > ⚠️ **坐标系配准红线（本次调查最关键的坑）**：`_place_T_w2r` 会算出**单一**的
   > world→robot(base) 刚体变换 `T_place`，把 `target_hand` 的抓取点对齐到
   > `intent_place_target=[0.5,0,0.05]`，再把它烘进**所有**几何（含 `p_target`，是 base 相对量，
   > `intent_processor.py:706-741, 788-833`）。
   > 若左右各**独立**算 `T_place`，两只手会各自被摆到"自己 base 的同一个 [0.5,0,0.05]"，
   > **丢失双手真实的相对空间关系**（两臂会重叠/错位）。
   > **正确做法**：两只手**共享同一套场景配准 `T_place`**（配准共享，目标各自独立），
   > 让左右 `p_target` 落在同一个一致坐标系里；具体 base 归属交给 Stage B 按 `robot_idx` 处理。

2. **Stage B：按手用对应 `robot_idx` 求解**，各自 base 天然正确。
   - 让 `StageBProcessor` 建 `PhantomBimanual` env，右手 `robot_idx=0`、左手 `robot_idx=1`；
     `MujocoPandaArm(env, robot_idx)` 已能读 `robot{idx}_base`（`stageb_processor.py:97-101`），
     把上面共享配准下的目标转换到各臂 base（**无需改 FK 类**）。
   - `PhantomBimanual` 两臂 base 是对称肩位（robot0=右 / robot1=左），由 env 定义、
     `robot_base_height/offset`（`twin_bimanual_robot.py:216-218`）。
   - 对两只手各解一次，输出 `stageb_processor/q_trajectory_left.npz` / `_right.npz`。
   - `_make_single_arm_env` → 需要一个 `_make_bimanual_env`（或直接复用 TwinBimanualRobot 的 env）。
   - **注意**：现在 `StageBProcessor.__init__` 写死 `robot_idx=0`（`stageb_processor.py:146-148`），
     且单臂 base 与 `PhantomBimanual` 的两臂 base 不同 → 目标要在**共享世界系**里给，
     由各臂 base 各自换算，别再沿用单臂 base 相对量。

3. **Retarget：一次性渲染两条臂 + 双侧 label。**
   - `RetargetInpaintProcessor` 在 bimanual 分支用 `TwinBimanualRobot`
     （复用 `RobotInpaintProcessor._initialize_robot` 的初始化路径）。
   - `_render_trajectory` 每帧调 `move_to_joint_positions(q_right[t], q_left[t])`
     （`twin_bimanual_robot.py:520-535`），取分臂 seg mask 合成到同一背景。
   - label 两侧都填真实值（现在右填值、左填零 → 改为双侧填 `q`/`FK(q)`）。
   - `_quality_gate` 对两条轨迹分别判定；任一不过 → 整条 demo 剔除（或按需保留单臂）。

4. **导出：`export_lerobot_retarget.py` 扩到双臂 action 维度**（现在只导单侧）。

> 工作量：中—大。查证后风险重排：
> - **物体/接触/抓取的按手归属**：几乎零成本（hand_sam2 + `_hands_for_contact` 已自动）。
> - **最大风险 = 坐标系配准**：左右必须共享同一套 `T_place`，否则两臂丢失真实相对位置；
>   且要把「单臂 base 相对量」改成「共享世界系 + 各臂 base 换算」。
> - 次要风险：确认 `PhantomBimanual` 两臂 base 在 ego 肩装视角下与人手位置吻合；
>   `objects.json` 若左右不同物体且需 YOLO 兜底则要按手拆。
> - 但环境/注入/mask/FK-base 这些底座都现成，整体风险可控。

### 方案 B（快速验证用）：两遍单臂渲染 + 合成

- 保持单臂 base，右/左各跑一遍现有单臂三段，得到两段 overlay + 两套 label，
  再按 seg mask 合成到同一背景。
- 优点：**几乎零核心改动**，能最快看到「双手都被替换」的效果，便于先验证 intent/stageb
  对左手是否 work。
- 缺点：两臂共用同一 base（单肩出两臂，物理不对）、跨臂遮挡不准、label 需外部拼接。
- 定位：作为方案 A 之前的**冒烟验证**，或左手 intent 调不通时的排查手段。

### 建议

先用**方案 B 跑通左手 intent/stageb**（确认左手抓取标定 work），再落**方案 A** 做正式双臂。

---

## 问题二：机械臂遮挡视野

### 根因（两点叠加）

1. **视角 + base 几何**：肩部第一视角相机 + 整条 7-DOF 臂从近处伸入，
   **粗大的上臂/肘部横扫画面顶部**（demo 1 第 2、3 帧最典型）。
2. **无深度硬贴**：`retarget_use_depth: false` → 走 `_process_robot_overlay`，
   `robot_mask=(seg>0)`（整条臂所有 link）直接盖到背景，不判深度
   （`robotinpaint_processor.py:619-622`）：

```python
overlay_mask = (robot_mask == 1) | (gripper_mask == 1)
img_robot_overlay[overlay_mask] = rgb_img_sim[overlay_mask]
```

结果：本应在机械臂**前面**的物体/桌面也被盖住。

### 方案一（推荐，见效快）：打开深度感知遮挡

- 配置 `retarget_use_depth: true`，走 `_process_robot_overlay_with_depth` +
  `_create_overlay_mask`（`robotinpaint_processor.py:626-732`）：机器人像素比真实场景更近才覆盖。
- **前置条件**：真实深度要对齐到渲染分辨率并接进 overlay。现有 DA3 `depth.npy` 可用，
  但当前 `_render_trajectory` 调深度 overlay 时喂的是零、真实场景深度没接上
  （`retarget_inpaint_processor.py:127-131`）。
- **改动点**：
  1. `RetargetInpaintProcessor` 加「加载真实深度序列并对齐到 `output_resolution`」，
     复用 `_load_background` 的裁剪/resize（`retarget_inpaint_processor.py:269-308`）。
  2. `_render_trajectory` 把该帧真实深度传给 `_process_robot_overlay_with_depth` 的 `img_depth`。
  3. 校准两套深度尺度/零点：sim 深度来自 `get_real_depth_map`，真实深度来自 DA3
     `depth.npy`（Stage A 已做 metric 对齐，需确认单位一致）。
- **收益**：物体重新「浮」到机械臂前，遮挡显著缓解。
- **风险**：深度尺度不一致 → 该盖没盖/不该盖却盖，用诊断图逐帧校验。
- 注：双臂 twin 也已有深度 + 分臂 mask，方案与单臂一致，做双臂时同样适用。

### 方案二（互补）：调机器人 base / 挂载，让臂从画面外进入

- 把 base 往后/往下挪，或换更贴合 ego 的外参，让只有前臂+夹爪进画面。
- **一致性**：base 一改，Stage B 与 Stage C 必须同步（见问题一的一致性红线），需重跑 stageb。
- 双臂下这里就是「调 `PhantomBimanual` 两臂 base / 肩宽」，属于方案 A 的标定环节。

### 方案三（轻量补丁）：只渲染 distal link / 收窄 robot_mask

- `_render_joint_positions` 里按 body/geom 过滤 seg，只留前臂+手+夹爪。
- 优点：局部、无需重跑 stageb。缺点：手臂视觉「悬空」，仅作临时可视化。

---

## 建议执行顺序（双手已确定）

1. **~~先做问题二方案一（深度遮挡）~~（已落地）**：`RetargetInpaintProcessor` 加载并对齐 DA3 `depth.npy`，把该帧真实深度传给 `_process_robot_overlay_with_depth`；`egodex_panda_intent.yaml` 的 `retarget_use_depth: true`。重跑 `mode=retarget_inpaint` 后用诊断图校验深度尺度（首帧会打 sim/real median 日志）。
2. **~~方案 B 冒烟~~（已跑 demo 1 `target_hand=left`）**：见下方落地记录。左手 GT 全帧在、三段管线能出 q/overlay；但左手几乎不动 → 无接触/无 grasp，不能当「左手也在操作」的视觉证明。正式双臂仍走方案 A。
3. **~~落方案 A（正式双臂）~~（代码已接）**：`contact_bimanual: true`。intent 左右各跑一遍、共享一套 `T_place`；stageb 建 `PhantomBimanual`、目标经 robot0 进同一 MuJoCo 世界、按 `robot_idx` 各解（无 grasp 则 park）；retarget 一次注入双臂 q。闲置手不挡质量门。
4. **~~调肩位（问题二方案二）~~（已落地）**：新 layout `shoulders_ego`：工作原点改成躯干（不再绑 robot0，否则挪肩相机跟着走、遮挡不变）；两臂 base 往后/两侧/略下。Stage B 与 Stage C 共用 `bimanual_torso_RT`。配置 `contact_bimanual_setup: shoulders_ego`，需重跑 stageb+retarget。
5. **实心臂**：`retarget_use_depth: false`。DA3 深度是原视频（含人手），inpaint 后 RGB 已是桌面，深度比较会把 Panda 中间挖空。关深度后整臂硬贴。

---

## 落地记录

### 问题二方案一（demo 1，`mode=retarget_inpaint`）

- 质量门通过：`key_pos=0.9cm`，160 帧渲染完成。
- 场景深度对齐 `(160, 1080, 1920)`，median `0.45m` / p95 `0.85m`（米，与 Stage A 一致）。
- 机器人像素：`sim median=0.14m` vs `real median=0.61m`。近处大臂**本来就比桌面近**，深度不会把它抠掉（问题二根因 1：肩装几何），这是预期。
- 有变化的是物体近处的帧：诊断图第 4 行 overlay 机器人像素 `7.8% → 4.5%`（桌面/盒盖从臂前露出来）。对比图：`retarget_processor/retarget_diagnostic_depth_compare.png`（旧剪影备份为 `*_nodepth.png`）。

### 方案 A（demo 1，`contact_bimanual: true`）

- intent `sides=['right','left']`，**一套** `T_place`（右手 grasp 锚到 `[0.5,0,0.05]`）。
- `intent_right.npz`：`grasp_valid=True`，kf 66/122；`intent_left.npz`：闲置、`grasp_valid=False`。
- stageb `PhantomBimanual` shoulders：右手 `robot_idx=0` 求解；左手 **park** 在 `init_qpos`（`q_trajectory_left.npz`）。
- retarget 质量门 `accept=True`（只卡右手，`key_pos=0.8cm`），一次注入双臂，160 帧 overlay 已写出。
- 本 task 左手不操作，画面里左臂是肩部初始姿态，不是第二次抓取。真双手 demo 才会两条活动轨迹。

### 问题二方案二（demo 1，`shoulders_ego`，只重跑 stageb+retarget）

- 工作原点 `torso = [0,0,1.5]`（不再绑 robot0）；右/左 base 约 `(-0.18, ±0.30, ~1.40)`，yaw=0。
- 右手 `pos_err mean=0.8cm max=6.7cm`；质量门 `key_pos=1.5cm`（此前 shoulders 约 0.8cm），仍通过。sim 深度 median `0.25m`（此前 `0.14m`）——臂整体离相机更远。
- 诊断图 overlay 像素占比（相对左列去手背景）：行 1–5 从 nodepth 的 `13.5 / 41.4 / 41.5 / 7.8 / 17.2%` 降到 `4.2 / 31.1 / 32.6 / 4.6 / 3.2%`。接近桌面的第 4 行明显变干净；伸臂中段第 2–3 行大臂仍占约 1/3 画面，若还要更干净可再加大 `|Y|`/`|X|` 或上方案三 distal mask。
- overlay：`retarget_processor/video_overlay.mkv` / `retarget_diagnostic.png`。

### 方案 B 冒烟（demo 1 左手，独立目录以免覆盖右手产物）

目录：`test_phantom_processed/egodex_basic_pick_place_left/1/`，`target_hand=left`，`intent_reuse_masks=false`。

- 左手 GT `160/160` 检测到，intent/stageb/retarget **三段都跑通**。
- 但左手几乎闲置：`moving_frames=0/160`，`contact_frames=0`，`grasp_valid=False`，全程 `phase=free`。种子跟到了 `box lid` 且 `hand_ov=0`（不是手里的物体）。
- Stage B 仍给出平滑 q（`pos_err max=1.4mm`），那是跟一条几乎静止的 EE 目标，不是一次抓取。
- 抽检 `0/1/10`：右手才是操作手（moving ~49–75 帧）；左手 net 位移 2–4cm。本 task 不适合当「双手各抓各物」的视觉样例；方案 A 仍要用共享 `T_place` + 分臂求解。闲置手在正式双臂里应走质量门/不渲染，而不是硬跟一个桌面物体。

---

## 涉及文件清单（供改代码时定位）

- `phantom/phantom/processors/retarget_inpaint_processor.py`
  - `__init__`（`use_depth` 开关、单臂警告）:45-63
  - `process_one_demo`（加载 q、质量门、渲染入口）:66-111
  - `_render_trajectory`（双臂 label、深度实参）:114-162
  - `_render_joint_positions`（seg→robot_mask、depth 产出）:165-205
  - `_load_background`（裁剪/resize，可复用于真实深度对齐）:269-308
- `phantom/phantom/processors/robotinpaint_processor.py`
  - `_initialize_robot`（single_arm vs 双臂分支，可复用）:87-149
  - `_process_robot_overlay`（无深度硬贴）:590-624
  - `_process_robot_overlay_with_depth` / `_create_overlay_mask`（深度遮挡）:626-732
- `phantom/phantom/twin_bimanual_robot.py`
  - `__init__` / env 构建（`PhantomBimanual`、两 robot、base）:91-218
  - `move_to_joint_positions`（双臂关节直注入）:520-557
  - 分臂 seg mask 提取:635-707
- `phantom/phantom/processors/stageb_processor.py`
  - `_make_single_arm_env`（需扩双臂 env）:34-72
  - `MujocoPandaArm`（已按 `robot_idx` 读 base，可直接复用）:75-130
  - `StageBProcessor.__init__`（`robot_idx=0` 写死处）:144-170
  - `process_one_demo`（intent→world→优化）:172-199+
- `phantom/phantom/twin_robot.py`
  - `DEFAULT_ROBOT_BASE_POS`:84；单机器人 env 配置:128-166
- `phantom/phantom/processors/intent_processor.py`
  - `target_hand` 取手逻辑:1149/1197/2243/2900（双手需按手各跑一次）
  - `_hands_for_contact`（只留 target_hand，接触归属自动分手）:2899-2904
  - `_place_T_w2r` / 应用 T_place（★配准红线，需共享）:706-741, 766-833
  - `_get_object_nouns` / `_get_object_prompt`（单一全局 prompt）:329-381
  - `_load_hands`（左右两手都加载，双手数据已在）:3035-3065
  - `intent_place_target` / `place_workspace` 读取:179-181
- `phantom/phantom/processors/paths.py`
  - `intent` / `joint_trajectory` / `retarget_video_overlay` 等（双臂需加分手路径）:72,85,90,108
- `phantom/b/configs/egodex_panda_intent.yaml`
  - `bimanual_setup` / `target_hand` / `retarget_use_depth` / 相机外参:30-36,116
- `phantom/b/run_contact_retarget.sh`（`hydra_mode_arg`/循环需支持双手编排）
- `phantom/b/export_egodex_hand_gt.py`（双手 GT 已导出，双臂可直接用）
- `phantom/b/export_lerobot_retarget.py`（双臂需扩展 action 维度）
