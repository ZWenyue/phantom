# EgoDex 双手去除与双臂动作映射

## 目标

Human-to-robot 数据需要同时满足：

- 人左手轨迹映射到左 Panda（`robot_idx=1`）。
- 人右手轨迹映射到右 Panda（`robot_idx=0`）。
- 原视频中的左右人手和手臂都由 ProPainter 去除。
- 确实静止的手可以停放对应机械臂；没有检测到 grasp 但手在运动时，机械臂仍应跟随。

## 原因

`egodex_panda_intent` 使用 `bimanual_setup: single_arm` 保留 contact-retarget
管线自己的 `shoulders_ego` 双臂布局。旧的 arm segmentation 因此只加载
`target_hand: right` 的 HaMeR 序列，`masks_arm.npy` 不含左手/左臂。

Contact intent 和 Stage B 已经按手分别保存
`intent_{left,right}.npz` 和 `q_trajectory_{left,right}.npz`，左右映射也是正确的。
问题在于 Stage B 过去把所有 `grasp_valid=False` 的手直接视为闲置并 park，
即使 intent 的 `p_source=hand` 已包含运动轨迹，也不会求解对应机械臂。

## 代码改动

### 双手去除

`BaseProcessor.process_both_hands()` 在以下任一条件成立时返回 true：

- `inpaint_both_hands: true`
- `contact_bimanual: true`

`BaseSegmentationProcessor._load_hamer_data()` 据此同时加载
`hand_data_left.npz` 和 `hand_data_right.npz`。`ArmSegmentationProcessor`
原有逻辑会分别初始化 SAM2，再把两侧传播结果 OR 到同一个
`segmentation_processor/masks_arm.npy`。若 Detectron2 人体 mask 与手关键点坐标系
不一致（720p EgoDex 上常见，左右 init 都会失败、mask 全空），则改用该手 bbox 做
单帧 SAM2 提示再时序传播。ProPainter 无需修改，直接使用该 union mask
生成 `inpaint_processor/video_human_inpaint.mkv`。

EgoDex/EPIC bbox 阶段本来就从 `hand_det.pkl` 读取左右检测，hand2d 阶段也本来就输出
左右 HaMeR 序列；本次修复的是后续 arm segmentation 的单手加载限制。

配置：

```yaml
inpaint_both_hands: true
```

### 无 grasp 时仍跟随运动手

Stage B 不再用 `grasp_valid=False` 作为唯一 park 条件，而是统计
`intent.npz` 中满足以下条件的目标：

- `p_valid=True`
- `p_source == "hand"`
- `p_target` 为有限 3D 坐标

用各轴 5%–95% 分位区间的对角线长度作为鲁棒运动 span。只有无有效 grasp 且
hand span 小于阈值（或有效手目标帧数不足）时才 park。超过阈值时照常运行轨迹优化，
因此左手驱动左臂、右手驱动右臂；无 grasp 时夹爪保持 intent 给出的打开状态。

配置：

```yaml
stageb_park_idle_arms: true
stageb_idle_hand_span_thresh: 0.02
stageb_idle_hand_min_frames: 5
```

阈值单位为米。若希望所有检测到的手都驱动机械臂，可设
`stageb_park_idle_arms: false`。若闲置臂受跟踪抖动影响而移动，则提高
`stageb_idle_hand_span_thresh`。

### 人臂（含袖子）完整进入 mask

上面的 bbox-SAM 提示只在 Detectron2 初始化失败时兜底，但它以手的 bbox 为种子，
SAM2 常常只吃到皮肤像素，黑色袖子留在画面里。修复后
`ArmSegmentationProcessor._find_best_init_frame` 在关键点包含判定失败时，不再直接
退到"只有手"的 box mask：它会保留与手 bbox 有实际重叠的最高分 Detectron2 人体
mask（重叠判定见新增的 `_mask_overlaps_bbox`），并与 box mask 取并集作为 init 帧。
只有连重叠人体 mask 都拿不到时才使用纯 box mask，并打 warning 说明袖子可能不全。

验收看 `masks_arm.npy` 内暗像素占比：修复前 0.6%–2%（只有皮肤），修复后 14%–68%，
且所有帧的 mask 都触及画面下边缘（手臂延伸出画外）。

### 自身机械臂遮挡视野

两条臂都开始跟随之后，`shoulders_ego` 下的机械臂中段会占满 egocentric 视野。
根因是底座离相机太近：相机在躯干原点 `(0, 0, 1.5)`，而旧底座在 `(-0.18, ±0.30, 1.40)`，
只在相机后方 0.18 m、下方 0.10 m，link5/link6 摆动时正好扫过视锥。

**按比例缩小机械臂不可取。** 实测 Panda + Robotiq85 到 grip site 的最大触及是
1.324 m（0.855 m 是 Panda 法兰自身规格，不含夹爪），bowls 任务只用到 0.784 m
即 59%，臂展上确实有缩放余量；但缩放会让夹爪与碗的尺寸对应失真，而抓取接触正是
策略最依赖的视觉线索，同时只缩视觉网格还会让各 link 之间出现断缝。

采用的方案是把肩部按真人躯干比例后移下移：

```text
robot{0,1}_base: (-0.18, ±0.30, 1.40)  ->  (-0.35, ±0.40, 1.15)
```

即相机后方 0.35 m、下方 0.35 m、外侧 0.40 m。该位置下 demo 31 的最大臂展需求为
1.049 m，占实测触及的 79%，留有约 20% 余量供姿态控制。改动在
`phantom_bimanual.py` 的 `bimanual_setup == "shoulders_ego"` 分支。
底座变了会改变 FK，因此必须重跑 `stageb`，不能只重跑 `retarget_inpaint`。

挪肩把整臂遮挡从"几乎占满视野"降到 demo 31 的均值 25.6%、中位数 15.2%。

### 腕部遮挡的下界

挪肩之后剩余的遮挡主要来自腕部 `link6`，且**这是几何下界，不能靠调整位姿消除**。
两项测量支持这个结论：

1. 扫描基座位置时，相机到手臂线段的最小距离恒定在 0.27–0.30 m。肩宽从 0.40 m
   加到 0.60 m 只改变 7 mm，基座下移 0.20 m 只改变 20 mm。原因是这个最小值出现在
   **手端而不是肩端**——相机在 `(0.11, 0.185, 1.76)`，末端目标在 `x∈[0.33,0.44]`、
   `z∈[1.47,1.59]`，即人正低头看约 0.3 m 外的双手。该距离由人的演示决定。
2. 六条 demo 的逐 body 像素归因显示 `link6` 占机器人像素的均值 36%–77%、峰值
   85%–99%。唯一例外是 demo 20（6%），而它恰好是左臂被 park、只有单臂在动。

Panda 的 link6 直径约 0.10–0.15 m，停在距镜头 0.3 m 处必然占据大片画面。
**当前选择是保留全部 link、接受这一遮挡**，视其为真实机器人自视角的正常现象。

若某个任务无法接受，`RetargetInpaintProcessor` 支持只渲染远端 link：

```yaml
retarget_distal_only: false
retarget_distal_body_tokens: [link7, eef, gripper, finger, knuckle]
```

打开后会用 element segmentation 过滤掉近端 link，并把它们的 geom alpha 置 0，
避免这些不进 mask 的近端几何体在深度上仍然遮挡夹爪（实测该 alpha 处理让 frame 108
的远端可见像素从 87k 涨到 249k）。这要求 `twin_robot.py` / `twin_bimanual_robot.py`
同时申请 `instance` 和 `element` 两种分割。按上述六条 demo 估算，打开后总遮挡降到
均值 2.5%–14%，代价是手腕以后的手臂不可见。

### 零空间摆肘（已实现但默认关闭）

`TrajOptConfig.w_reg` 现在接受标量或 `(n_dof,)` 序列，配合
`stageb_elbow_swivel_deg` / `stageb_w_reg_elbow` 可以把 Panda 的大臂滚转
（joint index 2，绕肩—腕轴摆肘的冗余自由度）偏置到视锥外，左右镜像。

该功能对本任务无效，默认为 0，原因记录如下，避免重复尝试：

- 遮挡主体是腕部 `link6` 而不是肘部，摆肘改变不了它。
- 只把偏置写进 `q_neutral`（热启动种子）时，左臂优化仍停在原分支，
  `q[:,2]` 均值 +0.070 而目标中性位是 +0.698。
- 改用强正则（`w_reg_elbow=0.3`）强行拉过去，左臂会打满 `max_nfev=2000`，
  单臂耗时从 5 秒涨到 4.5 分钟，而 `pos_err` 没有任何改善。

实测覆盖率也基本不变（均值 25.3% 对 25.6%，p90 反而从 48.8% 升到 55.4%）。

### 总流程

`run_human2robot_all.sh` 的默认阶段现在是：

```text
inpaint → retarget → narration → lerobot
```

其中 `inpaint` 调用 `run_hand_inpaint.sh`，执行
`bbox,hand2d,arm_segmentation,hand_inpaint`。这保证 retarget overlay 在渲染双臂前
能够读到最新的双手去除背景。
如果某个 task 的 inpaint 阶段失败，总脚本会跳过该 task 的 retarget 和导出，避免把
旧的单手去除背景误发布到 LeRobot。

DA3 深度仍应从原始含手视频计算，不应从 inpaint 后的视频重新估计。

## 重跑

先用单个 demo 验证：

```bash
cd /home/a26160/SRC/phantom

bash b/run_hand_inpaint.sh \
  --task stack_unstack_bowls \
  --demo-num 2 \
  --data-root /tmp/zwy/DATA/test_phantom \
  --processed-root /tmp/zwy/DATA/test_phantom_processed

bash b/run_contact_retarget.sh \
  --task stack_unstack_bowls \
  --demo-num 2 \
  --data-root /tmp/zwy/DATA/test_phantom \
  --processed-root /tmp/zwy/DATA/test_phantom_processed \
  --step intent,stageb,retarget_inpaint
```

全量重跑 bowls：

```bash
cd /home/a26160/SRC/phantom
bash b/run_human2robot_all.sh \
  --task stack_unstack_bowls \
  --stage inpaint,retarget,lerobot
```

不要给首次修复重跑加 `--skip`，否则旧的单手 `masks_arm.npy` 和
`video_human_inpaint.mkv` 可能被保留。

如果双手 mask 已重新生成、只需要重算动作和 overlay：

```bash
bash b/run_human2robot_all.sh \
  --task stack_unstack_bowls \
  --stage retarget,lerobot
```

## 验收

每个 demo 检查：

1. `segmentation_processor/video_masks_arm.mkv`：左右手臂均被覆盖。
2. `inpaint_processor/video_human_inpaint.mkv`：两侧人手/人臂均已去除。
3. `stageb_processor/q_trajectory_left.npz`：左手运动时 `parked=False`。
4. `stageb_processor/q_trajectory_right.npz`：右手运动时 `parked=False`。
5. `retarget_processor/video_overlay.mkv`：左右机械臂与对应人手方向一致，
   且机械臂不应占满视野。demo 31 挪肩后的覆盖率为：均值 25.6%、中位数 15.2%、
   p90 48.8%、最大 64.4%，仅 3 帧超过 60%。抓取时刻腕部占掉约半屏属于预期，
   见上文"腕部遮挡的下界"。
6. LeRobot 的 `joint_pos_left/right` 均来自对应侧轨迹，只有真正闲置侧允许零 range。

如果某一侧仍被 park，查看 Stage B 日志中的
`hand_span`、`hand_frames`、`grasp_valid` 和 `park`。如果机械臂参与求解但最终未导出，
查看 `retarget_processor/quality_report.npz`；此时是轨迹质量门失败，不再是错误的
“无 grasp 即 park”。

