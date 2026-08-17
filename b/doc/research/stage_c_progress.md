# Stage C 进展与验证 — Contact-Grounded Retargeting

> 配套设计文档：`contact_grounded_retargeting.md`（§3 Stage C、§5 质量剪枝）
> 前置：`stage_a_progress.md`（产出 `intent.npz`）、`stage_b_progress.md`（产出 `q_trajectory.npz`）
> Status: **已实现并端到端验证**（Zed RGB-D pick_and_place，184 帧，单臂，EGL 离屏渲染）。

---

## 0. TL;DR

Stage C = **渲染 `FK(q_t)` + 翻转标签 + 轨迹级质量剪枝**，是 contact-grounded 三段管线的收尾，取代旧 `robot_inpaint` 的「逐帧 IK/OSC 追人手目标 + 超阈丢帧」。

三件事：
1. **精确注入渲染**：把 Stage B 的 `q_{1:T}` 直接 `sim.data.qpos=q_t; sim.forward()` 注入**与 Stage B FK 同一套单臂 `Phantom` 环境**，用 `_get_observations` 取 RGB/分割/深度离屏渲染——渲染像素就是 `FK(q_t)`，**obs-action 一致性由构造保证**（无 OSC 残差、无控制器逼近误差）。
2. **标签翻转**：`joint_pos = q_t`、任务空间 `action = FK(q_t)`（robot 系，承自 `q_trajectory.npz`）。标签描述**渲染出来的机器人**，而非（可能不可达的）人手目标。
3. **轨迹级质量剪枝**：`TRACKING_ERROR_THRESHOLD` 逐帧丢帧退休（Stage B 已保证无 IK 失败、无丢帧）；改为按关键帧残差/关节速度/jerk/限位**整条 demo 保留或丢弃**，并存质量报告。

---

## 1. 实现

### 1.1 新增/改动文件

| 文件 | 改动 | 说明 |
|---|---|---|
| `phantom/processors/retarget_inpaint_processor.py` | **新增** | `RetargetInpaintProcessor(RobotInpaintProcessor)`——复用相机/叠图/存视频，重写取数据、注入渲染、标签、剪枝 |
| `phantom/process_data.py` | 改 | 注册 `retarget_inpaint` mode（enum + 两个顺序表，置于 `hand_inpaint` 后、`robot_inpaint` 前 + registry） |
| `phantom/processors/paths.py` | 改 | 新增 `retarget_processor/`：`video_overlay.mkv`、`training_data.npz`、`quality_report.npz`、`retarget_diagnostic.png` |
| `b/configs/pickplace_intent.yaml` | 改 | 新增 `retarget_*` 配置（质量门阈 + 深度遮挡开关） |
| `phantom/processors/stageb_processor.py` | 改 | **环境一致性修复**：Stage B FK 改到单臂 `Phantom` 环境（详见 `stage_b_progress.md` §1.3） |

> 命名：按用户要求**不叫** `StageCProcessor`；用 `RetargetInpaintProcessor`（mode `retarget_inpaint`），与旧 `RobotInpaintProcessor`/`robot_inpaint` 平行、便于对照与回滚。

### 1.2 复用 vs 重写（子类化 `RobotInpaintProcessor`）
- **复用**：`_initialize_robot`（单臂 `TwinRobot` + 标定相机）、`_get_mujoco_camera_params`、`_process_robot_overlay(_with_depth)`（分割掩码合成）、`_compute_gripper_actions`、`_save_video`、相机内外参换算。
- **重写**：
  - `process_one_demo`：读 `q_trajectory.npz`（+ `intent.npz` 取 `gripper_open_width`）→ 质量门 → 载背景 → 逐帧渲染 → 存。
  - `_render_joint_positions(q, width, open_width)`：**注入 qpos 渲染**（下详）。
  - 标签构造：`joint_pos_{hand}=q_t`、`action_pos/orixyzw_{hand}=FK(q_t)` robot 系；另一手全零。
  - `_quality_gate` / `_save_quality_report`：轨迹级剪枝。

### 1.3 注入渲染（核心）
```python
env = self.twin_robot.env.env      # robosuite Phantom 单臂 env（含标定相机）
sim.data.qpos[robot.joint_indexes] = q     # 直接注入臂关节
self._set_gripper_qpos(sim, width, open_width)   # 夹爪开合由 intent 宽度映射到指关节 qpos
sim.forward()                                # 只做运动学前向，不步进控制器
obs = env._get_observations(force_update=True)   # 离屏渲染当前 qpos
rgb = obs["frontview_image"]                 # (H,W,3) uint8（原始 robosuite HWC 布局）
seg = obs["frontview_segmentation_instance"] # 实例分割
```
- **与 `move_to_target_state` 的区别**：旧路径发 OSC 目标、步进 `n_steps` 物理直到收敛，鲜有精确到 `q`；这里**不经控制器**，渲染的就是 `q_t` 本身。
- **布局差异**：`_get_observations` 返回**原始 HWC uint8**（robomimic `env.step` 是 CHW float），故不复用 `TwinRobot.get_image` 的 `transpose`，自建 obs→results。
- **分割掩码**：单臂场景里前景实例即机器人 → `robot_mask = (seg>0)`，`gripper_mask=0`（合成时二者取并集，等价于整臂剪影）；`square` 时按管线约定中心裁剪。
- **夹爪**：`gripper_open_width` 与逐帧 `gripper_width` 线性映射为 `closure∈[0,1]`，写到夹爪指关节 `qpos`（按 `jnt_range` 插值）。

### 1.4 帧对齐与背景
- Stage A `intent`/Stage B `q` 均**逐帧对应 `original_images/`（全部帧，square 裁剪）**，无 `union_indices` 重映射，`T` = 视频总帧数。
- 背景优先用**去手的 `video_human_inpaint`**（若存在且帧数 == `T`），否则回退**原始帧**（人手仍可见，仅用于功能/姿态目视）。当前 demo 未跑 `hand_inpaint`，故用原始帧；接线顺序已把 `retarget_inpaint` 放在 `hand_inpaint` 之后，`all` 跑时会自动用去手背景。

### 1.5 轨迹级质量门（替代逐帧丢帧）
从 `q_trajectory.npz` 聚合判定**整条 demo**接受/剪除：

| 判据 | 默认阈 | 含义 |
|---|---|---|
| `key_pos` | 0.03 m | grasp/release 帧位置误差 max（够不到物体则剪） |
| `vmax` | 0.5 rad/帧 | 逐帧 `|Δq|` max（异常跳变） |
| `jerk` | 0.05 | 关节 jerk RMS（不平滑） |
| `viol` | 0 | 关节限位违反计数 |

- 全部通过 → 接受整条并存训练数据；任一超标 → **剪除整条**（不静默丢帧），原因写入 `quality_report.npz`。
- 另存逐帧 `frame_ok = pos_err ≤ retarget_frame_pos_thresh`（0.05m）供下游可选加权/掩码，但**不**据此丢帧。

### 1.6 配置项（`retarget_*`）
```yaml
retarget_use_depth: false            # 深度遮挡（需人手深度对齐到渲染分辨率；默认关，走剪影合成）
retarget_key_pos_thresh: 0.03        # 接受 demo 的 grasp/release 位置误差上限 (m)
retarget_vel_thresh: 0.5             # 接受 demo 的逐帧 |dq| 上限 (rad/帧)
retarget_jerk_thresh: 0.05           # 接受 demo 的 jerk RMS 上限
retarget_frame_pos_thresh: 0.05      # 逐帧质量掩码阈 (m)
```

### 1.7 输出（`{demo}/retarget_processor/`）

| 文件 | 内容 |
|---|---|
| `video_overlay.mkv` | 机器人 `FK(q_t)` 叠到背景帧（ffv1 无损，15fps） |
| `training_data.npz` | `TrainingDataSequence`：`joint_pos_*`=`q_t`、`action_pos/orixyzw_*`=`FK(q_t)` robot 系、`gripper_*`、`valid` |
| `quality_report.npz` | `accept`/`reasons`/`key_pos`/`vmax`/`jerk`/`viol`/`frame_ok`/`pos_err` |
| `retarget_diagnostic.png` | 关键帧「原始 | 叠图」并排（f0/grasp/中点/release/末帧）目视校验 |

---

## 2. 如何运行
```bash
cd phantom
# 需先 export EGL 后端（见 b/run_process.sh L66-69）：
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"
export __EGL_VENDOR_LIBRARY_DIRS="${HOME}/.local/share/glvnd/egl_vendor.d"
export MUJOCO_GL=egl; export PYOPENGL_PLATFORM=egl
# 需先有 Stage B 产出的 q_trajectory.npz（mode=stageb）
python process_data.py --config-path=../b/configs \
    --config-name=pickplace_intent mode=retarget_inpaint demo_num=0
```

---

## 3. 验证结果（`data/raw/pick_and_place`，184 帧，左臂）

### 3.1 端到端渲染
- **EGL 离屏渲染跑通**：184/184 帧渲染成功（~2.4 帧/秒），产出 overlay 视频 + 训练数据 + 质量报告 + 诊断图。

### 3.2 obs-action 一致性（数值）
| 校验 | 结果 |
|---|---|
| `joint_pos_left == q`（Stage B） | ✅ 完全相等 |
| `action_pos_left == ee_pos_robot`（= `FK(q)` robot 系） | ✅ 完全相等 |
| `action_orixyzw_left` 单位四元数 | ✅ 范数 1.0 |
| 另一手（right）全零 | ✅ |

→ 标签**由构造**与渲染像素同源（都来自同一 `q_t` 与同一环境 FK），彻底消除旧 pipeline「渲染用收敛后姿态、标签用人手目标」的漂移。

### 3.3 目视校验（`retarget_diagnostic.png`）
- **grasp 帧**：机械爪精确落在物体（玻璃容器内蓝色物）上、手指环绕——与 Stage B grasp 0.04cm/0.2° 数值吻合，**夹爪约定（`Rz90`）正确**。
- free/transport/release：机械臂随接触点平滑移动，位置合理。
- 已知视觉现象（预期）：本 demo 未跑 `hand_inpaint`，背景为原始帧，人臂仍可见；跑过去手步骤后自动切干净背景。

### 3.4 质量门
`accept=True`，`reasons=[]`：`key_pos=1.07cm`(<3)、`vmax=0.067`(<0.5)、`jerk=0.0036`(<0.05)、`viol=0`，`frame_ok=182/184`（2 个 free 帧略超 5cm，不影响接受）。

### 3.5 EgoDex `basic_pick_place` demo 0（头戴 + `T_place`）

`egodex_panda_intent.yaml`、`mode=retarget_inpaint`。质量门 `accept=True`（`key_pos=1.36cm`、`vmax=0.11`、`jerk=0.005`、`viol=0`）；`frame_ok=124/126`（t=89/90 是放完切回手跟随的 free 尖峰）。标签 `joint_pos_right==q`、`action_pos_right==FK(q)`。背景用已有 `video_human_inpaint`（去手）。渲染相机按 Stage A 的逐帧 `T_cam2robot_seq`（`T_place @ T_c2w`），不是肩部 JSON。诊断图 grasp 帧夹爪贴订书钉、transport 提起、release 放进盒盖。产出：`retarget_processor/video_overlay.mkv`。

---

## 4. 已知遗留 / 后续
1. **去手背景**：当前 demo 未跑 `hand_inpaint`，回退原始帧（人手可见）；正式产数据应先跑 `hand_inpaint` 让背景干净。代码已自动检测 `video_human_inpaint` 帧数对齐。
2. **深度遮挡默认关**：`retarget_use_depth=false`（剪影合成）。开启需把人手侧 `depth.npy` 对齐到渲染分辨率，再走 `_process_robot_overlay_with_depth`（物体遮挡夹爪时更真实，尤其 grasp）。
3. **夹爪开合朝向**：`_set_gripper_qpos` 按 `jnt_range` 线性插值指关节，假定「qpos 增大=闭合」；目视合理，若个别夹爪型号方向相反需翻转 `closure`。
4. **双臂**：随 Stage A/B 单臂现状，Stage C 亦单臂；双臂需并行注入两臂 qpos + 合成两套掩码。
5. **姿态跟物体旋转**：承 `stage_a_progress.md` §5.2 门控待办（平移主导任务下影响极小）。

---

## 5. 里程碑对齐（设计文档 §7）
- **M1（RGB-D 主线打通）**：Stage A ✅ + Stage B ✅ + **Stage C ✅** —— contact-grounded 三段管线 `intent → stageb → retarget_inpaint` 端到端跑通并验证（可达、无丢帧、平滑、关键帧保真、obs-action 一致、轨迹级剪枝）。
- **下一步建议**：先跑 `hand_inpaint` 得干净背景重出一版；再在更多/更难的 demo（明显重定向、物体旋转）上压测质量门与 §4 遗留项。
