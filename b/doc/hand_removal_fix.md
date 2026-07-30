# Hand Removal Quality Fix

EgoDex 1080p 视频中人手去除不干净，原因涉及 arm_segmentation 和 hand_inpaint 两个阶段。以下是所有改动的说明。

## 问题分析

pipeline 流程: bbox → hand2d → arm_segmentation → hand_segmentation → hand3d → action → smoothing → **hand_inpaint** → robot_inpaint

hand_inpaint 的输入是 arm_segmentation 产出的 `masks_arm.npy`。最终 inpaint 质量取决于：
1. mask 是否正确覆盖了人手/手臂（segmentation 质量）
2. inpaint 模型是否能在对应分辨率下正常工作（inpaint 质量）

实际发现两层问题：

| 问题 | 根因 | 现象 |
|------|------|------|
| inpaint 网格伪影 | E2FGVI-HQ 设计用于 ~240p，在 1080p 下产生棋盘格 | 输出视频布满方块状纹理 |
| mask 覆盖错误对象 | SAM2 分割了背景布料而非人手 | 手没被去掉，反而背景被涂黑 |

SAM2 分割错误的原因链：
- Detectron2 给出的是 "person" bbox（几乎覆盖整个画面），而非 hand bbox
- 初始化帧的 hand bbox 也可能过大（如 `[805, 0, 1920, 1080]`，占画面 58%）
- HaMeR 关键点在很多帧全为 (0,0)
- 只传了 1 个点（手腕）给 SAM2，在手持布料的第一人称视角下，SAM2 优先分割了更大的布料

## 改动一览

### 1. `phantom/processors/segmentation_processor.py`

#### 1.1 新增 `_select_init_frame` 方法

选择 SAM2 初始化帧时，过滤掉：
- 手未被检测到的帧
- bbox 面积超过画面 25% 的帧（说明检测失败）
- 关键点全为 (0,0) 的帧

在剩余候选帧中选 `bbox_min_dist_to_edge` 最大的（手离画面边缘最远）。如果没有帧通过过滤，退回到原始的 argmax。

```python
@staticmethod
def _select_init_frame(bboxes, bbox_min_dist, hand_detected, kpts_2d,
                       frame_shape, max_bbox_ratio=0.25):
    img_area = float(frame_shape[0] * frame_shape[1])
    candidates = []
    for i in range(len(bboxes)):
        if not hand_detected[i]:
            continue
        x1, y1, x2, y2 = bboxes[i]
        if float(max(x2-x1,0)) * float(max(y2-y1,0)) / img_area > max_bbox_ratio:
            continue
        if np.allclose(kpts_2d[i], 0):
            continue
        candidates.append(i)
    if candidates:
        return int(max(candidates, key=lambda i: bbox_min_dist[i]))
    return int(np.argmax(bbox_min_dist))
```

#### 1.2 使用 hand bbox 替代 Detectron2 bbox

原代码优先用 Detectron2 的 "person" bbox 初始化 SAM2。在第一人称视角下，person bbox 通常覆盖大半个画面，导致 SAM2 锁定背景布料。

改为始终使用 HaMeR 检测到的 hand bbox：

```python
# 改前
bbox_dets = det_bboxes[max_dist_idx]
if bbox_dets.sum() == 0:
    bbox_dets = bboxes[max_dist_idx]

# 改后
bbox_init = bboxes[max_dist_idx]
```

#### 1.3 传递全部 21 个关键点给 SAM2

原代码因 `zip(points, indices)` 的行为只传了 1 个点（手腕）。当 bbox 内同时包含手和布料时，单个点不足以让 SAM2 区分。

改为一次传入全部 21 个手部关键点作为正样本提示：

```python
# 改前: shape (21, 1, 2) → zip 只取第一个 → 1 个点
points = np.expand_dims(kpts_2d[max_dist_idx], axis=1)

# 改后: shape (1, 21, 2) → zip 取到 (21, 2) → 21 个点
points = kpts_2d[max_dist_idx].reshape(1, -1, 2)
```

#### 1.4 新增 `_filter_mask_by_keypoints` 方法

对 SAM2 输出的 mask 做后处理：用连通域分析找到各个独立区域，只保留包含手部关键点的区域。如果没有关键点命中任何区域，保留所有（fallback）。

### 2. `phantom/processors/handinpaint_processor.py`

#### 2.1 可配置 mask 膨胀参数

原来硬编码 `cv2.MORPH_ELLIPSE, size=3, iterations=4`。改为从 Hydra config 读取：

```python
self.mask_dilate_type = getattr(cv2, args.mask_dilate_kernel, cv2.MORPH_ELLIPSE)
self.mask_dilate_size = getattr(args, 'mask_dilate_size', 3)
self.mask_dilate_iterations = getattr(args, 'mask_dilate_iterations', 4)
```

#### 2.2 新增 `inpaint_resolution` 缩放

E2FGVI-HQ 在高分辨率下产生网格伪影。新增 `inpaint_resolution` 参数：
- 加载帧时按 `max(w,h)` 等比缩放到目标分辨率
- inpaint 完成后用 `cv2.INTER_LANCZOS4` 放大回原始分辨率

```python
self.inpaint_resolution = getattr(args, 'inpaint_resolution', 0)  # 0 = 不缩放
```

### 3. `b/configs/egodex.yaml`

新增配置项：

```yaml
inpaint_resolution: 480        # E2FGVI 工作分辨率（0=原始分辨率）
mask_dilate_kernel: "MORPH_ELLIPSE"
mask_dilate_size: 11           # 膨胀核大小
mask_dilate_iterations: 3      # 膨胀迭代次数
```

### 4. `b/run_process.sh`

`default_workers()` 中为 `hand_inpaint` 添加专用 worker 数：

```bash
hand_inpaint) echo "$((NUM_GPUS * 1))" ;;
```

原来 `hand_inpaint` 走 default 分支得到 `NUM_GPUS * 8`，在 1080p 下 OOM。

### 5. `b/run_reinpaint.sh`

重新跑 segmentation + inpaint 的便捷脚本。主要改动：
- 添加 `arm_segmentation` 步骤（先 segmentation 再 inpaint）
- 清理时同时删除 `segmentation_processor` 和 `inpaint_processor` 输出
- 添加 conda 环境激活
- 直接调用 `process_data.py` 按 episode 执行，支持 `--episodes`, `--hand-only`, `--dry-run`

用法：
```bash
bash b/run_reinpaint.sh --episodes 0 --hand-only    # 只跑 episode 0 的 segmentation + inpaint
bash b/run_reinpaint.sh --episodes 0-5              # 跑 0-5 含 robot_inpaint
bash b/run_reinpaint.sh --dry-run                    # 只打印不执行
```

## 效果对比

| 指标 | 修改前 | 修改后 |
|------|--------|--------|
| mask 目标 | 背景布料 (500K+ px) | 人手/手臂 (15K-90K px) |
| mask 覆盖帧数 | 471/471（全错） | 412/471（正确目标） |
| inpaint 质量 | 棋盘格伪影 | 干净，无伪影 |
| 人手去除 | 手完全保留 | 大部分帧手被正确去除 |

---

## 改动二：用 Detectron2 pred_masks 替代 SAM2 做 arm segmentation

### 问题

上面的改动修复了 SAM2 分割错误对象的问题，但 SAM2 用 HaMeR 的 21 个手部关键点初始化，只能分割到**手掌皮肤区域**，无法分割穿着衣袖的手臂。在第一人称视角下，inpaint 后手被去掉了，但灰色衣袖/手臂仍然可见。

另外 SAM2 方案存在严重的性能问题：
- 每个 episode 需要提取全部视频帧为 JPEG（`convert_video_to_images`），写到 `/mnt/r/`
- SAM2 `init_state` 再把所有 JPEG 读回来
- 双手 × 双向 = 4 轮 SAM2 propagation
- 首个 episode 约 12 分钟（6 worker 并发时）

### 根因

Detectron2 (`cascade_mask_rcnn_vitdet_h`) 本身就是 Mask R-CNN，输出中包含 `pred_masks`（实例分割 mask），但原代码只提取了 `pred_bboxes`，完全没有使用 mask。在第一人称视角下，Detectron2 检测到的 "person" 实例就是手臂+手，其 mask 正好覆盖整条手臂（含衣袖）。

### 改动

#### 1. `phantom/detectors/detector_detectron2.py`

新增 `get_person_masks` 方法，提取 `pred_masks`：

```python
def get_person_masks(self, img):
    det_out = self.detectron2(img)
    det_instances = det_out["instances"]
    valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
    pred_masks = det_instances.pred_masks[valid_idx].cpu().numpy()  # (K, H, W) bool
    pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
    pred_scores = det_instances.scores[valid_idx].cpu().numpy()
    return pred_masks, pred_bboxes, pred_scores
```

#### 2. `phantom/processors/segmentation_processor.py`

重写 `ArmSegmentationProcessor.process_one_demo`，用 Detectron2 逐帧分割替代 SAM2：

**新流程：**
```
load video → load HaMeR keypoints →
for each frame:
    Detectron2 → person masks →
    保留包含手部关键点的 mask →
    OR 合并
→ save masks_arm.npy
```

**关键变化：**
- 不再调用 `_setup_original_images()`（不需要提取 JPEG）
- 不再调用 `get_detectron_bboxes()`（不需要 bbox 匹配）
- 不再使用 SAM2（不需要 `_run_sam_segmentation`、`_filter_mask_by_keypoints`）
- 新增 `_get_detectron_arm_masks()`：逐帧运行 Detectron2，用 HaMeR 关键点匹配 person mask
- 新增 `_collect_hand_keypoints()`：收集每帧的有效手部关键点
- 新增 `_mask_contains_any_keypoint()`：检查 mask 是否包含关键点

**匹配逻辑：**
- 有手部关键点的帧：只保留包含关键点的 person mask
- 无关键点的帧（手未检测到）：保留所有 person mask（fallback，手臂大概率仍可见）

**删除的方法：** `_process_single_arm`, `_process_bimanual`, `_process_hand_data`, `_run_sam_segmentation`, `_select_init_frame`, `_filter_mask_by_keypoints`, `get_detectron_bboxes` 及其所有辅助方法。

**未修改：** `_save_results`, `_create_visualization`, `_validate_output_consistency`，输出格式 `masks_arm.npy` 不变，下游 `handinpaint_processor.py` 和 `robotinpaint_processor.py` 无需改动。

### 性能对比

| | SAM2 方案 | Detectron2 方案 |
|---|---|---|
| 首个 episode | ~12 min（含模型加载+I/O竞争） | ~2.5 min |
| 后续 episode | ~1.5 min | ~2.5 min |
| JPEG I/O | 539帧×2方向 写+读×4轮 | 无 |
| GPU 操作 | 4轮 SAM2 propagation | 逐帧 Detectron2（~0.28s/帧） |
| mask 覆盖 | 仅手掌皮肤 | 手臂+手（含衣袖） |

---

## 改动三：提升 Detectron2 arm mask 覆盖率

### 问题

改动二后实测 episode 0（539 帧），仍有 ~90 帧没有 mask（手臂完全可见），分析发现：

1. **60 帧**：Detectron2 检测到 person 但 score 在 0.25-0.49，被 `> 0.5` 阈值过滤
2. **30 帧**：Detectron2 完全没有 person 检测（model 内部 `test_score_thresh=0.25` 已丢弃）
3. **部分帧**：只检测到一条手臂，另一条漏检（手抓物体时不被识别为 person）

### 改动

#### 1. 降低 Detectron2 模型阈值

`phantom/detectors/detector_detectron2.py`：`test_score_thresh` 从 0.25 降到 0.05，让模型返回更多低置信度检测。

`get_person_masks` 增加 `score_thresh` 参数做二次过滤：
- 有关键点的帧：`score_thresh=0.1`（关键点验证兜底，可以放宽）
- 无关键点的帧：`score_thresh=0.3`（没有验证，稍严格防误检）

#### 2. 关键点匹配 fallback

当有关键点但没有 mask 匹配上时（mask 覆盖了另一只手），fallback 到保留所有 person mask，而不是丢弃。

#### 3. 时序填充 `_temporal_fill`

对 Detectron2 完全检测不到的帧，从最近的有 mask 帧复制 mask。相邻帧手臂位置变化小，填充效果可接受。

#### 4. HaMeR 关键点凸包补充 mask

对 Detectron2 漏检的手（有关键点但没有对应 mask），用 21 个关键点的凸包 + 膨胀生成补充 mask。覆盖 Detectron2 无法检测到的手部区域（如手抓物体时）。

### 效果

| | 改动二 | 改动三 |
|---|---|---|
| 有 mask 帧数 | 449/539 (83%) | 539/539 (100%) |
| 平均 mask 覆盖率 | 12.16% | 13.67% |
| 覆盖率提升 >1% 的帧 | — | 121 帧 |

---

## 改动四：Detectron2 + SAM2 混合方案

### 问题

改动三（纯 Detectron2 逐帧检测）的两个局限：

1. **逐帧独立检测无时序信息** — 部分帧漏检需要 temporal fill（简单复制邻近帧 mask），不准确
2. **膨胀过大覆盖手持物体** — 为弥补 mask 边界不够，`mask_dilate_iterations` 需要设到 8-12，导致手上拿的东西（三明治等）也被 inpaint 涂掉

原版代码用 SAM2 时序传播能做到 100% 帧覆盖且 mask 精确，但用手部关键点初始化 → 只分割手掌皮肤，衣袖不覆盖。

### 方案

结合两者优势：用 Detectron2 的 `pred_mask`（覆盖整条手臂含衣袖）作为 SAM2 的 mask prompt 初始化，让 SAM2 把这个 arm-level mask 时序传播到全部帧。

### 改动

#### 1. `phantom/processors/segmentation_processor.py`

重写 `ArmSegmentationProcessor.process_one_demo`，新流程：

```
load video → extract JPEG frames (SAM2 需要) →
for each hand (left, right):
    1. 在 top-K 候选帧（按 bbox_min_dist 排序）上跑 Detectron2
    2. 找到最佳初始帧（score 最高 + mask 包含该手关键点）
    3. 用该帧的 Detectron2 pred_mask 作为 SAM2 mask prompt
    4. SAM2 正向传播 + 反向传播
    5. 合并双向 masks
→ 左右手 mask OR 合并 → masks_arm.npy
```

新增方法：
- `_get_sam_arm_masks()` — 主方法，按手分别处理
- `_find_best_init_frame()` — 在候选帧上跑 Detectron2 选最佳帧
- `_run_sam_from_mask()` — 调用 `segment_video_from_mask` 的封装

删除不再需要的方法：`_get_detectron_arm_masks`, `_temporal_fill`, `_add_convex_hull_mask`, `_add_convex_hull_masks`, `_collect_per_hand_keypoints`

#### 2. `phantom/detectors/detector_sam2.py`

`segment_video_from_mask` 添加 `torch.cuda.empty_cache()` 防止 GPU OOM。

#### 3. `b/configs/egodex.yaml`

`mask_dilate_iterations` 恢复为 3（SAM2 mask 精确，不需要大膨胀）。

### 预期效果

| | Detectron2-only（改动三） | Detectron2 + SAM2（改动四） |
|---|---|---|
| 帧覆盖率 | ~90% + temporal fill | 100%（SAM2 传播） |
| mask 覆盖 | 手臂+衣袖 | 手臂+衣袖 |
| 时序一致性 | 无 | 有 |
| 膨胀需求 | iterations=8-12 | iterations=3 |
| 手持物体 | 被过度覆盖 | 精确保留 |
| 速度 | ~2.5 min/ep | ~5-8 min/ep |

### 实测结果

Episode 0（539 帧）验证通过：SAM2 4 轮传播（左手 forward/reverse + 右手 forward/reverse），mask 精确覆盖手臂+衣袖，手持物体不被过度覆盖。

#### 2 轮传播尝试（已放弃）

尝试将左右手 init mask 合并，只跑 2 轮 SAM2 传播（forward + reverse）。实现：`detector_sam2.py` 新增 `segment_video_from_masks()` 方法，支持在同一次 propagation 前用 `add_new_mask` 添加多个 conditioning mask。

结果：效果不如 4 轮。原因是左右手的最佳初始帧通常不同，合在一起传播时 SAM2 对两条手臂的跟踪质量下降。已恢复为每只手独立跑 forward + reverse（4 轮）。

`segment_video_from_masks()` 保留在代码中备用。

#### 新增/修改的方法总结

**`phantom/detectors/detector_sam2.py`：**
- `segment_video_from_mask()` — 添加 `torch.cuda.empty_cache()`，内部委托给 `segment_video_from_masks`
- `segment_video_from_masks()` — 新增，支持多个 `(mask, frame_idx)` 对初始化

**`phantom/processors/segmentation_processor.py`：**
- `_get_sam_arm_masks()` — 新增，替代 `_get_detectron_arm_masks`
- `_find_best_init_frame()` — 新增，在 top-K 候选帧上跑 Detectron2 选最佳初始帧
- `_run_sam_from_mask()` — 新增，封装 `segment_video_from_mask` 调用
- 删除：`_get_detectron_arm_masks`, `_temporal_fill`, `_add_convex_hull_mask`, `_add_convex_hull_masks`, `_collect_per_hand_keypoints`
- 保留：`_collect_hand_keypoints`, `_mask_contains_any_keypoint`（初始帧选择用）
