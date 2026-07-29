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
