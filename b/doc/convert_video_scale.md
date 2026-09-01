# Convert 阶段按比例缩放视频 + 内参

> Status: **已接入** `b/run_convert.sh` / `b/convert_egodex.py`。
> 缩放视频时必须同步缩放 \(K\)（\(f_x,f_y,c_x,c_y\)），否则 3D 反投影、接触、MuJoCo 叠臂都会错位。FOV 不变。

---

## 0. 为什么在 convert 做

EgoDex 源视频是 1080p（1920×1080）。整条流水线（DA3 深度、bbox/hand2d/SAM2、ProPainter、intent、retarget 渲染）都读 convert 写出的 `video_L.mp4`。在源头降分辨率：

- 后续所有阶段都跑低分辨率，加速最彻底；
- I/O 和磁盘也一起变小。

只缩视频、不改 \(K\) 会错。内参 JSON 默认是 1080p：

```
fx = fy ≈ 736.63
cx = 960, cy = 540
```

若画面变成 1280×720（\(s = 2/3\)），像素坐标按 \(s\) 缩小，\(K\) 必须同样乘 \(s\)。YAML 里的 `input_resolution` / `output_resolution` 也要改成新的高度。

Hydra 配置仍默认 1080。缩放后 convert 会写 `convert_meta.json`；`run_hand_inpaint.sh` 和 `run_contact_retarget.sh`（因此 `run_human2robot_all.sh`）发现该文件会自动带上分辨率和 `camera_intrinsics` 覆盖。

---

## 1. 数据流

```
EgoDex  0.mp4 + 0.hdf5 (camera/intrinsic 仍是 native)
        │  --height 720 或 --scale 0.5
        ▼
  ffmpeg 缩放 → video_L.mp4          （实文件，不再软链原片）
  K' = diag(sx,sy,1) @ K             hand_det.pkl 用 K' 投影
  camera_intrinsics_egodex_{H}p.json  fx,fy,cx,cy *= s；FOV 不变
  convert_meta.json                  给下游 Hydra
        │
        ▼
  bbox / DA3 / intent / inpaint / retarget
  全部按 {H}p 的画面和 K' 工作
```

- `--scale` 与 `--height` **互斥**。
- 不传缩放参数：行为与以前相同，`video_L.mp4` 软链原片，不写 `convert_meta.json`。
- 宽高会取偶数（x264 / yuv420p）。
- `hand_det.pkl` 的 bbox 是归一化坐标；只要 \(K\) 和 `img_w/img_h` 一起缩放，bbox 与 native 一致。
- HDF5 里的 `camera/intrinsic` **不改**（源数据只读）。`export_egodex_hand_gt.py` 会按 `video_L.mp4` 实际宽高把 \(K\) 缩到像素坐标再写 `kpts_2d`。

---

## 2. 改了哪些文件

| 文件 | 改动 |
|---|---|
| `b/convert_egodex.py` | `--scale` / `--height`；ffmpeg 编码；缩放 \(K\) 写 hand_det；写出 `{H}p` 内参 JSON 和 `convert_meta.json` |
| `b/run_convert.sh` | 转发 `--scale` / `--height` |
| `b/export_egodex_hand_gt.py` | 按 `video_L.mp4` 尺寸缩放 HDF5 \(K\)，再投影 2D |
| `b/run_hand_inpaint.sh` | 读 `convert_meta.json`，Hydra 覆盖 `input_resolution` / `camera_intrinsics` |
| `b/run_contact_retarget.sh` | 同上（`run_human2robot_all.sh` 走这条） |
| `phantom/processors/robotinpaint_processor.py` | `_get_image_dimensions`：任意高度按 16:9 算宽，不再写死 1080 |

未改：`camera_intrinsics_egodex.json`（1080p 原件保留）。缩放结果写到旁路文件 `camera_intrinsics_egodex_{H}p.json`。

---

## 3. 用法

从 `phantom/`：

```bash
# 720p（1280×720）+ 缩放 K；已有 demo 必须 --overwrite
bash b/run_convert.sh --height 720 --overwrite

# 或统一比例（1080p × 0.5 → 960×540）
bash b/run_convert.sh --scale 0.5 --overwrite

# 也可写在脚本默认里：HEIGHT="720" 或 SCALE="0.5"（不要两个都填）
```

直接调 Python：

```bash
python b/convert_egodex.py --task stack_unstack_tupperware \
  --egodex-root /home/a26160/DATA/Ego-Dex/test \
  --output-root /home/a26160/DATA/tmp/test_phantom \
  --height 720 --overwrite
```

依赖：本机 `ffmpeg`（缩放时）、`ffprobe` 或 OpenCV（探测源分辨率）。

手动跑 `process_data.py` 时自己带覆盖，例如 720p：

```bash
python process_data.py --config-path=../b/configs --config-name=egodex_panda_intent \
  input_resolution=720 output_resolution=720 \
  camera_intrinsics=camera/camera_intrinsics_egodex_720p.json
```

---

## 4. 写出的文件

以 `--height 720`、\(s = 720/1080 = 2/3\) 为例：

**内参（Phantom JSON）** — `phantom/phantom/camera/camera_intrinsics_egodex_720p.json`

- \(f_x' = f_y' \approx 491.09\)
- \(c_x' = 640\)，\(c_y' = 360\)
- `v_fov` / `h_fov` 不改

任务目录：

```
{DATA_ROOT}/egodex_<task>/
  convert_meta.json
  camera_intrinsics.json          # 同上内容的副本
  0/video_L.mp4                   # 1280×720 实文件
  0/hand_det.pkl
```

`convert_meta.json` 示例：

```json
{
  "scale_x": 0.666...,
  "scale_y": 0.666...,
  "src_wh": [1920, 1080],
  "dst_wh": [1280, 720],
  "input_resolution": 720,
  "output_resolution": 720,
  "camera_intrinsics": "camera/camera_intrinsics_egodex_720p.json"
}
```

`input_resolution` 在 Phantom 里表示**高度**。720 → 宽 `720 * 16 // 9 = 1280`。

---

## 5. 下游怎么接到 K'

`run_hand_inpaint.sh` / `run_contact_retarget.sh` 在 `DATA_ROOT/egodex_<task>/convert_meta.json` 存在时追加：

```
input_resolution={H}
output_resolution={H}
camera_intrinsics=camera/camera_intrinsics_egodex_{H}p.json
```

工作目录是 `phantom/phantom/`，相对路径 `camera/...` 能对上。

没有 `convert_meta.json`（native convert）则继续用 yaml 里的 1080 + `camera_intrinsics_egodex.json`。

Intent 叠臂读的是去手后的 `video_human_inpaint.mkv`（若已跑过 `run_hand_inpaint.sh`），不是 EgoDex 原片。convert 缩的是 `video_L.mp4`；inpaint / overlay 会跟着变成同一分辨率。

---

## 6. 注意

- 已 convert 过的 demo：必须 `--overwrite`，否则跳过，分辨率不会变。
- processed 里若还是旧的 1080 软链/拷贝，需要重新 link 或重跑 inpaint / retarget。
- 不要只改 yaml 的 `input_resolution` 却继续喂 1080 视频，或只缩视频却仍用 1080 的 JSON。
- 过低（例如 &lt; 360p）时小物体 OWLv2/SAM2 容易漏检。720p 是速度和精度的折中。
- `--scale` 和 `--height` 不能同时传；`run_convert.sh` 里 `SCALE` 与 `HEIGHT` 也不要同时设非空。
