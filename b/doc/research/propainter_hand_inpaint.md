# ProPainter 替换 E2FGVI 去手

> Status: **已接入** `mode=hand_inpaint`。推理在独立 conda `propainter` 里跑，不把依赖装进 phantom。
> 下游：`retarget_inpaint` 仍读 `inpaint_processor/video_human_inpaint.mkv`。

---

## 0. 为什么换

E2FGVI-HQ 面向 ~240–480p。EgoDex 1080p 上容易网格伪影、去手不干净（见 `b/doc/hand_removal_fix.md`）。ProPainter（ICCV 2023）做视频 object removal 更稳。

不把 ProPainter **import 进 phantom**：torch / deform-conv 和 phantom 环境容易打架。采用 **subprocess + 另一套 python**。

---

## 1. 数据流

```
video_rgb_imgs + masks_arm.npy
        │  phantom：膨胀 mask、可选降分辨率、dump PNG
        ▼
  tmp video_frames/*.png  +  tmp masks/*.png
        │  cwd=ProPainter repo，python=propainter conda
        ▼
  inference_propainter.py --mask_dilation 0 --save_frames --fp16
        │  phantom：读 PNG，升回原分辨率
        ▼
  inpaint_processor/video_human_inpaint.mkv  (ffv1, 15fps)
        │  帧数必须 == T
        ▼
  retarget_inpaint 叠机器人背景
```

- mask 只膨胀一次：phantom 的 `mask_dilate_*`；CLI 传 `--mask_dilation 0`。
- DA3 `depth.npy` 仍用**带手的原视频**；不要对 inpaint 后的 RGB 再估深度。
- `run_human2robot_all.sh` **不会**自动去手。用 `b/run_hand_inpaint.sh`（默认 `bbox → hand2d → arm_segmentation → hand_inpaint`），再（重）跑 `retarget_inpaint`。
- processed 目录若已由 DA3/intent 建好，`copytree` 不会带上 convert 的 `hand_det.pkl`。脚本会在跑前把 raw 的 `hand_det.pkl`（及 `video_L.mp4`）软链到 processed。缺 pkl 时 epic `bbox` 会 `FileNotFoundError`。

---

## 2. 改了哪些文件

| 文件 | 改动 |
|---|---|
| `phantom/processors/handinpaint_processor.py` | 默认 `inpaint_backend=propainter`；dump 帧/mask 后 subprocess；E2FGVI 改为按需 import |
| `b/run_hand_inpaint.sh` | 默认 `mode=[bbox,hand2d,arm_segmentation,hand_inpaint]`；`--inpaint-only` / `--seg-only` |
| `b/configs/egodex.yaml` | ProPainter 路径 / fp16 / subvideo 等 |
| `b/configs/egodex_panda_intent.yaml` | 同上（human2robot 用这份） |
| `b/configs/egodex_panda.yaml`、`egodex_r1pro*.yaml` | 同上 |

未改：`retarget_inpaint_processor.py`（已按「mkv 存在且帧数 == T」选背景）、未把 ProPainter 拷进 phantom。

---

## 3. 配置

```yaml
inpaint_backend: propainter          # 或 e2fgvi
propainter_root: "/home/a26160/SRC/ProPainter"
propainter_python: ""                # 必填，或环境变量 PROPAINTER_PYTHON
propainter_fp16: true
propainter_subvideo_length: 80
propainter_neighbor_length: 10
propainter_ref_stride: 10
inpaint_resolution: 480              # 降采样控显存；0=原分辨率
mask_dilate_kernel: "MORPH_ELLIPSE"
mask_dilate_size: 11
mask_dilate_iterations: 3
```

`propainter_python` 必须是 **propainter conda 的 python 二进制**，不能是 phantom 的 `sys.executable`。脚本会尝试 `conda run -n propainter python -c 'import sys; print(sys.executable)'`。

独立环境安装（只做一次）：

```bash
conda create -n propainter python=3.8 -y
conda activate propainter
cd /home/a26160/SRC/ProPainter
pip install -r requirements.txt
# 权重可首次推理自动下到 weights/，或手动放到该目录
```

回退 E2FGVI：`inpaint_backend=e2fgvi`（仍需 phantom 里已装的 E2FGVI）。

---

## 4. 怎么跑

`b/run_hand_inpaint.sh` 默认扫 `DATA_ROOT` 下全部 `egodex_*`、每个任务全部 demo、GPU `0-5`。并发默认 **每卡 4 路**（含 ProPainter，6 卡共 24）；`--seg-only` 每卡 8 路。H200 显存够用，旧逻辑曾把 `--jobs` 卡成「不超过 GPU 数」。OOM 再降 `--per-gpu`。

```bash
# 全部任务、全部 demo
bash b/run_hand_inpaint.sh

# 单个任务仍可指定
bash b/run_hand_inpaint.sh --task stack
bash b/run_hand_inpaint.sh --tasks stack,vertical_pick_place --per-gpu 6
bash b/run_hand_inpaint.sh --task stack --demo-num 0

# 已有 masks_arm.npy 时只 inpaint
bash b/run_hand_inpaint.sh --inpaint-only

# 只要 mask、先不去手
bash b/run_hand_inpaint.sh --seg-only
```

等价 Hydra：

```bash
cd phantom/phantom
python process_data.py --config-path=../b/configs --config-name=egodex_panda_intent \
  'mode=[bbox,hand2d,arm_segmentation,hand_inpaint]' demo_num=0 demo_name=egodex_stack \
  propainter_python=/path/to/envs/propainter/bin/python
```

然后重跑叠图（已有 overlay 不会自动换背景）：

```bash
bash b/run_contact_retarget.sh --task stack --demo-num 0 --step retarget_inpaint --skip-export
```

OOM：降低 `inpaint_resolution` 或 `propainter_subvideo_length`，保持 `--fp16`。

---

## 5. 校验

- `video_human_inpaint.mkv` 帧数 = 原视频 / Stage A 的 T。
- 目视手/臂被抹掉，被抓物体尽量还在（mask 过大仍会抹物体，属 segmentation 问题）。
- `retarget_inpaint` 日志应有 `background = human-inpaint video`；诊断图左列无人手。
