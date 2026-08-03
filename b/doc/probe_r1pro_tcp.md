# R1 Pro 短 TCP / 基座 / OSC 探针

## 概述

`b/probe_r1pro_tcp.py` 用少量采样帧快速测 OSC 跟踪误差，**不渲染视频**。用于调：

- 双臂基座位姿（`PhantomBimanual` `r1pro` setup）
- OSC `kp`（尤其是位置 vs 姿态增益比）
- 夹爪 TCP（`eef` / `grip_site`）变更后的可达性

全量 `robot_inpaint` 约 10–15 分钟/demo；探针通常 1 分钟内扫完一组候选。

## 背景：为什么需要探针

| 配置 | 现象 |
|---|---|
| `eef z=0.155`（抄自 Robotiq） | tracking 数字好看，但夹爪相对人手系统性偏 |
| `eef z=0.06`（几何抓取中心）+ 等权 `kp` | 人手姿态常不可达 → 腕关节顶死 → grip_site 被拖偏 10–20cm |
| `eef z=0.06` + **位置优先** `kp=[300,300,300,5,5,5]` | 位置误差可到 mm 级；539 帧全过 |

结论：短 TCP 下必须 **位置优先**；不要退回 `0.155` 假长。

## 依赖

- conda 环境：`phantom`
- 已生成的 smoothed actions（`*_r1pro.npz`）
- `MUJOCO_GL=egl`（无头 GPU 机）

## 用法

```bash
cd /mnt/r/share/zwy/Projects/phantom/phantom
eval "$(conda shell.bash hook)" && conda activate phantom

PYTHONUNBUFFERED=1 MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 \
  python ../b/probe_r1pro_tcp.py
```

### 当前脚本行为

1. 从固定 demo 读左右手目标：
   ```
   /mnt/r/DATA/EgoDex/test_phantom_processed/egodex_make_sandwich_test/0_useful/
     smoothing_processor/smoothed_actions_{left,right}_r1pro.npz
   ```
2. 采样帧：`[0, 30, 60, 100, 200, 300, 400]`（共 7 帧）
3. 用 `install_bases()` monkey-patch `PhantomBimanual._load_model` 注入基座
4. 建 `TwinBimanualRobot`（`epic=True`，`n_steps_short=80`）
5. **在 init 之后** 用 `apply_kp()` 覆盖控制器 `kp`/`kd`（重要：`TwinBimanualRobot` 构造时会写死一份 config，探针必须事后覆盖）
6. 对每组 `kp` 打印每帧 `L/R` 位置误差与 `pass n/7`

阈值与 pipeline 一致：`THRESH = 0.05` m。

## 如何改候选

编辑脚本顶部常量即可。

### 基座 `BASE_CFG`

```python
BASE_CFG = dict(
    r_pos=(-0.28, -0.12, 1.74), r_rot=(0.0, 0.0, np.pi / 2),  # 右臂
    l_pos=(-0.22, 0.20, 1.56), l_rot=(0.0, 0.0, -np.pi / 2),  # 左臂
)
```

与 `phantom_bimanual.py` 里 `"r1pro"` 分支应对齐（探针验证通过后再写回正式代码）。

### OSC 增益列表

```python
for name, kp in [
    ("scalar200", 200),
    ("pos300_ori5", [300, 300, 300, 5, 5, 5]),  # 推荐
    ("pos300_ori1", [300, 300, 300, 1, 1, 1]),
    ("pos400_ori0", [400, 400, 400, 0, 0, 0]),  # 纯位置
]:
```

格式：`[kx, ky, kz, kax, kay, kaz]`。

### 采样帧 / demo 路径

```python
DEMO = Path(".../0_useful")
FRAMES = [0, 30, 60, 100, 200, 300, 400]
```

换 demo 时改 `DEMO`，并确认存在 `smoothed_actions_*_r1pro.npz`。

## 读结果

```
=== pos300_ori5 ===
  controller kp= [300. 300. 300.   5.   5.   5.]
  frame   0: L=0.003 R=0.001 OK
  ...
  pass 7/7  mean L=0.003 R=0.005
```

- `OK` / `FAIL`：单帧是否两侧都 ≤ 5cm  
- `pass n/7`：采样通过率（不是全视频通过率）  
- `mean L/R`：平均位置误差  

探针通过后，再用正式 pipeline 验证全量：

```bash
bash b/run_process.sh \
  --step robot_inpaint \
  --task make_sandwich_test \
  --config egodex_r1pro_bimanual \
  --demo-num 0_useful \
  --no-skip
```

## 已知坑

1. **必须事后 `apply_kp`**  
   `TwinBimanualRobot` 对 `r1pro` 会设置 `kp=[300,300,300,5,5,5]`；若在探针里只改 `load_controller_config` 返回值，可能被构造逻辑覆盖。以 `ctrl.kp=` 打印为准。

2. **stdout 缓冲**  
   管道/tee 时加 `PYTHONUNBUFFERED=1`，否则失败帧日志会延迟刷出，统计易误判。

3. **TCP 与基座要一起调**  
   改 `eef` 长度后旧基座往往失效；先探针再全量。

4. **姿态不可达 ≠ 位置不可达**  
   等权 `kp=200` 时误差常 10–20cm；把 ori 增益降到 1–5 后同一基座可到 mm 级。腕部朝向会变软，overlay 以夹爪中心贴手为准。

5. **与 `compute_reachability.py` 的关系**  
   - `compute_reachability.py`：FK 采样，看目标是否在几何可达云内  
   - `probe_r1pro_tcp.py`：真实 OSC 闭环，看控制器能否在限时步数内跟上  
   两者互补；FK 可达仍可能因 ori/初值/步数导致 OSC 失败。

## 当前推荐配置（已写入正式代码）

| 项 | 值 |
|---|---|
| TCP | `eef pos="0 0 0.06"` |
| 右基座 | `(-0.28, -0.12, 1.74)`, yaw `+π/2` |
| 左基座 | `(-0.22, 0.20, 1.56)`, yaw `-π/2` |

## 实验：去掉关节限位

配置：`b/configs/egodex_r1pro_bimanual_nolimit.yaml`（`bimanual_setup: r1pro_nolimit`）。

- 运行时清除臂关节 MuJoCo `jnt_limited`；OSC `kp=[300,300,300,40,40,40]`，`n_steps_short=20`
- 复用已有 `*_r1pro.npz` 轨迹；输出 `*_r1pro_nolimit.*`
- Probe 全 539 帧位置误差 ≤5 cm（max≈3.8 cm）
- 仅仿真实验，不可映射真机

```bash
bash b/run_process.sh --config egodex_r1pro_bimanual_nolimit \
  --step robot_inpaint --demo-num 0 --no-skip --gpus 1 --workers 1
```

| OSC kp | `[300, 300, 300, 5, 5, 5]` |
| `uncouple_pos_ori` | `True` |
| `n_steps_short` | `80` |
| `init_qpos` | 相对伸展（见 `r1_pro_*_arm_robot.py`） |

详见 `b/doc/r1_pro.md`。
