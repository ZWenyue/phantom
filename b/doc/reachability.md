# R1 Pro 可达空间计算工具

## 概述

`b/compute_reachability.py` 通过 MuJoCo 正运动学（FK）均匀采样关节空间，计算 R1 Pro 机械臂末端执行器（EEF）的可达位置，并保存为 `.npz` 文件和 3D 散点图。

用于解决 IK 求解失败的问题：在规划目标位置前，先确认该位置是否在机械臂可达空间内。

## 依赖

- Python 环境：`phantom` conda 环境（`/mnt/r/share/zwy/conda/envs/phantom`）
- 依赖包：`mujoco`, `numpy`, `matplotlib`, `scipy`（仅 `--bimanual` 模式需要）

## 用法

```bash
PYTHON=/mnt/r/share/zwy/conda/envs/phantom/bin/python
```

### 计算可达空间

```bash
# 单臂（臂基座坐标系）
$PYTHON b/compute_reachability.py --arm left --samples 500000
$PYTHON b/compute_reachability.py --arm right --samples 500000

# 双臂（世界坐标系，应用 phantom_bimanual.py 中的基座变换）
$PYTHON b/compute_reachability.py --arm both --samples 500000 --bimanual
```

### 查询目标点是否可达

```bash
# 查询点 (0.3, 0.1, 1.5) 是否在左臂可达范围内（默认容差 2cm）
$PYTHON b/compute_reachability.py --arm left --samples 500000 --bimanual --query 0.3,0.1,1.5

# 自定义容差（5cm）
$PYTHON b/compute_reachability.py --arm left --samples 500000 --bimanual --query 0.3,0.1,1.5 --tol 0.05
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--arm` | `both` | 计算哪条臂：`left` / `right` / `both` |
| `--samples` | `500000` | FK 采样数量，越大覆盖越密，50 万通常足够 |
| `--bimanual` | 关 | 应用双臂基座位姿变换，输出世界坐标系下的 EEF 位置 |
| `--output` | 自动生成 | 输出 `.npz` 文件路径（不含后缀） |
| `--query` | 无 | 查询点坐标，格式 `x,y,z` |
| `--tol` | `0.02` | 可达性查询容差（米） |
| `--seed` | `42` | 随机种子 |
| `--no-plot` | 关 | 跳过生成可视化图片 |

## 输出文件

### `.npz` 文件

```python
import numpy as np

data = np.load("b/reachability_bimanual.npz")

# 左臂
left_positions = data["left_positions"]   # (N, 3) EEF 可达位置
left_joints = data["left_joints"]         # (N, 7) 对应关节角配置

# 右臂
right_positions = data["right_positions"]
right_joints = data["right_joints"]

# 是否使用了 bimanual 基座变换
bimanual = data["bimanual"][0]            # bool
```

### `.png` 可视化

自动生成 3D 散点图，包含各臂独立视图和双臂合并视图。颜色映射到 Z 轴高度。

## 在代码中使用

### 过滤不可达目标

```python
import numpy as np

# 加载预计算的可达空间
data = np.load("b/reachability_bimanual.npz")
left_pos = data["left_positions"]

def is_reachable(target_xyz, positions, tol=0.02):
    """判断目标点是否在可达空间内"""
    dists = np.linalg.norm(positions - target_xyz, axis=1)
    return dists.min() <= tol

# 示例
target = np.array([0.3, 0.1, 1.5])
if is_reachable(target, left_pos):
    print("目标可达，可以尝试 IK 求解")
else:
    print("目标不可达，跳过")
```

### 找到最近可达点

```python
def nearest_reachable(target_xyz, positions, joints):
    """找到离目标最近的可达点及其关节角"""
    dists = np.linalg.norm(positions - target_xyz, axis=1)
    idx = dists.argmin()
    return positions[idx], joints[idx], dists[idx]

nearest_pos, nearest_jnt, dist = nearest_reachable(target, left_pos, data["left_joints"])
print(f"最近可达点: {nearest_pos}, 距离: {dist:.4f}m")
print(f"对应关节角: {nearest_jnt}")
```

## 坐标系说明

- **不加 `--bimanual`**：EEF 位置在臂基座坐标系下，原点为臂的 base link
- **加 `--bimanual`**：EEF 位置在世界坐标系下，基座位姿来自 `phantom_bimanual.py`：
  - 右臂 base: pos=(-0.28, -0.12, 1.74), euler=(0, 0, π/2)
  - 左臂 base: pos=(-0.22, 0.20, 1.56), euler=(0, 0, -π/2)

如果你的目标点是在世界坐标系下（如从相机标定得到的），使用 `--bimanual`。

## 性能

在当前机器上，FK 速率约 **10k-100k 次/秒**，50 万采样约需 5-30 秒。

## 注意事项

- 采样是均匀随机的，可达空间边界附近采样密度较低，增加 `--samples` 可改善覆盖
- `--tol` 设置过小可能导致可达点误判为不可达（采样间隙），建议 ≥ 1cm
- 该工具仅计算位置可达性，不考虑末端姿态（orientation）约束
- 不做碰撞检测（自碰撞/环境碰撞），实际可用空间可能更小
