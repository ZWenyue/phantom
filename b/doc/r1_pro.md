# R1 Pro Phantom Integration

## Overview

将 R1 Pro 双臂人形机器人接入 Phantom 框架（robot digital twin overlay system），使用双臂（bimanual）模式。

R1 Pro 每条臂 7DOF，左右臂运动学镜像（J2 关节限位翻转，部分 link 偏移取反）。肩关节 J0 水平安装（轴向 Y），与 Kinova3/Panda 的竖直安装不同，因此单臂模式工作空间无法覆盖目标位置，必须使用双臂模式。

## 文件结构

### 新增文件

- `b/robots/r1_pro_left_arm/robot.xml` — 左臂 MJCF 模型
- `b/robots/r1_pro_left_arm/meshes/` — 左臂 mesh 文件（STL 格式，从 `b/urdf/r1pro/meshes/` 复制）
- `b/robots/r1_pro_right_arm/robot.xml` — 右臂 MJCF 模型（运动学镜像）
- `b/robots/r1_pro_right_arm/meshes/` — 右臂 mesh 文件（STL 格式）
- `b/robots/r1_pro_right_arm_robot.py` — 右臂 robosuite ManipulatorModel
- `b/robots/default_r1_pro_right_arm.json` — 右臂控制器配置
- `b/robots/r1_pro_gripper/r1_pro_gripper.xml` — R1 Pro 原生平行夹爪 MJCF 模型
- `b/robots/r1_pro_gripper/meshes/` — 夹爪 mesh 文件（STL 格式）
- `submodules/phantom-robosuite/robosuite/models/grippers/r1_pro_gripper.py` — R1ProGripper GripperModel 类
- `b/configs/egodex_r1pro_bimanual.yaml` — R1 Pro 双臂 pipeline 配置
- `submodules/phantom-robosuite/robosuite/controllers/config/default_r1_pro_right_arm.json` — 符号链接 → `b/robots/default_r1_pro_right_arm.json`

### 修改文件

- `submodules/phantom-robosuite/robosuite/models/robots/manipulators/__init__.py` — 注册 R1ProRightArm/LeftArm
- `submodules/phantom-robosuite/robosuite/models/grippers/__init__.py` — 注册 R1ProGripper
- `submodules/phantom-robosuite/robosuite/robots/__init__.py` — 添加 `"R1ProRightArm": SingleArm` 到 ROBOT_CLASS_MAPPING
- `submodules/phantom-robosuite/robosuite/robots/manipulator.py` — 修复 `grip_action` 支持单 actuator 夹爪
- `submodules/phantom-robosuite/robosuite/environments/manipulation/phantom_bimanual.py` — 添加 `"r1pro"` bimanual_setup（设置双臂基座位置和朝向）
- `phantom/twin_bimanual_robot.py` — 支持 robot_name 为 list + R1Pro 夹爪动作映射
- `phantom/twin_robot.py` — 添加 R1Pro 夹爪动作映射
- `phantom/processors/robotinpaint_processor.py` — R1Pro 双臂机器人名称映射 + 相机名称修复
- `phantom/processors/action_processor.py` — 添加 `"r1pro"` 到 neutral_configs

## 关键技术细节

### URDF 与 Mesh 文件

源文件位于 `b/urdf/r1pro/`（新版 URDF + 已修正坐标轴的 mesh）。

旧版 URDF（`b/urdf/r1_pro/`）的 OBJ mesh 使用 Y-up 坐标系（CAD 导出），与 MuJoCo Z-up 不一致，导致渲染"散架"。新版 URDF 提供了正确的 STL mesh 文件，直接使用即可，无需 euler 旋转 hack。

### 关节阻尼

MuJoCo 仿真中出现数值不稳定（"Nan, Inf or huge value in QACC at DOF 2"），通过增大关节阻尼解决：
- 主关节 J1-J5：damping 0.1 → 0.5
- 腕关节 J6-J7：damping 0.01 → 0.1

### 双臂模式

`TwinBimanualRobot` 修改为接受 robot name list，使左右臂可以使用不同的模型：
- 右臂：`R1ProRightArm`（robot 0）
- 左臂：`R1ProLeftArm`（robot 1）

`PhantomBimanual` 环境中 `"r1pro"` setup 的基座配置：
- Robot 0（右臂）：pos=(-0.2, -0.1, 1.8), rot=(0, 0, pi/2)
- Robot 1（左臂）：pos=(-0.2, 0.2, 1.7), rot=(0, 0, -pi/2)

### 夹爪

使用 R1 Pro 原生平行夹爪（`R1ProGripper`），替代 Robotiq85。
- 2 个棱柱关节（prismatic），沿 Y 轴对称开合，行程 ±50mm
- 1 个 actuator 驱动 finger_joint1，通过 equality constraint 耦合 finger_joint2
- `eef` body（grip_site/IK 控制点）位于 `pos="0 0 0.155"` from base_mount — 与之前 Robotiq85 的 grip_site 位置一致，保证 IK 行为不变
- `right_hand` body 使用 identity quat（无旋转），原生夹爪 mesh 朝向已正确

### 控制器

使用 OSC_POSE 控制器（笛卡尔空间，DOF 无关），由 TwinRobot/TwinBimanualRobot 硬编码，忽略 robot 的 default_controller_config。

## 运行方式

```bash
# 生成 overlay 视频
python phantom/process_data.py \
  --config-path ../b/configs \
  --config-name egodex_r1pro_bimanual \
  "mode=[robot_inpaint]" \
  demo_num=0 \
  demo_name=egodex_make_sandwich
```

需要先跑完前置步骤（bbox → hand2d → hand3d → action → smoothing）生成 smoothed_actions 数据。

## 当前状态

- Pipeline 端到端可运行，约 93.3% 帧通过 tracking error 阈值（5cm）
- 渲染质量正确：机械臂连接正常，R1 Pro 原生夹爪朝向正确
- 待优化：双臂基座位置/朝向调优，减少 tracking error 帧数
