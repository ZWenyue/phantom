#!/usr/bin/env python3
"""Quick test: frantik + MuJoCo refine on demo 0."""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "phantom"))

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
from robosuite.controllers import load_controller_config
from robomimic.envs.env_robosuite import EnvRobosuite
import robomimic.utils.obs_utils as ObsUtils
from phantom.panda_frantik_ik import PandaFrantikIKSolver, ReachabilityIndex

ObsUtils.initialize_obs_utils_with_obs_specs(
    obs_modality_specs=dict(obs=dict(low_dim=["robot0_eef_pos"], rgb=["zed_image"]))
)
options = dict(
    env_name="PhantomBimanual",
    bimanual_setup="shoulders",
    robots=["Panda", "Panda"],
    gripper_types=["Robotiq85Gripper", "Robotiq85Gripper"],
    controller_configs=load_controller_config(default_controller="OSC_POSE"),
    camera_heights=480,
    camera_widths=480,
    camera_segmentations="instance",
    direct_gripper_control=True,
    use_depth_obs=True,
    camera_pos=np.array([0, 0, 1.5]),
    camera_quat_wxyz=np.array([1, 0, 0, 0]),
    camera_sensorsize=np.array([6.0, 6.0]),
    camera_principalpixel=np.array([0.0, 0.0]),
    camera_focalpixel=np.array([400.0, 400.0]),
)
env = EnvRobosuite(
    **options,
    render=False,
    render_offscreen=True,
    use_image_obs=True,
    camera_names=["zed"],
    control_freq=20,
)
env.reset()

npz = os.path.join(ROOT, "b", "reachability_panda_bimanual.npz")
reach = ReachabilityIndex.load(npz) if os.path.isfile(npz) else None
solver = PandaFrantikIKSolver(env, reachability=reach)

demo = "/home/a26160/DATA/test_phantom_processed/egodex_basic_pick_place/0/smoothing_processor"
right = np.load(f"{demo}/smoothed_actions_right_shoulders.npz")
left = np.load(f"{demo}/smoothed_actions_left_shoulders.npz")

errs_r, errs_l = [], []
for i in range(min(30, len(right["ee_pts"]))):
    q_r, q_l, e_r, e_l = solver.solve_bimanual(
        right["ee_pts"][i], right["ee_oris"][i], left["ee_pts"][i], left["ee_oris"][i]
    )
    errs_r.append(e_r)
    errs_l.append(e_l)
    print(f"frame {i}: R={e_r:.4f} L={e_l:.4f}")

print(f"R: mean={np.mean(errs_r):.4f} max={np.max(errs_r):.4f} <5cm={sum(e<0.05 for e in errs_r)}/{len(errs_r)}")
print(f"L: mean={np.mean(errs_l):.4f} max={np.max(errs_l):.4f} <5cm={sum(e<0.05 for e in errs_l)}/{len(errs_l)}")
env.env.close()
