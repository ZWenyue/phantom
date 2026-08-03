#!/usr/bin/env python3
"""
Compute reachable workspace of R1 Pro arms via MuJoCo FK.

Uniformly samples joint space, computes forward kinematics for each sample,
and records the end-effector positions. Outputs:
  1. An .npz file with all reachable EEF positions (and joint configs)
  2. A 3D scatter plot visualization of the workspace

Usage:
    python b/compute_reachability.py --arm left --samples 500000
    python b/compute_reachability.py --arm right --samples 500000
    python b/compute_reachability.py --arm both --samples 500000

    # With bimanual base transforms (as in phantom_bimanual.py):
    python b/compute_reachability.py --arm both --samples 500000 --bimanual

    # Query whether a specific point is reachable:
    python b/compute_reachability.py --arm left --query 0.1,0.0,-0.3
"""

import argparse
import os
import tempfile
import time

import mujoco
import numpy as np


ARM_CONFIGS = {
    "left": {
        "joint_limits": np.array([
            [-4.4506, 1.309],    # J1 axis Y
            [-0.1745, 3.1416],   # J2 axis X
            [-2.3562, 2.3562],   # J3 axis Z
            [-2.0944, 0.3491],   # J4 axis Y
            [-2.3562, 2.3562],   # J5 axis Z
            [-1.0472, 1.0472],   # J6 axis Y
            [-1.5708, 1.5708],   # J7 axis X
        ]),
        "eef_body": "right_hand",
        # Bimanual base pose from phantom_bimanual.py
        "base_pos": np.array([-0.22, 0.20, 1.56]),
        "base_euler": np.array([0.0, 0.0, -np.pi / 2]),
    },
    "right": {
        "joint_limits": np.array([
            [-4.4506, 1.309],    # J1 axis Y
            [-3.1416, 0.1745],   # J2 axis X (mirrored)
            [-2.3562, 2.3562],   # J3 axis Z
            [-2.0944, 0.3491],   # J4 axis Y
            [-2.3562, 2.3562],   # J5 axis Z
            [-1.0472, 1.0472],   # J6 axis Y
            [-1.5708, 1.5708],   # J7 axis X
        ]),
        "eef_body": "right_hand",
        "base_pos": np.array([-0.28, -0.12, 1.74]),
        "base_euler": np.array([0.0, 0.0, np.pi / 2]),
    },
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def build_arm_xml(arm: str) -> str:
    """Build a standalone MuJoCo XML for one arm using the project's mesh files."""
    mesh_dir = os.path.join(SCRIPT_DIR, f"robots/r1_pro_{arm}_arm")
    robot_xml_path = os.path.join(mesh_dir, "robot.xml")

    xml = f"""<mujoco model="r1_pro_{arm}_arm_reachability">
  <compiler angle="radian" meshdir="{mesh_dir}/"/>
  <option gravity="0 0 0"/>
  <include file="{robot_xml_path}"/>
</mujoco>"""
    return xml


def compute_fk(arm: str, joint_configs: np.ndarray, bimanual: bool = False) -> np.ndarray:
    """
    Compute FK for an array of joint configurations.

    Args:
        arm: "left" or "right"
        joint_configs: (N, 7) array of joint angles
        bimanual: if True, transform EEF positions to world frame using bimanual base pose

    Returns:
        (N, 3) array of EEF positions
    """
    xml_str = build_arm_xml(arm)

    with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
        f.write(xml_str)
        tmp_path = f.name

    try:
        model = mujoco.MjModel.from_xml_path(tmp_path)
        data = mujoco.MjData(model)
    finally:
        os.unlink(tmp_path)

    cfg = ARM_CONFIGS[arm]
    eef_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg["eef_body"])
    if eef_id == -1:
        raise RuntimeError(f"EEF body '{cfg['eef_body']}' not found in model")

    n = joint_configs.shape[0]
    positions = np.empty((n, 3), dtype=np.float64)

    nq = model.nq
    joint_start = 0
    for i in range(model.njnt):
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE:
            joint_start = model.jnt_qposadr[i]
            break

    for i in range(n):
        data.qpos[:] = 0
        data.qpos[joint_start:joint_start + 7] = joint_configs[i]
        mujoco.mj_kinematics(model, data)
        positions[i] = data.xpos[eef_id].copy()

    if bimanual:
        from scipy.spatial.transform import Rotation
        R = Rotation.from_euler("xyz", cfg["base_euler"]).as_matrix()
        t = cfg["base_pos"]
        positions = (R @ positions.T).T + t

    return positions


def sample_joint_space(arm: str, n_samples: int, seed: int = 42) -> np.ndarray:
    """Uniformly sample the joint space within joint limits."""
    rng = np.random.default_rng(seed)
    limits = ARM_CONFIGS[arm]["joint_limits"]
    lo, hi = limits[:, 0], limits[:, 1]
    samples = rng.uniform(lo, hi, size=(n_samples, 7))
    return samples


def query_reachable(positions: np.ndarray, point: np.ndarray, tol: float = 0.02) -> dict:
    """Check if a query point is within the sampled reachable workspace."""
    dists = np.linalg.norm(positions - point, axis=1)
    min_dist = dists.min()
    nearest_idx = dists.argmin()
    return {
        "reachable": min_dist <= tol,
        "min_distance": min_dist,
        "nearest_idx": nearest_idx,
        "nearest_pos": positions[nearest_idx],
    }


def visualize(positions_dict: dict, output_path: str, bimanual: bool = False):
    """3D scatter plot of reachable workspace."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 6))
    arms = list(positions_dict.keys())
    n_plots = len(arms)

    for idx, arm_name in enumerate(arms):
        ax = fig.add_subplot(1, n_plots + (1 if n_plots > 1 else 0), idx + 1, projection="3d")
        pos = positions_dict[arm_name]
        # Subsample for plotting
        max_plot = 50000
        if len(pos) > max_plot:
            plot_idx = np.random.default_rng(0).choice(len(pos), max_plot, replace=False)
            pos_plot = pos[plot_idx]
        else:
            pos_plot = pos

        ax.scatter(pos_plot[:, 0], pos_plot[:, 1], pos_plot[:, 2],
                   s=0.1, alpha=0.3, c=pos_plot[:, 2], cmap="viridis")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(f"{arm_name} arm ({len(pos)} pts)")
        ax.set_aspect("equal")

    if n_plots > 1:
        ax = fig.add_subplot(1, n_plots + 1, n_plots + 1, projection="3d")
        for arm_name in arms:
            pos = positions_dict[arm_name]
            max_plot = 30000
            if len(pos) > max_plot:
                plot_idx = np.random.default_rng(0).choice(len(pos), max_plot, replace=False)
                pos_plot = pos[plot_idx]
            else:
                pos_plot = pos
            ax.scatter(pos_plot[:, 0], pos_plot[:, 1], pos_plot[:, 2],
                       s=0.1, alpha=0.3, label=arm_name)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("Combined workspace")
        ax.legend()
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved visualization: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute R1 Pro arm reachable workspace")
    parser.add_argument("--arm", type=str, default="both", choices=["left", "right", "both"])
    parser.add_argument("--samples", type=int, default=500000, help="Number of FK samples")
    parser.add_argument("--bimanual", action="store_true",
                        help="Apply bimanual base transforms (world frame)")
    parser.add_argument("--output", type=str, default=None, help="Output .npz path")
    parser.add_argument("--query", type=str, default=None,
                        help="Query point x,y,z (e.g. 0.1,0.0,-0.3)")
    parser.add_argument("--tol", type=float, default=0.02,
                        help="Tolerance for reachability query (meters)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plot", action="store_true", help="Skip visualization")
    args = parser.parse_args()

    arms = ["left", "right"] if args.arm == "both" else [args.arm]
    output_base = args.output or os.path.join(
        SCRIPT_DIR, f"reachability_{'bimanual' if args.bimanual else args.arm}")

    all_positions = {}
    all_joints = {}

    for arm in arms:
        print(f"\n{'='*60}")
        print(f"Computing FK for {arm} arm ({args.samples} samples)...")
        t0 = time.time()

        joints = sample_joint_space(arm, args.samples, seed=args.seed)
        positions = compute_fk(arm, joints, bimanual=args.bimanual)

        elapsed = time.time() - t0
        print(f"Done in {elapsed:.1f}s ({args.samples / elapsed:.0f} FK/s)")

        all_positions[arm] = positions
        all_joints[arm] = joints

        # Stats
        print(f"  X range: [{positions[:, 0].min():.4f}, {positions[:, 0].max():.4f}]")
        print(f"  Y range: [{positions[:, 1].min():.4f}, {positions[:, 1].max():.4f}]")
        print(f"  Z range: [{positions[:, 2].min():.4f}, {positions[:, 2].max():.4f}]")

        if args.query:
            point = np.array([float(x) for x in args.query.split(",")])
            result = query_reachable(positions, point, tol=args.tol)
            status = "REACHABLE" if result["reachable"] else "NOT REACHABLE"
            print(f"\n  Query {point} -> {status}")
            print(f"    Min distance: {result['min_distance']:.4f} m")
            print(f"    Nearest point: {result['nearest_pos']}")

    # Save
    npz_path = output_base + ".npz"
    save_dict = {}
    for arm in arms:
        save_dict[f"{arm}_positions"] = all_positions[arm]
        save_dict[f"{arm}_joints"] = all_joints[arm]
    save_dict["bimanual"] = np.array([args.bimanual])
    np.savez_compressed(npz_path, **save_dict)
    print(f"\nSaved {npz_path}")

    # Visualize
    if not args.no_plot:
        png_path = output_base + ".png"
        visualize(all_positions, png_path, bimanual=args.bimanual)


if __name__ == "__main__":
    main()
