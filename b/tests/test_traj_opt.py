"""Unit tests for the Stage B trajectory optimizer core (traj_opt.py).

Uses a synthetic 7-DoF arm with an exact, analytic FK/Jacobian so the optimizer
math (position + orientation residuals, jacr sign/convention, 2nd-order
smoothness, joint-limit bounds) is verified without MuJoCo:

    pos = q[:3]                 (Cartesian gantry)      jacp = [I3 | 0]
    R   = Rz(q[3])              (wrist about world z)   jacr col3 = [0,0,1]
    q[4:7] unused               (exercise the regularizer)

Usage (from phantom/):
    python b/tests/test_traj_opt.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "phantom")))

from phantom.traj_opt import TrajectoryOptimizer, TrajOptConfig
from scipy.spatial.transform import Rotation


class GantryZArm:
    n_dof = 7

    def __init__(self):
        self.q_min = np.array([-3, -3, -3, -np.pi, -3, -3, -3], dtype=float)
        self.q_max = np.array([3, 3, 3, np.pi, 3, 3, 3], dtype=float)
        self.q_neutral = np.zeros(7)

    def fk(self, q):
        pos = q[:3].copy()
        R = Rotation.from_euler("z", q[3]).as_matrix()
        return pos, R

    def jac(self, q):
        jacp = np.zeros((3, 7)); jacp[0, 0] = jacp[1, 1] = jacp[2, 2] = 1.0
        jacr = np.zeros((3, 7)); jacr[2, 3] = 1.0  # world angular vel about z per dq3
        return jacp, jacr


class GantryXArm:
    n_dof = 7

    def __init__(self):
        self.q_min = np.array([-3, -3, -3, -np.pi, -3, -3, -3], dtype=float)
        self.q_max = np.array([3, 3, 3, np.pi, 3, 3, 3], dtype=float)
        self.q_neutral = np.zeros(7)

    def fk(self, q):
        pos = q[:3].copy()
        R = Rotation.from_euler("x", q[3]).as_matrix()
        return pos, R

    def jac(self, q):
        jacp = np.zeros((3, 7)); jacp[0, 0] = jacp[1, 1] = jacp[2, 2] = 1.0
        jacr = np.zeros((3, 7)); jacr[0, 3] = 1.0  # world angular vel about x per dq3
        return jacp, jacr


def _rz_targets(thetas):
    return np.stack([Rotation.from_euler("z", th).as_matrix() for th in thetas])


def _rx_targets(thetas):
    return np.stack([Rotation.from_euler("x", th).as_matrix() for th in thetas])


def test_position_tracking():
    kin = GantryZArm()
    n = 30
    t = np.linspace(0, 1, n)
    p_target = np.stack([0.5 * np.sin(2 * t), 0.5 * t, 0.3 + 0.1 * t], axis=1)
    R_target = _rz_targets(np.zeros(n))
    w_p = np.ones(n); w_r = np.zeros(n)  # position only
    opt = TrajectoryOptimizer(kin, TrajOptConfig(w_smooth=0.05, w_reg=1e-3))
    out = opt.optimize(p_target, R_target, w_p, w_r)
    assert out["success"] or out["nfev"] > 0
    assert out["pos_err"].max() < 5e-3, f"pos_err too high: {out['pos_err'].max()}"
    q = out["q"]
    assert np.all(q >= kin.q_min - 1e-6) and np.all(q <= kin.q_max + 1e-6), "joint limits violated"
    print(f"[pos] pos_err mean={out['pos_err'].mean():.5f} max={out['pos_err'].max():.5f} "
          f"cost {float(out['cost_initial']):.3g}->{float(out['cost_final']):.3g}")


def test_orientation_tracking():
    kin = GantryXArm()
    n = 25
    t = np.linspace(0, 1, n)
    thetas = 1.2 * np.sin(np.pi * t)  # smooth tilt sweep within limits
    p_target = np.zeros((n, 3))
    R_target = _rx_targets(thetas)
    w_p = np.ones(n); w_r = 5.0 * np.ones(n)
    opt = TrajectoryOptimizer(kin, TrajOptConfig(w_smooth=0.05, w_reg=1e-4))
    out = opt.optimize(p_target, R_target, w_p, w_r)
    assert out["ori_err"].max() < 1e-2, f"ori_err too high: {out['ori_err'].max()}"
    # recovered wrist angle should match the target tilt.
    q3 = out["q"][:, 3]
    assert np.allclose(q3, thetas, atol=1e-2), f"tilt mismatch, max |dq3|={np.abs(q3-thetas).max()}"
    print(f"[ori] ori_err mean={out['ori_err'].mean():.5f} max={out['ori_err'].max():.5f} "
          f"tilt_err_max={np.abs(q3-thetas).max():.5f}")


def test_parallel_jaw_spin_is_free():
    kin = GantryZArm()
    n = 20
    t = np.linspace(0, 1, n)
    thetas = 1.0 * np.sin(np.pi * t)
    p_target = np.zeros((n, 3))
    R_target = _rz_targets(thetas)
    w_p = np.ones(n); w_r = 5.0 * np.ones(n)
    opt = TrajectoryOptimizer(kin, TrajOptConfig(w_smooth=0.05, w_reg=1e-4))
    out = opt.optimize(p_target, R_target, w_p, w_r)
    q3 = out["q"][:, 3]
    # Rotation about target z is intentionally unconstrained.
    assert np.allclose(q3, 0.0, atol=2e-2), q3
    reduced = np.array([
        np.linalg.norm(opt._project_parallel_jaw_rotvec(R_target[i], kin.fk(out["q"][i])[1]))
        for i in range(n)
    ])
    assert reduced.max() < 1e-3, reduced.max()


def test_parallel_jaw_180_flip_is_equivalent():
    kin = GantryXArm()
    p_target = np.zeros((1, 3))
    R_target = Rotation.from_euler("xz", [0.4, np.pi]).as_matrix()[None]
    w_p = np.ones(1); w_r = 5.0 * np.ones(1)
    opt = TrajectoryOptimizer(kin, TrajOptConfig(w_smooth=0.0, w_reg=1e-4))
    out = opt.optimize(p_target, R_target, w_p, w_r)
    # The arm can only realize the x tilt; the extra 180° spin about z is free.
    assert abs(float(out["q"][0, 3]) - 0.4) < 1e-2, out["q"][0, 3]
    reduced = np.linalg.norm(opt._project_parallel_jaw_rotvec(R_target[0], kin.fk(out["q"][0])[1]))
    assert reduced < 1e-3, reduced


def test_smoothness_denoises():
    """A noisy warm start on a smooth target should come out smoother (low jerk)
    while still tracking position."""
    kin = GantryZArm()
    rng = np.random.default_rng(0)
    n = 40
    t = np.linspace(0, 1, n)
    p_target = np.stack([0.4 * np.sin(2 * t), 0.4 * np.cos(2 * t), 0.3 + 0.0 * t], axis=1)
    R_target = _rz_targets(np.zeros(n))
    w_p = np.ones(n); w_r = np.zeros(n)
    q_init = np.zeros((n, 7))
    q_init[:, :3] = p_target + rng.normal(0, 0.05, size=(n, 3))  # jittery init
    opt = TrajectoryOptimizer(kin, TrajOptConfig(w_smooth=2.0, w_reg=1e-4))
    out = opt.optimize(p_target, R_target, w_p, w_r, q_init=q_init)
    init_jerk = np.sqrt(np.mean((q_init[2:] - 2 * q_init[1:-1] + q_init[:-2]) ** 2))
    assert out["jerk_rms"] < init_jerk, f"jerk not reduced: {out['jerk_rms']} vs {init_jerk}"
    assert out["pos_err"].mean() < 2e-2, f"pos tracking lost: {out['pos_err'].mean()}"
    print(f"[smooth] jerk {init_jerk:.4f}->{float(out['jerk_rms']):.4f} "
          f"pos_err mean={out['pos_err'].mean():.4f}")


def test_per_joint_regularizer():
    """A per-joint w_reg must match the scalar form when uniform, and must pull
    only the weighted joint when it is not.

    Stage B uses this to bias one redundant joint (the Panda upper-arm roll)
    without touching the joints that carry EE tracking.
    """
    kin = GantryZArm()
    n = 12
    p_target = np.tile([0.5, 0.2, 0.3], (n, 1)).astype(float)
    R_target = _rz_targets(np.zeros(n))
    w_p = np.ones(n); w_r = np.zeros(n)

    def solve(w_reg):
        opt = TrajectoryOptimizer(kin, TrajOptConfig(w_smooth=0.0, w_reg=w_reg))
        return opt.optimize(p_target, R_target, w_p, w_r)["q"]

    q_scalar = solve(0.01)
    q_vector = solve([0.01] * kin.n_dof)
    assert np.allclose(q_scalar, q_vector, atol=1e-6), "uniform vector w_reg diverged from scalar"

    # Weighting joint 0 alone trades its tracking for the neutral posture. With
    # w_p=1 and q_neutral=0 the stationary point is q0 = 0.5 / (1 + w_reg0^2).
    q_biased = solve([1.0] + [1e-4] * (kin.n_dof - 1))
    assert abs(q_biased[:, 0].mean() - 0.25) < 1e-3, q_biased[:, 0].mean()
    assert abs(q_biased[:, 1].mean() - 0.2) < 1e-3, q_biased[:, 1].mean()
    assert abs(q_biased[:, 2].mean() - 0.3) < 1e-3, q_biased[:, 2].mean()
    print(f"[reg] scalar==vector ok; weighted joint0 pulled {q_scalar[:, 0].mean():.4f}"
          f"->{q_biased[:, 0].mean():.4f} while joint1/2 tracking held")


if __name__ == "__main__":
    test_position_tracking()
    test_orientation_tracking()
    test_parallel_jaw_spin_is_free()
    test_parallel_jaw_180_flip_is_equivalent()
    test_smoothness_denoises()
    test_per_joint_regularizer()
    print("OK: traj_opt smoke tests passed")
