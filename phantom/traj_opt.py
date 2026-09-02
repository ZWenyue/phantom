"""Stage B: whole-trajectory joint-space optimization (contact-grounded retargeting).

Consumes the Stage A *intent* (per-frame EE targets + phase-dependent weights,
``intent.npz``) and solves for a feasible, smooth joint trajectory ``q_{1:T}``
with forward kinematics in the loop (design doc §2.6):

    minimize over q_{1:T}
        Σ_t  w_p(t)·‖FK_pos(q_t) − p_t*‖²                 (position fidelity)
           + w_r(t)·d_SO3(FK_ori(q_t), R_t*)²             (orientation fidelity)
           + w_s ·‖q_{t+1} − 2 q_t + q_{t−1}‖²            (2nd-order smoothness)
           + w_reg·‖q_t − q_neutral‖²                     (posture regularizer)
    subject to  q_min ≤ q_t ≤ q_max   (+ optional soft velocity limit)

Unlike per-frame analytical IK there is no hard "no solution": infeasible targets
degrade to a best-effort minimum-residual solution (feasible by construction), so
there are **no dropped frames**, and the saved labels (``q_t`` / ``FK(q_t)``) are
**consistent with the rendered robot** by construction (design §2.5).

The optimizer is decoupled from MuJoCo via a small kinematics interface
(:class:`ArmKinematics`) so the math is unit-testable with a synthetic arm; the
real Panda FK/Jacobian provider lives in ``processors/stageb_processor.py``.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


class ArmKinematics:
    """Interface a Stage B kinematics provider must implement (world frame).

    Attributes:
        n_dof: number of actuated arm joints (7 for Panda).
        q_min, q_max: (n_dof,) joint limits.
        q_neutral: (n_dof,) neutral posture for the regularizer / default init.

    Methods:
        fk(q)  -> (pos(3,), R(3,3))   EE position + rotation in the world frame.
        jac(q) -> (jacp(3,n), jacr(3,n))  world-frame position + angular Jacobians.
    """

    n_dof: int
    q_min: np.ndarray
    q_max: np.ndarray
    q_neutral: np.ndarray

    def fk(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:  # pragma: no cover - interface
        raise NotImplementedError

    def jac(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class TrajOptConfig:
    w_smooth: float = 1.0        # 2nd-order smoothness weight (replaces GP/SLERP)
    # Posture regularizer weight (toward q_neutral). A scalar weights every joint
    # equally; an (n_dof,) sequence lets a single redundant joint (e.g. the Panda
    # upper-arm roll that swivels the elbow) be pulled harder than the rest.
    w_reg: Union[float, Sequence[float]] = 0.01
    w_vel: float = 0.0           # soft velocity-limit penalty weight (0 disables)
    dq_max: float = 0.3          # per-joint velocity cap (rad/frame) for the soft penalty
    max_nfev: int = 200          # max least_squares function evaluations
    xtol: float = 1e-8
    ftol: float = 1e-8
    verbose: int = 0             # scipy least_squares verbosity (0/1/2)


class TrajectoryOptimizer:
    """Gauss-Newton / Levenberg-Marquardt whole-trajectory optimizer.

    Solves the least-squares problem above with :func:`scipy.optimize.least_squares`
    (``method='trf'`` so joint limits enter as box bounds) and an analytic sparse
    Jacobian: position blocks use the world position Jacobian ``jacp``; orientation
    blocks use the world angular Jacobian ``jacr`` (first-order GN approximation of
    the SO(3) log-map residual); smoothness/regularizer blocks are constant.
    """

    def __init__(self, kin: ArmKinematics, cfg: Optional[TrajOptConfig] = None):
        self.kin = kin
        self.cfg = cfg or TrajOptConfig()

    # ------------------------------------------------------------------
    @staticmethod
    def _project_parallel_jaw_rotvec(R_target: np.ndarray, R_cur: np.ndarray) -> np.ndarray:
        """Reduced SO(3) residual for a parallel-jaw gripper.

        Two symmetries are intentionally ignored:
        1. rotation about the approach axis (target z / gripper self-spin);
        2. swapping jaws via a 180-degree flip around that same axis.
        """
        Rt = np.asarray(R_target, dtype=float).reshape(3, 3)
        Rc = np.asarray(R_cur, dtype=float).reshape(3, 3)
        z = Rt[:, 2]
        z = z / max(float(np.linalg.norm(z)), 1e-9)
        P = np.eye(3) - np.outer(z, z)

        R_pi = Rotation.from_rotvec(np.pi * z).as_matrix()
        cands = [Rt, Rt @ R_pi]
        best = None
        for R_ref in cands:
            rv = Rotation.from_matrix(R_ref @ Rc.T).as_rotvec()
            rv_proj = P @ rv
            norm = float(np.linalg.norm(rv_proj))
            if best is None or norm < best[0]:
                best = (norm, rv_proj)
        return best[1] if best is not None else np.zeros(3, dtype=float)

    def optimize(
        self,
        p_target: np.ndarray,     # (n,3) world-frame EE position targets
        R_target: np.ndarray,     # (n,3,3) world-frame EE rotation targets
        w_p: np.ndarray,          # (n,) position weights
        w_r: np.ndarray,          # (n,) orientation weights
        p_valid: Optional[np.ndarray] = None,  # (n,) bool; zeroes the position term where False
        q_init: Optional[np.ndarray] = None,   # (n,m) warm start (else q_neutral tiled)
    ) -> dict:
        n = len(p_target)
        m = self.kin.n_dof
        p_target = np.asarray(p_target, dtype=float).reshape(n, 3)
        R_target = np.asarray(R_target, dtype=float).reshape(n, 3, 3)
        w_p = np.asarray(w_p, dtype=float).reshape(n)
        w_r = np.asarray(w_r, dtype=float).reshape(n)
        if p_valid is None:
            p_valid = np.ones(n, dtype=bool)
        p_valid = np.asarray(p_valid, dtype=bool).reshape(n)
        wp_eff = w_p * p_valid  # drop position term on invalid frames

        if q_init is None:
            q_init = np.tile(self.kin.q_neutral, (n, 1))
        q_init = np.clip(np.asarray(q_init, dtype=float).reshape(n, m),
                         self.kin.q_min, self.kin.q_max)

        cfg = self.cfg
        ws = float(cfg.w_smooth)
        wreg = np.broadcast_to(
            np.asarray(cfg.w_reg, dtype=float).reshape(-1), (m,)
        ).astype(float)
        wvel = float(cfg.w_vel)
        q_neutral = self.kin.q_neutral

        # Residual layout (row offsets).
        n_pos = 3 * n
        n_ori = 3 * n
        n_smooth = m * max(n - 2, 0)
        n_reg = m * n
        n_vel = m * max(n - 1, 0) if wvel > 0 else 0
        off_pos, off_ori = 0, n_pos
        off_smooth = off_ori + n_ori
        off_reg = off_smooth + n_smooth
        off_vel = off_reg + n_reg
        n_res = off_vel + n_vel

        def _fk_all(Q):
            pos = np.empty((n, 3)); Rs = np.empty((n, 3, 3))
            for t in range(n):
                pos[t], Rs[t] = self.kin.fk(Q[t])
            return pos, Rs

        def residual(x):
            Q = x.reshape(n, m)
            pos, Rs = _fk_all(Q)
            r = np.zeros(n_res)
            # position + orientation fidelity
            for t in range(n):
                r[off_pos + 3 * t: off_pos + 3 * t + 3] = wp_eff[t] * (pos[t] - p_target[t])
                if w_r[t] != 0.0:
                    rv = self._project_parallel_jaw_rotvec(R_target[t], Rs[t])
                    r[off_ori + 3 * t: off_ori + 3 * t + 3] = w_r[t] * rv
            # 2nd-order smoothness on interior frames
            if n_smooth:
                acc = Q[2:] - 2.0 * Q[1:-1] + Q[:-2]     # (n-2, m)
                r[off_smooth: off_smooth + n_smooth] = (ws * acc).ravel()
            # posture regularizer
            r[off_reg: off_reg + n_reg] = (wreg * (Q - q_neutral)).ravel()
            # soft velocity limit (one-sided)
            if n_vel:
                dq = Q[1:] - Q[:-1]                       # (n-1, m)
                over = np.maximum(np.abs(dq) - cfg.dq_max, 0.0) * np.sign(dq)
                r[off_vel: off_vel + n_vel] = (wvel * over).ravel()
            return r

        def jac(x):
            Q = x.reshape(n, m)
            J = lil_matrix((n_res, n * m))
            for t in range(n):
                jacp, jacr = self.kin.jac(Q[t])          # (3,m), (3,m)
                c0 = t * m
                if wp_eff[t] != 0.0:
                    J[off_pos + 3 * t: off_pos + 3 * t + 3, c0:c0 + m] = wp_eff[t] * jacp
                if w_r[t] != 0.0:
                    z = np.asarray(R_target[t], dtype=float)[:, 2]
                    z = z / max(float(np.linalg.norm(z)), 1e-9)
                    P = np.eye(3) - np.outer(z, z)
                    # d/dq rotvec(R_tgt R_cur^T) ≈ -jacr; drop the free spin axis.
                    J[off_ori + 3 * t: off_ori + 3 * t + 3, c0:c0 + m] = -w_r[t] * (P @ jacr)
            # smoothness: constant tri-diagonal blocks (+1, -2, +1)
            for k in range(max(n - 2, 0)):
                r0 = off_smooth + k * m
                for jj in range(m):
                    J[r0 + jj, (k + 0) * m + jj] = ws * 1.0
                    J[r0 + jj, (k + 1) * m + jj] = ws * -2.0
                    J[r0 + jj, (k + 2) * m + jj] = ws * 1.0
            # regularizer: block diagonal
            for t in range(n):
                r0 = off_reg + t * m
                for jj in range(m):
                    J[r0 + jj, t * m + jj] = wreg[jj]
            # velocity soft penalty (subgradient where active)
            if n_vel:
                dq = Q[1:] - Q[:-1]
                active = np.abs(dq) > cfg.dq_max
                s = np.sign(dq)
                for k in range(n - 1):
                    r0 = off_vel + k * m
                    for jj in range(m):
                        if active[k, jj]:
                            J[r0 + jj, (k + 1) * m + jj] = wvel * s[k, jj]
                            J[r0 + jj, (k + 0) * m + jj] = -wvel * s[k, jj]
            return J.tocsr()

        lb = np.tile(self.kin.q_min, n)
        ub = np.tile(self.kin.q_max, n)
        x0 = np.clip(q_init.ravel(), lb, ub)

        r0 = residual(x0)
        cost0 = 0.5 * float(r0 @ r0)
        sol = least_squares(
            residual, x0, jac=jac, bounds=(lb, ub), method="trf",
            max_nfev=cfg.max_nfev, xtol=cfg.xtol, ftol=cfg.ftol, verbose=cfg.verbose,
        )
        Q = sol.x.reshape(n, m)

        # Report per-frame tracking residuals (unweighted, physical units).
        # Orientation uses the *symmetry-reduced* parallel-jaw error (approach-
        # axis self-spin + 180-degree jaw flip ignored) so the metric matches
        # what the optimizer penalizes; a full-SO(3) norm would inflate to ~pi
        # for freely-spun-but-physically-identical grasps.
        pos, Rs = _fk_all(Q)
        pos_err = np.linalg.norm(pos - p_target, axis=1)               # meters
        ori_err = np.array([
            float(np.linalg.norm(self._project_parallel_jaw_rotvec(R_target[t], Rs[t])))
            for t in range(n)
        ])                                                             # radians
        jerk = (Q[2:] - 2 * Q[1:-1] + Q[:-2]) if n > 2 else np.zeros((0, m))

        logger.info(
            "[stageb] optimized T=%d: cost %.4g -> %.4g | pos_err mean=%.4f max=%.4f m | "
            "ori_err mean=%.3f max=%.3f rad | nfev=%d success=%s",
            n, cost0, float(sol.cost), float(pos_err.mean()), float(pos_err.max()),
            float(ori_err.mean()), float(ori_err.max()), int(sol.nfev), bool(sol.success),
        )
        return {
            "q": Q.astype(np.float32),
            "pos_err": pos_err.astype(np.float32),
            "ori_err": ori_err.astype(np.float32),
            "ee_pos_world": pos.astype(np.float32),
            "ee_R_world": Rs.astype(np.float32),
            "cost_initial": np.float32(cost0),
            "cost_final": np.float32(sol.cost),
            "jerk_rms": np.float32(np.sqrt(np.mean(jerk ** 2)) if jerk.size else 0.0),
            "nfev": np.int64(sol.nfev),
            "success": bool(sol.success),
        }
