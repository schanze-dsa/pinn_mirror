#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
boundary_conditions.py
----------------------
Penalty energy for displacement (Dirichlet) boundary conditions.

Core model (per boundary sample k):
    r_k = M_k ⊙ ( u(X_k) - u_target_k )                      # M_k is {0,1}^3 dof mask
    E_bc = 0.5 * alpha * Σ w_k * || r_k ||^2

Where:
    - X_bc      : (N,3) coordinates of boundary samples
    - dof_mask  : (N,3) boolean/0-1 mask; e.g., [1,1,1] for ENCASTRE; [1,0,0] fix ux only
    - u_target  : (N,3) target displacements (often zeros)
    - w_bc      : (N,) area/line weights (can be uniform 1.0 if unknown)
    - alpha     : penalty stiffness (large -> "hard" constraint)
    - extra_w   : (N,) optional multiplicative weights for Weighted PINN

Usage:
    bc = BoundaryPenalty(BoundaryConfig(alpha=1e3, dtype="float32"))
    bc.build_from_numpy(X_bc, dof_mask, u_target, w_bc, extra_w=None)
    E_bc, stats = bc.energy(u_fn, params)

Notes:
- This module does NOT parse Abaqus *Boundary directly; mapping from INP to (X_bc, mask, u_target, w)
  will be handled by a higher-level sampler/adapter.
- Differentiable w.r.t. model parameters through u_fn.

Author: you
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import numpy as np
import tensorflow as tf


# -----------------------------
# Config
# -----------------------------

@dataclass
class BoundaryConfig:
    alpha: float = 1.0e3     # penalty stiffness
    dtype: str = "float32"
    mode: str = "penalty"    # 'penalty' | 'hard' | 'alm'
    mu: float = 1.0e3        # ALM 增广系数（mode='alm' 时生效）


# -----------------------------
# Operator
# -----------------------------

class BoundaryPenalty:
    """
    Displacement boundary penalty energy.

    Similar to :class:`TiePenalty`, the constructor accepts a superset of the
    modern configuration so that older call-sites (e.g. ``BoundaryPenalty(alpha
    =..., dof1=..., dof2=..., kind=...)``) keep working.  ``attach_ties_bcs``
    still relies on that signature when constructing penalties from Abaqus
    ``*Boundary`` definitions.
    """

    def __init__(
        self,
        cfg: Optional[BoundaryConfig] = None,
        *,
        alpha: Optional[float] = None,
        dtype: Optional[str] = None,
        mode: Optional[str] = None,
        dof1: Optional[int] = None,
        dof2: Optional[int] = None,
        kind: Optional[str] = None,
        **_legacy_kwargs,
    ):
        if cfg is None:
            cfg = BoundaryConfig()

        if alpha is not None:
            cfg = BoundaryConfig(alpha=alpha, dtype=cfg.dtype, mode=cfg.mode)
        if dtype is not None:
            cfg = BoundaryConfig(alpha=cfg.alpha, dtype=dtype, mode=cfg.mode)
        if mode is not None:
            cfg = BoundaryConfig(alpha=cfg.alpha, dtype=cfg.dtype, mode=mode)

        self.cfg = cfg
        self.dtype = tf.float32 if self.cfg.dtype == "float32" else tf.float64

        # Store legacy boundary metadata so the ``build`` alias can reconstruct
        # masks similar to the historical implementation.
        self._legacy_dof1 = dof1
        self._legacy_dof2 = dof2
        self._legacy_kind = kind.upper() if isinstance(kind, str) else None

        # per-batch tensors
        self.X: Optional[tf.Tensor] = None          # (N,3)
        self.mask: Optional[tf.Tensor] = None       # (N,3) 0/1
        self.u_target: Optional[tf.Tensor] = None   # (N,3)
        self.w: Optional[tf.Tensor] = None          # (N,)
        self.alpha = tf.Variable(self.cfg.alpha, dtype=self.dtype, trainable=False, name="alpha_bc")
        self.mu = tf.Variable(self.cfg.mu, dtype=self.dtype, trainable=False, name="mu_bc")
        self.lmbda: Optional[tf.Variable] = None     # (N,3) ALM 乘子

        self._N = 0

    # ---------- build ----------

    def build_from_numpy(
        self,
        X_bc: np.ndarray,
        dof_mask: np.ndarray,
        u_target: Optional[np.ndarray],
        w_bc: Optional[np.ndarray],
        extra_w: Optional[np.ndarray] = None,
    ):
        """
        Prepare tensors from NumPy arrays.
        - If u_target is None, defaults to zeros.
        - If w_bc is None, defaults to ones.
        """
        assert X_bc.ndim == 2 and X_bc.shape[1] == 3, "X_bc must be (N,3)"
        assert dof_mask.shape == X_bc.shape, "dof_mask must be (N,3) 0/1 or bool"

        N = X_bc.shape[0]
        self.X = tf.convert_to_tensor(X_bc, dtype=self.dtype)
        self.mask = tf.convert_to_tensor(dof_mask, dtype=self.dtype)

        if u_target is None:
            self.u_target = tf.zeros((N, 3), dtype=self.dtype)
        else:
            assert u_target.shape == X_bc.shape
            self.u_target = tf.convert_to_tensor(u_target, dtype=self.dtype)

        if w_bc is None:
            self.w = tf.ones((N,), dtype=self.dtype)
        else:
            assert w_bc.shape[0] == N
            self.w = tf.convert_to_tensor(w_bc, dtype=self.dtype)

        if extra_w is not None:
            ew = tf.convert_to_tensor(extra_w, dtype=self.dtype)
            self.w = self.w * ew

        self._N = N

        # 初始化或调整 ALM 乘子形状
        if self.lmbda is None or tuple(self.lmbda.shape) != (N, 3):
            self.lmbda = tf.Variable(
                tf.zeros((N, 3), dtype=self.dtype), trainable=False, name="lambda_bc"
            )

    def reset_for_new_batch(self):
        self.X = self.mask = self.u_target = self.w = None
        self._N = 0

    # ---- compatibility aliases -------------------------------------------------

    def _legacy_mask(self, N: int) -> np.ndarray:
        mask = np.zeros((N, 3), dtype=np.float32)

        if self._legacy_kind == "ENCASTRE":
            mask.fill(1.0)
            return mask

        if self._legacy_dof1 is None:
            return mask

        d1 = int(self._legacy_dof1)
        d2 = int(self._legacy_dof2) if self._legacy_dof2 is not None else d1

        for dof in range(min(d1, d2), max(d1, d2) + 1):
            if 1 <= dof <= 3:
                mask[:, dof - 1] = 1.0

        return mask

    def build(
        self,
        X_bc: np.ndarray,
        dof_mask: Optional[np.ndarray] = None,
        u_target: Optional[np.ndarray] = None,
        w_bc: Optional[np.ndarray] = None,
        extra_w: Optional[np.ndarray] = None,
    ):
        """Backward-compatible build helper used by ``attach_ties_bcs``."""

        X_np = np.asarray(X_bc)
        if dof_mask is None:
            dof_mask = self._legacy_mask(X_np.shape[0])
        self.build_from_numpy(X_np, dof_mask, u_target, w_bc, extra_w)

    # ---------- energy ----------

    def energy(self, u_fn, params=None) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Compute boundary energy. 支持两种模式：
            - penalty: E_bc = 0.5 * alpha * Σ w * || M ⊙ (u - u_tgt) ||^2
            - hard   : 直接将有约束的自由度投影为 u_tgt，返回 0 能量，仅记录残差统计
        """
        if self.X is None or self.mask is None or self.u_target is None or self.w is None:
            raise RuntimeError("[BoundaryPenalty] build_from_numpy must be called first.")

        u = u_fn(self.X, params)                                # (N,3)
        r_raw = (u - self.u_target) * self.mask                # (N,3)
        r_raw2 = tf.reduce_sum(r_raw * r_raw, axis=1)          # (N,)

        mode = (self.cfg.mode or "penalty").lower()
        if mode == "hard":
            # 直接投影到目标位移：u_proj = u - stop_grad(r_raw)
            # 这样受限自由度被强制为 u_target，且梯度对未约束自由度仍然透明。
            u = u - tf.stop_gradient(r_raw)
            r = (u - self.u_target) * self.mask  # (N,3) -> 理论上全为 0
            r2 = tf.reduce_sum(r * r, axis=1)
            E_bc = tf.cast(0.0, self.dtype)
        elif mode == "alm":
            if self.lmbda is None:
                self.lmbda = tf.Variable(
                    tf.zeros_like(self.mask), trainable=False, name="lambda_bc"
                )
            lmbda = tf.cast(self.lmbda, self.dtype)
            mu = tf.cast(self.mu, self.dtype)
            r = r_raw
            r2 = r_raw2
            E_bc = tf.reduce_sum(self.w[:, None] * (lmbda * r + 0.5 * mu * r * r))
        else:
            r = r_raw
            r2 = r_raw2
            E_bc = 0.5 * self.alpha * tf.reduce_sum(self.w * r2)    # scalar

        # stats 仍然报告“投影前”的残差，以便监控硬约束的偏离
        abs_r = tf.sqrt(tf.maximum(r_raw2, tf.cast(0.0, self.dtype)))  # (N,)
        stats = {
            "bc_rms": tf.sqrt(tf.reduce_mean(abs_r * abs_r) + 1e-20),
            "bc_max": tf.reduce_max(abs_r),
        }
        return E_bc, stats

    def update_multipliers(self, u_fn, params=None):
        """ALM 外层更新：λ ← λ + μ r，仅在 mode='alm' 时启用。"""
        if (self.cfg.mode or "penalty").lower() != "alm":
            return
        if self.X is None or self.mask is None or self.u_target is None:
            return
        if self.lmbda is None:
            self.lmbda = tf.Variable(
                tf.zeros_like(self.mask), trainable=False, name="lambda_bc"
            )
        u = u_fn(self.X, params)
        r = (u - self.u_target) * self.mask
        mu = tf.cast(self.mu, r.dtype)
        self.lmbda.assign_add(mu * r)

    # ---------- setters ----------

    def set_alpha(self, alpha: float):
        self.alpha.assign(tf.cast(alpha, self.dtype))

    def multiply_weights(self, extra_w: np.ndarray):
        """Multiply current per-sample weights (Weighted PINN hook)."""
        ew = tf.convert_to_tensor(extra_w, dtype=self.dtype)
        self.w.assign(self.w * ew)
