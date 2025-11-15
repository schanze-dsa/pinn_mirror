#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tie_constraints.py
------------------
Penalty energy for Tie (surface-to-surface displacement continuity).

Mathematical form (per sample k):
    r_k = M_k ⊙ [ (x_s + u(x_s)) - (x_m + u(x_m)) ]         # M_k ∈ {0,1}^3, DOF mask
    E_tie = 0.5 * alpha * Σ w_k * || r_k ||^2

Inputs per batch (NumPy arrays):
    xs        : (N,3) slave points on tie slave surface
    xm        : (N,3) master points (e.g., closest points on tie master surface)
    w_area    : (N,) area weights on the slave side (Monte Carlo)
    dof_mask  : (N,3) optional {0,1} mask per sample (default: all ones)
    extra_w   : (N,) optional multiplicative weights (for Weighted PINN)

Typical usage:
    tie = TiePenalty(TieConfig(alpha=1e3, dtype="float32"))
    tie.build_from_numpy(xs, xm, w_area, dof_mask=None, extra_w=None)
    E_tie, stats = tie.energy(u_fn, params)

Notes:
- This module focuses on the penalty formulation for convenience and stability.
- Building xs/xm/w_area can reuse the sampling+projection pipeline used for contact pairs
  (e.g., mesh.contact_pairs.build_contact_pair_data with a dedicated specs list for *Tie pairs).

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
class TieConfig:
    alpha: float = 1.0e3      # penalty stiffness for tie (make it large to emulate hard constraint)
    dtype: str = "float32"


# -----------------------------
# Operator
# -----------------------------

class TiePenalty:
    """
    Tie (displacement continuity) penalty energy.

    The constructor is intentionally flexible so legacy call-sites such as
    ``TiePenalty(alpha=..., dtype=...)`` continue to function after the module
    rewrite.  ``attach_ties_bcs.py`` still instantiates the class with
    positional/keyword arguments instead of a ``TieConfig`` instance, therefore
    we accept the most common keywords and fold them into the dataclass config
    before proceeding with the modern initialisation path.
    """

    def __init__(
        self,
        cfg: Optional[TieConfig] = None,
        *,
        alpha: Optional[float] = None,
        dtype: Optional[str] = None,
        **_legacy_kwargs,
    ):
        if cfg is None:
            cfg = TieConfig()

        # Legacy constructors used ``TiePenalty(alpha=..., dtype=...)``.  Allow
        # those keywords to override the dataclass defaults when present.
        if alpha is not None:
            cfg = TieConfig(alpha=alpha, dtype=cfg.dtype)
        if dtype is not None:
            cfg = TieConfig(alpha=cfg.alpha, dtype=dtype)

        self.cfg = cfg
        self.dtype = tf.float32 if self.cfg.dtype == "float32" else tf.float64

        # per-batch tensors
        self.xs: Optional[tf.Tensor] = None       # (N,3)
        self.xm: Optional[tf.Tensor] = None       # (N,3)
        self.w: Optional[tf.Tensor] = None        # (N,)
        self.mask: Optional[tf.Tensor] = None     # (N,3) 0/1
        self.alpha = tf.Variable(self.cfg.alpha, dtype=self.dtype, trainable=False, name="alpha_tie")

        self._N = 0

    # ---------- build ----------

    def build_from_numpy(
        self,
        xs: np.ndarray,
        xm: np.ndarray,
        w_area: np.ndarray,
        dof_mask: Optional[np.ndarray] = None,
        extra_w: Optional[np.ndarray] = None,
    ):
        """
        Prepare per-batch tensors from NumPy arrays.
        """
        assert xs.shape == xm.shape and xs.shape[1] == 3, "xs/xm must be (N,3)"
        assert w_area.shape[0] == xs.shape[0], "w_area must be (N,)"

        self.xs = tf.convert_to_tensor(xs, dtype=self.dtype)
        self.xm = tf.convert_to_tensor(xm, dtype=self.dtype)
        self.w  = tf.convert_to_tensor(w_area, dtype=self.dtype)

        if dof_mask is None:
            self.mask = tf.ones_like(self.xs, dtype=self.dtype)
        else:
            assert dof_mask.shape == xs.shape, "dof_mask must be (N,3)"
            self.mask = tf.convert_to_tensor(dof_mask, dtype=self.dtype)

        if extra_w is not None:
            ew = tf.convert_to_tensor(extra_w, dtype=self.dtype)
            self.w = self.w * ew

        self._N = int(xs.shape[0])

    def reset_for_new_batch(self):
        self.xs = self.xm = self.w = self.mask = None
        self._N = 0

    # ---- compatibility aliases -------------------------------------------------

    def build(
        self,
        xs: np.ndarray,
        xm: np.ndarray,
        w_area: np.ndarray,
        dof_mask: Optional[np.ndarray] = None,
        extra_w: Optional[np.ndarray] = None,
    ):
        """Backward-compatible alias used by older sampling pipelines."""

        self.build_from_numpy(xs, xm, w_area, dof_mask=dof_mask, extra_w=extra_w)

    def build_from_points(
        self,
        xs: np.ndarray,
        xm: np.ndarray,
        w_area: np.ndarray,
        dof_mask: Optional[np.ndarray] = None,
        extra_w: Optional[np.ndarray] = None,
    ):
        """Alias for libraries that expose a ``build_from_points`` API."""

        self.build_from_numpy(xs, xm, w_area, dof_mask=dof_mask, extra_w=extra_w)

    # ---------- energy ----------

    def energy(self, u_fn, params=None) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Compute tie penalty energy:
            E_tie = 0.5 * alpha * Σ w * || M ⊙ [ (xs+u(xs)) - (xm+u(xm)) ] ||^2
        """
        if self.xs is None or self.xm is None or self.w is None or self.mask is None:
            raise RuntimeError("[TiePenalty] build_from_numpy must be called first.")

        us = u_fn(self.xs, params)                          # (N,3)
        um = u_fn(self.xm, params)                          # (N,3)
        r = ((self.xs + us) - (self.xm + um)) * self.mask   # (N,3)
        r2 = tf.reduce_sum(r * r, axis=1)                   # (N,)
        E_tie = 0.5 * self.alpha * tf.reduce_sum(self.w * r2)

        # stats
        abs_r = tf.sqrt(tf.maximum(r2, tf.cast(0.0, self.dtype)))
        stats = {
            "tie_rms": tf.sqrt(tf.reduce_mean(abs_r * abs_r) + 1e-20),
            "tie_max": tf.reduce_max(abs_r),
        }
        return E_tie, stats

    # ---------- setters ----------

    def set_alpha(self, alpha: float):
        self.alpha.assign(tf.cast(alpha, self.dtype))

    def multiply_weights(self, extra_w: np.ndarray):
        ew = tf.convert_to_tensor(extra_w, dtype=self.dtype)
        self.w.assign(self.w * ew)
