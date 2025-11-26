#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
contact_normal_alm.py
---------------------
Augmented Lagrangian (ALM) normal-contact energy (frictionless, normal gap only).

本模块实现的是“能量型 + ALM”的法向接触项，可与 DFEM/ PINN 解耦使用：

- 接受 ContactMap 采样得到的接触点信息 (xs, xm, n, w_area)；
- 通过 u_fn(xs, params)、u_fn(xm, params) 获得节点位移；
- 计算几何间隙 g = ((xs + u(xs)) - (xm + u(xm))) · n；
- 使用平滑负部 φ(g) = softplus(-g; β) 近似 max(0, -g)；
- 法向接触能量：
      En = Σ w · [ λ · φ(g) + 0.5 · μ_n · φ(g)^2 ]
- 外层 ALM 更新：
      λ ← max(0, λ + η · μ_n · φ(g))

其中 η 为外部可控的步长（增强型拉格朗日法），默认 η=1。

支持特性：
- 支持通过 w_area 和额外的 extra_weights 组合成最终权重；
- energy(...) 中支持额外的 per-sample 权重 extra_weights，便于 weighted PINN；
- update_multipliers(...) 保持接口简洁，同时增加 step_scale 控制更新步长；
- 不依赖 Jacobian、与 DFEM 内核解耦，只要 u_fn(X, params) 接口一致即可。

典型用法（每个训练 step 内）::

    op = NormalContactALM(cfg)
    op.build_from_cat(contact_cat_dict)         # 或 build_from_numpy(...)
    En, stats_cn = op.energy(u_fn, params, extra_weights=cn_weights)
    # 反向传播 En
    if step % k == 0:
        op.update_multipliers(u_fn, params, step_scale=1.0)

Notes
-----
- 本算子是“每批次状态”，如果每步都重采样接触点，则每次 build_* 都会重置 λ。
  若希望 λ 跨 step 继承，可在 Trainer 层控制不重置 / 以某种方式 carry over。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #

def _to_tf(x, dtype=tf.float32):
    """Convert NumPy/TF input to a TF tensor with given dtype."""
    if isinstance(x, tf.Tensor):
        return tf.cast(x, dtype)
    return tf.convert_to_tensor(x, dtype=dtype)


def softplus_neg(x: tf.Tensor, beta: tf.Tensor) -> tf.Tensor:
    """
    Smooth negative-part: softplus(-x; beta) ≈ max(0, -x).

    具体形式：
        φ(x) = softplus(-β x) / β = (1/β) log(1 + exp(-β x))

    参数
    ----
    x : (...,) tensor
        gap 值 g。
    beta : scalar tensor
        越大则越接近硬的 max(0, -g)。
    """
    return tf.nn.softplus(-x * beta) / (beta + 1e-12)


@dataclass
class NormalALMConfig:
    """Hyperparameters for the normal-contact ALM operator."""
    mode: str = "penalty"          # "penalty" (softplus) or "alm"
    beta: float = 100.0         # softplus steepness (增大默认值以逼近硬接触)
    mu_n: float = 1.0e3         # ALM coefficient (normal penalty)
    enforce_nonneg_lambda: bool = True
    dtype: str = "float32"


class NormalContactALM:
    """
    Normal-contact ALM energy operator.

    接口保持与旧版兼容，但实现上调整为更清晰的“能量 + ALM”形式，并增加：

    - energy(..., extra_weights=None):
        额外 per-sample 权重入口，不改变内部 self.w，用于 weighted PINN；
    - update_multipliers(..., step_scale=1.0):
        外部可控的 ALM 步长因子 η。
    """

    # ------------------------------------------------------------------ #
    # 构造与批次构建
    # ------------------------------------------------------------------ #

    def __init__(self, cfg: Optional[NormalALMConfig] = None):
        self.cfg = cfg or NormalALMConfig()
        self.dtype = tf.float32 if self.cfg.dtype == "float32" else tf.float64

        # Per-batch tensors (TF)
        self.xs: Optional[tf.Tensor] = None   # (N,3) slave/sample points
        self.xm: Optional[tf.Tensor] = None   # (N,3) master projection points
        self.n:  Optional[tf.Tensor] = None   # (N,3) normals (unit, may be auto-flipped)
        self.w:  Optional[tf.Tensor] = None   # (N,) base weights (e.g., area)

        # State (multipliers λ_n)
        self.lmbda: Optional[tf.Variable] = None  # (N,)
        self._built_N: int = 0

        # Schedules/coeffs (scalars on TF)
        self.beta = tf.Variable(self.cfg.beta, dtype=self.dtype, trainable=False, name="beta_n")
        self.mu_n = tf.Variable(self.cfg.mu_n, dtype=self.dtype, trainable=False, name="mu_n")

        # Stats cache
        self._last_gap: Optional[tf.Tensor] = None
        self._auto_flip_done: bool = False

    # ---------- building ----------

    def build_from_numpy(
        self,
        xs: np.ndarray,
        xm: np.ndarray,
        n: np.ndarray,
        w_area: np.ndarray,
        extra_weights: Optional[np.ndarray] = None,
        auto_orient: bool = True,
    ):
        """
        Initialize per-batch tensors from NumPy arrays.

        参数
        ----
        xs, xm : (N,3)
            分别为从从表面采样的点（slave）及其在主表面上的投影点（master）。
        n : (N,3)
            主表面的单位法向量。
        w_area : (N,)
            从从表面采样得到的面积权重。
        extra_weights : (N,), optional
            额外权重（例如 IRLS / edge weighting），会在 build 阶段乘到 w_area 上。
            若希望每一步动态变化的权重，请使用 energy(..., extra_weights=...)。
        auto_orient : bool
            是否在零位移状态下根据 g0 = (xs - xm)·n 的中位数自动翻转法向。
        """
        assert xs.shape == xm.shape and xs.shape[1] == 3
        assert n.shape == xs.shape and w_area.shape[0] == xs.shape[0]

        Xs = _to_tf(xs, self.dtype)
        Xm = _to_tf(xm, self.dtype)
        Nn = _to_tf(n,  self.dtype)
        W  = _to_tf(w_area, self.dtype)

        if extra_weights is not None:
            W = W * _to_tf(extra_weights, self.dtype)

        # Normalize normals to unit (defensive)
        Nn = Nn / (tf.norm(Nn, axis=1, keepdims=True) + tf.cast(1e-12, self.dtype))

        # Assign
        self.xs, self.xm, self.n, self.w = Xs, Xm, Nn, W
        self._built_N = int(Xs.shape[0])

        # (Re)init multipliers (per-batch state)
        self.lmbda = tf.Variable(
            tf.zeros((self._built_N,), dtype=self.dtype),
            trainable=False,
            name="lambda_n",
        )

        # Auto-orient normals (once per batch): ensure median((xs-xm)·n) >= 0 at zero displacement
        if auto_orient:
            self._auto_orient_normals()

        self._last_gap = None

    def build_from_cat(
        self,
        cat: Dict[str, np.ndarray],
        extra_weights: Optional[np.ndarray] = None,
        auto_orient: bool = True,
    ):
        """
        从 ContactMap.concatenate() 返回的字典构建：
            cat['xs'], cat['xm'], cat['n'], cat['w_area']  (N,*)

        其中 extra_weights 同 build_from_numpy，用于一次性的静态重权。
        """
        self.build_from_numpy(
            cat["xs"],
            cat["xm"],
            cat["n"],
            cat["w_area"],
            extra_weights=extra_weights,
            auto_orient=auto_orient,
        )

    def _auto_orient_normals(self):
        """Flip all normals if median zero-displacement gap is negative."""
        # g0 = (xs - xm) · n
        xs = tf.cast(self.xs, self.dtype)
        xm = tf.cast(self.xm, self.dtype)
        n = tf.cast(self.n, self.dtype)

        g0 = tf.reduce_sum((xs - xm) * n, axis=1)
        med = tfp_median(g0)
        flip = med < tf.cast(0.0, self.dtype)

        def _flip():
            self.n.assign(-self.n)
            return tf.constant(0, dtype=tf.int32)

        def _noflip():
            return tf.constant(0, dtype=tf.int32)

        _ = tf.cond(flip, _flip, _noflip)
        self._auto_flip_done = True

    # ------------------------------------------------------------------ #
    # 核心计算：gap / energy / multipliers
    # ------------------------------------------------------------------ #

    def _gap(self, u_fn, params=None) -> tf.Tensor:
        """
        Compute signed gap:
            g = ((xs + u(xs)) - (xm + u(xm))) · n

        返回：
            g: (N,) tensor
        """
        xs = tf.cast(self.xs, self.dtype)
        xm = tf.cast(self.xm, self.dtype)
        n = tf.cast(self.n,  self.dtype)

        u_s = tf.cast(_ensure_2d(u_fn(xs, params)), self.dtype)
        u_m = tf.cast(_ensure_2d(u_fn(xm, params)), self.dtype)

        g = tf.reduce_sum(((xs + u_s) - (xm + u_m)) * n, axis=1)
        self._last_gap = g
        return g

    def energy(
        self,
        u_fn,
        params=None,
        extra_weights: Optional[tf.Tensor] = None,
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Compute ALM normal-contact energy (scalar) and stats (dict):

        - 若 mode="alm":
              En = Σ w_eff · [ λ · φ(g) + 0.5 · μ_n · φ(g)^2 ]
        - 若 mode="penalty":
              En = Σ w_eff · 0.5 · μ_n · φ(g)^2

        其中：
            - w_eff = w * extra_weights（若 extra_weights 为 None，则 w_eff = w）；
            - φ(g) = softplus(-g; β)。

        参数
        ----
        u_fn : callable
            u_fn(X, params) -> (N,3)，DFEM/ PINN 统一的位移预测接口。
        params : Any
            传给 u_fn 的参数，可为字典或 dataclass。
        extra_weights : tensor or None
            形状 (N,)；若提供，则作为本次 energy 调用的额外权重，
            不会永久写回 self.w。可用于 weighted PINN 中基于残差
            的自适应权重。
        """
        g = self._gap(u_fn, params)
        phi = softplus_neg(g, self.beta)         # (N,)

        # 有效权重 w_eff：保留 self.w 作为几何/面积基权重，额外权重只在本次调用中生效
        w_eff = self.w
        if extra_weights is not None:
            w_eff = w_eff * tf.cast(extra_weights, self.dtype)

        p_eff = self._compute_effective_pressure(phi)

        if self.cfg.mode.lower() == "alm":
            En = tf.reduce_sum(w_eff * (self.lmbda * phi + 0.5 * self.mu_n * phi * phi))
        else:
            En = tf.reduce_sum(w_eff * (0.5 * self.mu_n * phi * phi))

        # Stats for logging
        stats = {
            "cn_min_gap": tf.reduce_min(g),
            "cn_mean_gap": tf.reduce_mean(g),
            "cn_pen_ratio": tf.reduce_mean(tf.cast(phi > 0.0, self.dtype)),  # fraction with penetration (phi>0)
            "cn_phi_mean": tf.reduce_mean(phi),
            "cn_mean_weight": tf.reduce_mean(w_eff),
            "cn_mean_pressure": tf.reduce_mean(p_eff),
        }
        return En, stats

    @tf.function(jit_compile=False)
    def update_multipliers(
        self,
        u_fn,
        params=None,
        step_scale: float = 1.0,
    ):
        """
        Outer-loop ALM update (not part of gradient path):

            λ ← max(0, λ + η · μ_n · φ(g)),  其中 η = step_scale。

        参数
        ----
        u_fn, params : 同 energy(...)
        step_scale : float
            增强型拉格朗日的步长因子；可以在 Trainer 中根据步数递增/递减。
        """
        if self.cfg.mode.lower() != "alm":
            # 纯软化罚函数模式下不需要更新乘子，但仍刷新 gap 统计
            self._gap(u_fn, params)
            return

        g = self._gap(u_fn, params)
        phi = softplus_neg(g, self.beta)

        eta = tf.cast(step_scale, self.dtype)
        new_lmbda = self.lmbda + eta * self.mu_n * phi

        if self.cfg.enforce_nonneg_lambda:
            new_lmbda = tf.maximum(new_lmbda, tf.cast(0.0, self.dtype))
        self.lmbda.assign(new_lmbda)

    # ------------------------------------------------------------------ #
    # Schedules / setters / misc
    # ------------------------------------------------------------------ #

    def set_beta(self, beta: float):
        """Set softplus steepness β."""
        self.beta.assign(tf.cast(beta, self.dtype))

    def set_mu_n(self, mu_n: float):
        """Set ALM coefficient μ_n."""
        self.mu_n.assign(tf.cast(mu_n, self.dtype))

    def multiply_weights(self, extra_w: np.ndarray):
        """
        Permanently multiply current base weights self.w by extra_w (e.g., IRLS / edge weighting).

        若只是想在某次 energy 调用中使用临时权重，请使用
        energy(..., extra_weights=...)，而不是本函数。
        """
        ew = _to_tf(extra_w, self.dtype)
        self.w.assign(self.w * ew)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _compute_effective_pressure(self, phi: tf.Tensor) -> tf.Tensor:
        """p_eff = max(0, λ + μ_n φ)（ALM）或 μ_n φ（softplus penalty）。"""
        p = self.mu_n * phi
        if self.cfg.mode.lower() == "alm":
            p = self.lmbda + p
        return tf.maximum(p, tf.cast(0.0, self.dtype))

    def effective_normal_pressure(self, u_fn, params=None) -> tf.Tensor:
        """Expose effective compressive pressure for friction算子复用。"""
        g = self._gap(u_fn, params)
        phi = softplus_neg(g, self.beta)
        return self._compute_effective_pressure(phi)

    def reset_for_new_batch(self):
        """
        Clear internal tensors/state so the operator can be rebuilt with a new batch.

        注意：这会丢弃当前批次的 λ，需要在下一次 build_* 之后重新优化。
        """
        self.xs = self.xm = self.n = self.w = None
        self.lmbda = None
        self._built_N = 0
        self._last_gap = None
        self._auto_flip_done = False


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def tfp_median(x: tf.Tensor) -> tf.Tensor:
    """
    Quick median without depending on tfp: sort and take middle.
    Assumes 1-D x.
    """
    x_sorted = tf.sort(x)
    n = tf.shape(x_sorted)[0]
    mid = n // 2
    # If even, average two middles
    even = (n % 2) == 0
    mid_val = tf.cond(
        even,
        lambda: 0.5 * (x_sorted[mid - 1] + x_sorted[mid]),
        lambda: x_sorted[mid],
    )
    return tf.cast(mid_val, x.dtype)


def _ensure_2d(u: tf.Tensor) -> tf.Tensor:
    """Ensure u has shape (N,3)."""
    if u.shape.rank == 1:
        return tf.reshape(u, (-1, 3))
    return u
