#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
contact_operator.py
-------------------
Unified contact operator that wraps:
  - NormalContactALM   (normal, frictionless, ALM)
  - FrictionContactALM (tangential, Coulomb / smooth friction, ALM)

典型用法（每个训练 batch）::

    op = ContactOperator(cfg)
    op.build_from_cat(cat_dict, extra_weights=..., auto_orient=True)

    # 在损失里：
    E_c, parts_c, stats_cn, stats_ct = op.energy(u_fn, params)
    # parts_c: {"E_n": En, "E_t": Et}
    # stats_cn: 法向残差 / 间隙统计
    # stats_ct: 摩擦 stick/slip 比例、τ 等

    # 在外层 ALM 更新时（比如每 K 步一次）：
    op.update_multipliers(u_fn, params)

Weighted PINN:
    - extra_weights: np.ndarray, shape (N,)
    - 会在 build_from_cat 时与面积权重相乘，用于法向和摩擦两部分能量
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf

from .contact_normal_alm import NormalContactALM, NormalALMConfig
from .contact_friction_alm import FrictionContactALM, FrictionALMConfig


# -----------------------------
# Config for the unified operator
# -----------------------------

@dataclass
class ContactOperatorConfig:
    # 子模块超参数
    normal: NormalALMConfig = NormalALMConfig(beta=50.0, mu_n=1.0e3, dtype="float32")
    friction: FrictionALMConfig = FrictionALMConfig(
        mu_f=0.15, k_t=5.0e2, mu_t=1.0e3, dtype="float32"
    )

    # ALM 外层更新节奏：若 <=0，则每一步都更新；否则每 update_every_steps 步更新一次
    update_every_steps: int = 150

    # 摩擦相关选项（可选，用于和 FrictionContactALM 协同）
    use_smooth_friction: bool = False      # True 时偏向使用 C^1 平滑摩擦伪势
    fric_weight_mode: str = "residual"     # 后续在 Trainer 里可根据该字段选择加权策略

    # 精度
    dtype: str = "float32"


class ContactOperator:
    """
    Combine normal-ALM and friction-ALM into a single, convenient interface.

    关键接口：
        - build_from_cat(cat, extra_weights=None, auto_orient=True)
        - energy(u_fn, params=None)
        - update_multipliers(u_fn, params=None)
        - multiply_weights(extra_w)  # runtime 再叠乘一层权重（比如 IRLS）
    """

    def __init__(self, cfg: Optional[ContactOperatorConfig] = None):
        self.cfg = cfg or ContactOperatorConfig()
        self.dtype = tf.float32 if self.cfg.dtype == "float32" else tf.float64

        # sub-operators
        self.normal = NormalContactALM(self.cfg.normal)
        self.friction = FrictionContactALM(self.cfg.friction)
        self.friction.link_normal(self.normal)

        # 如果 FrictionContactALM 已经实现平滑摩擦开关，这里做一次同步（有则用，无则跳过）
        if hasattr(self.friction, "set_smooth_friction"):
            try:
                self.friction.set_smooth_friction(self.cfg.use_smooth_friction)  # type: ignore[attr-defined]
            except Exception:
                pass
        elif hasattr(self.friction, "cfg") and hasattr(self.friction.cfg, "use_smooth_friction"):
            try:
                self.friction.cfg.use_smooth_friction = self.cfg.use_smooth_friction  # type: ignore[assignment]
            except Exception:
                pass

        # bookkeeping
        self._built: bool = False
        self._N: int = 0
        self._step: int = 0
        self._meta: Dict[str, np.ndarray] = {}

    # ---------- build per batch ----------

    def build_from_cat(
        self,
        cat: Dict[str, np.ndarray],
        extra_weights: Optional[np.ndarray] = None,
        auto_orient: bool = True,
    ):
        """
        Build both normal and friction operators from concatenated contact arrays.

        Parameters
        ----------
        cat : dict
            必须包含键: "xs", "xm", "n", "t1", "t2", "w_area"
        extra_weights : np.ndarray, shape (N,), optional
            额外的加权（如 weighted PINN 或 IRLS 权重），会与 w_area 相乘。
        auto_orient : bool
            若为 True，normal ALM 会在 build 阶段根据零位移间隙自动翻转法向。
        """
        required = ["xs", "xm", "n", "t1", "t2", "w_area"]
        for k in required:
            if k not in cat:
                raise KeyError(f"[ContactOperator] cat missing key '{k}'")

        # normal
        self.normal.build_from_cat(
            {"xs": cat["xs"], "xm": cat["xm"], "n": cat["n"], "w_area": cat["w_area"]},
            extra_weights=extra_weights,
            auto_orient=auto_orient,
        )

        # friction (linked to normal)
        self.friction.build_from_cat(
            {"xs": cat["xs"], "xm": cat["xm"], "t1": cat["t1"], "t2": cat["t2"], "w_area": cat["w_area"]},
            extra_weights=extra_weights,
        )

        self._N = int(cat["xs"].shape[0])
        self._built = True
        self._step = 0
        keep_keys = ["pair_id", "slave_tri_idx", "master_tri_idx", "w_area", "xs", "xm", "n", "t1", "t2"]
        self._meta = {k: v for k, v in cat.items() if k in keep_keys}

    def reset_for_new_batch(self):
        """Clear internal state so you can rebuild with a new set of contact samples."""
        self.normal.reset_for_new_batch()
        self.friction.reset_for_new_batch()
        self._built = False
        self._N = 0
        self._step = 0
        self._meta = {}

    # ---------- energy & update ----------

    def energy(
        self,
        u_fn,
        params=None,
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor], Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
        """
        Compute total contact energy and return:

            E_contact_total, part_dict, stats_cn, stats_ct

        Returns
        -------
        E_contact_total : tf.Tensor (scalar)
            总接触能量 En + Et
        part_dict : dict
            {"E_n": En, "E_t": Et}
        stats_cn : dict
            法向 ALM 的统计量（由 NormalContactALM.energy 返回）
        stats_ct : dict
            摩擦 ALM 的统计量（由 FrictionContactALM.energy 返回）

        注意：此函数是可微的，通常直接参与 PINN 损失；ALM 乘子更新在
              `update_multipliers` 中完成（不可微）。
        """
        if not self._built:
            raise RuntimeError("[ContactOperator] call build_from_cat() before energy().")

        En, stats_cn = self.normal.energy(u_fn, params)
        Et, stats_ct = self.friction.energy(u_fn, params)

        E = En + Et
        parts = {"E_n": En, "E_t": Et}

        return E, parts, stats_cn, stats_ct

    def update_multipliers(self, u_fn, params=None):
        """
        Outer-loop ALM update for both normal and friction.

        - 若 cfg.update_every_steps <= 0：每一次调用都更新一次乘子；
        - 否则：只有在 self._step % update_every_steps == 0 时才真正更新。
        """
        if not self._built:
            raise RuntimeError("[ContactOperator] call build_from_cat() before update_multipliers().")

        do_update = False
        if self.cfg.update_every_steps is None or self.cfg.update_every_steps <= 0:
            do_update = True
        else:
            if (self._step % self.cfg.update_every_steps) == 0:
                do_update = True

        if do_update:
            self.normal.update_multipliers(u_fn, params)
            self.friction.update_multipliers(u_fn, params)

        self._step += 1

    # ---------- residual export (for adaptive sampling) ----------

    def last_sample_metrics(self) -> Dict[str, np.ndarray]:
        """
        Return per-sample residual-like metrics from the latest energy call.

        Useful for residual-adaptive resampling (RAR):
            - "gap": raw normal gap g (negative => penetration)
            - "fric_res": ||r_t|| or |s_t| depending on friction mode
        If contact has not been evaluated since the last build/reset, returns empty dict.
        """

        metrics: Dict[str, np.ndarray] = {}
        if getattr(self.normal, "_last_gap", None) is not None:
            try:
                metrics["gap"] = np.asarray(self.normal._last_gap.numpy()).reshape(-1)
            except Exception:
                pass
        if getattr(self.friction, "_last_r_norm", None) is not None:
            try:
                metrics["fric_res"] = np.asarray(self.friction._last_r_norm.numpy()).reshape(-1)
            except Exception:
                pass
        return metrics

    def last_meta(self) -> Dict[str, np.ndarray]:
        """Return shallow-copied metadata for the current batch of contact samples."""

        return dict(self._meta)

    # ---------- staged-loading helpers ----------

    def snapshot_stage_state(self) -> Dict[str, np.ndarray]:
        """Snapshot frictional state so staged preload can carry order-dependent stick/slip."""
        if hasattr(self.friction, "snapshot_state"):
            return self.friction.snapshot_state()
        return {}

    def restore_stage_state(self, state: Dict[str, np.ndarray]):
        """Restore frictional state from :meth:`snapshot_stage_state`."""
        if hasattr(self.friction, "restore_state"):
            self.friction.restore_state(state)

    def last_friction_slip(self):
        """Expose cached tangential slip for staged path-penalty construction."""
        if hasattr(self.friction, "last_slip"):
            return self.friction.last_slip()
        return None

    # ---------- schedules / setters ----------

    def set_beta(self, beta: float):
        """Set softplus steepness for normal contact."""
        self.normal.set_beta(beta)

    def set_mu_n(self, mu_n: float):
        """Set ALM coefficient for normal part."""
        self.normal.set_mu_n(mu_n)

    def set_mu_t(self, mu_t: float):
        """Set ALM coefficient for tangential residual energy."""
        self.friction.set_mu_t(mu_t)

    def set_k_t(self, k_t: float):
        """Set tangential penalty stiffness for trial traction."""
        self.friction.set_k_t(k_t)

    def set_mu_f(self, mu_f: float):
        """Set Coulomb friction coefficient μ_f."""
        # 小心之前版本里的拼写错误，这里直接对 variable 赋值
        self.friction.mu_f.assign(tf.cast(mu_f, self.dtype))

    def multiply_weights(self, extra_w: np.ndarray):
        """
        Multiply extra weights to both normal and friction energies (Weighted PINN hook).

        注意：这是在 build 之后、训练过程中「再叠一层」权重的接口，
        典型用法是 IRLS 或残差自适应加权。
        """
        self.normal.multiply_weights(extra_w)
        self.friction.multiply_weights(extra_w)

    # ---------- convenience ----------

    @property
    def N(self) -> int:
        return self._N

    @property
    def built(self) -> bool:
        return self._built


# -----------------------------
# Minimal smoke test (optional)
# -----------------------------
if __name__ == "__main__":
    # 仅做 API 连通性检查，不跑真实接触（缺少 u_fn）
    import numpy as np

    N = 10
    cat = {
        "xs": np.random.randn(N, 3),
        "xm": np.random.randn(N, 3),
        "n": np.tile(np.array([0.0, 0.0, 1.0]), (N, 1)),
        "t1": np.tile(np.array([1.0, 0.0, 0.0]), (N, 1)),
        "t2": np.tile(np.array([0.0, 1.0, 0.0]), (N, 1)),
        "w_area": np.ones((N,), dtype=np.float64),
    }

    op = ContactOperator()
    op.build_from_cat(cat, extra_weights=None, auto_orient=True)

    # dummy u_fn: zero displacement
    def u_fn(X, params=None):
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        return tf.zeros_like(X)

    E_c, parts_c, stats_cn, stats_ct = op.energy(u_fn)
    print("E_contact =", float(E_c.numpy()))
    print("parts:", {k: float(v.numpy()) for k, v in parts_c.items()})
    print("stats_cn keys:", list(stats_cn.keys()))
    print("stats_ct keys:", list(stats_ct.keys()))
