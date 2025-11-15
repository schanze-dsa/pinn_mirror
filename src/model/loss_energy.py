#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
loss_energy.py
--------------
Total potential energy assembly for DFEM/PINN with contact & preloads.

Composition (默认的线性组合形式，可以在外部被 loss_weights 覆盖)：
    Π = w_int * E_int
        + w_cn * E_cn
        + w_ct * E_ct
        + w_tie * E_tie
        + w_bc  * E_bc
        - w_pre * W_pre

Public usage (typical):
    # 1) Build sub-operators per batch
    elas.build_from_numpy(...) / build_dfem_subcells(...)
    contact.build_from_cat(cat_dict, extra_weights=..., auto_orient=True)
    tie.build_from_numpy(xs, xm, w_area, dof_mask=None)
    bc.build_from_numpy(X_bc, dof_mask, u_target, w_bc)

    # 2) Assemble total energy
    total = TotalEnergy()
    total.attach(elasticity=elas, contact=contact, preload=preload, ties=[tie], bcs=[bc])

    # 3) Compute energy & update multipliers in training loop
    Pi, parts, stats = total.energy(model.u_fn, params={"P": [P1,P2,P3]})
    # 若使用自适应权重，可在外部调用 loss_weights.update_loss_weights / combine_loss
    if step % total.cfg.update_every_steps == 0:
        total.update_multipliers(model.u_fn, params)

Weighted PINN:
    - You can multiply extra per-sample weights into components:
        contact.multiply_weights(w_contact)
        for t in ties: t.multiply_weights(w_tie)
        for b in bcs:  b.multiply_weights(w_bc)
    - If you need to reweight volume points, see TotalEnergy.scale_volume_weights().
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import tensorflow as tf

# sub-operators
from physics.elasticity_energy import ElasticityEnergy, ElasticityConfig
from physics.contact.contact_operator import ContactOperator, ContactOperatorConfig
from physics.tie_constraints import TiePenalty, TieConfig
from physics.boundary_conditions import BoundaryPenalty, BoundaryConfig
from physics.preload_model import PreloadWork, PreloadConfig


# -----------------------------
# Config for total energy
# -----------------------------

@dataclass
class TotalConfig:
    # coefficients for each term (基础权重；如用 loss_weights，可作为 base_weights)
    w_int: float = 1.0
    w_cn: float = 1.0            # normal contact  -> E_cn
    w_ct: float = 1.0            # frictional      -> E_ct
    w_tie: float = 1.0
    w_bc: float = 1.0
    w_pre: float = 1.0           # multiplies the subtracted W_pre

    # ALM outer update cadence for contact (can be used by training loop)
    update_every_steps: int = 150

    # dtype
    dtype: str = "float32"


# -----------------------------
# Total energy assembler
# -----------------------------

class TotalEnergy:
    """
    Assemble total potential energy from provided operators.

    - energy(...) 负责计算各个分量能量/残差，并返回：
        Π_total, parts_dict, stats_dict
      其中 parts_dict 可直接交给 train/loss_weights.py 做自适应加权。

    - update_multipliers(...) 只负责外层 ALM 乘子更新（接触/摩擦）。
    """

    def __init__(self, cfg: Optional[TotalConfig] = None):
        self.cfg = cfg or TotalConfig()
        self.dtype = tf.float32 if self.cfg.dtype == "float32" else tf.float64

        # sub-ops (optional ones can be None)
        self.elasticity: Optional[ElasticityEnergy] = None
        self.contact: Optional[ContactOperator] = None
        self.ties: List[TiePenalty] = []
        self.bcs: List[BoundaryPenalty] = []
        self.preload: Optional[PreloadWork] = None

        # trainable (non) scalars as TF vars so they can be scheduled
        self.w_int = tf.Variable(self.cfg.w_int, dtype=self.dtype, trainable=False, name="w_int")
        self.w_cn  = tf.Variable(self.cfg.w_cn,  dtype=self.dtype, trainable=False, name="w_cn")
        self.w_ct  = tf.Variable(self.cfg.w_ct,  dtype=self.dtype, trainable=False, name="w_ct")
        self.w_tie = tf.Variable(self.cfg.w_tie, dtype=self.dtype, trainable=False, name="w_tie")
        self.w_bc  = tf.Variable(self.cfg.w_bc,  dtype=self.dtype, trainable=False, name="w_bc")
        self.w_pre = tf.Variable(self.cfg.w_pre, dtype=self.dtype, trainable=False, name="w_pre")

        self._built = False

    # ---------- wiring ----------

    def attach(
        self,
        elasticity: Optional[ElasticityEnergy] = None,
        contact: Optional[ContactOperator] = None,
        preload: Optional[PreloadWork] = None,
        ties: Optional[List[TiePenalty]] = None,
        bcs: Optional[List[BoundaryPenalty]] = None,
    ):
        """
        Attach sub-components built for the current batch.
        """
        if elasticity is not None:
            self.elasticity = elasticity
        if contact is not None:
            self.contact = contact
        if preload is not None:
            self.preload = preload
        if ties is not None:
            self.ties = list(ties)
        if bcs is not None:
            self.bcs = list(bcs)

        self._built = True

    def reset(self):
        """Detach everything (e.g., before building a new batch)."""
        self.elasticity = None
        self.contact = None
        self.preload = None
        self.ties = []
        self.bcs = []
        self._built = False

    # ---------- optional helpers ----------

    def scale_volume_weights(self, factor: float):
        """
        Multiply all volume quadrature weights by 'factor' (coarse reweighting).
        Use this if you want Weighted PINN-like emphasis on volume PDE residuals.
        """
        # 注意：DFEM 版本里，体积分权重通常在 ElasticityEnergy 内部管理。
        if getattr(self.elasticity, "w_tf", None) is None:
            return
        self.elasticity.w_tf.assign(self.elasticity.w_tf * tf.cast(factor, self.dtype))

    # ---------- energy ----------

    def energy(self, u_fn, params=None, tape=None):
        """
        Compute total potential and return:
            Π_total, parts_dict, stats_dict

        parts_dict（为 loss_weights 提供 Hook）:
            {
              'E_int': E_int,
              'E_cn':  E_cn,
              'E_ct':  E_ct,
              'E_tie': E_tie,
              'E_bc':  E_bc,
              'W_pre': W_pre,
              'R_fric_comp':  ... (可选),
              'R_contact_comp': ... (可选),
            }

        stats_dict: merged sub-stats with prefixes / original keys.
        """
        if not self._built:
            raise RuntimeError("[TotalEnergy] attach(...) must be called before energy().")

        parts: Dict[str, tf.Tensor] = {}
        stats: Dict[str, tf.Tensor] = {}

        # -------------------- Elasticity --------------------
        E_int = tf.cast(0.0, self.dtype)
        if self.elasticity is not None:
            # 关键：把 tape 传进弹性项（需要梯度时）
            E_int, estates = self.elasticity.energy(u_fn, params, tape=tape)
            parts["E_int"] = E_int
            # estates 里一般是一些标量统计（节点数、单元数等），加 el_ 前缀防冲突
            stats.update({f"el_{k}": v for k, v in estates.items()})

        # -------------------- Contact (normal + friction) --------------------
        E_cn = tf.cast(0.0, self.dtype)
        E_ct = tf.cast(0.0, self.dtype)
        if self.contact is not None:
            # 新接口：E_c_total, cparts, stats_cn, stats_ct
            E_c, cparts, stats_cn, stats_ct = self.contact.energy(u_fn, params)

            # cparts 里按约定有 "E_n", "E_t" 或 "E_cn", "E_ct"
            if "E_cn" in cparts:
                E_cn = tf.cast(cparts["E_cn"], self.dtype)
            elif "E_n" in cparts:
                E_cn = tf.cast(cparts["E_n"], self.dtype)

            if "E_ct" in cparts:
                E_ct = tf.cast(cparts["E_ct"], self.dtype)
            elif "E_t" in cparts:
                E_ct = tf.cast(cparts["E_t"], self.dtype)

            parts["E_cn"] = E_cn
            parts["E_ct"] = E_ct

            # 统计信息：直接合并 normal / friction 的 stats
            stats.update(stats_cn)
            stats.update(stats_ct)

            # 若摩擦 stats 中提供了 R_fric_comp，把它也写入 parts 供 loss_weights 使用
            if "R_fric_comp" in stats_ct:
                parts["R_fric_comp"] = tf.cast(stats_ct["R_fric_comp"], self.dtype)
            # 将来若在 NormalContactALM 里加入 R_contact_comp，这里也顺手写入
            if "R_contact_comp" in stats_cn:
                parts["R_contact_comp"] = tf.cast(stats_cn["R_contact_comp"], self.dtype)

        # -------------------- Tie (sum over multiple) --------------------
        E_tie = tf.cast(0.0, self.dtype)
        if self.ties:
            tie_acc = []
            for i, t in enumerate(self.ties):
                Ei, si = t.energy(u_fn, params)
                tie_acc.append(Ei)
                stats.update({f"tie{i+1}_{k}": v for k, v in si.items()})
            E_tie = tf.add_n(tie_acc)
            parts["E_tie"] = E_tie

        # -------------------- Boundary (sum over multiple) --------------------
        E_bc = tf.cast(0.0, self.dtype)
        if self.bcs:
            bc_acc = []
            for i, b in enumerate(self.bcs):
                Ei, si = b.energy(u_fn, params)
                bc_acc.append(Ei)
                stats.update({f"bc{i+1}_{k}": v for k, v in si.items()})
            E_bc = tf.add_n(bc_acc)
            parts["E_bc"] = E_bc

        # -------------------- Preload work (positive value to be subtracted) --------------------
        W_pre = tf.cast(0.0, self.dtype)
        if self.preload is not None:
            W_pre, pstats = self.preload.energy(u_fn, params)
            parts["W_pre"] = W_pre
            stats.update({f"pre_{k}": v for k, v in pstats.items()})

        # -------------------- Assemble Π (baseline linear combination) --------------------
        # 这里依然给出一个“基础版” Π_total，方便不使用 loss_weights 时直接训练；
        # 若用自适应权重，可以在 Trainer 中忽略 Pi，而用 loss_weights.combine_loss(parts, state)。
        Pi = (
            self.w_int * E_int
            + self.w_cn * E_cn
            + self.w_ct * E_ct
            + self.w_tie * E_tie
            + self.w_bc * E_bc
            - self.w_pre * W_pre
        )

        return Pi, parts, stats

    # ---------- outer updates ----------

    def update_multipliers(self, u_fn, params=None):
        """
        Run ALM outer-loop updates for contact (and anything else in future).
        Call this every cfg.update_every_steps steps in your training loop.
        """
        if self.contact is not None:
            self.contact.update_multipliers(u_fn, params)

    # ---------- setters / schedules ----------

    def set_coeffs(
        self,
        w_int: Optional[float] = None,
        w_cn: Optional[float] = None,
        w_ct: Optional[float] = None,
        w_tie: Optional[float] = None,
        w_bc: Optional[float] = None,
        w_pre: Optional[float] = None,
    ):
        """Set any subset of coefficients on the fly (e.g., curriculum)."""
        if w_int is not None:
            self.w_int.assign(tf.cast(w_int, self.dtype))
        if w_cn is not None:
            self.w_cn.assign(tf.cast(w_cn, self.dtype))
        if w_ct is not None:
            self.w_ct.assign(tf.cast(w_ct, self.dtype))
        if w_tie is not None:
            self.w_tie.assign(tf.cast(w_tie, self.dtype))
        if w_bc is not None:
            self.w_bc.assign(tf.cast(w_bc, self.dtype))
        if w_pre is not None:
            self.w_pre.assign(tf.cast(w_pre, self.dtype))


# -----------------------------
# Minimal smoke test
# -----------------------------
if __name__ == "__main__":
    # 仅做 API 连通性检查；真正的 DFEM/接触细节在各子模块里。
    from dataclasses import dataclass
    import numpy as np
    from physics.contact.contact_operator import ContactOperator

    @dataclass
    class DummyBlock:
        elem_type: str
        connectivity: list

    @dataclass
    class DummyPart:
        name: str
        element_blocks: list

    class DummyAsm:
        def __init__(self):
            # 4 节点四面体：节点编号故意使用非 0 基，以测试映射
            self.nodes = {
                10: (0.0, 0.0, 0.0),
                11: (1.0, 0.0, 0.0),
                12: (0.0, 1.0, 0.0),
                13: (0.0, 0.0, 1.0),
            }
            block = DummyBlock("C3D4", [[10, 11, 12, 13]])
            part = DummyPart(name="demo", element_blocks=[block])
            self.parts = {"demo": part}

    asm = DummyAsm()
    materials = {"steel": (210000.0, 0.3)}
    part2mat = {"demo": "steel"}
    elas = ElasticityEnergy(asm=asm, part2mat=part2mat, materials=materials, cfg=ElasticityConfig())

    # 2) Contact (random placeholders)
    cat = {
        "xs": np.random.randn(8, 3),
        "xm": np.random.randn(8, 3),
        "n": np.tile(np.array([0.0, 0.0, 1.0]), (8, 1)),
        "t1": np.tile(np.array([1.0, 0.0, 0.0]), (8, 1)),
        "t2": np.tile(np.array([0.0, 1.0, 0.0]), (8, 1)),
        "w_area": np.ones((8,), dtype=np.float64),
    }
    contact = ContactOperator()
    contact.build_from_cat(cat, extra_weights=None, auto_orient=True)

    # 3) Dummy tie / bc / preload（可以根据需要删除）
    tie = TiePenalty(TieConfig())
    tie.build_from_numpy(xs=cat["xs"], xm=cat["xm"], w_area=cat["w_area"])
    bc = BoundaryPenalty(BoundaryConfig())
    X_bc = np.random.randn(4, 3)
    mask = np.ones((4, 3))
    bc.build_from_numpy(X_bc, mask, u_target=None, w_bc=None)
    pl = PreloadWork()

    # 4) Dummy u_fn
    def u_fn(X, params=None):
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        uz = 1e-3 * (X[:, 0:1] ** 2 + X[:, 1:2] ** 2)
        return tf.concat([tf.zeros_like(uz), tf.zeros_like(uz), uz], axis=1)

    total = TotalEnergy(TotalConfig(update_every_steps=50))
    total.attach(elasticity=elas, contact=contact, preload=pl, ties=[tie], bcs=[bc])

    P = tf.constant([300.0, 500.0, 700.0], dtype=tf.float32)
    Pi, parts, stats = total.energy(u_fn, params={"P": P})
    print("Π =", float(Pi.numpy()))
    print("parts:", {k: float(v.numpy()) for k, v in parts.items() if parts[k].shape == ()})
