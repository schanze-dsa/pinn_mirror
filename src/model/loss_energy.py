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

    adaptive_scheme: str = "contact_only"

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

        If ``params`` contains a staged sequence (``{"stages": [...]}``) the
        energy is evaluated incrementally for each stage and accumulated so that
        different tightening orders can influence the loss.
        """
        if not self._built:
            raise RuntimeError("[TotalEnergy] attach(...) must be called before energy().")

        if isinstance(params, dict) and params.get("stages"):
            Pi, parts, stats = self._energy_staged(u_fn, params["stages"], params, tape)
            return Pi, parts, stats

        parts, stats = self._compute_parts(u_fn, params or {}, tape)
        Pi = self._combine_parts(parts)
        return Pi, parts, stats

    def _compute_parts(self, u_fn, params, tape=None):
        """Evaluate all energy components for a given parameter dictionary."""
        dtype = self.dtype
        zero = tf.cast(0.0, dtype)
        parts: Dict[str, tf.Tensor] = {
            "E_int": zero,
            "E_cn": zero,
            "E_ct": zero,
            "E_tie": zero,
            "E_bc": zero,
            "W_pre": zero,
        }
        stats: Dict[str, tf.Tensor] = {}

        if self.elasticity is not None:
            E_int, estates = self.elasticity.energy(u_fn, params, tape=tape)
            parts["E_int"] = tf.cast(E_int, dtype)
            stats.update({f"el_{k}": v for k, v in estates.items()})

        if self.contact is not None:
            _, cparts, stats_cn, stats_ct = self.contact.energy(u_fn, params)
            if "E_cn" in cparts:
                parts["E_cn"] = tf.cast(cparts["E_cn"], dtype)
            elif "E_n" in cparts:
                parts["E_cn"] = tf.cast(cparts["E_n"], dtype)

            if "E_ct" in cparts:
                parts["E_ct"] = tf.cast(cparts["E_ct"], dtype)
            elif "E_t" in cparts:
                parts["E_ct"] = tf.cast(cparts["E_t"], dtype)

            stats.update(stats_cn)
            stats.update(stats_ct)

            if "R_fric_comp" in stats_ct:
                parts["R_fric_comp"] = tf.cast(stats_ct["R_fric_comp"], dtype)
            if "R_contact_comp" in stats_cn:
                parts["R_contact_comp"] = tf.cast(stats_cn["R_contact_comp"], dtype)

        if self.ties:
            tie_terms = []
            for i, t in enumerate(self.ties):
                Ei, si = t.energy(u_fn, params)
                tie_terms.append(tf.cast(Ei, dtype))
                stats.update({f"tie{i+1}_{k}": v for k, v in si.items()})
            if tie_terms:
                parts["E_tie"] = tf.add_n(tie_terms)

        if self.bcs:
            bc_terms = []
            for i, b in enumerate(self.bcs):
                Ei, si = b.energy(u_fn, params)
                bc_terms.append(tf.cast(Ei, dtype))
                stats.update({f"bc{i+1}_{k}": v for k, v in si.items()})
            if bc_terms:
                parts["E_bc"] = tf.add_n(bc_terms)

        if self.preload is not None:
            W_pre, pstats = self.preload.energy(u_fn, params)
            parts["W_pre"] = tf.cast(W_pre, dtype)
            stats.update({f"pre_{k}": v for k, v in pstats.items()})

        return parts, stats

    def _combine_parts(self, parts: Dict[str, tf.Tensor]) -> tf.Tensor:
        return (
            self.w_int * parts.get("E_int", tf.cast(0.0, self.dtype))
            + self.w_cn * parts.get("E_cn", tf.cast(0.0, self.dtype))
            + self.w_ct * parts.get("E_ct", tf.cast(0.0, self.dtype))
            + self.w_tie * parts.get("E_tie", tf.cast(0.0, self.dtype))
            + self.w_bc * parts.get("E_bc", tf.cast(0.0, self.dtype))
            - self.w_pre * parts.get("W_pre", tf.cast(0.0, self.dtype))
        )

    def _energy_staged(self, u_fn, stages, root_params, tape=None):
        """Accumulate energy across staged preload applications.

        与先前的“望向最终态”做差不同，这里直接累加每个阶段的能量/功，
        同时为相邻阶段的开口量变化乘以载荷变化添加一项耗散式惩罚，
        使得不同的加载顺序在无数据训练时也能对总损失产生影响。
        """
        dtype = self.dtype
        keys = ["E_int", "E_cn", "E_ct", "E_tie", "E_bc", "W_pre"]
        totals: Dict[str, tf.Tensor] = {k: tf.cast(0.0, dtype) for k in keys}
        stats_all: Dict[str, tf.Tensor] = {}
        path_penalty = tf.cast(0.0, dtype)

        if isinstance(stages, dict):
            stage_tensor_P = stages.get("P")
            stage_tensor_feat = stages.get("P_hat")
            stage_tensor_rank = stages.get("stage_rank")
            stage_tensor_mask = stages.get("stage_mask")
            stage_tensor_last = stages.get("stage_last")
            if stage_tensor_P is None or stage_tensor_feat is None:
                stage_seq: List[Dict[str, tf.Tensor]] = []
            else:
                stacked_rank = None
                if stage_tensor_rank is not None:
                    stacked_rank = tf.convert_to_tensor(stage_tensor_rank)
                stacked_mask = None
                if stage_tensor_mask is not None:
                    stacked_mask = tf.convert_to_tensor(stage_tensor_mask)
                stacked_last = None
                if stage_tensor_last is not None:
                    stacked_last = tf.convert_to_tensor(stage_tensor_last)
                stage_seq = []
                for idx, (p, z) in enumerate(
                    zip(tf.unstack(stage_tensor_P, axis=0), tf.unstack(stage_tensor_feat, axis=0))
                ):
                    entry = {"P": p, "P_hat": z}
                    if stacked_rank is not None:
                        if stacked_rank.shape.rank == 2:
                            entry["stage_rank"] = stacked_rank[idx]
                        else:
                            entry["stage_rank"] = stacked_rank
                    if stacked_mask is not None:
                        entry["stage_mask"] = stacked_mask[idx]
                    if stacked_last is not None:
                        entry["stage_last"] = stacked_last[idx]
                    stage_seq.append(entry)
        else:
            stage_seq = []
            for item in stages:
                if isinstance(item, dict):
                    stage_seq.append(item)
                else:
                    p_val, z_val = item
                    stage_seq.append({"P": p_val, "P_hat": z_val})

        if not stage_seq:
            return self._combine_parts(totals), totals, stats_all

        prev_parts: Optional[Dict[str, tf.Tensor]] = None
        prev_bolt_deltas: Optional[tf.Tensor] = None
        prev_P: Optional[tf.Tensor] = None
        prev_slip: Optional[tf.Tensor] = None

        contact_state = None
        base_contact_state = None
        if self.contact is not None and hasattr(self.contact, "snapshot_stage_state"):
            contact_state = self.contact.snapshot_stage_state()
            base_contact_state = contact_state
        for idx, stage_params in enumerate(stage_seq):
            if contact_state is not None and hasattr(self.contact, "restore_stage_state"):
                self.contact.restore_stage_state(contact_state)

            stage_parts, stage_stats = self._compute_parts(u_fn, stage_params, tape)
            for k, v in stage_stats.items():
                stats_all[f"s{idx+1}_{k}"] = v

            for key in keys:
                cur = tf.cast(stage_parts.get(key, tf.cast(0.0, dtype)), dtype)
                totals[key] = totals[key] + cur
                stats_all[f"s{idx+1}_{key}"] = cur
                stats_all[f"s{idx+1}_cum{key}"] = totals[key]

            bolt_deltas = None
            pre_entry = stage_stats.get("pre_preload")
            if isinstance(pre_entry, dict) and "bolt_deltas" in pre_entry:
                bolt_deltas = tf.cast(pre_entry["bolt_deltas"], dtype)

            P_vec = tf.cast(tf.convert_to_tensor(stage_params.get("P", [])), dtype)
            slip_t = None
            if self.contact is not None and hasattr(self.contact, "last_friction_slip"):
                slip_t = self.contact.last_friction_slip()

            if idx > 0:
                load_jump = tf.reduce_sum(tf.abs(P_vec - prev_P)) if prev_P is not None else tf.cast(0.0, dtype)

                if bolt_deltas is not None and prev_bolt_deltas is not None:
                    disp_jump = tf.reduce_sum(tf.abs(bolt_deltas - prev_bolt_deltas))
                    stage_path = disp_jump * load_jump
                    path_penalty = path_penalty + stage_path
                    stats_all[f"s{idx+1}_path_penalty"] = stage_path

                if slip_t is not None and prev_slip is not None:
                    slip_jump = tf.reduce_sum(tf.abs(slip_t - prev_slip))
                    fric_path = slip_jump * load_jump
                    path_penalty = path_penalty + fric_path
                    stats_all[f"s{idx+1}_fric_path_penalty"] = fric_path

            if bolt_deltas is not None:
                prev_bolt_deltas = bolt_deltas
            if tf.size(P_vec) > 0:
                prev_P = P_vec
            if slip_t is not None:
                prev_slip = slip_t

            prev_parts = stage_parts

            if contact_state is not None and hasattr(self.contact, "snapshot_stage_state"):
                contact_state = self.contact.snapshot_stage_state()

        if base_contact_state is not None and hasattr(self.contact, "restore_stage_state"):
            self.contact.restore_stage_state(base_contact_state)

        if isinstance(root_params, dict):
            if "stage_order" in root_params:
                stats_all["stage_order"] = root_params["stage_order"]
            if "stage_rank" in root_params:
                stats_all["stage_rank"] = root_params["stage_rank"]
            if "stage_count" in root_params:
                stats_all["stage_count"] = root_params["stage_count"]

        stats_all["path_penalty_total"] = path_penalty

        Pi = self._combine_parts(totals) + path_penalty
        return Pi, totals, stats_all

    # ---------- outer updates ----------
    def update_multipliers(self, u_fn, params=None):
        """
        Run ALM outer-loop updates for contact (and anything else in future).
        Call this every cfg.update_every_steps steps in your training loop.
        """
        target_params = params
        if isinstance(params, dict) and params.get("stages"):
            stages = params["stages"]
            if isinstance(stages, dict):
                stage_tensor_P = stages.get("P")
                stage_tensor_feat = stages.get("P_hat")
                if stage_tensor_P is not None:
                    target_params = {"P": stage_tensor_P[-1]}
                    if stage_tensor_feat is not None:
                        target_params["P_hat"] = stage_tensor_feat[-1]
            elif isinstance(stages, (list, tuple)) and stages:
                last_stage = stages[-1]
                if isinstance(last_stage, dict):
                    target_params = last_stage
                else:
                    p_val, z_val = last_stage
                    target_params = {"P": p_val, "P_hat": z_val}
        if self.contact is not None:
            self.contact.update_multipliers(u_fn, target_params)

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
