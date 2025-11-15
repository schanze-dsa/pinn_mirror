# src/train/loss_weights.py
# -*- coding: utf-8 -*-
"""
loss_weights.py
---------------
Loss-weight scheduler for TotalEnergy / Trainer.

职责：
- 保存当前各个能量/残差项的权重（current）以及基础权重（base）；
- 维护接触/摩擦等关键项的 EMA 残差；
- 根据 adaptive_scheme 自动更新权重；
- 提供 combine_loss() 将分项能量组合成一个标量损失。

设计假定：
- TotalEnergy.energy(...) 会返回一个 dict parts，例如：
    parts = {
        "E_int": E_int,
        "E_cn":  E_cn,
        "E_ct":  E_ct,
        "E_bc":  E_bc,
        "E_tie": E_tie,
        "W_pre": W_pre,
        "R_fric_comp":  R_fric,      # 可选
        "R_contact_comp": R_cont,    # 可选
        ...
    }
  其中 key 名就是我们在 loss_weights 里识别的名字。

- contact_operator.energy(...) 已经返回 stats_cn / stats_ct，
  你可以在 Trainer 里把它们打包传进来作为 stats，方便以后扩展。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple

import numpy as np
import tensorflow as tf


def _to_float(x: Any) -> float:
    """Convert tf.Tensor/np scalar/Python scalar to float."""
    if isinstance(x, tf.Tensor):
        # 停止梯度并取 numpy 值
        return float(x.numpy())
    if isinstance(x, np.ndarray):
        return float(x.astype(np.float64))
    return float(x)


@dataclass
class LossWeightState:
    """
    Loss-weight状态容器。

    字段说明
    -------
    base : dict
        基础权重（通常来自 config.yaml 的 loss_weights 段）。
        例：
            {
              "E_int": 1.0,
              "E_cn":  1.0,
              "E_ct":  1.0,
              "E_bc":  1.0,
              "E_tie": 1.0,
              "W_pre": 1.0,
              "R_fric_comp": 0.0,
              "R_contact_comp": 0.0,
            }

    current : dict
        当前实际使用的权重（会在训练过程中被自动更新）。

    adaptive_scheme : str
        自适应策略：
            "off"          : 完全不用自适应，current = base；
            "contact_only" : 只对接触/摩擦相关权重做自适应（E_cn / E_ct）；
            "basic"        : 目前等价于 contact_only，预留以后扩展。

    ema_contact / ema_fric : float
        接触/摩擦的 EMA 残差（这里默认用 E_cn/E_ct 或 R_contact_comp/R_fric_comp 代表）。

    decay : float
        EMA 衰减系数（越接近 1，记忆越长）。

    min_factor / max_factor : float
        自适应权重相对 base 的缩放上下限。

    gamma : float
        softmax 的“锐化”系数，越大则越偏向 residual 更大的项。

    step : int
        已更新次数计数，仅作参考/调试。
    """

    base: Dict[str, float] = field(default_factory=dict)
    current: Dict[str, float] = field(default_factory=dict)

    adaptive_scheme: str = "off"

    ema_contact: float = 0.0
    ema_fric: float = 0.0
    decay: float = 0.95

    min_factor: float = 0.25
    max_factor: float = 4.0
    gamma: float = 2.0

    update_every: int = 1
    focus_terms: Tuple[str, ...] = tuple()

    step: int = 0

    # 方便调试记录最近一次因子
    last_factor_cn: float = 1.0
    last_factor_ct: float = 1.0
    last_factors: Dict[str, float] = field(default_factory=dict)

    # 通用 EMA 容器（focus_terms 自适应时使用）
    ema_terms: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_config(
        cls,
        base_weights: Dict[str, float],
        adaptive_scheme: str = "off",
        ema_decay: float = 0.95,
        min_factor: float = 0.25,
        max_factor: float = 4.0,
        gamma: float = 2.0,
        focus_terms: Tuple[str, ...] | None = None,
        update_every: int = 1,
    ) -> "LossWeightState":
        """
        从配置初始化一个 LossWeightState。

        Parameters
        ----------
        base_weights : dict
            每个分项的基础权重（来自 config.yaml / TotalConfig.loss_weights）。
        adaptive_scheme : {"off", "contact_only", "basic"}
            自适应策略，见类说明。
        ema_decay : float
            EMA 衰减系数。
        """
        base = dict(base_weights)
        current = dict(base_weights)
        return cls(
            base=base,
            current=current,
            adaptive_scheme=adaptive_scheme,
            decay=ema_decay,
            min_factor=min_factor,
            max_factor=max_factor,
            gamma=gamma,
            focus_terms=tuple(focus_terms or tuple()),
            update_every=max(int(update_every), 1),
        )

    def as_dict(self) -> Dict[str, float]:
        """返回当前权重字典（方便给 TotalConfig / logger 使用）。"""
        return dict(self.current)


# --------------------------------------------------------------------------- #
# 核心：更新权重
# --------------------------------------------------------------------------- #

def update_loss_weights(
    state: LossWeightState,
    parts: Dict[str, tf.Tensor],
    stats: Dict[str, Any] | None = None,
) -> None:
    """
    根据当前分项能量/残差和历史 EMA，更新 state.current 内的权重。

    Parameters
    ----------
    state : LossWeightState
        权重状态，会被原地修改。
    parts : dict[str, tf.Tensor]
        TotalEnergy.energy(...) 返回的分项能量与残差，
        例如：
            {
                "E_int": E_int,
                "E_cn":  E_cn,
                "E_ct":  E_ct,
                "E_bc":  E_bc,
                "E_tie": E_tie,
                "W_pre": W_pre,
                "R_fric_comp":  R_fric,    # 可选
                "R_contact_comp": R_cont,  # 可选
                ...
            }
    stats : dict, optional
        额外的统计量（如 contact_operator.energy 返回的 stats_cn/stats_ct）。
        当前实现只用 parts 中的量即可，stats 先作为预留参数。
    """
    state.step += 1

    # 1) 若不启用自适应，直接回到 base
    if state.adaptive_scheme == "off":
        state.current = dict(state.base)
        state.last_factor_cn = 1.0
        state.last_factor_ct = 1.0
        state.last_factors = {}
        return

    # 每 update_every 步才真正更新一次权重；EMA 会在每次调用时更新
    should_update = state.step % max(state.update_every, 1) == 0

    # 若 focus_terms 非空，则采用通用自适应逻辑
    if state.focus_terms:
        residual_aliases = {
            "E_cn": ("R_contact_comp",),
            "E_ct": ("R_fric_comp",),
        }

        values = []
        term_order = []
        d = state.decay
        for term in state.focus_terms:
            # 依次尝试 term 自身及别名，找到第一个存在的分量
            aliases = (term,) + residual_aliases.get(term, tuple())
            val = 0.0
            for alias in aliases:
                part = parts.get(alias)
                if isinstance(part, tf.Tensor) and part.shape.rank == 0:
                    val = abs(_to_float(part))
                    break
                if isinstance(part, (float, int, np.generic)):
                    val = abs(_to_float(part))
                    break
            prev = state.ema_terms.get(term, 0.0)
            ema = d * prev + (1.0 - d) * val
            state.ema_terms[term] = ema
            values.append(ema)
            term_order.append(term)

        # 同步兼容字段（便于调试输出）
        state.ema_contact = state.ema_terms.get("E_cn", state.ema_contact)
        state.ema_fric = state.ema_terms.get("E_ct", state.ema_fric)

        if not should_update:
            return

        if not any(v > 1e-16 for v in values):
            state.current = dict(state.base)
            state.last_factors = {term: 1.0 for term in term_order}
            state.last_factor_cn = state.last_factors.get("E_cn", 1.0)
            state.last_factor_ct = state.last_factors.get("E_ct", 1.0)
            return

        vals = np.array(values, dtype=np.float64)
        mean = float(np.mean(vals)) if np.mean(vals) > 1e-16 else 1.0
        x = vals / mean

        logits = state.gamma * x
        logits = logits - np.max(logits)
        exp = np.exp(logits)
        soft = exp / np.sum(exp)

        avg = float(len(soft)) if len(soft) > 0 else 1.0
        factors = avg * soft
        factors = np.clip(factors, state.min_factor, state.max_factor)

        new_current = dict(state.base)
        state.last_factors = {}
        for term, factor in zip(term_order, factors):
            if term in new_current:
                new_current[term] = new_current[term] * float(factor)
                state.last_factors[term] = float(factor)

        state.current = new_current
        state.last_factor_cn = state.last_factors.get("E_cn", state.last_factor_cn)
        state.last_factor_ct = state.last_factors.get("E_ct", state.last_factor_ct)
        return

    # --------------------------------------------------
    # 2) 计算接触/摩擦的“残差代表值”：
    #    优先使用 R_contact_comp / R_fric_comp；
    #    若没有，则退回到 E_cn / E_ct 的绝对值。
    # --------------------------------------------------
    # contact
    if "R_contact_comp" in parts:
        r_cn_now = abs(_to_float(parts["R_contact_comp"]))
    elif "E_cn" in parts:
        r_cn_now = abs(_to_float(parts["E_cn"]))
    else:
        r_cn_now = 0.0

    # friction
    if "R_fric_comp" in parts:
        r_ct_now = abs(_to_float(parts["R_fric_comp"]))
    elif "E_ct" in parts:
        r_ct_now = abs(_to_float(parts["E_ct"]))
    else:
        r_ct_now = 0.0

    # 更新 EMA
    d = state.decay
    state.ema_contact = d * state.ema_contact + (1.0 - d) * r_cn_now
    state.ema_fric = d * state.ema_fric + (1.0 - d) * r_ct_now

    # 若两个都是 0（例如一开始），直接回到 base
    if state.ema_contact <= 1e-16 and state.ema_fric <= 1e-16:
        state.current = dict(state.base)
        state.last_factor_cn = 1.0
        state.last_factor_ct = 1.0
        state.last_factors = {}
        return

    # --------------------------------------------------
    # 3) 根据 EMA 残差构造自适应因子（软最大化风格）
    #    简单策略：对 [ema_cn, ema_ct] 做 softmax，
    #              然后平移到平均因子 ~1，并限制在 [min_factor, max_factor]。
    # --------------------------------------------------
    if state.adaptive_scheme in ("contact_only", "basic"):
        if not should_update:
            return

        vals = np.array([state.ema_contact, state.ema_fric], dtype=np.float64)
        mean = float(np.mean(vals)) if np.mean(vals) > 1e-16 else 1.0
        x = vals / mean  # 归一化到 ~O(1)

        logits = state.gamma * x
        logits = logits - np.max(logits)
        exp = np.exp(logits)
        soft = exp / np.sum(exp)

        # soft 在 [0,1] 且和为 1，我们希望平均因子为 1，因此乘上 2：
        factors = 2.0 * soft
        f_cn, f_ct = float(factors[0]), float(factors[1])

        f_cn = float(np.clip(f_cn, state.min_factor, state.max_factor))
        f_ct = float(np.clip(f_ct, state.min_factor, state.max_factor))

        # 更新 current 权重
        new_current = dict(state.base)
        if "E_cn" in new_current:
            new_current["E_cn"] = new_current["E_cn"] * f_cn
        if "E_ct" in new_current:
            new_current["E_ct"] = new_current["E_ct"] * f_ct

        # 其他项保持 base
        state.current = new_current
        state.last_factor_cn = f_cn
        state.last_factor_ct = f_ct
        state.last_factors = {"E_cn": f_cn, "E_ct": f_ct}
        return

    # 其它未知 scheme，退回 base
    state.current = dict(state.base)
    state.last_factor_cn = 1.0
    state.last_factor_ct = 1.0
    state.last_factors = {}


# --------------------------------------------------------------------------- #
# 将分项能量组合成总损失
# --------------------------------------------------------------------------- #

def combine_loss(
    parts: Dict[str, tf.Tensor],
    state: LossWeightState,
) -> tf.Tensor:
    """
    根据 state.current 中的权重，将各项能量/残差组合成一个标量损失。

    用法示例（在 Trainer 里）::

        E_parts, contact_stats = total_energy.energy(u_fn, params)
        update_loss_weights(weight_state, E_parts, contact_stats)
        loss = combine_loss(E_parts, weight_state)

    规则：
    - 对于在 state.current / state.base 中出现的 key，尝试获取对应权重；
    - 未配置的项默认权重为 0，不参与损失（但仍可用于监控）。
    """
    loss = tf.constant(0.0, dtype=tf.float32)

    # Certain energy contributions (e.g. external work) enter the potential with
    # a negative sign.  Keep a small map of such terms so combine_loss stays
    # consistent with TotalEnergy.Pi's baseline formulation even when adaptive
    # weighting is enabled.
    sign_overrides = {
        "W_pre": -1.0,
    }

    for name, value in parts.items():
        # 只组合标量项
        if not isinstance(value, tf.Tensor):
            continue
        if value.shape.rank != 0:
            # 如果是非标量（比如某些向量残差），这里只跳过，保留给其它模块处理
            continue

        w = None
        if name in state.current:
            w = state.current[name]
        elif name in state.base:
            w = state.base[name]

        if w is None or abs(w) <= 0.0:
            continue

        sign = tf.cast(sign_overrides.get(name, 1.0), tf.float32)
        loss = loss + sign * tf.cast(w, tf.float32) * tf.cast(value, tf.float32)

    return loss
