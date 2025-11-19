# -*- coding: utf-8 -*-
"""
physics/preload_model.py

从螺栓上下表面（来自 INP/Assembly 的 SurfaceDef 或现成点集）采样点云，
并在训练时计算预紧功/位移差等统计量。

关键点：
  - _fetch_surface_points(): 优先调用 assembly.surfaces.get_surface_points()
    自动把 SurfaceDef（ELEMENT 面）转换为 (X,N,w)。
  - _coerce_surface_like_to_points(): 统一把多种“表面样式”落成点集，需要时利用 asm。
  - energy(): 使用 (X + u) · N 的积分形式累加，权重为 w，全部使用 tf.float32。
  - _u_fn_chunked(): 对 u_fn 前向做 micro-batch，避免一次性大批量前向引起显存峰值。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf


# ---------- 规格 ----------
@dataclass
class BoltSurfaceSpec:
    name: str
    up_key: str
    down_key: Optional[str] = None


@dataclass
class PreloadConfig:
    epsilon: float = 1e-12     # 数值安全项
    work_coeff: float = 1.0    # 预紧功系数，可按需要扩展
    rank_relaxation: float = 0.0  # 顺序相关的松弛系数 (0 -> 不考虑顺序)
    # 可选：前向分块大小（若未在 cfg 里设置，也可由 _u_fn_chunked 内部默认取 2048）
    # forward_chunk: Optional[int] = None


@dataclass
class BoltSampleData:
    name: str
    # 上/下采样点与法向、权重（允许下侧缺省）
    X_up: np.ndarray
    N_up: np.ndarray
    w_up: np.ndarray
    X_dn: Optional[np.ndarray] = None
    N_dn: Optional[np.ndarray] = None
    w_dn: Optional[np.ndarray] = None


# ---------- 辅助：把各种“表面样式”转为 (X,N,w) ----------
def _coerce_surface_like_to_points(surface_like: Any,
                                   n_points_each: Optional[int] = None,
                                   asm: Any = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将 surface_like 统一转为 (X, N, w)，需要时使用 asm（比如 SurfaceDef -> 元素面采样）。

    支持：
      - numpy.ndarray: 直接视为 X，N 取 z 轴，w=1
      - SurfaceDef/具有 items/raw_lines 的对象：需要 asm，走 assembly.surfaces.surface_def_to_points()
      - 带 to_points()/sample_points()/points 属性的方法对象：尝试调用得到 X/N/w
    """
    import numpy as _np

    # 1) numpy 数组：直接当点集
    if isinstance(surface_like, _np.ndarray):
        X = _np.asarray(surface_like, dtype=_np.float32).reshape(-1, 3)
        N = _np.zeros_like(X, dtype=_np.float32)
        N[:, 2] = 1.0
        w = _np.ones((X.shape[0],), dtype=_np.float32)
        return X, N, w

    # 2) SurfaceDef（或类似，带 items/raw_lines）
    if surface_like is not None and (hasattr(surface_like, "items") or hasattr(surface_like, "raw_lines")):
        if asm is None:
            raise TypeError("SurfaceDef 转点需要 asm，但 asm=None。")
        from assembly.surfaces import surface_def_to_points  # 延迟导入
        X, N, w = surface_def_to_points(asm, surface_like, n_points_each or 1)
        return X.astype(_np.float32), N.astype(_np.float32), w.astype(_np.float32)

    # 3) 带方法的自定义类型
    for attr in ("to_points", "sample_points", "points"):
        if hasattr(surface_like, attr) and callable(getattr(surface_like, attr)):
            out = getattr(surface_like, attr)()
            # 允许返回 X 或 (X,N) 或 (X,N,w)
            if isinstance(out, (list, tuple)):
                if len(out) == 3:
                    X, N, w = out
                elif len(out) == 2:
                    X, N = out
                    w = _np.ones((X.shape[0],), dtype=_np.float32)
                else:
                    X = out[0]
                    N = _np.zeros_like(X, dtype=_np.float32); N[:, 2] = 1.0
                    w = _np.ones((X.shape[0],), dtype=_np.float32)
            else:
                X = _np.asarray(out, dtype=_np.float32)
                N = _np.zeros_like(X, dtype=_np.float32); N[:, 2] = 1.0
                w = _np.ones((X.shape[0],), dtype=_np.float32)
            return X.astype(_np.float32), N.astype(_np.float32), w.astype(_np.float32)

    # 4) 都不匹配
    tname = type(surface_like).__name__
    raise TypeError(f"无法把对象类型 {tname} 转为表面点集。请提供 ndarray/SurfaceDef/可 to_points 的对象。")


class PreloadWork:
    def __init__(self, cfg: Optional[PreloadConfig] = None):
        self.cfg = cfg or PreloadConfig()
        self._bolts: List[BoltSampleData] = []

    # --------- 构建 ---------
    def build_from_specs(self, asm, specs: List[BoltSurfaceSpec],
                         n_points_each: int = 800, seed: int = 0) -> None:
        """
        从装配的 surfaces 中根据 specs 采样出每个螺栓的上/下表面点集合。
        保留用户在 INP 中的原始键名（包括 'bolt2 uo' 这样的笔误），不做重命名。
        """
        rng = np.random.RandomState(seed)
        bolts: List[BoltSampleData] = []
        for sp in specs:
            X_up, N_up, w_up = self._fetch_surface_points(asm, sp.up_key, n_points_each)
            X_dn = N_dn = w_dn = None
            if sp.down_key:
                X_dn, N_dn, w_dn = self._fetch_surface_points(asm, sp.down_key, n_points_each)

            # 打乱（可选）
            idx = np.arange(X_up.shape[0]); rng.shuffle(idx)
            X_up, N_up, w_up = X_up[idx], N_up[idx], w_up[idx]
            if X_dn is not None:
                idy = np.arange(X_dn.shape[0]); rng.shuffle(idy)
                X_dn, N_dn, w_dn = X_dn[idy], N_dn[idy], w_dn[idy]

            bolts.append(BoltSampleData(
                name=sp.name,
                X_up=X_up.astype(np.float32), N_up=N_up.astype(np.float32), w_up=w_up.astype(np.float32),
                X_dn=None if X_dn is None else X_dn.astype(np.float32),
                N_dn=None if N_dn is None else N_dn.astype(np.float32),
                w_dn=None if w_dn is None else w_dn.astype(np.float32),
            ))
        self._bolts = bolts

    # --------- 采样辅助 ---------
    def _fetch_surface_points(self, asm, key: str, n_points_each: int
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        统一返回 (X, N, w)，均为 float32。

        优先级：
          1) assembly.surfaces.get_surface_points(asm, key, n)  -> (X,N,w)
          2) asm.surfaces[key] 存在：交给 _coerce_surface_like_to_points(..., asm=asm)
          3) asm.get_surface_points(key, n) 兜底
        """
        # 1) 首选高层入口
        try:
            from assembly.surfaces import get_surface_points as _get_pts
            X, N, w = _get_pts(asm, key, n_points_each)
            return X.astype(np.float32), N.astype(np.float32), w.astype(np.float32)
        except Exception:
            pass  # 继续向下兜底

        # 2) 直接从映射取
        mp = getattr(asm, "surfaces", None) or getattr(asm, "surface_map", None) or {}
        if isinstance(mp, dict) and key in mp:
            val = mp[key]
            X, N, w = _coerce_surface_like_to_points(val, n_points_each, asm)  # <<< 接受 asm
            return X.astype(np.float32), N.astype(np.float32), w.astype(np.float32)

        # 3) 兜底：装配自带方法
        if hasattr(asm, "get_surface_points") and callable(getattr(asm, "get_surface_points")):
            out = asm.get_surface_points(key, n_points_each)
            if isinstance(out, (list, tuple)):
                if len(out) == 3:
                    X, N, w = out
                elif len(out) == 2:
                    X, N = out
                    w = np.ones((X.shape[0],), dtype=np.float32)
                else:
                    X = out[0]
                    N = np.zeros_like(X, dtype=np.float32); N[:, 2] = 1.0
                    w = np.ones((X.shape[0],), dtype=np.float32)
            else:
                X = np.asarray(out, dtype=np.float32)
                N = np.zeros_like(X, dtype=np.float32); N[:, 2] = 1.0
                w = np.ones((X.shape[0],), dtype=np.float32)
            return X.astype(np.float32), N.astype(np.float32), w.astype(np.float32)

        # 4) 失败
        raise KeyError(f"[PreloadWork] 找不到表面 '{key}' 的点集。"
                       f" 请检查 asm.surfaces / assembly.surfaces.get_surface_points / asm.get_surface_points。")

    # --------- 前向分块（避免显存峰值） ---------
    def _u_fn_chunked(self, u_fn, params, X, batch: int = None) -> tf.Tensor:
        """
        对大批量坐标 X 分块调用位移网络 u_fn，避免一次性前向造成显存峰值。
        - u_fn: 形如 u_fn(X, params) -> (N,3)
        - params: 训练时传入的额外参数（如预紧力编码）
        - X: (N,3) 张量或 ndarray；允许已是 float16/float32 Tensor
        - batch: 每个 micro-batch 的大小；None 时使用 cfg.forward_chunk 或 2048
        返回: (N,3) 与输入顺序一致的拼接结果（float32）
        """
        if batch is None:
            batch = int(getattr(self.cfg, "forward_chunk", 2048))
        batch = max(1, int(batch))

        # 注意：若 X 已是 Tensor 且 dtype=fp16，tf.convert_to_tensor(..., dtype=fp32) 会报错；
        # 正确做法是先 convert，再显式 cast。
        X = tf.convert_to_tensor(X)
        if X.dtype != tf.float32:
            X = tf.cast(X, tf.float32)

        n = int(X.shape[0])

        outs = []
        for s in range(0, n, batch):
            e = min(n, s + batch)
            Xi = X[s:e]                   # (m,3) float32
            Ui = u_fn(Xi, params)         # (m,3)（网络内部可用混合精度）
            outs.append(tf.cast(Ui, tf.float32))
        return tf.concat(outs, axis=0)    # (N,3) float32

    # --------- 物理量计算 ---------
    def _bolt_delta(self, u_fn, params, bolt: BoltSampleData) -> tf.Tensor:
        """
        计算单个螺栓的“轴向开口量”近似：Δ = ∫[(X_up+u_up)·N_up] w_up - ∫[(X_dn+u_dn)·N_dn] w_dn
        若无下侧，则仅用上侧项。
        返回标量 tf.float32。
        """
        policy = tf.keras.mixed_precision.global_policy()
        compute_dtype = getattr(policy, "compute_dtype", tf.float32)

        # 上侧
        X_up = tf.cast(tf.convert_to_tensor(bolt.X_up), compute_dtype)  # (m,3)
        N_up = tf.cast(tf.convert_to_tensor(bolt.N_up), compute_dtype)  # (m,3)
        w_up = tf.cast(tf.convert_to_tensor(bolt.w_up), compute_dtype)  # (m,)
        u_up = tf.cast(
            self._u_fn_chunked(u_fn, params, X_up, batch=int(getattr(self.cfg, "forward_chunk", 2048))),
            compute_dtype
        )  # (m,3)
        s_up = tf.reduce_sum((X_up + u_up) * N_up, axis=1) * w_up       # (m,)
        I_up = tf.cast(tf.reduce_sum(s_up), tf.float32)

        if bolt.X_dn is None:
            return I_up

        # 下侧
        X_dn = tf.cast(tf.convert_to_tensor(bolt.X_dn), compute_dtype)
        N_dn = tf.cast(tf.convert_to_tensor(bolt.N_dn), compute_dtype)
        w_dn = tf.cast(tf.convert_to_tensor(bolt.w_dn), compute_dtype)
        u_dn = tf.cast(
            self._u_fn_chunked(u_fn, params, X_dn, batch=int(getattr(self.cfg, "forward_chunk", 2048))),
            compute_dtype
        )  # (m,3)
        s_dn = tf.reduce_sum((X_dn + u_dn) * N_dn, axis=1) * w_dn
        I_dn = tf.cast(tf.reduce_sum(s_dn), tf.float32)

        return I_up - I_dn

    def energy(self, u_fn, params: Dict[str, tf.Tensor]):
        """
        预紧功近似：W_pre = Σ_i  P_i * Δ_i
        其中 P_i 来自 params["P"] (shape=(3,))，Δ_i 来自 _bolt_delta。
        返回 (W_pre, stats)；stats 里附带每个 bolt 的 Δ。
        """
        if not self._bolts:
            zero = tf.constant(0.0, dtype=tf.float32)
            return zero, {"preload": {"bolt_deltas": tf.zeros((0,), tf.float32)}}

        P = tf.convert_to_tensor(params.get("P", [0.0, 0.0, 0.0]), dtype=tf.float32)  # (3,)
        nb = len(self._bolts)
        nb_tf = tf.constant(nb, dtype=tf.int32)
        # 截断/补零到 nb（保持在图模式下使用张量逻辑，避免 Python 布尔比较）
        p_len = tf.shape(P)[0]

        def _pad():
            pad = nb_tf - p_len
            zeros = tf.zeros((pad,), dtype=tf.float32)
            return tf.concat([P, zeros], axis=0)

        def _truncate():
            return P[:nb]

        P = tf.cond(p_len < nb_tf, _pad, _truncate)
        P = P[:nb]

        stage_rank = params.get("stage_rank", None)
        if stage_rank is not None:
            rank_vec = tf.convert_to_tensor(stage_rank, dtype=tf.float32)
            rank_vec = rank_vec[:nb]
            relax = float(getattr(self.cfg, "rank_relaxation", 0.0) or 0.0)
            if relax != 0.0:
                if nb > 1:
                    coeff = tf.constant(relax, dtype=tf.float32)
                    center = tf.constant(0.5, dtype=tf.float32)
                    scale = 1.0 + coeff * (center - rank_vec)
                else:
                    scale = tf.ones_like(rank_vec)
                P = P * scale

        deltas = []
        for bolt in self._bolts:
            di = self._bolt_delta(u_fn, params, bolt)   # 标量
            deltas.append(di)
        delta_vec = tf.stack(deltas, axis=0)            # (nb,)

        W_pre = tf.reduce_sum(P[:nb] * delta_vec) * tf.constant(self.cfg.work_coeff, dtype=tf.float32)
        stats = {"preload": {"bolt_deltas": delta_vec}}
        return W_pre, stats
