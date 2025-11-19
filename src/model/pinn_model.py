#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pinn_model.py
-------------
Displacement field network for DFEM/PINN with preload conditioning.

Components:
- ParamEncoder: encodes normalized preload vector (P1,P2,P3) -> condition z
- GaussianFourierFeatures: optional positional encoding for coordinates
- DisplacementNet: MLP with residual skips; inputs [x_feat, z] -> u(x; P)

Public factory:
    model = create_displacement_model(cfg)      # returns DisplacementModel
    u = model.u_fn(X, params)                   # X: (N,3) mm (normalized outside if needed)
                                               # params: dict; must contain either:
                                               #   "P_hat": (3,) or (N,3) normalized preload
                                               # or "P": (3,) with "preload_shift/scale" in cfg

Notes:
- This file只关注“网络前向”，不做物理装配；训练循环将把本模型与能量/接触算子组合。
- 激活默认 SiLU；可选 GELU/RELU/Tanh/SIREN（sine）。
- 混合精度可选（'float16' 或 'bfloat16'）；权重保持 float32，数值稳定。

Author: you
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
import tensorflow as tf


# -----------------------------
# Config dataclasses
# -----------------------------

@dataclass
class FourierConfig:
    num: int = 8              # number of Gaussian frequencies per axis; 0 -> disable
    sigma: float = 3.0        # std for frequency sampling (larger -> higher freq coverage)

@dataclass
class EncoderConfig:
    in_dim: int = 3           # (P1,P2,P3) normalized
    width: int = 64
    depth: int = 2
    act: str = "silu"         # silu|gelu|relu|tanh
    out_dim: int = 64         # condition vector size

@dataclass
class FieldConfig:
    in_dim_coord: int = 3     # xyz (normalized outside if需要)
    fourier: FourierConfig = FourierConfig()
    cond_dim: int = 64
    width: int = 256
    depth: int = 7
    act: str = "silu"         # silu|gelu|relu|tanh|sine(SIREN)
    residual_skips: Tuple[int, int] = (3, 6)  # add skip from input features to these layers
    out_dim: int = 3          # displacement ux,uy,uz
    w0: float = 30.0          # only for SIREN (sine) first layer frequency
    use_graph: bool = True    # 是否改用 GCN 主干而非传统 MLP
    graph_k: int = 12         # kNN 图中的邻居数量
    graph_layers: int = 4     # 图卷积层数
    graph_width: int = 192    # 每层的隐藏特征维度
    graph_dropout: float = 0.0
    graph_max_nodes: int = 4096  # 节点超过该阈值时自动回退到 MLP，防止构图 OOM

@dataclass
class ModelConfig:
    encoder: EncoderConfig = EncoderConfig()
    field: FieldConfig = FieldConfig()
    mixed_precision: Optional[str] = None      # None|'float16'|'bfloat16'
    preload_shift: float = 200.0               # for P normalization if only "P" is given
    preload_scale: float = 800.0               # P_hat = (P - shift)/scale


# -----------------------------
# Utilities
# -----------------------------

def _get_activation(name: str):
    name = (name or "silu").lower()
    if name == "silu":
        return tf.nn.silu
    if name == "gelu":
        return tf.nn.gelu
    if name == "relu":
        return tf.nn.relu
    if name == "tanh":
        return tf.nn.tanh
    if name == "sine":
        # SIREN-style activation
        def sine(x):
            return tf.sin(x)
        return sine
    raise ValueError(f"Unknown activation '{name}'")

def _maybe_mixed_precision(policy: Optional[str]):
    if policy:
        try:
            tf.keras.mixed_precision.set_global_policy(policy)
            print(f"[pinn_model] Mixed precision policy set to: {policy}")
        except Exception as e:
            print(f"[pinn_model] Failed to set mixed precision '{policy}': {e}")


# -----------------------------
# Layers
# -----------------------------

class GaussianFourierFeatures(tf.keras.layers.Layer):
    """
    Map 3D coordinates x -> [sin(Bx), cos(Bx), x] with B ~ N(0, sigma^2).
    Mixed precision 兼容策略：
    - 统一在 float32 中进行 matmul/sin/cos/concat，再 cast 回输入 dtype（通常是 float16）。
    """
    def __init__(self, in_dim: int, num: int, sigma: float, **kwargs):
        super().__init__(**kwargs)
        self.in_dim = in_dim
        self.num = int(num)
        self.sigma = float(sigma)
        self.B: Optional[tf.Variable] = None  # (in_dim, num)

    def build(self, input_shape):
        if self.num <= 0:
            return
        # fixed random matrix (not trainable)
        rng = tf.random.Generator.from_non_deterministic_state()
        B_np = rng.normal(shape=(self.in_dim, self.num), dtype=tf.float32) * self.sigma
        self.B = tf.Variable(B_np, trainable=False, name="B_fourier")

    def call(self, x: tf.Tensor) -> tf.Tensor:
        if self.num <= 0 or self.B is None:
            return x
        # ---- 修复 dtype 不匹配：在 float32 里计算，最后再 cast 回来 ----
        x32 = tf.cast(x, tf.float32)      # (N, in_dim)
        B32 = tf.cast(self.B, tf.float32) # (in_dim, num)
        xb32 = tf.matmul(x32, B32)        # (N, num) float32
        s32 = tf.sin(xb32)
        c32 = tf.cos(xb32)
        feat32 = tf.concat([s32, c32, x32], axis=-1)  # (N, 2*num + in_dim) float32
        return tf.cast(feat32, x.dtype)   # 回到与输入一致的 dtype（mixed_float16 下为 float16）

    @property
    def out_dim(self) -> int:
        return (0 if self.num <= 0 else self.num * 2) + self.in_dim


class MLP(tf.keras.layers.Layer):
    """Simple MLP block with configurable depth/width/activation."""

    def __init__(
        self,
        width: int,
        depth: int,
        act: str,
        final_dim: Optional[int] = None,
        w0: float = 30.0,
        siren: bool = False,
        dtype: tf.dtypes.DType = tf.float32,
    ):
        super().__init__()
        self.width = width
        self.depth = depth
        self.act = _get_activation(act)
        self.final_dim = final_dim
        self.siren = siren
        self.w0 = w0
        self._dense_dtype = dtype

        self.layers_dense = []
        for i in range(depth):
            self.layers_dense.append(
                tf.keras.layers.Dense(
                    width,
                    kernel_initializer=self._kernel_init(i),
                    dtype=self._dense_dtype,
                )
            )
        if final_dim is not None:
            self.final_dense = tf.keras.layers.Dense(
                final_dim,
                kernel_initializer="glorot_uniform",
                dtype=self._dense_dtype,
            )
        else:
            self.final_dense = None

    def _kernel_init(self, layer_idx: int):
        if self.siren:
            # SIREN initialization (Sitzmann et al.)
            def siren_init(shape, dtype=None):
                in_dim = shape[0]
                if layer_idx == 0:
                    scale = 1.0 / in_dim
                else:
                    scale = tf.sqrt(6.0 / in_dim) / self.w0
                return tf.random.uniform(shape, minval=-scale, maxval=scale, dtype=dtype)
            return siren_init
        # default
        return "he_uniform"

    def call(self, x: tf.Tensor) -> tf.Tensor:
        y = x
        if self.siren:
            # first layer with w0 scaling
            y = self.layers_dense[0](y)
            y = tf.sin(self.w0 * y)
            for i in range(1, self.depth):
                y = self.layers_dense[i](y)
                y = tf.sin(y)
        else:
            for i in range(self.depth):
                y = self.layers_dense[i](y)
                y = self.act(y)
        if self.final_dense is not None:
            y = self.final_dense(y)
        return y


class GraphConvLayer(tf.keras.layers.Layer):
    """简单的消息传递层：聚合 kNN 邻居并结合相对坐标统计。"""

    def __init__(self, hidden_dim: int, k: int, act: str, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k = max(int(k), 1)
        self.act = _get_activation(act)
        self.dropout = float(max(dropout, 0.0))
        self.lin = tf.keras.layers.Dense(
            hidden_dim,
            kernel_initializer="he_uniform",
            dtype=tf.float32,
        )

    def call(
        self,
        feat: tf.Tensor,
        coords: tf.Tensor,
        knn_idx: tf.Tensor,
        training: bool | None = False,
    ) -> tf.Tensor:
        """
        feat   : (N, C)
        coords : (N, 3)
        knn_idx: (N, K)
        """
        neighbors = tf.gather(feat, knn_idx)                 # (N, K, C)
        agg = tf.reduce_mean(neighbors, axis=1)              # (N, C)

        nbr_coords = tf.gather(coords, knn_idx)              # (N, K, 3)
        rel = nbr_coords - tf.expand_dims(coords, axis=1)    # (N, K, 3)
        rel_mean = tf.reduce_mean(rel, axis=1)
        rel_std = tf.math.reduce_std(rel, axis=1)
        rel_feat = tf.concat([rel_mean, rel_std], axis=-1)   # (N, 6)
        rel_feat = tf.cast(rel_feat, feat.dtype)

        mix = tf.concat([feat, agg, rel_feat], axis=-1)
        out = self.lin(mix)
        out = self.act(out)
        if self.dropout > 0.0 and training:
            out = tf.nn.dropout(out, rate=self.dropout)
        return out


def _pairwise_distances(x: tf.Tensor) -> tf.Tensor:
    x = tf.cast(x, tf.float32)
    x_sq = tf.reduce_sum(tf.square(x), axis=1, keepdims=True)
    dist = x_sq - 2.0 * tf.matmul(x, x, transpose_b=True) + tf.transpose(x_sq)
    dist = tf.maximum(dist, 0.0)
    return dist


def _build_knn_graph(x: tf.Tensor, k: int) -> tf.Tensor:
    """返回每个点的 k 个邻居索引 (N, k)。"""
    x = tf.cast(x, tf.float32)
    n = tf.shape(x)[0]
    k = max(int(k), 1)
    k_const = tf.constant(k, dtype=tf.int32)

    dist = _pairwise_distances(x)
    mask = tf.eye(n, dtype=tf.float32) * 1e9
    dist = dist + mask
    idx = tf.argsort(dist, axis=-1)[:, :k]  # (N, <=k)
    cur_k = tf.shape(idx)[1]
    def _pad_needed():
        deficit = k_const - cur_k
        pad = tf.tile(idx[:, -1:], [1, deficit])
        return tf.concat([idx, pad], axis=1)

    idx = tf.cond(tf.less(cur_k, k_const), _pad_needed, lambda: idx)
    return tf.cast(idx, tf.int32)


# -----------------------------
# Networks
# -----------------------------

class ParamEncoder(tf.keras.layers.Layer):
    """Encode normalized preload vector (P_hat) to a condition vector z."""
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.mlp = MLP(
            width=cfg.width,
            depth=cfg.depth,
            act=cfg.act,
            final_dim=cfg.out_dim,
            dtype=tf.float32,
        )

    def call(self, P_hat: tf.Tensor) -> tf.Tensor:
        # Ensure 2D: (B,3)
        if P_hat.shape.rank == 1:
            P_hat = tf.reshape(P_hat, (1, -1))
        return self.mlp(P_hat)  # (B, out_dim)


class DisplacementNet(tf.keras.Model):
    """
    Core field network: input features = [x_feat, z_broadcast] -> u
    - x_feat = pe(x) if PE enabled else x
    - z is per-parameter vector; we broadcast to match number of spatial samples
    """
    def __init__(self, cfg: FieldConfig):
        super().__init__()
        self.cfg = cfg
        self.is_siren = (cfg.act.lower() == "sine")
        self.use_graph = bool(cfg.use_graph)
        self.residual_skips = set(cfg.residual_skips or tuple())

        # Fourier PE
        self.pe = GaussianFourierFeatures(
            in_dim=cfg.in_dim_coord, num=cfg.fourier.num, sigma=cfg.fourier.sigma
        )

        in_dim_total = self.pe.out_dim + cfg.cond_dim
        self.act = _get_activation(cfg.act)
        self.w0 = cfg.w0

        # 始终构建 MLP 主干（即使启用 GCN，也可在节点过多时退回）
        self.in_linear = tf.keras.layers.Dense(
            cfg.width,
            kernel_initializer="he_uniform",
            dtype=tf.float32,
        )
        self.blocks = []
        for _ in range(cfg.depth):
            self.blocks.append(
                tf.keras.layers.Dense(
                    cfg.width,
                    kernel_initializer="he_uniform",
                    dtype=tf.float32,
                )
            )
        self.out_linear = tf.keras.layers.Dense(
            cfg.out_dim,
            kernel_initializer="glorot_uniform",
            dtype=tf.float32,
        )

        if self.use_graph:
            self.graph_proj = tf.keras.layers.Dense(
                cfg.graph_width,
                kernel_initializer="he_uniform",
                dtype=tf.float32,
            )
            self.graph_layers = [
                GraphConvLayer(
                    hidden_dim=cfg.graph_width,
                    k=cfg.graph_k,
                    act=cfg.act,
                    dropout=cfg.graph_dropout,
                )
                for _ in range(cfg.graph_layers)
            ]
            self.graph_norm = tf.keras.layers.LayerNormalization(axis=-1)
            self.graph_out = tf.keras.layers.Dense(
                cfg.out_dim,
                kernel_initializer="glorot_uniform",
                dtype=tf.float32,
            )

    def call(self, x: tf.Tensor, z: tf.Tensor, training: bool | None = False) -> tf.Tensor:
        """
        x : (N,3) coordinates (already normalized if you采用归一化)
        z : (B,cond_dim) or (cond_dim,)
        Returns:
            u: (N,3)
        """
        # Mixed-precision policies may feed float16 inputs even though the
        # network itself operates in float32. Force the spatial samples and
        # conditioning vectors back to float32 before further processing.
        x = tf.cast(x, tf.float32)

        # Broadcast z to N samples
        if z.shape.rank == 1:
            z = tf.reshape(z, (1, -1))
        z = tf.cast(z, tf.float32)
        # If B>1 and N>1, support either B==N (per-point conditioning) or B==1 (global)
        N = tf.shape(x)[0]
        B = tf.shape(z)[0]
        if tf.not_equal(B, 1) and tf.not_equal(B, N):
            # fallback: repeat first row
            z = z[:1, :]
            B = 1
        if tf.equal(B, 1):
            zb = tf.repeat(z, repeats=N, axis=0)
        else:
            zb = z  # assume B==N
        zb = tf.cast(zb, tf.float32)

        # positional encoding
        # Positional encoding can inherit a mixed policy dtype (float16) from
        # the caller even though the dense trunk expects float32.  Always cast
        # the encoded features back to float32 before concatenation to avoid
        # dtype mismatches when `zb` stays in float32.
        x_feat = tf.cast(self.pe(x), tf.float32)  # (N, pe_dim)
        h = tf.concat([x_feat, zb], axis=-1)

        def mlp_forward():
            h0 = self.in_linear(h)
            if self.cfg.act == "sine":
                h0 = tf.sin(self.w0 * h0)
            else:
                h0 = self.act(h0)

            hcur = h0
            for idx, dense in enumerate(self.blocks, start=1):
                hcur = dense(hcur)
                if self.cfg.act == "sine":
                    hcur = tf.sin(hcur)
                else:
                    hcur = self.act(hcur)
                if idx in self.residual_skips:
                    hcur = hcur + h0  # simple residual skip

            return self.out_linear(hcur)

        def graph_forward():
            coords = tf.cast(x, tf.float32)
            knn_idx = _build_knn_graph(coords, self.cfg.graph_k)
            hcur = self.graph_proj(h)
            for layer in self.graph_layers:
                hcur = layer(hcur, coords, knn_idx, training=training)
            hcur = self.graph_norm(hcur)
            return self.graph_out(hcur)

        if self.use_graph:
            max_nodes = int(getattr(self.cfg, "graph_max_nodes", 0) or 0)
            if max_nodes > 0:
                max_nodes_const = tf.constant(max_nodes, dtype=tf.int32)
                use_graph = tf.less_equal(tf.shape(x)[0], max_nodes_const)
                u = tf.cond(use_graph, graph_forward, mlp_forward)
            else:
                u = graph_forward()
        else:
            u = mlp_forward()
        # 在半精度策略下，强制回落到 float32，避免在物理能量项中产生 NaN
        return tf.cast(u, tf.float32)


# -----------------------------
# Wrapper model with unified u_fn
# -----------------------------

class DisplacementModel:
    """
    High-level wrapper that holds:
      - ParamEncoder (P_hat -> z)
      - DisplacementNet ([x_feat, z] -> u)

    Provides:
      - u_fn(X, params): unified forward callable for energy modules.
    """
    def __init__(self, cfg: ModelConfig):
        _maybe_mixed_precision(cfg.mixed_precision)
        self.cfg = cfg
        self.encoder = ParamEncoder(cfg.encoder)
        # Ensure field.cond_dim == encoder.out_dim
        if cfg.field.cond_dim != cfg.encoder.out_dim:
            print(f"[pinn_model] Adjust cond_dim from {cfg.field.cond_dim} -> {cfg.encoder.out_dim}")
            cfg.field.cond_dim = cfg.encoder.out_dim
        self.field = DisplacementNet(cfg.field)

    @tf.function(jit_compile=False)
    def u_fn(self, X: tf.Tensor, params: Optional[Dict] = None) -> tf.Tensor:
        """
        Unified forward:
            X: (N,3) float tensor (coordinates; normalized outside if采用归一化)
            params: dict with either
                - 'P_hat': (3,) or (N,3) normalized preload
                - or 'P': (3,) real preload in N + cfg.preload_shift/scale provided
        """
        if params is None:
            raise ValueError("params must contain 'P_hat' or 'P'.")

        if "P_hat" in params:
            P_hat = params["P_hat"]
        elif "P" in params:
            # normalize: (P - shift)/scale
            shift = tf.cast(self.cfg.preload_shift, tf.float32)
            scale = tf.cast(self.cfg.preload_scale, tf.float32)
            P_hat = (tf.convert_to_tensor(params["P"], dtype=tf.float32) - shift) / scale
        else:
            raise ValueError("params must have 'P_hat' or 'P'.")

        P_hat = tf.convert_to_tensor(P_hat, dtype=tf.float32)
        z = self.encoder(P_hat)          # (B, cond_dim)
        u = self.field(tf.cast(X, tf.float32), z)   # (N,3)
        return u


def create_displacement_model(cfg: Optional[ModelConfig] = None) -> DisplacementModel:
    """Factory function to create the high-level displacement model."""
    return DisplacementModel(cfg or ModelConfig())


# -----------------------------
# Minimal smoke test
# -----------------------------
if __name__ == "__main__":
    cfg = ModelConfig(
        encoder=EncoderConfig(in_dim=3, width=64, depth=2, act="silu", out_dim=64),
        field=FieldConfig(
            in_dim_coord=3,
            fourier=FourierConfig(num=8, sigma=3.0),
            cond_dim=64,
            width=256, depth=7, act="silu", residual_skips=(3,6),
            out_dim=3
        ),
        mixed_precision=None,
        preload_shift=200.0, preload_scale=800.0
    )

    model = create_displacement_model(cfg)

    # Fake inputs
    N = 1024
    X = tf.random.uniform((N, 3), minval=-1.0, maxval=1.0)     # assume normalized coords
    P = tf.constant([500.0, 800.0, 300.0], dtype=tf.float32)   # N
    out = model.u_fn(X, {"P": P})
    print("u shape:", out.shape)  # expect (N,3)
