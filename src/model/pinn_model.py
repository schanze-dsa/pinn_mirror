#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pinn_model.py
-------------
Displacement field network for DFEM/PINN with preload conditioning.

Components:
- ParamEncoder: encodes normalized preload features -> condition z
  - GaussianFourierFeatures: optional positional encoding for coordinates
  - DisplacementNet: Graph neural network (GCN) backbone; inputs [x_feat, z] -> u(x; P)

Public factory:
    model = create_displacement_model(cfg)      # returns DisplacementModel
    u = model.u_fn(X, params)                   # X: (N,3) mm (normalized outside if needed)
                                               # params: dict; must contain either:
                                               #   "P_hat": preload feature vector; staged 情况下
                                               #           包含 [P_hat, mask, last, rank]，长度
                                               #           为 4*n_bolts
                                               # or "P": (3,) with "preload_shift/scale" in cfg

Notes:
- This file只关注“网络前向”，不做物理装配；训练循环将把本模型与能量/接触算子组合。
- 激活默认 SiLU；可选 GELU/RELU/Tanh。
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
    sigmas: Optional[Tuple[float, ...]] = (1.0, 10.0, 50.0)  # multi-scale sigmas; if set, overrides sigma
    trainable: bool = False    # whether to learn B instead of keeping it frozen

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
    # 以下 legacy MLP 参数仅保留兼容性；当前实现始终走 GCN 主干
    width: int = 256
    depth: int = 7
    act: str = "silu"
    residual_skips: Tuple[int, int] = (3, 6)
    out_dim: int = 3          # displacement ux,uy,uz
    stress_out_dim: int = 6   # 应力分量输出维度（默认 6: σxx,σyy,σzz,σxy,σyz,σxz）；<=0 关闭应力头
    use_graph: bool = True    # 是否启用 GCN 主干；若为 False 将报错
    graph_k: int = 12         # kNN 图中的邻居数量
    graph_knn_chunk: int = 1024  # 构建 kNN/图卷积时每批处理的节点数量
    graph_precompute: bool = False  # 是否在构建阶段预计算全局邻接并缓存
    graph_layers: int = 4     # 图卷积层数
    graph_width: int = 192    # 每层的隐藏特征维度
    graph_dropout: float = 0.0
    # 基于条件向量的 FiLM 调制
    use_film: bool = True
    # 仅为兼容旧版残差开关（已移除残差实现）；保留字段避免加载旧配置时报错
    graph_residual: bool = False
    # 简单硬约束掩码：以圆孔为例，半径内强制位移为 0，可选开启
    hard_bc_radius: Optional[float] = None
    hard_bc_center: Tuple[float, float] = (0.0, 0.0)
    hard_bc_dims: Tuple[bool, bool, bool] = (True, True, True)
    # 输出缩放：网络预测无量纲位移后再乘以该尺度，便于微小量级的数值稳定
    output_scale: float = 1.0e-2
    output_scale_trainable: bool = False
    
    # DFEM mode: use learnable node embeddings instead of spatial coordinates
    dfem_mode: bool = False           # Enable pure DFEM mode
    n_nodes: Optional[int] = None     # Total number of mesh nodes (required if dfem_mode=True)
    node_emb_dim: int = 64            # Dimension of learnable node embeddings

@dataclass
class ModelConfig:
    encoder: EncoderConfig = EncoderConfig()
    field: FieldConfig = FieldConfig()
    mixed_precision: Optional[str] = None      # None|'float16'|'bfloat16'
    preload_shift: float = 500.0               # for P normalization if only "P" is given
    preload_scale: float = 1500.0              # P_hat = (P - shift)/scale


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
    Map 3D coordinates x -> concat_k [sin(B_k x), cos(B_k x)] with B_k ~ N(0, sigma_k^2).
    - 支持多尺度 sigma_k（例如 [1,10,50]），每个尺度采样 num 个频率后拼接。
    - 可选让 B_k 变为 trainable，以便网络自适应频段。默认保持冻结。
    Mixed precision 兼容策略：
    - 统一在 float32 中进行 matmul/sin/cos/concat，再 cast 回输入 dtype（通常是 float16）。
    """

    def __init__(
        self,
        in_dim: int,
        num: int,
        sigma: float,
        sigmas: Optional[Tuple[float, ...]] = None,
        trainable: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_dim = in_dim
        self.num = int(num)
        self.sigma = float(sigma)
        self.sigmas = tuple(sigmas) if sigmas is not None else None
        self.trainable_B = bool(trainable)
        self.B_list: list[tf.Variable] = []

    def build(self, input_shape):
        if self.num <= 0:
            return
        rng = tf.random.Generator.from_non_deterministic_state()
        sigmas = self.sigmas if self.sigmas else (self.sigma,)
        for idx, sig in enumerate(sigmas):
            B_np = rng.normal(shape=(self.in_dim, self.num), dtype=tf.float32) * float(sig)
            self.B_list.append(
                tf.Variable(B_np, trainable=self.trainable_B, name=f"B_fourier_{idx}")
            )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        if self.num <= 0 or not self.B_list:
            return x
        # ---- 修复 dtype 不匹配：在 float32 里计算，最后再 cast 回来 ----
        x32 = tf.cast(x, tf.float32)      # (N, in_dim)
        feat_bands = []
        for B in self.B_list:
            B32 = tf.cast(B, tf.float32)  # (in_dim, num)
            xb32 = tf.matmul(x32, B32)    # (N, num) float32
            feat_bands.append(tf.sin(xb32))
            feat_bands.append(tf.cos(xb32))
        feat32 = tf.concat(feat_bands + [x32], axis=-1)
        return tf.cast(feat32, x.dtype)   # 回到与输入一致的 dtype（mixed_float16 下为 float16）

    @property
    def out_dim(self) -> int:
        if self.num <= 0:
            return self.in_dim
        n_bands = len(self.sigmas) if self.sigmas else 1
        return n_bands * self.num * 2 + self.in_dim


class MLP(tf.keras.layers.Layer):
    """Simple MLP block with configurable depth/width/activation."""

    def __init__(
        self,
        width: int,
        depth: int,
        act: str,
        final_dim: Optional[int] = None,
        dtype: Optional[tf.dtypes.DType] = None,
    ):
        super().__init__()
        self.width = width
        self.depth = depth
        self.act = _get_activation(act)
        self.final_dim = final_dim
        self._dense_dtype = dtype

        self.layers_dense = []
        for i in range(depth):
            dense_kwargs = {
                "units": width,
                "kernel_initializer": "he_uniform",
            }
            if self._dense_dtype is not None:
                dense_kwargs["dtype"] = self._dense_dtype
            self.layers_dense.append(tf.keras.layers.Dense(**dense_kwargs))
        if final_dim is not None:
            final_kwargs = {
                "units": final_dim,
                "kernel_initializer": "glorot_uniform",
            }
            if self._dense_dtype is not None:
                final_kwargs["dtype"] = self._dense_dtype
            self.final_dense = tf.keras.layers.Dense(**final_kwargs)
        else:
            self.final_dense = None

    def call(self, x: tf.Tensor) -> tf.Tensor:
        y = x
        for i in range(self.depth):
            y = self.layers_dense[i](y)
            y = self.act(y)
        if self.final_dense is not None:
            y = self.final_dense(y)
        return y


class GraphConvLayer(tf.keras.layers.Layer):
    """简单的消息传递层：聚合 kNN 邻居并结合相对坐标统计。"""

    def __init__(
        self,
        hidden_dim: int,
        k: int,
        act: str,
        dropout: float = 0.0,
        chunk_size: int | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k = max(int(k), 1)
        self.act = _get_activation(act)
        self.dropout = float(max(dropout, 0.0))
        # chunk_size 参数仅为向后兼容保留；新实现为一次性并行求解
        self._unused_chunk = chunk_size
        self.lin = tf.keras.layers.Dense(
            hidden_dim,
            kernel_initializer="he_uniform",
        )

    def call(
        self,
        feat: tf.Tensor,
        coords: tf.Tensor,
        knn_idx: tf.Tensor,
        adj: tf.sparse.SparseTensor | None = None,
        training: bool | None = False,
    ) -> tf.Tensor:
        """
        feat   : (N, C)
        coords : (N, 3)
        knn_idx: (N, K)
        adj    : (N, N) Normalized SparseTensor (Optional, preferred for memory efficiency)
        """
        input_dtype = feat.dtype
        feat = tf.ensure_shape(feat, (None, self.hidden_dim))
        coords = tf.cast(coords, input_dtype)
        coords = tf.ensure_shape(coords, (None, 3))
        knn_idx = tf.ensure_shape(knn_idx, (None, self.k))

        # --- Optimization: Use Sparse MatMul if adj provided ---
        if adj is not None:
             # adj scale is 1/k, so matmul performs the mean aggregation
             # sparse_dense_matmul requires matching dtypes. adj is typically float32.
             # If feat is float16 (mixed precision), we must cast to float32 temporarily.
             if adj.values.dtype != feat.dtype:
                 agg = tf.sparse.sparse_dense_matmul(adj, tf.cast(feat, adj.values.dtype))
                 agg = tf.cast(agg, feat.dtype)
             else:
                 agg = tf.sparse.sparse_dense_matmul(adj, feat)  # (N, C)
        else:
             neighbors = tf.gather(feat, knn_idx)  # (N, K, C)
             neighbors.set_shape([None, self.k, self.hidden_dim])
             agg = tf.reduce_mean(neighbors, axis=1)  # (N, C)
        
        agg.set_shape([None, self.hidden_dim])

        if adj is not None:
            # Compute rel_mean and rel_std using sparse ops to avoid gather(coords)
            # which produces sparse gradients (IndexedSlices) and triggers warnings.
            # rel_mean = mean(x_j) - x_i
            # rel_std = std(x_j - x_i) = std(x_j)
            
            c_dtype = coords.dtype
            # Ensure float32 for matmul if mixed precision
            if adj.values.dtype != c_dtype:
                 coords_32 = tf.cast(coords, adj.values.dtype)
                 mean_x = tf.sparse.sparse_dense_matmul(adj, coords_32)
                 mean_x = tf.cast(mean_x, c_dtype)
                 
                 # E[x^2]
                 mean_sq_x = tf.sparse.sparse_dense_matmul(adj, tf.square(coords_32))
                 mean_sq_x = tf.cast(mean_sq_x, c_dtype)
            else:
                 mean_x = tf.sparse.sparse_dense_matmul(adj, coords)
                 # E[x^2]
                 mean_sq_x = tf.sparse.sparse_dense_matmul(adj, tf.square(coords))

            rel_mean = mean_x - coords
            # Var = E[x^2] - E[x]^2. Use relu for numerical stability.
            var_x = tf.nn.relu(mean_sq_x - tf.square(mean_x))
            rel_std = tf.sqrt(var_x)
        else:
            nbr_coords = tf.gather(coords, knn_idx)  # (N, K, 3)
            nbr_coords.set_shape([None, self.k, 3])
            rel = nbr_coords - tf.expand_dims(coords, axis=1)
            rel_mean = tf.reduce_mean(rel, axis=1)
            rel_std = tf.math.reduce_std(rel, axis=1)
        rel_feat = tf.concat([rel_mean, rel_std], axis=-1)  # (N, 6)
        rel_feat.set_shape([None, 6])

        mix = tf.concat([feat, agg, rel_feat], axis=-1)
        out = self.lin(mix)
        out = self.act(out)
        if self.dropout > 0.0 and training:
            out = tf.nn.dropout(out, rate=self.dropout)
        return out


def _build_knn_graph(x: tf.Tensor, k: int, chunk_size: int) -> tf.Tensor:
    """
    返回每个点的 k 个邻居索引 (N, k)。

    早期实现即便做了按行分块，依旧需要为每个行块一次性构造大小为
    (chunk × N) 的距离矩阵，N 动辄上万时会产生数百 MB 的瞬时分配，从而
    触发 GPU OOM。这里改为 *双层分块*：对于每个行块，再按列块遍历全集，
    仅保留当前行块的 top-k 中间结果，使得任一时刻只需保存
    (chunk × chunk) 的距离矩阵，内存需求降到线性级别。
    """

    x = tf.cast(x, tf.float32)
    n = tf.shape(x)[0]
    k = max(int(k), 1)
    chunk = max(int(chunk_size), 1)
    chunk = min(chunk, 1024)
    k_const = tf.constant(k, dtype=tf.int32)
    chunk_const = tf.constant(chunk, dtype=tf.int32)
    large_val = tf.constant(1e30, dtype=tf.float32)

    def _empty():
        return tf.zeros((0, k), dtype=tf.int32)

    def _build():
        with tf.device("/CPU:0"):
            x_sq = tf.reduce_sum(tf.square(x), axis=1)  # (N,)
            ta = tf.TensorArray(
                dtype=tf.int32,
                size=0,
                dynamic_size=True,
                clear_after_read=False,
                element_shape=None,  # Allow variable-sized chunks
                infer_shape=False,    # Disable shape inference in while_loop
            )

            def _cond(start, *_):
                return tf.less(start, n)

            def _body(start, ta_handle, write_idx):
                end = tf.minimum(n, start + chunk_const)
                rows = tf.range(start, end)
                chunk_len = tf.shape(rows)[0]
                x_chunk = tf.gather(x, rows)
                chunk_sq = tf.gather(x_sq, rows)
                best_shape = tf.stack([chunk_len, k_const])
                best_dist = tf.fill(best_shape, large_val)
                best_idx = tf.zeros(best_shape, dtype=tf.int32)

                def _inner_cond(col_start, *_):
                    return tf.less(col_start, n)

                def _inner_body(col_start, best_d, best_i):
                    col_end = tf.minimum(n, col_start + chunk_const)
                    cols = tf.range(col_start, col_end)
                    x_cols = tf.gather(x, cols)
                    col_sq = tf.gather(x_sq, cols)
                    dist = (
                        tf.expand_dims(chunk_sq, 1)
                        + tf.expand_dims(col_sq, 0)
                        - 2.0 * tf.matmul(x_chunk, x_cols, transpose_b=True)
                    )
                    dist = tf.maximum(dist, 0.0)
                    same = tf.cast(
                        tf.equal(tf.expand_dims(rows, 1), tf.expand_dims(cols, 0)),
                        dist.dtype,
                    )
                    dist = dist + same * 1e9

                    combined_dist = tf.concat([best_d, dist], axis=1)
                    tiled_cols = tf.tile(
                        tf.expand_dims(tf.cast(cols, tf.int32), 0), [chunk_len, 1]
                    )
                    combined_idx = tf.concat([best_i, tiled_cols], axis=1)

                    neg_dist = -combined_dist
                    vals, top_idx = tf.math.top_k(neg_dist, k=k)
                    new_best_dist = -vals
                    new_best_idx = tf.gather(combined_idx, top_idx, batch_dims=1)
                    return col_end, new_best_dist, new_best_idx

                start_inner = tf.constant(0, dtype=tf.int32)
                _, best_final, idx_final = tf.while_loop(
                    _inner_cond,
                    _inner_body,
                    (start_inner, best_dist, best_idx),
                    parallel_iterations=1,
                )
                ta_handle = ta_handle.write(write_idx, idx_final)
                return end, ta_handle, write_idx + 1

            start0 = tf.constant(0, dtype=tf.int32)
            write0 = tf.constant(0, dtype=tf.int32)
            _, ta_final, _ = tf.while_loop(
                _cond, _body, (start0, ta, write0), parallel_iterations=1
            )
            return ta_final.concat()

    return tf.cond(tf.equal(n, 0), _empty, _build)


def _knn_to_adj(knn_idx: tf.Tensor, n_nodes: int | tf.Tensor) -> tf.sparse.SparseTensor:
    """
    Convert (N, K) knn indices to normalized (N, N) sparse adjacency matrix.
    Values are 1.0/K (row-normalized).
    """
    knn_idx = tf.cast(knn_idx, tf.int64)
    N = tf.shape(knn_idx)[0]
    K = tf.shape(knn_idx)[1]
    
    # Construct indices: (row, col)
    # rows: [0,0,..,0, 1,1,..,1, ...]
    row_idx = tf.repeat(tf.range(N, dtype=tf.int64), repeats=K)
    col_idx = tf.reshape(knn_idx, [-1])
    
    indices = tf.stack([row_idx, col_idx], axis=1) # (N*K, 2)
    
    # Values: 1/K
    val = tf.cast(1.0 / tf.cast(K, tf.float32), tf.float32)
    values = tf.fill([N * K], val)
    
    # Sort indices (required for sparse operations)
    # Since we constructed row_idx sequentially, it should be sorted by row, 
    # but strictly allow sparse_reorder to ensure correctness if col order matters or implementation changes.
    sp = tf.sparse.SparseTensor(
        indices=indices,
        values=values,
        dense_shape=[tf.cast(n_nodes, tf.int64), tf.cast(n_nodes, tf.int64)]
    )
    return tf.sparse.reorder(sp)


# -----------------------------
# Networks
# -----------------------------

class ParamEncoder(tf.keras.layers.Layer):
    """Encode normalized preload vector (P_hat) to a condition vector z."""
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.in_dim = int(getattr(cfg, "in_dim", 0) or 0)
        self.mlp = MLP(
            width=cfg.width,
            depth=cfg.depth,
            act=cfg.act,
            final_dim=cfg.out_dim,
        )

    def call(self, P_hat: tf.Tensor) -> tf.Tensor:
        # Ensure 2D: (B,3)
        if P_hat.shape.rank == 1:
            P_hat = tf.reshape(P_hat, (1, -1))
        P_hat = self._normalize_dim(P_hat)
        return self.mlp(P_hat)  # (B, out_dim)

    def _normalize_dim(self, P_hat: tf.Tensor) -> tf.Tensor:
        """Pad/trim P_hat to the configured input dim so encoder weight shapes一致。"""

        target = self.in_dim
        if target <= 0:
            return P_hat

        # 静态形状已匹配则直接返回
        if P_hat.shape.rank is not None and P_hat.shape[-1] == target:
            P_hat.set_shape((None, target))
            return P_hat

        cur = tf.shape(P_hat)[-1]
        target_tf = tf.cast(target, tf.int32)

        # Avoid tf.cond to prevent trace-time Optional type inconsistencies when using
        # mixed precision (half vs int32). We pad with zeros only when needed, then
        # slice to the target width so both under- and over-length inputs are handled
        # in a single branch with consistent dtypes.
        pad_width = tf.maximum(target_tf - cur, 0)
        pad_zeros = tf.zeros((tf.shape(P_hat)[0], pad_width), dtype=P_hat.dtype)
        padded = tf.concat([P_hat, pad_zeros], axis=-1)
        adjusted = padded[:, :target_tf]
        adjusted.set_shape((None, target))
        return adjusted


class DisplacementNet(tf.keras.Model):
    """
    Core field network: input features = [x_feat, z_broadcast] -> u
    - x_feat = pe(x) if PE enabled else x
    - z is per-parameter vector; we broadcast to match number of spatial samples
    """
    def __init__(self, cfg: FieldConfig):
        super().__init__()
        self.cfg = cfg
        self.use_graph = bool(cfg.use_graph)
        self.use_film = bool(getattr(cfg, "use_film", False))

        # Fourier PE (used if not in DFEM mode)
        self.pe = GaussianFourierFeatures(
            in_dim=cfg.in_dim_coord,
            num=cfg.fourier.num,
            sigma=cfg.fourier.sigma,
            sigmas=cfg.fourier.sigmas,
            trainable=cfg.fourier.trainable,
        )

        # DFEM mode: learnable node embeddings instead of positional encoding
        self.dfem_mode = cfg.dfem_mode
        if self.dfem_mode:
            if cfg.n_nodes is None or cfg.n_nodes <= 0:
                raise ValueError(
                    "FieldConfig.dfem_mode=True requires n_nodes > 0, "
                    f"got {cfg.n_nodes}"
                )
            self.n_nodes = cfg.n_nodes
            # Learnable embeddings for each node
            self.node_embeddings = tf.Variable(
                tf.random.normal((self.n_nodes, cfg.node_emb_dim), stddev=0.02),
                trainable=True,
                name="node_embeddings"
            )
            in_dim_total = cfg.node_emb_dim + cfg.cond_dim
        else:
            in_dim_total = self.pe.out_dim + cfg.cond_dim
            
        if not self.use_graph:
            raise ValueError(
                "FieldConfig.use_graph=False 已不再支持；请开启 GCN 主干以运行位移网络。"
            )

        self.graph_proj = tf.keras.layers.Dense(
            cfg.graph_width,
            kernel_initializer="he_uniform",
        )
        self.graph_layers = [
            GraphConvLayer(
                hidden_dim=cfg.graph_width,
                k=cfg.graph_k,
                act=cfg.act,
                dropout=cfg.graph_dropout,
                chunk_size=cfg.graph_knn_chunk,
            )
            for _ in range(cfg.graph_layers)
        ]
        # FiLM 调制：为每层准备 γ/β，初始为恒等（γ=1, β=0）
        self.film_gamma: list[tf.keras.layers.Layer] = []
        self.film_beta: list[tf.keras.layers.Layer] = []
        if self.use_film:
            for li in range(cfg.graph_layers):
                self.film_gamma.append(
                    tf.keras.layers.Dense(
                        cfg.graph_width,
                        kernel_initializer="zeros",
                        bias_initializer="ones",
                        name=f"film_gamma_{li}",
                    )
                )
                self.film_beta.append(
                    tf.keras.layers.Dense(
                        cfg.graph_width,
                        kernel_initializer="zeros",
                        bias_initializer="zeros",
                        name=f"film_beta_{li}",
                    )
                )
        self.graph_norm = tf.keras.layers.LayerNormalization(axis=-1)
        self.graph_out = tf.keras.layers.Dense(
            cfg.out_dim,
            kernel_initializer="glorot_uniform",
        )
        self.stress_out = None
        if cfg.stress_out_dim > 0:
            self.stress_out = tf.keras.layers.Dense(
                cfg.stress_out_dim,
                kernel_initializer="glorot_uniform",
                name="stress_head",
            )
        # 全局邻接缓存（可选）
        self._global_knn_idx: Optional[tf.Tensor] = None
        self._global_adj: Optional[tf.sparse.SparseTensor] = None
        self._global_knn_n: Optional[int] = None

        # 输出缩放（可选可训练），便于微小位移量级的数值稳定
        scale_init = tf.constant(getattr(cfg, "output_scale", 1.0), dtype=tf.float32)
        if getattr(cfg, "output_scale_trainable", False):
            self.output_scale = tf.Variable(scale_init, trainable=True, name="output_scale")
        else:
            self.output_scale = tf.cast(scale_init, tf.float32)
    
    def prebuild_adjacency(self, X_nodes: tf.Tensor | np.ndarray):
        """
        Pre-build and cache the adjacency graph using node coordinates.
        Should be called once during initialization with all mesh node coordinates.
        
        Args:
            X_nodes: (N_nodes, 3) node coordinates
        """
        if not self.dfem_mode:
            # For traditional PINN, this is optional but can still cache
            pass
            
        X_nodes = tf.convert_to_tensor(X_nodes, dtype=tf.float32)
        n_nodes = tf.shape(X_nodes)[0]
        
        # Build KNN graph
        knn_idx = _build_knn_graph(X_nodes, self.cfg.graph_k, self.cfg.graph_knn_chunk)
        adj = _knn_to_adj(knn_idx, n_nodes)
        
        # Cache
        self._global_knn_idx = knn_idx
        self._global_adj = adj
        self._global_knn_n = int(n_nodes.numpy() if hasattr(n_nodes, 'numpy') else n_nodes)
        
        print(f"[DisplacementNet] Pre-built adjacency graph: {self._global_knn_n} nodes, k={self.cfg.graph_k}")

    def call(
        self,
        x: tf.Tensor,
        z: tf.Tensor,
        training: bool | None = False,
        return_stress: bool = False,
    ) -> tf.Tensor | Tuple[tf.Tensor, tf.Tensor]:
        """
        x : (N,3) coordinates (already normalized if you采用归一化)
        z : (B,cond_dim) or (cond_dim,)
        Returns:
            u: (N,3)
        """
        x = tf.convert_to_tensor(x)
        z = tf.convert_to_tensor(z)
        
        # Ensure z is 2D: (B, cond_dim)
        # Static shape check if possible, otherwise dynamic
        if z.shape.rank is not None and z.shape.rank == 1:
            z = tf.reshape(z, (1, -1))
        
        # Broadcast z to N samples
        # logic: if B != 1 and B != N, fallback to B=1; then broadcast B=1 to N
        
        N = tf.shape(x)[0]
        B = tf.shape(z)[0]

        # --- 修复点 1：处理 Fallback 逻辑 ---
        # 原代码: if tf.not_equal(B, 1) and tf.not_equal(B, N): ...
        # 新代码: 使用 tf.cond
        condition_fallback = tf.logical_and(tf.not_equal(B, 1), tf.not_equal(B, N))
        z = tf.cond(condition_fallback, lambda: z[:1], lambda: z)
        
        # 更新 B (因为 z 可能变了)
        B = tf.shape(z)[0]

        # --- 修复点 2：处理广播逻辑 (你现在的报错点) ---
        # 原代码: if tf.equal(B, 1): ... else: ...
        # 新代码: 使用 tf.cond
        zb = tf.cond(
            tf.equal(B, 1), 
            lambda: tf.repeat(z, repeats=N, axis=0), 
            lambda: z
        )

        # --- 后续逻辑保持不变 ---
        feat_dtype = x.dtype
        if zb.dtype != feat_dtype:
            zb = tf.cast(zb, feat_dtype)

        # DFEM mode: use node embeddings; Traditional: use positional encoding
        if self.dfem_mode:
            # x should contain node indices in DFEM mode: (N,) or (N,1) or (N,3) ignored
            # We use implicit indexing: x[i] corresponds to node i
            node_indices = tf.range(N, dtype=tf.int32)
            x_feat = tf.gather(self.node_embeddings, node_indices)  # (N, node_emb_dim)
        else:
            # Traditional PINN: positional encoding of coordinates
            x_feat = self.pe(x)  # (N, fourier_dim)
            
        if x_feat.dtype != feat_dtype:
            x_feat = tf.cast(x_feat, feat_dtype)
        
        h = tf.concat([x_feat, zb], axis=-1)

        def graph_forward():
            coords = x
            use_cached = (
                self._global_knn_idx is not None
                and self._global_knn_n is not None
                and tf.shape(coords)[0] == self._global_knn_idx.shape[0]
            )
            if use_cached:
                knn_idx = tf.cast(self._global_knn_idx, tf.int32)
                adj = self._global_adj
            else:
                knn_idx = _build_knn_graph(coords, self.cfg.graph_k, self.cfg.graph_knn_chunk)
                # Dynamic construct adj
                adj = _knn_to_adj(knn_idx, tf.shape(coords)[0])
                
            hcur = self.graph_proj(h)
            film_gamma = self.film_gamma if self.use_film else None
            film_beta = self.film_beta if self.use_film else None
            for li, layer in enumerate(self.graph_layers):
                hcur = layer(hcur, coords, knn_idx, adj=adj, training=training)
                if film_gamma is not None and film_beta is not None:
                    gamma = film_gamma[li](zb)
                    beta = film_beta[li](zb)
                    gamma = tf.cast(gamma, hcur.dtype)
                    beta = tf.cast(beta, hcur.dtype)
                    hcur = gamma * hcur + beta
            hcur = self.graph_norm(hcur)
            u_out = self.graph_out(hcur)

            # 输出缩放：网络预测无量纲位移，再映射到物理量级
            scale = tf.cast(self.output_scale, u_out.dtype)
            u_out = u_out * scale

            # 可选硬约束：以圆孔为例，在半径内直接将位移投影为 0，减少软约束漏出
            if self.cfg.hard_bc_radius is not None and float(self.cfg.hard_bc_radius) > 0.0:
                cx, cy = self.cfg.hard_bc_center
                dx = coords[:, 0] - tf.cast(cx, coords.dtype)
                dy = coords[:, 1] - tf.cast(cy, coords.dtype)
                r2 = dx * dx + dy * dy
                mask = tf.cast(r2 > tf.cast(self.cfg.hard_bc_radius, coords.dtype) ** 2, u_out.dtype)
                dof_mask = tf.convert_to_tensor(self.cfg.hard_bc_dims, dtype=u_out.dtype)
                u_out = u_out * mask[:, None] * dof_mask
            if return_stress:
                if self.stress_out is None:
                    raise ValueError("stress head disabled (stress_out_dim<=0)")
                sigma_out = self.stress_out(hcur)
                return u_out, sigma_out
            return u_out

        return graph_forward()

    def set_global_graph(self, coords: tf.Tensor):
        """预计算并缓存全局 kNN 邻接，用于整图前向以避免分块断图。"""

        coords = tf.convert_to_tensor(coords, dtype=tf.float32)
        k = self.cfg.graph_k
        self._global_knn_idx = _build_knn_graph(coords, k, self.cfg.graph_knn_chunk)
        self._global_knn_n = int(coords.shape[0]) if coords.shape.rank else None
        
        # Precompute sparse adj
        self._global_adj = _knn_to_adj(self._global_knn_idx, self._global_knn_n)


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
        # Alias stress head for backward compatibility with previously traced graphs
        # that referenced `self.stress_out` directly.
        self.stress_out = self.field.stress_out

    def _normalize_inputs(self, X: tf.Tensor, params: Optional[Dict]) -> Tuple[tf.Tensor, tf.Tensor]:
        """Validate/convert inputs and ensure stable shapes for tf.function trace reuse."""
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
        if P_hat.shape.rank == 1:
            P_hat = tf.expand_dims(P_hat, axis=0)

        # P_hat may include staged metadata (mask/last/rank) -> length 4*n_bolts; avoid
        # over-constraining the last dimension. We only ensure rank-2 here and let
        # ParamEncoder._normalize_dim pad/trim to cfg.encoder.in_dim when set.
        P_hat = tf.ensure_shape(P_hat, (None, None))

        X = tf.convert_to_tensor(X, dtype=tf.float32)
        if X.shape.rank == 1:
            X = tf.expand_dims(X, axis=0)
        X = tf.ensure_shape(X, (None, 3))

        return X, P_hat

    @tf.function(
        jit_compile=False,
        reduce_retracing=True,
        input_signature=(
            tf.TensorSpec(shape=(None, 3), dtype=tf.float32, name="X"),
            tf.TensorSpec(shape=(None, None), dtype=tf.float32, name="P_hat"),
        ),
    )
    def _u_fn_compiled(self, X: tf.Tensor, P_hat: tf.Tensor) -> tf.Tensor:
        z = self.encoder(P_hat)          # (B, cond_dim)
        u = self.field(X, z)             # (N,3)
        # Physics operators和能量算子都假定输入为 float32；若启用混合精度，
        # 网络内部会在 float16/bfloat16 下计算，此处统一 cast 回 float32，
        # 以避免如 tie/boundary 约束中出现 "expected float but got half" 的报错。
        return tf.cast(u, tf.float32)

    def u_fn(self, X: tf.Tensor, params: Optional[Dict] = None) -> tf.Tensor:
        """
        Unified forward:
            X: (N,3) float tensor (coordinates; normalized outside if采用归一化)
            params: dict with either
                - 'P_hat': (3,) or (N,3) normalized preload
                - or 'P': (3,) real preload in N + cfg.preload_shift/scale provided
        """
        X, P_hat = self._normalize_inputs(X, params)
        return self._u_fn_compiled(X, P_hat)

    @tf.function(
        jit_compile=False,
        reduce_retracing=True,
        input_signature=(
            tf.TensorSpec(shape=(None, 3), dtype=tf.float32, name="X"),
            tf.TensorSpec(shape=(None, None), dtype=tf.float32, name="P_hat"),
        ),
    )
    def _us_fn_compiled(self, X: tf.Tensor, P_hat: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        if self.field.stress_out is None:
            raise ValueError("stress head disabled (stress_out_dim<=0)")
        z = self.encoder(P_hat)          # (B, cond_dim)
        u, sigma = self.field(X, z, return_stress=True)
        return tf.cast(u, tf.float32), tf.cast(sigma, tf.float32)

    def us_fn(self, X: tf.Tensor, params: Optional[Dict] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        带应力头的前向：返回位移 u 和预测的应力分量 sigma。
        sigma 的维度由 cfg.field.stress_out_dim 决定（默认 6）。
        """
        X, P_hat = self._normalize_inputs(X, params)
        return self._us_fn_compiled(X, P_hat)


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
