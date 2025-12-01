# -*- coding: utf-8 -*-
# src/physics/elasticity_energy.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import tensorflow as tf

# 约定：dfem_utils.build_dfem_subcells 负责从 AssemblyModel 中预处理 DFEM 子单元
# 返回 B 矩阵、权重、材料参数和 DOF 索引等（见下方 __init__ 的说明）
try:
    from .dfem_utils import build_dfem_subcells
except Exception:  # 兼容直接脚本运行
    from dfem_utils import build_dfem_subcells


# ---------------------------------------------------------------------------- #
# 配置
# ---------------------------------------------------------------------------- #
@dataclass
class ElasticityConfig:
    """
    线弹性 DFEM 内能项的配置。

    - coord_scale:   坐标尺度（目前仅作为统计/占位，不改变 X 的物理值）
    - chunk_size:    评估神经网络 u_fn 时的前向分块大小（降低峰值显存）
    - use_pfor:      仅为向后兼容 trainer 中的统计/配置，不在 DFEM 中实际使用
    - check_nan:     是否在关键张量处进行数值检查
    - n_points_per_step:
        每步用于 DFEM 能量积分的子单元上限：
          * None 或 0 表示使用所有子单元；
          * 正整数 m 表示本步随机子采样 m 个子单元，并做无偏缩放。
    """
    coord_scale: float = 1.0
    chunk_size: int = 8192
    use_pfor: bool = False
    check_nan: bool = False
    n_points_per_step: Optional[int] = None
    # 应力监督：若模型提供应力输出头，可用线弹性理论应力指导，0 表示关闭
    stress_loss_weight: float = 1.0


class ElasticityEnergy:
    r"""
    基于 DFEM 的线弹性内能

        E_int = ∫_Ω [ 1/2 * λ * (tr ε)^2 + μ * (ε : ε) ] dΩ,

    其中小应变张量 ε 由**离散 DOF 与预计算的 B 矩阵**得到：

        ε_k = B_k · u_local,k

    本实现要点：

    - 初始化时，通过 dfem_utils.build_dfem_subcells(...) 对装配模型进行 DFEM 预处理：
        * 将 C3D4 / C3D8 等单元拆成四面体子单元；
        * 对每个子单元预计算：
            B_k      : (6, 12)   Voigt 形式应变-位移矩阵
            w_k      : ()        子单元体积 / 积分权重
            λ_k, μ_k : ()        对应材料的 Lamé 参数
            dof_idx_k: (12,)     指向全局 DOF 向量 u_d 的索引
        * 同时构造全局节点坐标 X_nodes (N,3)，N 为节点数。

    - 训练时，每一步只需要：
        * 用 u_fn 在所有节点坐标 X_nodes 上评估位移，得到全局 DOF 向量 u_d；
        * 子采样若干 DFEM 子单元，利用 B_k、λ_k、μ_k 与 u_local,k 计算能量密度并积分。
    """

    # ------------------------------------------------------------------------ #
    # 初始化：DFEM 预处理
    # ------------------------------------------------------------------------ #
    def __init__(
        self,
        asm: Any,
        part2mat: Dict[str, str],
        materials: Dict[str, Dict[str, float] | tuple],
        cfg: ElasticityConfig,
    ) -> None:
        """
        参数
        ----
        asm : AssemblyModel
            从 inp_parser.load_inp 之类接口得到的装配模型。
        part2mat : dict[str, str]
            零件名 → 材料名 的映射，例如 {"mirror": "jingmian", "auto": "zhijia", ...}。
        materials : dict
            材料名 → 材料属性。
            至少需要提供弹性模量和泊松比，例如：
                {
                    "jingmian": {"E": 7.2e10, "nu": 0.33, ...},
                    "steel":    {"E": 2.1e11, "nu": 0.30, ...},
                    ...
                }
        cfg : ElasticityConfig
            弹性内能相关超参数。
        """
        self.cfg = cfg
        self._scale = float(max(self.cfg.coord_scale, 1e-8))

        # 由 dfem_utils 构建 DFEM 子单元信息。
        # 约定 build_dfem_subcells 返回一个 dict：
        #   {
        #       "X_nodes":  (N,3)    float64/float32，节点物理坐标
        #       "B":        (K,6,12) float32，子单元 B 矩阵
        #       "w":        (K,)     float32，子单元权重 / 体积
        #       "lam":      (K,)     float32，Lamé λ
        #       "mu":       (K,)     float32，Lamé μ
        #       "dof_idx":  (K,12)   int32，全局 DOF 索引（3 自由度 × 4 节点）
        #   }
        # 其中 K 为 DFEM 子单元数量，N 为节点数量。
        dfem_data = build_dfem_subcells(asm, part2mat, materials)

        X_nodes = np.asarray(dfem_data["X_nodes"], dtype=np.float32)
        B = np.asarray(dfem_data["B"], dtype=np.float32)
        w = np.asarray(dfem_data["w"], dtype=np.float32)
        lam = np.asarray(dfem_data["lam"], dtype=np.float32)
        mu = np.asarray(dfem_data["mu"], dtype=np.float32)
        dof_idx = np.asarray(dfem_data["dof_idx"], dtype=np.int32)

        # 基本形状检查
        if X_nodes.ndim != 2 or X_nodes.shape[1] != 3:
            raise ValueError(
                f"[ElasticityEnergy] X_nodes 形状应为 (N,3)，得到 {X_nodes.shape}"
            )
        if B.ndim != 3 or B.shape[1:] != (6, 12):
            raise ValueError(
                f"[ElasticityEnergy] B 形状应为 (K,6,12)，得到 {B.shape}"
            )
        if w.ndim != 1 or lam.ndim != 1 or mu.ndim != 1 or dof_idx.ndim != 2:
            raise ValueError(
                "[ElasticityEnergy] w / lam / mu / dof_idx 维度错误："
                f"{w.shape}, {lam.shape}, {mu.shape}, {dof_idx.shape}"
            )
        if not (B.shape[0] == w.shape[0] == lam.shape[0] == mu.shape[0] == dof_idx.shape[0]):
            raise ValueError("[ElasticityEnergy] DFEM 子单元数量不一致")

        if not np.all(np.isfinite(X_nodes)):
            raise ValueError("[ElasticityEnergy] X_nodes 含 NaN/Inf")
        if not np.all(np.isfinite(B)):
            raise ValueError("[ElasticityEnergy] B 矩阵含 NaN/Inf")
        if not np.all(np.isfinite(w)):
            raise ValueError("[ElasticityEnergy] 权重 w 含 NaN/Inf")
        if not np.all(np.isfinite(lam)) or not np.all(np.isfinite(mu)):
            raise ValueError("[ElasticityEnergy] 材料参数 λ/μ 含 NaN/Inf")

        # 缓存为 Tensor
        self.X_nodes_tf = tf.convert_to_tensor(X_nodes, dtype=tf.float32)   # (N,3)
        self.B_tf = tf.convert_to_tensor(B, dtype=tf.float32)               # (K,6,12)
        self.w_tf = tf.convert_to_tensor(w, dtype=tf.float32)               # (K,)
        self.lam_tf = tf.convert_to_tensor(lam, dtype=tf.float32)           # (K,)
        self.mu_tf = tf.convert_to_tensor(mu, dtype=tf.float32)             # (K,)
        self.dof_idx_tf = tf.convert_to_tensor(dof_idx, dtype=tf.int32)     # (K,12)

        # 记录规模信息与总权重
        self.n_nodes: int = int(X_nodes.shape[0])
        self.n_cells: int = int(B.shape[0])
        self.w_total_tf = tf.reduce_sum(self.w_tf)
        self._forced_indices: Optional[tf.Tensor] = None
        self._last_sample_metrics: Dict[str, Any] = {}

    # -------------------------------------------------------------------- #
    # RAR 支持接口
    # -------------------------------------------------------------------- #
    def set_sample_indices(self, indices: Optional[np.ndarray | tf.Tensor]):
        """指定下一步积分所使用的 DFEM 子单元索引，用于 RAR。"""

        if indices is None:
            self._forced_indices = None
            return
        try:
            arr = np.asarray(indices, dtype=np.int32).reshape(-1)
        except Exception:
            self._forced_indices = None
            return
        if arr.size == 0:
            self._forced_indices = None
            return
        self._forced_indices = tf.convert_to_tensor(arr, dtype=tf.int32)

    def last_sample_metrics(self) -> Dict[str, Any]:
        return dict(self._last_sample_metrics) if self._last_sample_metrics else {}

    # ------------------------------------------------------------------------ #
    # 顶层接口：计算内能
    # ------------------------------------------------------------------------ #
    def energy(
        self,
        u_fn,
        params: Optional[Dict[str, tf.Tensor]] = None,
        tape: Optional[tf.GradientTape] = None,  # 为兼容旧签名，此处不再使用
        return_cache: bool = False,
    ):
        """
        计算 DFEM 弹性内能 E_int 以及一些统计信息 stats。

        与旧实现相比：
        - 不再计算 Jacobian / ∇u，完全依赖预计算的 B 矩阵；
        - 仅需一次前向传播得到所有节点位移；
        - 可选对子采样 DFEM 子单元（n_points_per_step），并做无偏缩放。
        """
        del tape  # 不再使用，仅为兼容旧调用保留形参

        # 1) 在所有节点上评估位移 u(X_nodes)，按节点拼成全局 DOF 向量
        u_nodes = self._eval_u_on_nodes(u_fn, params)  # (N,3)
        # 展平成一维：全局 DOF 向量 u_d，约定 DOF 排布为 [ux0,uy0,uz0, ux1,uy1,uz1, ...]
        u_flat = tf.reshape(tf.cast(u_nodes, tf.float32), (-1,))  # (3N,)

        # 2) 选择本步参与积分的 DFEM 子单元（支持 RAR 强制索引）
        K = self.n_cells
        if self.cfg.n_points_per_step and self.cfg.n_points_per_step > 0:
            m = min(int(self.cfg.n_points_per_step), K)
        else:
            m = K

        forced_idx = self._forced_indices
        self._forced_indices = None
        if forced_idx is not None:
            idx = tf.reshape(tf.cast(forced_idx, tf.int32), (-1,))
            idx = tf.clip_by_value(idx, 0, max(K - 1, 0))
            if tf.shape(idx)[0] < m:
                pad = tf.random.shuffle(tf.range(K, dtype=tf.int32))
                idx = tf.concat([idx, pad], axis=0)
            idx = idx[:m]
        else:
            base_idx = tf.range(K, dtype=tf.int32)
            if m < K:
                base_idx = tf.random.shuffle(base_idx)
            idx = base_idx[:m]

        B_sel = tf.gather(self.B_tf, idx)         # (M,6,12)
        w_sel = tf.gather(self.w_tf, idx)         # (M,)
        lam_sel = tf.gather(self.lam_tf, idx)     # (M,)
        mu_sel = tf.gather(self.mu_tf, idx)       # (M,)
        dof_idx_sel = tf.gather(self.dof_idx_tf, idx)  # (M,12)

        # 3) 取出每个子单元的局部 DOF，计算应变 ε 与能量密度 ψ
        # u_local: (M,12)，每行对应一个子单元 4 节点 × 3 DOF
        u_local = tf.gather(u_flat, dof_idx_sel)

        # ε_vec: (M,6)，Voigt 形式 [ε_xx, ε_yy, ε_zz, γ_yz, γ_xz, γ_xy]
        # einsum: (M,6,12) · (M,12) -> (M,6)
        eps_vec = tf.einsum("mij,mj->mi", B_sel, u_local)

        if self.cfg.check_nan:
            tf.debugging.check_numerics(eps_vec, "[ElasticityEnergy] strain ε has NaN/Inf")

        # 能量密度 ψ = 1/2 λ (tr ε)^2 + μ (ε:ε)
        tr_eps = eps_vec[:, 0] + eps_vec[:, 1] + eps_vec[:, 2]         # (M,)
        eps_sq = tf.reduce_sum(eps_vec * eps_vec, axis=1)              # (M,)
        psi = 0.5 * lam_sel * (tr_eps ** 2.0) + mu_sel * eps_sq        # (M,)

        if self.cfg.check_nan:
            tf.debugging.check_numerics(psi, "[ElasticityEnergy] energy density ψ has NaN/Inf")

        # 4) 对选中的子单元积分，并按总权重做无偏缩放
        w_used_sum = tf.reduce_sum(w_sel)
        E_used = tf.reduce_sum(psi * w_sel)  # 仅基于子样本的能量

        # 若做子采样，则乘以 (总权重 / 本次样本权重) 作无偏估计
        scale = tf.math.divide_no_nan(self.w_total_tf, w_used_sum)
        E_int = E_used * scale

        try:
            self._last_sample_metrics = {
                "psi": psi.numpy(),
                "idx": tf.reshape(idx, (-1,)).numpy(),
            }
        except Exception:
            self._last_sample_metrics = {}

        if self.cfg.check_nan:
            tf.debugging.check_numerics(E_int, "[ElasticityEnergy] E_int has NaN/Inf")

        # 5) 一些统计信息（保留部分旧字段以兼容 logger）
        stats = {
            "N_nodes": int(self.n_nodes),
            "N_total": int(self.n_cells),   # 总 DFEM 子单元数
            "N_used": int(m),               # 本步实际参与积分的子单元数
            "chunk_size": int(self.cfg.chunk_size),
            "use_pfor": bool(self.cfg.use_pfor),
            "coord_scale": float(self._scale),
        }
        if not return_cache:
            return E_int, stats

        cache = {
            "eps_vec": eps_vec,
            "lam": lam_sel,
            "mu": mu_sel,
            "dof_idx": dof_idx_sel,
            "sample_idx": idx,
        }
        return E_int, stats, cache

    # ------------------------------------------------------------------------ #
    # 工具：按节点分块评估位移场
    # ------------------------------------------------------------------------ #
    def _eval_u_on_nodes(self, u_fn, params: Optional[Dict[str, tf.Tensor]]) -> tf.Tensor:
        """
        在所有节点 X_nodes 上评估神经网络位移 u(X_nodes)。

        - 使用 chunk_size 做前向分块，避免一次性评估过大导致显存峰值过高；
        - 默认假定 u_fn(X, params) → shape (batch, 3)，即每个节点 3 个自由度。
        """
        X = self.X_nodes_tf  # (N,3)
        N = self.n_nodes
        cfg_chunk = int(self.cfg.chunk_size)
        # chunk_size ≤0 表示整图前向
        chunk = N if cfg_chunk <= 0 else max(1, cfg_chunk)
        # 若使用 GCN 主干，则必须整图前向以保持全局邻接；忽略分块避免断图
        try:
            bound_model = getattr(u_fn, "__self__", None)
            field = getattr(bound_model, "field", None)
            if getattr(field, "use_graph", False):
                chunk = N
        except Exception:
            pass

        outs = []
        for s in range(0, N, chunk):
            e = min(N, s + chunk)
            Xc = X[s:e]  # (m,3)
            Uc = u_fn(Xc, params)  # 期望 (m,3)
            Uc = tf.cast(Uc, tf.float32)
            if self.cfg.check_nan:
                tf.debugging.check_numerics(Uc, "[ElasticityEnergy] u(X) has NaN/Inf")
            outs.append(Uc)

        if len(outs) == 1:
            return outs[0]
        return tf.concat(outs, axis=0)
