# src/physics/elasticity_energy.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional
import numpy as np
import tensorflow as tf


@dataclass
class ElasticityConfig:
    """
    线弹性能量项的配置：
      - coord_scale: 坐标缩放（避免 J 的尺度过大/过小带来的数值不稳）。
      - chunk_size:  计算 batch_jacobian 时的分块大小，避免 OOM。
      - use_pfor:    是否启用 TF 的 pfor 向量化 Jacobian；部分环境会报 variant 相关错，建议 False。
      - check_nan:   在关键张量上做数值检查，遇到 NaN/Inf 立即报错，便于定位数据问题。
    """
    coord_scale: float = 1.0
    chunk_size: int = 2048
    use_pfor: bool = False
    check_nan: bool = True


class ElasticityEnergy:
    """
    线弹性体的内能：
       E_int = ∫ [ 1/2 * λ * (tr ε)^2 + μ * ε:ε ] dΩ
    其中 ε = (∇u + ∇u^T)/2  为小应变张量；λ, μ 为拉梅参数。

    本类尽量鲁棒地从 `matlib` 提取 (E,ν)，支持：
      1) MaterialLibrary 风格对象（有 props_for_id / id2name / props_for_name 等接口）
      2) dict：
         - {int_id: (E,nu)}
         - {name: (E,nu)} + 通过 matlib.id2name / name_for_id 做回退
    """

    def __init__(
        self,
        X_vol: np.ndarray,          # (N,3) 体积积分点坐标
        w_vol: np.ndarray,          # (N,)   对应权重/体积
        mat_id: np.ndarray,         # (N,)   材料 id（整型）
        matlib: Any,                # 材料库（见上面的兼容说明）
        cfg: ElasticityConfig,
    ) -> None:
        # ---------- 严格输入校验（缺什么就直接报错） ----------
        if X_vol is None or w_vol is None:
            raise ValueError(
                "[ElasticityEnergy] 需要有效的体积分点与权重，但检测到 "
                f"X_vol={type(X_vol).__name__}, w_vol={type(w_vol).__name__}。"
                " 请检查构建阶段 build_volume_points，确保返回 (X_vol, w_vol, mat_id)。"
            )
        X_vol = np.asarray(X_vol)
        w_vol = np.asarray(w_vol)
        mat_id = np.asarray(mat_id)

        if X_vol.ndim != 2 or X_vol.shape[1] != 3:
            raise ValueError(f"[ElasticityEnergy] X_vol 形状应为 (N,3)，但得到 {X_vol.shape}。")
        if w_vol.ndim != 1:
            raise ValueError(f"[ElasticityEnergy] w_vol 形状应为 (N,)，但得到 {w_vol.shape}。")
        if mat_id.ndim != 1:
            raise ValueError(f"[ElasticityEnergy] mat_id 形状应为 (N,)，但得到 {mat_id.shape}。")

        N = X_vol.shape[0]
        if w_vol.shape[0] != N or mat_id.shape[0] != N:
            raise ValueError(
                "[ElasticityEnergy] X_vol / w_vol / mat_id 的长度不一致："
                f" X_vol={N}, w_vol={w_vol.shape[0]}, mat_id={mat_id.shape[0]}。"
            )

        # NaN/Inf 检查（numpy 侧）
        for name, arr in (("X_vol", X_vol), ("w_vol", w_vol), ("mat_id", mat_id)):
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"[ElasticityEnergy] 输入 {name} 含 NaN/Inf，请先清理。")

        # --- 基本数据缓存（统一转 float32/int32，避免混精度坑） ---
        self.X_np = X_vol.astype(np.float32, copy=False)   # (N,3)
        self.w_np = w_vol.astype(np.float32, copy=False)   # (N,)
        self.mat_id_np = mat_id.astype(np.int32, copy=False)  # (N,)
        self.cfg = cfg

        # --- 预计算每个 id 对应的拉梅参数（以表的形式），并展开为每个点的 λ/μ ---
        lam_tab, mu_tab = self._precompute_lame_params(self.mat_id_np, matlib)  # ndarray shape=(K,)
        # 映射到每个积分点
        self.lam_np = lam_tab[self.mat_id_np]  # (N,)
        self.mu_np = mu_tab[self.mat_id_np]    # (N,)

        # --- 转成 tf 常量（float32） ---
        self.X_tf = tf.convert_to_tensor(self.X_np)                  # (N,3)
        self.w_tf = tf.convert_to_tensor(self.w_np)                  # (N,)
        self.lam_tf = tf.convert_to_tensor(self.lam_np)              # (N,)
        self.mu_tf = tf.convert_to_tensor(self.mu_np)                # (N,)

        # 坐标缩放（用于稳定求导），内部用 float32 计算
        self._scale = float(max(self.cfg.coord_scale, 1e-8))

    # --------------------- 对外入口：计算内能 ---------------------
    def energy(self, u_fn, params: Optional[Dict[str, tf.Tensor]] = None) -> Tuple[tf.Tensor, Dict[str, Any]]:
        """
        计算内能标量（float32）及一些统计信息。
        参数：
            u_fn(X, params) -> (N,3) 位移，支持混精度；此处统一在 float32 下计算导数。
        返回：
            E_int: tf.Tensor 标量（float32）
            stats: dict，包含 batch 大小等信息
        """
        # 1) 分块求 J = ∂u/∂x，返回 float32
        J = self._batch_jacobian_chunked(u_fn, params)  # (N,3,3) float32
        if self.cfg.check_nan:
            tf.debugging.check_numerics(J, "Jacobian has NaN/Inf")

        # 2) 小应变 ε = 0.5*(J + Jᵀ)
        eps = 0.5 * (J + tf.transpose(J, perm=[0, 2, 1]))  # (N,3,3)
        if self.cfg.check_nan:
            tf.debugging.check_numerics(eps, "strain tensor has NaN/Inf")

        # 3) 能量密度：0.5*λ*(tr ε)^2 + μ*(ε:ε)
        tr_eps = tf.linalg.trace(eps)                         # (N,)
        eps_sq = tf.reduce_sum(eps * eps, axis=[1, 2])        # (N,)
        psi = 0.5 * self.lam_tf * (tr_eps ** 2.0) + self.mu_tf * eps_sq  # (N,)
        if self.cfg.check_nan:
            tf.debugging.check_numerics(psi, "energy density has NaN/Inf")

        # 4) 内能积分
        E_int = tf.reduce_sum(self.w_tf * psi)  # 标量，float32
        if self.cfg.check_nan:
            tf.debugging.check_numerics(E_int, "E_int has NaN/Inf")

        stats = {
            "N": int(self.X_np.shape[0]),
            "chunk_size": int(self.cfg.chunk_size),
            "use_pfor": bool(self.cfg.use_pfor),
            "coord_scale": float(self._scale),
        }
        return E_int, stats

    # --------------------- 内部：分块计算 Jacobian ---------------------
    def _batch_jacobian_chunked(self, u_fn, params: Optional[Dict[str, tf.Tensor]] = None) -> tf.Tensor:
        """
        为避免 OOM，对积分点分块，逐块用 GradientTape 计算 batch_jacobian。
        关键的数值细节：
          - 统一在 float32 下展开 tape，避免混精度在 autodiff 里的 dtype 不一致；
          - 对 X 先做缩放 Xs=X/scale，tape 对 Xs 求导，得到 J_scaled = ∂u/∂Xs；
            再除以 scale 还原 J = ∂u/∂x（链式法则）。
          - u_fn 的输入我们用 (Xs * scale) 还原到原始坐标系，保证网络看到的是原尺度。
        """
        N = self.X_np.shape[0]
        chunk = int(max(1, self.cfg.chunk_size))
        outs = []

        for s in range(0, N, chunk):
            e = min(N, s + chunk)
            Xc = self.X_tf[s:e]                      # (m,3), float32
            Xc_s = Xc / self._scale                  # (m,3), float32

            # 不能把 watch_accessed_variables 设成 False，否则 tape 会忽略网络参数，
            # 导致外层对 Π 的求导拿不到关于权重的高阶梯度，训练一开始梯度就恒为 0。
            with tf.GradientTape(persistent=False) as tape:
                tape.watch(Xc_s)
                # u_fn 要求输入坐标（原尺度），我们提供 X = Xs * scale
                Uc = u_fn(Xc_s * self._scale, params)    # 形状 (m,3)
                Uc = tf.cast(Uc, tf.float32)             # 统一 float32 以便求导/后续相乘
                if self.cfg.check_nan:
                    tf.debugging.check_numerics(Uc, "u(X) has NaN/Inf")

            # 对 Xc_s 求导，再按链式法则除以 scale
            J_scaled = tape.batch_jacobian(
                Uc, Xc_s, experimental_use_pfor=bool(self.cfg.use_pfor)
            )  # (m,3,3)
            J = J_scaled / self._scale
            outs.append(J)

        return tf.concat(outs, axis=0)  # (N,3,3) float32

    # --------------------- 内部：从材料库/字典提取拉梅参数 ---------------------
    @staticmethod
    def _E_nu_to_lame(E: float, nu: float) -> Tuple[float, float]:
        lam = float(E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))
        mu = float(E / (2.0 * (1.0 + nu)))
        return lam, mu

    def _precompute_lame_params(
        self, mat_id_np: np.ndarray, matlib: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成 “id 表” 形式的 λ/μ：返回两个 ndarray（长度 = max_id+1），便于用 mat_id 做索引。
        兼容多种 matlib 形态。
        """
        uniq = np.unique(mat_id_np.astype(np.int32))
        max_id = int(uniq.max()) if uniq.size > 0 else 0
        lam_tab = np.zeros((max_id + 1,), dtype=np.float32)
        mu_tab = np.zeros((max_id + 1,), dtype=np.float32)

        # --------- 可用的获取函数（按优先级尝试） ----------
        def _get_by_id(mid: int) -> Optional[Tuple[float, float]]:
            # 1) MaterialLibrary 风格：props_for_id(mid) -> (E,nu)
            if hasattr(matlib, "props_for_id") and callable(getattr(matlib, "props_for_id")):
                try:
                    E, nu = matlib.props_for_id(mid)
                    return float(E), float(nu)
                except Exception:
                    pass

            # 2) MaterialLibrary 风格：id2name + props_for_name(name) / materials[name]
            name = None
            if hasattr(matlib, "id2name"):
                try:
                    # id2name 既可能是 dict 也可能是 list
                    name = matlib.id2name[mid] if not isinstance(matlib.id2name, dict) else matlib.id2name.get(mid)
                except Exception:
                    name = None
            if name is None and hasattr(matlib, "name_for_id") and callable(getattr(matlib, "name_for_id")):
                try:
                    name = matlib.name_for_id(mid)
                except Exception:
                    name = None

            if isinstance(name, str):
                # 优先 props_for_name
                if hasattr(matlib, "props_for_name") and callable(getattr(matlib, "props_for_name")):
                    try:
                        E, nu = matlib.props_for_name(name)
                        return float(E), float(nu)
                    except Exception:
                        pass
                # 退化 materials / props / dict-like
                for attr in ("materials", "props", "name2props"):
                    if hasattr(matlib, attr):
                        store = getattr(matlib, attr)
                        try:
                            if isinstance(store, dict) and name in store:
                                E, nu = store[name]
                                return float(E), float(nu)
                        except Exception:
                            pass

            # 3) matlib 直接是 dict：可能是 {id: (E,nu)} 或 {name: (E,nu)}
            if isinstance(matlib, dict):
                # 3a) int-id 直接命中
                if mid in matlib:
                    try:
                        E, nu = matlib[mid]
                        return float(E), float(nu)
                    except Exception:
                        pass
                # 3b) name→(E,nu) + 需要 id→name 的映射（尝试 matlib['id2name'] 或全局 name 列表）
                cand = None
                # 内置 id2name
                if "id2name" in matlib:
                    try:
                        cand = matlib["id2name"][mid]
                    except Exception:
                        cand = None
                # 其它常见键
                if cand is None:
                    for k in ("names", "materials_order", "enum"):
                        if k in matlib:
                            try:
                                cand = matlib[k][mid]
                                break
                            except Exception:
                                cand = None
                if cand is not None and isinstance(cand, str) and cand in matlib:
                    try:
                        E, nu = matlib[cand]
                        return float(E), float(nu)
                    except Exception:
                        pass

            return None  # 所有路径都失败

        # 循环填表
        missing: list[int] = []
        for mid in uniq:
            pair = _get_by_id(int(mid))
            if pair is None:
                missing.append(int(mid))
            else:
                lam, mu = self._E_nu_to_lame(pair[0], pair[1])
                lam_tab[int(mid)] = lam
                mu_tab[int(mid)] = mu

        if missing:
            raise KeyError(
                "[ElasticityEnergy] 无法从材料库推断 id->(E,nu)。"
                f" 缺失 id: {missing}。"
                " 请确认：\n"
                "  - 传入的是带有 props_for_id / id2name / props_for_name 的 MaterialLibrary；或\n"
                "  - 传入 dict 时为 {int_id:(E,nu)} 或 {name:(E,nu)}，并能由 id→name 映射到 name。\n"
                "  - 也可改为在 Trainer 中构造 {id:(E,nu)} 的字典直接传进来。"
            )

        return lam_tab.astype(np.float32), mu_tab.astype(np.float32)
