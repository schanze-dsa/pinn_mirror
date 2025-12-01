# -*- coding: utf-8 -*-
"""
trainer.py — 主训练循环（精简日志 + 分阶段进度提示）。

该版本专注于保留关键构建/训练信息：
  - 初始化时报告是否启用 GPU。
  - 构建阶段仅输出必需的信息与接触汇总。
  - 单步训练进度条会标注当前阶段，便于观察训练流程。
"""
from __future__ import annotations
from train.attach_ties_bcs import attach_ties_and_bcs_from_inp

import os
import sys
import time
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Mapping, Sequence

import numpy as np
from tensorflow.python.util import deprecation
import tensorflow as tf
from tqdm.auto import tqdm  # 仅用 tqdm.auto，适配 PyCharm/终端

# ---------- TF 显存与分配器 ----------
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

try:
    import colorama
    colorama.just_fix_windows_console()
    _ANSI_WHITE = colorama.Fore.WHITE
    _ANSI_RESET = colorama.Style.RESET_ALL
except Exception:
    colorama = None
    _ANSI_WHITE = ""
    _ANSI_RESET = ""

import builtins as _builtins


def _wrap_white(text: str) -> str:
    if not _ANSI_WHITE:
        return text
    return f"{_ANSI_WHITE}{text}{_ANSI_RESET}"


def print(*values, sep: str = " ", end: str = "\n", file=None, flush: bool = False):
    """Module-local print that forces white foreground text on stdout/stderr."""

    target = sys.stdout if file is None else file
    msg = sep.join(str(v) for v in values)
    if target in (sys.stdout, sys.stderr):
        msg = _wrap_white(msg)
    _builtins.print(msg, end=end, file=target, flush=flush)

# 让 src 根目录可导入
_SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".", "."))
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

# ---------- 项目模块 ----------
from inp_io.inp_parser import load_inp, AssemblyModel
from mesh.volume_quadrature import build_volume_points
from mesh.contact_pairs import ContactPairSpec, build_contact_map, resample_contact_map
from physics.material_lib import MaterialLibrary
from model.pinn_model import create_displacement_model, ModelConfig, DisplacementModel
from physics.elasticity_energy import ElasticityEnergy, ElasticityConfig
from physics.contact.contact_operator import ContactOperator, ContactOperatorConfig
from physics.boundary_conditions import BoundaryPenalty, BoundaryConfig
from physics.tie_constraints import TiePenalty, TieConfig
from physics.preload_model import PreloadWork, PreloadConfig, BoltSurfaceSpec
from model.loss_energy import TotalEnergy, TotalConfig
from train.loss_weights import LossWeightState, update_loss_weights, combine_loss
from viz.mirror_viz import plot_mirror_deflection_by_name


# ----------------- 配置 -----------------
@dataclass
class TrainerConfig:
    inp_path: str = "data/shuangfan.inp"

    # 镜面名称（裸名字），以及对应 ASM 键名（可空，自动猜）
    mirror_surface_name: str = "MIRROR up"
    mirror_surface_asm_key: Optional[str] = None   # e.g. 'ASM::"MIRROR up"'

    # 材料库（名字 -> (E, nu)）
    materials: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "mirror": (70000.0, 0.33),
        "steel":  (210000.0, 0.30),
    })
    # 零件到材料名映射
    part2mat: Dict[str, str] = field(default_factory=lambda: {
        "MIRROR": "mirror",
        "BOLT1": "steel",
        "BOLT2": "steel",
        "BOLT3": "steel",
    })

    # 接触
    contact_pairs: List[Dict[str, str]] = field(default_factory=list)
    n_contact_points_per_pair: int = 6000
    contact_seed: int = 1234
    contact_rar_enabled: bool = True           # 是否启用接触残差驱动的自适应重采样
    contact_rar_fraction: float = 0.5          # 每次重采样中，多少比例来自残差加权抽样
    contact_rar_temperature: float = 1.0       # >1 平滑、<1 更尖锐
    contact_rar_floor: float = 1e-6            # 防止全零残差
    contact_rar_uniform_ratio: float = 0.3     # 保留多少比例的全局均匀点，避免过拟合热点
    contact_rar_fric_mix: float = 0.4          # 穿透 vs 摩擦残差的混合系数
    contact_rar_balance_pairs: bool = True     # 是否保持各接触对的样本占比

    # 体积分点（弹性能量）RAR
    volume_rar_enabled: bool = True            # 是否启用体积分点基于应变能密度的 RAR
    volume_rar_fraction: float = 0.5           # 每步 DFEM 子单元子采样中，多少比例来自 RAR
    volume_rar_temperature: float = 1.0        # >1 平滑、<1 更尖锐
    volume_rar_uniform_ratio: float = 0.2      # 保底均匀抽样比例
    volume_rar_floor: float = 1e-8             # 基础重要性，避免全零

    # 预紧
    preload_specs: List[Dict[str, str]] = field(default_factory=list)
    preload_n_points_each: int = 800

    # tie / 边界（如需）
    ties: List[Dict[str, Any]] = field(default_factory=list)
    bcs: List[Dict[str, Any]] = field(default_factory=list)
    bc_mode: str = "alm"                    # penalty | hard | alm
    bc_mu: float = 1.0e3                    # ALM 增广系数
    bc_alpha: float = 1.0e4                 # 罚函数/ALM 基础刚度

    # 预紧力范围（N）
    preload_min: float = 0.0
    preload_max: float = 2000.0
    preload_sequence: List[Any] = field(default_factory=list)
    preload_sequence_repeat: int = 1
    preload_sequence_shuffle: bool = False
    preload_sequence_jitter: float = 0.0

    # 预紧采样方式
    preload_sampling: str = "lhs"            # "lhs" | "uniform"
    preload_lhs_size: int = 64               # 每批次的拉丁超立方样本数量

    # 预紧顺序（分步加载）
    preload_use_stages: bool = False
    preload_randomize_order: bool = True

    # 物理项/模型配置
    model_cfg: ModelConfig = field(default_factory=ModelConfig)
    elas_cfg: ElasticityConfig = field(
        default_factory=lambda: ElasticityConfig(coord_scale=1.0, chunk_size=0, use_pfor=False)
    )
    contact_cfg: ContactOperatorConfig = field(default_factory=ContactOperatorConfig)
    preload_cfg: PreloadConfig = field(default_factory=PreloadConfig)
    total_cfg: TotalConfig = field(default_factory=lambda: TotalConfig(
        w_int=1.0, w_cn=1.0, w_ct=1.0, w_tie=1.0, w_pre=1.0, w_sigma=1.0
    ))

    # 损失加权（自适应）
    loss_adaptive_enabled: bool = True
    loss_update_every: int = 1
    loss_ema_decay: float = 0.95
    loss_min_factor: float = 0.25
    loss_max_factor: float = 4.0
    loss_gamma: float = 2.0
    loss_focus_terms: Tuple[str, ...] = field(default_factory=tuple)

    # 训练超参
    max_steps: int = 1000
    adam_steps: Optional[int] = None
    lr: float = 1e-3
    grad_clip_norm: Optional[float] = 1.0
    log_every: int = 1
    alm_update_every: int = 10
    resample_contact_every: int = 10
    lbfgs_enabled: bool = False
    lbfgs_max_iter: int = 200
    lbfgs_tolerance: float = 1e-6
    lbfgs_history_size: int = 50
    lbfgs_line_search: int = 50
    lbfgs_reuse_last_batch: bool = True

    # 进度条颜色（None 则禁用彩色，使用终端默认色）
    build_bar_color: Optional[str] = "cyan"
    train_bar_color: Optional[str] = "cyan"
    step_bar_color: Optional[str] = "green"

    # 精度/随机种子
    mixed_precision: Optional[str] = "mixed_float16"
    seed: int = 42

    # 输出
    out_dir: str = "outputs"
    ckpt_dir: str = "checkpoints"
    viz_samples_after_train: int = 6
    viz_title_prefix: str = "Total Deformation (trained PINN)"
    viz_style: str = "contour"             # 默认采用等值填充以获得平滑云图
    viz_colormap: str = "turbo"             # Abaqus-like rainbow palette
    viz_levels: int = 64                    # 等值线数量，提升平滑度
    viz_symmetric: bool = False             # displacement magnitude is nonnegative
    viz_units: str = "mm"
    viz_draw_wireframe: bool = False
    viz_surface_enabled: bool = True        # 是否渲染单一镜面云图
    viz_surface_source: str = "part_top"    # "surface" 使用 INP 表面；"part_top" 优先用零件外表面上表面
    viz_write_data: bool = True             # export displacement samples next to figure
    viz_write_surface_mesh: bool = False    # export reconstructed FE surface mesh next to figure
    viz_plot_full_structure: bool = False   # 导出全装配（或指定零件）的位移云图
    viz_full_structure_part: Optional[str] = "mirror1"  # None -> 全装配
    viz_write_full_structure_data: bool = False  # 记录全装配位移数据
    viz_retriangulate_2d: bool = False      # 兼容旧配置的占位符，不再使用
    viz_refine_subdivisions: int = 2        # 细分表面三角形以平滑云图
    viz_refine_max_points: int = 180_000    # guardrail against runaway refinement cost
    viz_eval_batch_size: int = 65_536       # batch PINN queries during visualization
    viz_eval_scope: str = "assembly"        # "surface" or "assembly"/"all"
    viz_diagnose_blanks: bool = False       # 是否在生成云图时自动诊断留白原因
    viz_auto_fill_blanks: bool = False      # 覆盖率低时自动用 2D 重新三角化填补留白（默认关闭以保留真实孔洞）
    viz_remove_rigid: bool = True           # 可视化时默认去除刚体平移/转动分量
    save_best_on: str = "Pi"   # or "E_int"


class Trainer:
    def __init__(self, cfg: TrainerConfig):
        self.cfg = cfg
        np.random.seed(cfg.seed)
        tf.random.set_seed(cfg.seed)

        self._preload_dim: int = 3
        self._preload_lhs_rng = np.random.default_rng(cfg.seed + 11)
        self._preload_lhs_points: np.ndarray = np.zeros((0, self._preload_dim), dtype=np.float32)
        self._preload_lhs_index: int = 0

        # 默认设备描述，避免后续日志访问属性时出错
        self.device_summary = "Unknown"
        self._step_stage_times: List[Tuple[str, float]] = []
        self._pi_baseline: Optional[float] = None
        self._pi_ema: Optional[float] = None
        self._prev_pi: Optional[float] = None
        self._preload_sequence: List[np.ndarray] = []
        self._preload_sequence_orders: List[Optional[np.ndarray]] = []
        self._preload_sequence_index: int = 0
        self._preload_sequence_hold: int = 0
        self._preload_current_target: Optional[np.ndarray] = None
        self._preload_current_order: Optional[np.ndarray] = None
        self._last_preload_order: Optional[np.ndarray] = None
        self._last_preload_case: Optional[Dict[str, np.ndarray]] = None
        self._train_vars: List[tf.Variable] = []
        self._contact_rar_cache: Optional[Dict[str, Any]] = None
        self._volume_rar_cache: Optional[Dict[str, Any]] = None
        self._current_contact_cat: Optional[Dict[str, np.ndarray]] = None

        if cfg.preload_specs:
            self._set_preload_dim(len(cfg.preload_specs))

        if cfg.preload_sequence:
            sanitized: List[np.ndarray] = []
            sanitized_orders: List[Optional[np.ndarray]] = []
            for idx, entry in enumerate(cfg.preload_sequence):
                order_entry = None
                values_entry: Any = entry
                if isinstance(entry, dict):
                    order_entry = entry.get("order")
                    for key in ("values", "loads", "P", "p", "preload", "forces"):
                        if key in entry:
                            values_entry = entry[key]
                            break
                try:
                    arr = np.array(values_entry, dtype=np.float32).reshape(-1)
                except Exception:
                    print(
                        f"[preload] 忽略 preload_sequence[{idx}]，无法解析为浮点数组：{entry}"
                    )
                    sanitized_orders.append(None)
                    continue
                if arr.size == 0:
                    print(f"[preload] 忽略 preload_sequence[{idx}]，未提供数值。")
                    sanitized_orders.append(None)
                    continue
                if arr.size == 1:
                    arr = np.repeat(arr, 3)
                if arr.size != 3:
                    print(
                        f"[preload] 忽略 preload_sequence[{idx}]，需要 3 个数值，实际 {arr.size} 个。"
                    )
                    sanitized_orders.append(None)
                    continue

                order_arr: Optional[np.ndarray] = None
                if order_entry is not None:
                    try:
                        order_raw = np.array(order_entry, dtype=np.int32).reshape(-1)
                    except Exception:
                        print(
                            f"[preload] 忽略 preload_sequence[{idx}] 的顺序字段，无法解析：{order_entry}"
                        )
                        order_raw = None
                    if order_raw is not None:
                        nb = arr.size
                        if order_raw.size != nb:
                            print(
                                f"[preload] 忽略 preload_sequence[{idx}] 的顺序字段，长度需为 {nb}。"
                            )
                        else:
                            if np.all(order_raw >= 1) and np.max(order_raw) <= nb and np.min(order_raw) >= 1:
                                order_raw = order_raw - 1
                            unique = sorted(set(order_raw.tolist()))
                            if unique != list(range(nb)):
                                print(
                                    f"[preload] 忽略 preload_sequence[{idx}] 的顺序字段，必须是 0~{nb-1} 的排列（或 1~{nb}）。"
                                )
                            else:
                                order_arr = order_raw.astype(np.int32)

                sanitized.append(arr.astype(np.float32))
                sanitized_orders.append(order_arr.copy() if order_arr is not None else None)

            if sanitized:
                if cfg.preload_sequence_shuffle:
                    perm = np.random.permutation(len(sanitized))
                    sanitized = [sanitized[i] for i in perm]
                    sanitized_orders = [sanitized_orders[i] for i in perm]
                self._preload_sequence = sanitized
                self._preload_sequence_orders = sanitized_orders
                self._preload_current_target = self._preload_sequence[0].copy()
                if self._preload_sequence_orders:
                    self._preload_current_order = (
                        None
                        if self._preload_sequence_orders[0] is None
                        else self._preload_sequence_orders[0].copy()
                    )
                hold = max(1, cfg.preload_sequence_repeat)
                print(
                    f"[preload] 已启用顺序载荷：{len(self._preload_sequence)} 组，",
                    f"每组持续 {hold} 步。"
                )
                if cfg.preload_sequence_jitter > 0:
                    print(
                        f"[preload] 顺序载荷将叠加 ±{cfg.preload_sequence_jitter}N 的均匀扰动。"
                    )
                self._set_preload_dim(self._preload_sequence[0].size)
            else:
                print("[preload] preload_sequence 中有效条目为空，改为随机采样。")
        if cfg.model_cfg.preload_scale:
            print(
                f"[preload] 归一化: shift={cfg.model_cfg.preload_shift:.2f}, "
                f"scale={cfg.model_cfg.preload_scale:.2f}"
            )

        # 显存增长
        gpus = tf.config.list_physical_devices('GPU')
        gpu_labels = []
        for idx, g in enumerate(gpus):
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass
            label = getattr(g, "name", None)
            if label:
                label = label.split("/")[-1]
                parts = label.split(":")
                if len(parts) >= 2:
                    label = ":".join(parts[-2:])
            else:
                label = f"GPU:{idx}"
            gpu_labels.append(label)

        if gpu_labels:
            self.device_summary = f"GPU ({', '.join(gpu_labels)})"
            print(f"[trainer] 使用 GPU 进行训练: {', '.join(gpu_labels)}")
        else:
            self.device_summary = "CPU"
            print("[trainer] 未检测到 GPU，将在 CPU 上训练。")

        # 混合精度
        if cfg.mixed_precision:
            try:
                tf.keras.mixed_precision.set_global_policy(cfg.mixed_precision)
                print(f"[pinn_model] Mixed precision policy set to: {cfg.mixed_precision}")
            except Exception as e:
                print("[pinn_model] Failed to set mixed precision:", e)

        os.makedirs(cfg.out_dir, exist_ok=True)
        os.makedirs(cfg.ckpt_dir, exist_ok=True)

        # 组件
        self.asm: Optional[AssemblyModel] = None
        self.matlib: Optional[MaterialLibrary] = None
        self.model = None
        self.optimizer = None

        self.elasticity: Optional[ElasticityEnergy] = None
        self.contact: Optional[ContactOperator] = None
        self.preload: Optional[PreloadWork] = None
        self.ties_ops: List[TiePenalty] = []
        self.bcs_ops: List[BoundaryPenalty] = []
        self._cp_specs: List[ContactPairSpec] = []

        self.ckpt = None
        self.ckpt_manager = None
        self.best_metric = float("inf")
        self.last_viz_data_path: Optional[str] = None
        self.last_viz_diag = None

        # —— 体检/调试可读
        self.X_vol = None
        self.w_vol = None
        self.mat_id = None
        self.enum_names: List[str] = []
        self.id2props_map: Dict[int, Tuple[float, float]] = {}
        # 自适应损失权重的状态（在 run() 里初始化）
        self.loss_state: Optional[LossWeightState] = None


    # ----------------- 辅助工具 -----------------

    def _set_preload_dim(self, nb: int):
        nb_int = int(nb)
        if nb_int <= 0:
            nb_int = 3
        if nb_int != getattr(self, "_preload_dim", None):
            self._preload_dim = nb_int
            self._preload_lhs_points = np.zeros((0, nb_int), dtype=np.float32)
            self._preload_lhs_index = 0

    def _generate_lhs_points(self, n_samples: int, n_dim: int, lo: float, hi: float) -> np.ndarray:
        """简单的拉丁超立方采样生成器，返回 (n_samples, n_dim)."""

        if n_samples <= 0:
            return np.zeros((0, n_dim), dtype=np.float32)
        unit = np.zeros((n_samples, n_dim), dtype=np.float32)
        for j in range(n_dim):
            seg = (np.arange(n_samples, dtype=np.float32) + self._preload_lhs_rng.random(n_samples)) / float(n_samples)
            self._preload_lhs_rng.shuffle(seg)
            unit[:, j] = seg
        scale = hi - lo
        return (lo + unit * scale).astype(np.float32)

    def _next_lhs_preload(self, n_dim: int, lo: float, hi: float) -> np.ndarray:
        batch = max(1, int(self.cfg.preload_lhs_size))
        if self._preload_lhs_points.shape[1] != n_dim or len(self._preload_lhs_points) == 0:
            self._preload_lhs_points = self._generate_lhs_points(batch, n_dim, lo, hi)
            self._preload_lhs_index = 0
        if self._preload_lhs_index >= len(self._preload_lhs_points):
            self._preload_lhs_points = self._generate_lhs_points(batch, n_dim, lo, hi)
            self._preload_lhs_index = 0
        out = self._preload_lhs_points[self._preload_lhs_index].copy()
        self._preload_lhs_index += 1
        return out

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        if seconds < 1e-3:
            return f"{seconds * 1e6:.0f}µs"
        if seconds < 1:
            return f"{seconds * 1e3:.1f}ms"
        return f"{seconds:.2f}s"

    @staticmethod
    def _short_device_name(device: Optional[str]) -> str:
        if not device:
            return "?"
        if "/device:" in device:
            return device.split("/device:")[-1]
        if device.startswith("/"):
            return device.split(":")[-1]
        return device

    @staticmethod
    def _wrap_bar_text(text: Optional[str]) -> Optional[str]:
        if text is None:
            return None
        return _wrap_white(str(text))

    def _set_pbar_desc(self, pbar, text: str) -> None:
        pbar.set_description_str(self._wrap_bar_text(text))

    def _set_pbar_postfix(self, pbar, text: str) -> None:
        if text is None:
            pbar.set_postfix_str(text)
            return
        pbar.set_postfix_str(self._wrap_bar_text(text))

    def _loss_weight_lookup(self) -> Dict[str, float]:
        """Assemble the latest per-term loss weights for logging."""

        weights = {
            "E_int": getattr(self.cfg.total_cfg, "w_int", 1.0),
            "E_cn": getattr(self.cfg.total_cfg, "w_cn", 1.0),
            "E_ct": getattr(self.cfg.total_cfg, "w_ct", 1.0),
            "E_tie": getattr(self.cfg.total_cfg, "w_tie", 1.0),
            "W_pre": getattr(self.cfg.total_cfg, "w_pre", 1.0),
            "E_sigma": getattr(self.cfg.total_cfg, "w_sigma", 1.0),
        }
        if self.loss_state is not None:
            for key, value in self.loss_state.current.items():
                try:
                    weights[key] = float(value)
                except Exception:
                    weights[key] = value
        return weights

    @staticmethod
    def _extract_part_scalar(parts: Mapping[str, tf.Tensor], *keys: str) -> Optional[float]:
        for key in keys:
            if key not in parts:
                continue
            value = parts[key]
            try:
                if isinstance(value, tf.Tensor):
                    return float(value.numpy())
                if isinstance(value, np.ndarray):
                    return float(value)
                return float(value)
            except Exception:
                continue
        return None

    def _format_energy_summary(self, parts: Mapping[str, tf.Tensor]) -> str:
        display = [
            ("E_int", "Eint"),
            ("E_cn", "Ecn"),
            ("E_ct", "Ect"),
            ("E_tie", "Etie"),
            ("W_pre", "Wpre"),
            ("E_sigma", "Esig"),
        ]
        aliases = {
            "E_cn": ("E_cn", "E_n"),
            "E_ct": ("E_ct", "E_t"),
        }
        weights = self._loss_weight_lookup()
        entries: List[str] = []
        for key, label in display:
            val = self._extract_part_scalar(parts, *aliases.get(key, (key,)))
            if val is None:
                continue
            weight = weights.get(key, 0.0)
            entries.append(f"{label}={val:.3e}(w={weight:.3g})")
        return " ".join(entries)

    def _format_train_log_postfix(
        self,
        P_np: np.ndarray,
        Pi: tf.Tensor,
        parts: Mapping[str, tf.Tensor],
        stats: Optional[Mapping[str, Any]],
        grad_val: float,
        rel_pi: float,
        rel_delta: Optional[float],
        order: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[str], str]:
        """Compose the detailed training log postfix for the outer progress bar.

        Returns a tuple of ``(postfix, note)`` where ``postfix`` is the formatted
        text (or ``None`` when formatting fails) and ``note`` summarises whether
        logging succeeded.
        """

        try:
            p1, p2, p3 = [int(x) for x in P_np.tolist()]
            pin = float(Pi.numpy())
            energy_disp = self._format_energy_summary(parts)

            bolt_txt = ""
            preload_stats = None
            if isinstance(stats, Mapping):
                preload_stats = stats.get("preload") or stats.get("preload_stats")
            if isinstance(preload_stats, Mapping):
                bd = preload_stats.get("bolt_deltas") or preload_stats.get("bolt_delta")
                if bd is not None:
                    if hasattr(bd, "numpy"):
                        bd = bd.numpy()
                    try:
                        b1, b2, b3 = [float(x) for x in list(bd)[:3]]
                        bolt_txt = f" Δ=[{b1:.3e},{b2:.3e},{b3:.3e}]"
                    except Exception:
                        pass

            pen_ratio = None
            stick_ratio = None
            slip_ratio = None
            mean_gap = None

            def _get_stat_float(*keys: str) -> Optional[float]:
                if not isinstance(stats, Mapping):
                    return None
                for key in keys:
                    val = stats.get(key)
                    if val is None:
                        continue
                    try:
                        if hasattr(val, "numpy"):
                            return float(val.numpy())
                        return float(val)
                    except Exception:
                        continue
                return None

            pen_ratio = _get_stat_float("n_pen_ratio", "cn_pen_ratio", "pen_ratio")
            stick_ratio = _get_stat_float("t_stick_ratio", "stick_ratio")
            slip_ratio = _get_stat_float("t_slip_ratio", "slip_ratio")
            mean_gap = _get_stat_float("n_mean_gap", "cn_mean_gap", "mean_gap")

            grad_disp = f"grad={grad_val:.2e}"
            rel_pct = rel_pi * 100.0 if rel_pi is not None else None
            rel_disp = (
                f"Πrel={rel_pct:.2f}%" if rel_pct is not None else "Πrel=--"
            )
            delta_disp = (
                f"ΔΠ={rel_delta * 100:+.1f}%" if rel_delta is not None else "ΔΠ=--"
            )
            pen_disp = (
                f"pen={pen_ratio * 100:.1f}%" if pen_ratio is not None else "pen=--"
            )
            stick_disp = (
                f"stick={stick_ratio * 100:.1f}%" if stick_ratio is not None else "stick=--"
            )
            slip_disp = (
                f"slip={slip_ratio * 100:.1f}%" if slip_ratio is not None else "slip=--"
            )
            gap_disp = (
                f"⟨gap⟩={mean_gap:.2e}" if mean_gap is not None else "⟨gap⟩=--"
            )

            order_txt = ""
            if order is not None:
                try:
                    order_list = [int(x) for x in list(order)]
                    human_order = "-".join(str(idx + 1) for idx in order_list)
                    ordered_values: Optional[List[int]] = None
                    if P_np is not None and len(order_list) == len(P_np):
                        ordered_values = []
                        for idx in order_list:
                            if 0 <= idx < len(P_np):
                                ordered_values.append(int(P_np[idx]))
                            else:
                                ordered_values = None
                                break
                    if ordered_values:
                        order_txt = (
                            f" order={human_order}(P序=["
                            + ",".join(str(val) for val in ordered_values)
                            + "])"
                        )
                    else:
                        order_txt = f" order={human_order}"
                except Exception:
                    order_txt = " order=?"
            parts_disp = energy_disp or ""
            postfix = (
                f"P=[{p1},{p2},{p3}]N{order_txt} Π={pin:.3e} | {parts_disp} {bolt_txt} "
                f"| {grad_disp} {pen_disp} {stick_disp} {slip_disp} {gap_disp}"
            )
            return postfix, "已记录"
        except Exception:
            return None, "记录异常"

    # ----------------- 采样三螺栓预紧力 -----------------
    def _sample_P(self) -> np.ndarray:
        if self._preload_sequence:
            if self._preload_current_target is None:
                idx = self._preload_sequence_index
                self._preload_current_target = self._preload_sequence[idx].copy()
                base_order = (
                    self._preload_sequence_orders[idx]
                    if idx < len(self._preload_sequence_orders)
                    else None
                )
                self._preload_current_order = (
                    None if base_order is None else base_order.copy()
                )
            target = self._preload_current_target.copy()
            current_order = (
                None if self._preload_current_order is None else self._preload_current_order.copy()
            )
            jitter = float(self.cfg.preload_sequence_jitter or 0.0)
            if jitter > 0.0:
                noise = np.random.uniform(-jitter, jitter, size=target.shape)
                target = target + noise.astype(np.float32)
            lo, hi = self.cfg.preload_min, self.cfg.preload_max
            target = np.clip(target, lo, hi)

            self._preload_sequence_hold += 1
            if self._preload_sequence_hold >= max(1, self.cfg.preload_sequence_repeat):
                self._preload_sequence_hold = 0
                self._preload_sequence_index = (self._preload_sequence_index + 1) % len(
                    self._preload_sequence
                )
                if self._preload_sequence_index == 0 and self.cfg.preload_sequence_shuffle:
                    perm = np.random.permutation(len(self._preload_sequence))
                    self._preload_sequence = [self._preload_sequence[i] for i in perm]
                    self._preload_sequence_orders = [
                        self._preload_sequence_orders[i] if i < len(self._preload_sequence_orders) else None
                        for i in perm
                    ]
                idx = self._preload_sequence_index
                self._preload_current_target = self._preload_sequence[idx].copy()
                base_order = (
                    self._preload_sequence_orders[idx]
                    if idx < len(self._preload_sequence_orders)
                    else None
                )
                self._preload_current_order = (
                    None if base_order is None else base_order.copy()
                )

            self._last_preload_order = None if current_order is None else current_order.copy()
            return target.astype(np.float32)

        lo, hi = self.cfg.preload_min, self.cfg.preload_max
        nb = int(self._preload_dim)
        sampling = (self.cfg.preload_sampling or "uniform").lower()
        if sampling == "lhs":
            out = self._next_lhs_preload(nb, lo, hi)
        else:
            out = np.random.uniform(lo, hi, size=(nb,)).astype(np.float32)
        self._last_preload_order = None
        return out.astype(np.float32)

    def _normalize_order(self, order: Optional[Any], nb: int) -> Optional[np.ndarray]:
        if order is None:
            return None
        arr = np.array(order, dtype=np.int32).reshape(-1)
        if arr.size != nb:
            raise ValueError(f"顺序长度需为 {nb}，收到 {arr.size}。")
        if np.all(arr >= 1) and np.max(arr) <= nb and np.min(arr) >= 1:
            arr = arr - 1
        unique = sorted(set(arr.tolist()))
        if unique != list(range(nb)):
            raise ValueError(
                f"顺序字段必须是 0~{nb-1}（或 1~{nb}）的排列，收到 {list(arr)}。"
            )
        return arr.astype(np.int32)

    def _build_stage_case(self, P: np.ndarray, order: np.ndarray) -> Dict[str, np.ndarray]:
        nb = int(P.shape[0])
        order = np.asarray(order, dtype=np.int32).reshape(-1)
        if order.size != nb:
            raise ValueError(f"顺序长度需为 {nb}，收到 {order.size}。")
        stage_loads = []
        stage_masks = []
        stage_last = []
        cumulative = np.zeros_like(P)
        mask = np.zeros_like(P)
        rank = np.zeros((nb,), dtype=np.float32)
        for pos, idx in enumerate(order):
            idx_int = int(idx)
            cumulative[idx_int] = P[idx_int]
            mask[idx_int] = 1.0
            stage_loads.append(cumulative.copy())
            stage_masks.append(mask.copy())
            onehot = np.zeros_like(P)
            onehot[idx_int] = 1.0
            stage_last.append(onehot)
            rank[idx_int] = float(pos)
        if nb > 1:
            rank = rank / float(nb - 1)
        else:
            rank = np.zeros_like(rank)
        rank_matrix = np.tile(rank.reshape(1, -1), (len(stage_loads), 1))
        return {
            "stages": np.stack(stage_loads).astype(np.float32),
            "stage_masks": np.stack(stage_masks).astype(np.float32),
            "stage_last": np.stack(stage_last).astype(np.float32),
            "stage_rank": rank.astype(np.float32),
            "stage_rank_matrix": rank_matrix.astype(np.float32),
        }

    def _sample_preload_case(self) -> Dict[str, np.ndarray]:
        P = self._sample_P()
        case: Dict[str, np.ndarray] = {"P": P}
        if not self.cfg.preload_use_stages:
            return case

        base_order = None if self._last_preload_order is None else self._last_preload_order.copy()
        if base_order is None:
            if self.cfg.preload_randomize_order:
                order = np.random.permutation(P.shape[0]).astype(np.int32)
            else:
                order = np.arange(P.shape[0], dtype=np.int32)
        else:
            order = base_order.astype(np.int32)

        case["order"] = order
        case.update(self._build_stage_case(P, order))
        return case

    def _make_preload_params(self, case: Dict[str, np.ndarray]) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "P": tf.convert_to_tensor(case["P"], dtype=tf.float32)
        }
        if not self.cfg.preload_use_stages or "stages" not in case:
            return params

        stages = case.get("stages")
        masks = case.get("stage_masks")
        lasts = case.get("stage_last")
        order_np = case.get("order")
        rank_np = case.get("stage_rank")
        rank_matrix_np = case.get("stage_rank_matrix")
        if order_np is None:
            order_np = np.arange(case["P"].shape[0], dtype=np.int32)
        order_tf = tf.convert_to_tensor(order_np, dtype=tf.int32)
        rank_tf = None
        if rank_np is not None:
            rank_tf = tf.convert_to_tensor(rank_np, dtype=tf.float32)
        rank_matrix_tf = None
        if rank_matrix_np is not None:
            rank_matrix_tf = tf.convert_to_tensor(rank_matrix_np, dtype=tf.float32)
        elif rank_tf is not None:
            rank_matrix_tf = tf.repeat(
                tf.expand_dims(rank_tf, axis=0), repeats=int(len(stages)), axis=0
            )
        mask_tensor = (
            tf.convert_to_tensor(masks, dtype=tf.float32) if masks is not None else None
        )
        last_tensor = (
            tf.convert_to_tensor(lasts, dtype=tf.float32) if lasts is not None else None
        )

        stage_params_P: List[tf.Tensor] = []
        stage_params_feat: List[tf.Tensor] = []
        stage_count = int(len(stages))
        shift = tf.cast(self.cfg.model_cfg.preload_shift, tf.float32)
        scale = tf.cast(self.cfg.model_cfg.preload_scale, tf.float32)
        n_bolts = int(case["P"].shape[0])
        feat_dim = n_bolts
        if masks is not None:
            feat_dim += n_bolts
        if lasts is not None:
            feat_dim += n_bolts
        if rank_tf is not None:
            feat_dim += n_bolts

        for idx in range(stage_count):
            p_stage = tf.convert_to_tensor(stages[idx], dtype=tf.float32)
            norm = (p_stage - shift) / scale
            feat_parts = [norm]
            if mask_tensor is not None:
                feat_parts.append(mask_tensor[idx])
            if last_tensor is not None:
                feat_parts.append(last_tensor[idx])
            if rank_tf is not None:
                feat_parts.append(rank_tf)
            features = tf.concat(feat_parts, axis=0)
            features.set_shape((feat_dim,))
            stage_params_P.append(p_stage)
            stage_params_feat.append(features)
        stage_tensor_P = tf.stack(stage_params_P, axis=0)
        stage_tensor_P.set_shape((stage_count, n_bolts))
        stage_tensor_feat = tf.stack(stage_params_feat, axis=0)
        stage_tensor_feat.set_shape((stage_count, feat_dim))
        stage_dict: Dict[str, tf.Tensor] = {
            "P": stage_tensor_P,
            "P_hat": stage_tensor_feat,
        }
        if mask_tensor is not None:
            mask_tensor.set_shape((stage_count, n_bolts))
            stage_dict["stage_mask"] = mask_tensor
        if last_tensor is not None:
            last_tensor.set_shape((stage_count, n_bolts))
            stage_dict["stage_last"] = last_tensor
        if rank_matrix_tf is not None:
            stage_dict["stage_rank"] = rank_matrix_tf
        params["stages"] = stage_dict
        params["stage_order"] = order_tf
        if rank_tf is not None:
            params["stage_rank"] = rank_tf
        params["stage_count"] = tf.constant(stage_count, dtype=tf.int32)
        return params

    @staticmethod
    def _static_last_dim(arr: Any) -> Optional[int]:
        try:
            dim = getattr(arr, "shape", None)
            if dim is None:
                return None
            last = dim[-1]
            return None if last is None else int(last)
        except Exception:
            return None

    def _infer_preload_feat_dim(self, params: Dict[str, Any]) -> Optional[int]:
        """静态推断 P_hat 的长度；优先 staged 特征，其次单步 P_hat/P。"""

        if not isinstance(params, dict):
            return None

        stages = params.get("stages")
        if isinstance(stages, dict):
            feat = stages.get("P_hat")
            dim = self._static_last_dim(feat)
            if dim:
                return dim

        if "P_hat" in params:
            dim = self._static_last_dim(params.get("P_hat"))
            if dim:
                return dim

        return self._static_last_dim(params.get("P"))

    def _extract_final_stage_params(
        self, params: Dict[str, Any], keep_context: bool = False
    ) -> Dict[str, Any]:
        """Return the last staged parameter set, optionally carrying context."""

        if not (
            self.cfg.preload_use_stages
            and isinstance(params, dict)
            and "stages" in params
        ):
            return params

        stages = params["stages"]
        final: Optional[Dict[str, tf.Tensor]] = None
        if isinstance(stages, dict) and stages:
            last_P = stages.get("P")
            last_feat = stages.get("P_hat")
            if last_P is not None and last_feat is not None:
                final = {"P": last_P[-1], "P_hat": last_feat[-1]}
                rank_tensor = stages.get("stage_rank")
                if rank_tensor is not None:
                    if getattr(rank_tensor, "shape", None) is not None and rank_tensor.shape.rank == 2:
                        final["stage_rank"] = rank_tensor[-1]
                    else:
                        final["stage_rank"] = rank_tensor
        elif isinstance(stages, (list, tuple)) and stages:
            last_stage = stages[-1]
            if isinstance(last_stage, dict):
                final = dict(last_stage)
            else:
                p_val, z_val = last_stage
                final = {"P": p_val, "P_hat": z_val}

        if final is None:
            return params

        if keep_context:
            for key in (
                "stage_order",
                "stage_rank",
                "stage_count",
                "stage_mask",
                "stage_last",
            ):
                if key in params and key not in final:
                    final[key] = params[key]
        return final

    def _make_warmup_case(self) -> Dict[str, np.ndarray]:
        mid = 0.5 * (float(self.cfg.preload_min) + float(self.cfg.preload_max))
        base = np.full((3,), mid, dtype=np.float32)
        case: Dict[str, np.ndarray] = {"P": base}
        if self.cfg.preload_use_stages:
            order = np.arange(base.shape[0], dtype=np.int32)
            case["order"] = order
            case.update(self._build_stage_case(base, order))
        return case

    # ----------------- 从 INP/Assembly 尝试自动发现接触对 -----------------
    def _autoguess_contacts_from_inp(self, asm: AssemblyModel) -> List[Dict[str, str]]:
        candidates = []
        try:
            # 0) 直接读取 asm.contact_pairs（通常是 ContactPair dataclass 列表）
            raw = getattr(asm, "contact_pairs", None)
            cand = self._normalize_pairs(raw)
            if cand:
                return cand

            # 1) 若模型实现了 autoguess_contact_pairs()
            if hasattr(asm, "autoguess_contact_pairs") and callable(asm.autoguess_contact_pairs):
                pairs = asm.autoguess_contact_pairs()
                cand = self._normalize_pairs(pairs)
                if cand:
                    return cand

            # 2) 兜底：常见属性名
            for attr in ["contacts", "contact_pairs", "interactions", "contact", "pairs"]:
                obj = getattr(asm, attr, None)
                cand = self._normalize_pairs(obj)
                if cand:
                    candidates.extend(cand)

            # 去重
            unique, seen = [], set()
            for d in candidates:
                key = (d.get("master_key"), d.get("slave_key"))
                if key not in seen and all(key):
                    unique.append(d);
                    seen.add(key)
            return unique
        except Exception:
            return []

    @staticmethod
    def _normalize_pairs(obj: Any) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        if obj is None:
            return out
        # 统一成可迭代
        seq = obj
        if isinstance(obj, dict):
            seq = [obj]
        elif not isinstance(obj, (list, tuple)):
            seq = [obj]

        for item in seq:
            # 1) 显式 (master, slave)
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                m, s = item[0], item[1]
                out.append({"master_key": str(m), "slave_key": str(s)})
                continue
            # 2) dict
            if isinstance(item, dict):
                keys = {k.lower(): v for k, v in item.items()}
                m = keys.get("master_key") or keys.get("master") or keys.get("a")
                s = keys.get("slave_key") or keys.get("slave") or keys.get("b")
                if m and s:
                    out.append({"master_key": str(m), "slave_key": str(s)})
                continue
            # 3) dataclass / 任意对象：有 .master / .slave 属性即可
            m = getattr(item, "master", None)
            s = getattr(item, "slave", None)
            if m is not None and s is not None:
                out.append({"master_key": str(m), "slave_key": str(s)})
                continue
        return out

    # ----------------- Build -----------------
    def build(self):
        cfg = self.cfg

        def _raise_vol_error(enum_names, X_vol, w_vol, mat_id):
            enum_str = ", ".join(f"{i}->{n}" for i, n in enumerate(enum_names))
            shapes = dict(
                X_vol=None if X_vol is None else tuple(getattr(X_vol, "shape", [])),
                w_vol=None if w_vol is None else tuple(getattr(w_vol, "shape", [])),
                mat_id=None if mat_id is None else tuple(getattr(mat_id, "shape", [])),
            )
            msg = (
                "\n[trainer] ERROR: build_volume_points 未返回有效体积分点；训练终止。\n"
                f"  - 材料枚举(按 part2mat 顺序)：{enum_str}\n"
                f"  - 返回 shapes: {shapes}\n"
                "  - 常见原因：\n"
                "      * INP 中的零件名与 part2mat 的键不一致（大小写/空格）。\n"
                "      * 材料名不在 materials 字典里。\n"
                "      * 网格上没有体积分点（或被过滤为空）。\n"
                "  - 建议：运行 sanity_check.py，确认第二步“体积分点 + 材料映射”为非 None。\n"
            )
            raise RuntimeError(msg)

        steps = [
            "Load INP", "Volume/Materials", "Elasticity",
            "Contact", "Preload", "Ties/BCs",
            "Model/Opt", "Checkpoint"
        ]

        print(f"[INFO] Build.start  inp_path={cfg.inp_path}")

        pb_kwargs = dict(total=len(steps), desc="Build", leave=True)
        if cfg.build_bar_color:
            pb_kwargs["colour"] = cfg.build_bar_color
        with tqdm(**pb_kwargs) as pb:
            # 1) INP
            self.asm = load_inp(cfg.inp_path)
            print(f"[INFO] Loaded INP: surfaces={len(self.asm.surfaces)} "
                  f"elsets={len(self.asm.elsets)} contact_pairs(raw)={len(getattr(self.asm, 'contact_pairs', []))}")
            pb.update(1)

            # 2) 体积分点 & 材料映射（严格检查）
            self.matlib = MaterialLibrary(cfg.materials)
            X_vol, w_vol, mat_id = build_volume_points(self.asm, cfg.part2mat, self.matlib)

            enum_names = list(dict.fromkeys(cfg.part2mat.values()))
            enum_str = ", ".join(f"{i}->{n}" for i, n in enumerate(enum_names))
            print(f"[trainer] Material enum (from part2mat order): {enum_str}")

            # —— 严格检查
            if X_vol is None or w_vol is None or mat_id is None:
                _raise_vol_error(enum_names, X_vol, w_vol, mat_id)

            n = getattr(X_vol, "shape", [0])[0]
            if getattr(w_vol, "shape", [0])[0] != n or getattr(mat_id, "shape", [0])[0] != n or n == 0:
                _raise_vol_error(enum_names, X_vol, w_vol, mat_id)

            # —— 暴露到 Trainer
            self.X_vol = X_vol
            self.w_vol = w_vol
            self.mat_id = mat_id
            self.enum_names = enum_names
            self.id2props_map = {i: tuple(map(float, cfg.materials[name])) for i, name in enumerate(enum_names)}

            pb.update(1)

            # 3) 弹性项 —— 改为 DFEM 构造方式
            # 注意：X_vol / w_vol / mat_id 依然保留在 Trainer 里用于可视化与检查，
            # 但不再传进 ElasticityEnergy，DFEM 内部自己做子单元积分。
            self.elasticity = ElasticityEnergy(
                asm=self.asm,
                part2mat=cfg.part2mat,
                materials=cfg.materials,
                cfg=cfg.elas_cfg,
            )
            pb.update(1)

            # 4) 接触（优先使用 cfg；否则尝试自动探测）
            self._cp_specs = []
            contact_source = ""
            if cfg.contact_pairs:
                try:
                    self._cp_specs = [ContactPairSpec(**d) for d in cfg.contact_pairs]
                except TypeError:
                    norm = self._normalize_pairs(cfg.contact_pairs)
                    self._cp_specs = [ContactPairSpec(**d) for d in norm] if norm else []
                contact_source = "配置"
            else:
                auto_pairs = self._autoguess_contacts_from_inp(self.asm)
                if auto_pairs:
                    self._cp_specs = [ContactPairSpec(**d) for d in auto_pairs]
                    contact_source = "自动识别"

            self.contact = None
            if self._cp_specs:
                try:
                    cmap = build_contact_map(
                        self.asm,
                        self._cp_specs,
                        cfg.n_contact_points_per_pair,
                        seed=cfg.contact_seed,
                    )
                    cat = cmap.concatenate()
                    self.contact = ContactOperator(cfg.contact_cfg)
                    self.contact.build_from_cat(cat, extra_weights=None, auto_orient=True)
                    self._current_contact_cat = cat
                    self._contact_rar_cache = None
                    total_pts = len(cmap)
                    src_txt = f"（{contact_source}）" if contact_source else ""
                    print(
                        f"[contact] 已加载 {len(self._cp_specs)} 对接触面{src_txt}，"
                        f"采样 {total_pts} 个点。"
                    )
                except Exception as exc:
                    print(f"[contact] 构建接触失败：{exc}")
                    self.contact = None
            else:
                print("[contact] 未找到接触信息，训练将不启用接触。")

            pb.update(1)

            # 5) 预紧（保留你的命名）
            if cfg.preload_specs:
                try:
                    specs = [BoltSurfaceSpec(**d) for d in cfg.preload_specs]
                    self.preload = PreloadWork(cfg.preload_cfg)
                    self.preload.build_from_specs(
                        self.asm,
                        specs,
                        n_points_each=cfg.preload_n_points_each,
                        seed=cfg.seed,
                    )
                    print(f"[preload] 已配置 {len(specs)} 个螺栓表面样本。")
                except Exception as exc:
                    print(f"[preload] 构建预紧样本失败：{exc}")
                    self.preload = None
            else:
                self.preload = None
                print("[preload] 未提供预紧配置。")
            pb.update(1)

            # 6) Ties/BCs（如需，可在 cfg 里填充）
            self.ties_ops, self.bcs_ops = [], []
            pb.update(1)

            # 6.5) 根据预紧特征维度统一 ParamEncoder 输入形状，避免 staged 特征长度变化
            self._warmup_case = self._make_warmup_case()
            self._warmup_params = self._make_preload_params(self._warmup_case)
            feat_dim = self._infer_preload_feat_dim(self._warmup_params)
            if feat_dim:
                old_dim = getattr(cfg.model_cfg.encoder, "in_dim", None)
                if old_dim != feat_dim:
                    print(
                        f"[model] 预紧特征维度 {old_dim} -> {feat_dim}，统一 ParamEncoder 输入。"
                    )
                    cfg.model_cfg.encoder.in_dim = feat_dim

            # 7) 模型 & 优化器
            if cfg.mixed_precision:
                cfg.model_cfg.mixed_precision = cfg.mixed_precision
            self.model = create_displacement_model(cfg.model_cfg)
            if getattr(cfg.model_cfg.field, "graph_precompute", False) and getattr(self, "elasticity", None):
                try:
                    self.model.field.set_global_graph(self.elasticity.X_nodes_tf)
                    print(
                        f"[graph] 已预计算全局 kNN 邻接: N={getattr(self.elasticity, 'n_nodes', '?')} k={cfg.model_cfg.field.graph_k}"
                    )
                except Exception as exc:
                    print(f"[graph] 预计算全局邻接失败，将退回动态构图：{exc}")
            base_optimizer = tf.keras.optimizers.Adam(cfg.lr)
            if cfg.mixed_precision:
                base_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(base_optimizer)
                print("[trainer] 已启用 LossScaleOptimizer 以配合混合精度训练。")
            self.optimizer = base_optimizer
            pb.update(1)

            # 8) checkpoint
            self.ckpt = tf.train.Checkpoint(encoder=self.model.encoder,
                                            field=self.model.field, opt=self.optimizer)
            self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, directory=cfg.ckpt_dir, max_to_keep=3)
            pb.update(1)

        # 预热网络，确保所有权重在进入梯度带之前已创建，从而可以被显式 watch
        try:
            warmup_n = min(2048, int(self.X_vol.shape[0])) if hasattr(self, "X_vol") else 0
        except Exception:
            warmup_n = 0
        if warmup_n > 0:
            X_sample = tf.convert_to_tensor(self.X_vol[:warmup_n], dtype=tf.float32)
            params = self._warmup_params or self._make_preload_params(self._make_warmup_case())
            eval_params = self._extract_final_stage_params(params)
            # 调用一次前向以创建所有变量；忽略实际输出
            _ = self.model.u_fn(X_sample, eval_params)

        self._train_vars = (
            list(self.model.encoder.trainable_variables)
            + list(self.model.field.trainable_variables)
        )
        if not self._train_vars:
            raise RuntimeError(
                "[trainer] 未发现可训练权重，请检查模型创建/预热流程是否成功。"
            )

        print(f"[trainer] GPU allocator = {os.environ.get('TF_GPU_ALLOCATOR', '(default)')}")
        print(
            f"[contact] 状态：{'已启用' if self.contact is not None else '未启用'}"
        )
        print(
            f"[preload] 状态：{'已启用' if self.preload is not None else '未启用'}"
        )

    # ----------------- 组装总能量 -----------------
    def _assemble_total(self) -> TotalEnergy:
        total = TotalEnergy(self.cfg.total_cfg)
        total.attach(
            elasticity=self.elasticity,
            contact=self.contact,
            preload=self.preload,
            ties=self.ties_ops,
            bcs=self.bcs_ops,
        )
        return total

    # ----------------- Contact-driven RAR -----------------

    def _update_contact_rar_cache(self):
        """Cache残差信息，供下一次接触重采样时做重要性抽样。"""

        if not self.cfg.contact_rar_enabled:
            self._contact_rar_cache = None
            return
        if self.contact is None or self._current_contact_cat is None:
            self._contact_rar_cache = None
            return

        metrics = self.contact.last_sample_metrics()
        if not metrics:
            self._contact_rar_cache = None
            return

        imp: Optional[np.ndarray] = None
        if "gap" in metrics:
            pen = np.maximum(-metrics["gap"], 0.0)
            imp = pen
        if "fric_res" in metrics:
            fr = np.abs(metrics["fric_res"])
            if imp is None:
                imp = fr
            else:
                alpha = float(np.clip(self.cfg.contact_rar_fric_mix, 0.0, 1.0))
                imp = (1.0 - alpha) * imp + alpha * fr

        if imp is None or imp.size == 0:
            self._contact_rar_cache = None
            return

        imp = np.where(np.isfinite(imp), imp, 0.0)
        self._contact_rar_cache = {
            "importance": imp,
            "cat": self._current_contact_cat,
            "meta": self.contact.last_meta(),
        }

    def _maybe_apply_contact_rar(
        self, cat_uniform: Dict[str, np.ndarray], step_index: int
    ) -> Tuple[Dict[str, np.ndarray], str]:
        """
        将上一批接触残差转化为重要性抽样，混合到本次接触样本中。

        Returns
        -------
        cat_new : dict
            可能重排/混合后的 contact cat。
        note : str
            用于进度条/日志的附加说明。
        """

        if (
            not self.cfg.contact_rar_enabled
            or self._contact_rar_cache is None
            or self._contact_rar_cache.get("cat") is None
        ):
            return cat_uniform, ""

        source_cat: Dict[str, np.ndarray] = self._contact_rar_cache.get("cat", {})
        importance: Optional[np.ndarray] = self._contact_rar_cache.get("importance")
        if importance is None or importance.shape[0] != source_cat.get("xs", np.zeros((0, 3))).shape[0]:
            return cat_uniform, ""

        total_n = int(cat_uniform.get("xs", np.zeros((0, 3))).shape[0])
        if total_n == 0:
            return cat_uniform, ""

        rar_frac = float(np.clip(self.cfg.contact_rar_fraction, 0.0, 1.0))
        min_uniform = int(np.round(total_n * np.clip(self.cfg.contact_rar_uniform_ratio, 0.0, 1.0)))
        n_rar = int(np.round(total_n * rar_frac))
        if n_rar + min_uniform > total_n:
            n_rar = max(0, total_n - min_uniform)
        n_uniform = max(0, total_n - n_rar)
        if n_rar <= 0:
            return cat_uniform, ""

        temp = max(self.cfg.contact_rar_temperature, 1e-6)
        weights = np.power(importance + self.cfg.contact_rar_floor, 1.0 / temp)
        weights = np.where(np.isfinite(weights), weights, 0.0)
        if float(weights.sum()) <= 0.0:
            return cat_uniform, ""

        rng = np.random.default_rng(self.cfg.contact_seed + step_index * 17)
        pair_ids = None
        meta = self._contact_rar_cache.get("meta") or {}
        if meta:
            pair_ids = meta.get("pair_id")
        if pair_ids is None:
            pair_ids = source_cat.get("pair_id")

        rar_indices: List[int] = []
        if self.cfg.contact_rar_balance_pairs and pair_ids is not None:
            pair_ids = np.asarray(pair_ids).reshape(-1)
            total_src = max(1, pair_ids.shape[0])
            for pid in np.unique(pair_ids):
                mask = pair_ids == pid
                if not np.any(mask):
                    continue
                quota = int(np.round(n_rar * float(mask.sum()) / float(total_src)))
                quota = max(1 if n_rar > 0 else 0, quota)
                probs = weights[mask]
                probs = probs / (probs.sum() + 1e-12)
                candidates = np.flatnonzero(mask)
                rar_indices.extend(list(rng.choice(candidates, size=quota, replace=True, p=probs)))
            if len(rar_indices) > n_rar:
                rar_indices = rar_indices[:n_rar]
        else:
            probs = weights / (weights.sum() + 1e-12)
            rar_indices = list(rng.choice(weights.shape[0], size=n_rar, replace=True, p=probs))

        if not rar_indices:
            return cat_uniform, ""

        rar_indices = np.asarray(rar_indices, dtype=np.int64)
        if rar_indices.shape[0] < n_rar:
            rar_indices = rng.choice(rar_indices, size=n_rar, replace=True)

        uni_indices = np.arange(cat_uniform["xs"].shape[0])
        if n_uniform < uni_indices.shape[0]:
            uni_indices = rng.choice(uni_indices, size=n_uniform, replace=False)

        cat_new: Dict[str, np.ndarray] = {}
        for key, arr in cat_uniform.items():
            src_arr = source_cat.get(key, arr)
            rar_part = src_arr[rar_indices] if rar_indices.size > 0 else src_arr[:0]
            uni_part = arr[uni_indices] if n_uniform > 0 else arr[:0]
            cat_new[key] = np.concatenate([rar_part, uni_part], axis=0)

        note = f"RAR {len(rar_indices)}/{total_n}"
        return cat_new, note

    # ----------------- Volume (strain energy) RAR -----------------

    def _update_volume_rar_cache(self):
        """基于上一批次的应变能密度，构造体积分点的重要性分布。"""

        if not self.cfg.volume_rar_enabled:
            self._volume_rar_cache = None
            return
        if self.elasticity is None:
            self._volume_rar_cache = None
            return
        if not hasattr(self.elasticity, "last_sample_metrics"):
            self._volume_rar_cache = None
            return

        metrics = self.elasticity.last_sample_metrics() or {}
        psi = metrics.get("psi")
        idx = metrics.get("idx")
        total_cells = int(getattr(self.elasticity, "n_cells", 0) or 0)
        if psi is None or idx is None or total_cells <= 0:
            self._volume_rar_cache = None
            return

        psi = np.asarray(psi, dtype=np.float64).reshape(-1)
        idx = np.asarray(idx, dtype=np.int64).reshape(-1)
        if psi.size == 0 or idx.size == 0 or psi.shape[0] != idx.shape[0]:
            self._volume_rar_cache = None
            return

        imp = np.full((total_cells,), float(self.cfg.volume_rar_floor), dtype=np.float32)
        valid = (idx >= 0) & (idx < total_cells) & np.isfinite(psi)
        if not np.any(valid):
            self._volume_rar_cache = None
            return
        np.add.at(imp, idx[valid], np.abs(psi[valid]).astype(np.float32))
        imp = np.where(np.isfinite(imp), imp, float(self.cfg.volume_rar_floor))
        self._volume_rar_cache = {"importance": imp}

    def _maybe_apply_volume_rar(self, step_index: int) -> Tuple[Optional[np.ndarray], str]:
        """返回一组 DFEM 子单元索引，按应变能密度进行重采样。"""

        if (
            not self.cfg.volume_rar_enabled
            or self._volume_rar_cache is None
            or self.elasticity is None
        ):
            return None, ""

        total_cells = int(getattr(self.elasticity, "n_cells", 0) or 0)
        target_n = getattr(getattr(self.elasticity, "cfg", None), "n_points_per_step", None)
        if total_cells <= 0 or target_n is None or target_n <= 0:
            return None, ""

        m = min(int(target_n), total_cells)
        importance = self._volume_rar_cache.get("importance")
        if importance is None or importance.shape[0] != total_cells:
            return None, ""

        rar_frac = float(np.clip(self.cfg.volume_rar_fraction, 0.0, 1.0))
        min_uniform = int(np.round(m * np.clip(self.cfg.volume_rar_uniform_ratio, 0.0, 1.0)))
        n_rar = int(np.round(m * rar_frac))
        if n_rar + min_uniform > m:
            n_rar = max(0, m - min_uniform)
        n_uniform = max(0, m - n_rar)
        if n_rar <= 0:
            return None, ""

        temp = max(self.cfg.volume_rar_temperature, 1e-6)
        weights = np.power(importance + float(self.cfg.volume_rar_floor), 1.0 / temp)
        weights = np.where(np.isfinite(weights), weights, 0.0)
        if float(weights.sum()) <= 0.0:
            return None, ""

        rng = np.random.default_rng(self.cfg.seed + step_index * 23)
        probs = weights / (weights.sum() + 1e-12)
        rar_indices = np.array(rng.choice(total_cells, size=n_rar, replace=True, p=probs), dtype=np.int64)

        if n_uniform > 0:
            uni_indices = rng.choice(total_cells, size=n_uniform, replace=False)
            combined = np.concatenate([rar_indices, uni_indices], axis=0)
        else:
            combined = rar_indices

        note = f"volRAR {len(rar_indices)}/{m}"
        return combined, note

    # 在 Trainer 类里新增/覆盖这个方法
    def _collect_trainable_variables(self):
        m = self.model

        # 1) 标准 keras.Model 路径
        if hasattr(m, "trainable_variables") and m.trainable_variables:
            return m.trainable_variables

        vars_list = []

        # 2) 常见容器属性（按你工程里常见命名，必要时可在这里增减）
        common_attrs = [
            "field", "net", "model", "encoder", "cond_encoder", "cond_enc",
            "embed", "embedding", "backbone", "trunk", "head",
            "blocks", "layers"
        ]
        for name in common_attrs:
            sub = getattr(m, name, None)
            if sub is None:
                continue
            if hasattr(sub, "trainable_variables"):
                vars_list += list(sub.trainable_variables)
            elif isinstance(sub, (list, tuple)):
                for layer in sub:
                    if hasattr(layer, "trainable_variables"):
                        vars_list += list(layer.trainable_variables)

        # 3) 去重
        seen, out = set(), []
        for v in vars_list:
            if v is None:
                continue
            vid = id(v)
            if vid in seen:
                continue
            seen.add(vid)
            out.append(v)

        # 4) 兜底（可能为空：例如图尚未 build）
        if not out:
            try:
                out = list(tf.compat.v1.trainable_variables())
            except Exception:
                out = []
        if not out:
            raise RuntimeError(
                "[trainer] 找不到可训练变量。请确认 DisplacementModel 的 Keras 子模块已构建完毕，"
                "如仍为空，可在 _collect_trainable_variables.common_attrs 中补充实际属性名。"
            )
        return out

    def _compute_total_loss(
        self,
        total: TotalEnergy,
        params: Dict[str, Any],
        *,
        adaptive: bool = True,
    ):
        stress_head_enabled = False
        try:
            stress_head_enabled = getattr(self.model.field.cfg, "stress_out_dim", 0) > 0
        except Exception:
            stress_head_enabled = False

        stress_fn = self.model.us_fn if stress_head_enabled and hasattr(self.model, "us_fn") else None

        Pi_raw, parts, stats = total.energy(
            self.model.u_fn, params=params, tape=None, stress_fn=stress_fn
        )
        Pi = Pi_raw
        if self.loss_state is not None:
            if adaptive:
                update_loss_weights(self.loss_state, parts, stats)
            Pi = combine_loss(parts, self.loss_state)
        reg = tf.add_n(self.model.losses) if getattr(self.model, "losses", None) else 0.0
        loss = Pi + reg
        return loss, Pi, parts, stats

    # 请用此代码完全覆盖 src/train/trainer.py 中的 _train_step 方法
    def _train_step(self, total, preload_case: Dict[str, np.ndarray]):
        model = self.model
        opt = self.optimizer
        train_vars = self._collect_trainable_variables()

        # 1. 生成完整参数 (包含 stages 字典)
        params = self._make_preload_params(preload_case)

        # 保持完整的阶段序列传入 total.energy，这样弹性/接触/预紧都会按 tighten 顺序逐
        # 阶段累积，不再随机抽取单一阶段，避免顺序信息被丢弃。

        with tf.GradientTape() as tape:
            # params 中保留了阶段序列（含 P_hat/顺序特征），总能量会按顺序累加
            loss, Pi, parts, stats = self._compute_total_loss(total, params, adaptive=True)

            use_loss_scale = hasattr(opt, "get_scaled_loss")
            if use_loss_scale:
                scaled_loss = opt.get_scaled_loss(loss)

        # 4) 反传 & 梯度缩放
        if use_loss_scale:
            scaled_grads = tape.gradient(scaled_loss, train_vars)
            grads = opt.get_unscaled_gradients(scaled_grads)
        else:
            grads = tape.gradient(loss, train_vars)

        if not any(g is not None for g in grads):
            raise RuntimeError(
                "[trainer] 所有梯度均为 None，训练无法继续。请确认损失在 tape 作用域内构建，且未用 .numpy()/np.* 切断图。"
            )

        # 5) 计算/裁剪梯度范数
        non_none = [(g, v) for g, v in zip(grads, train_vars) if g is not None]
        g_list, v_list = zip(*non_none)
        grad_norm = tf.linalg.global_norm(g_list)

        clip_norm = getattr(self, "clip_grad_norm", None) or getattr(self.cfg, "clip_grad_norm", None)
        if clip_norm is not None and float(clip_norm) > 0.0:
            g_list, _ = tf.clip_by_global_norm(g_list, clip_norm)

        opt.apply_gradients(zip(g_list, v_list))

        # 返回“当前权重下”的 Π，而不是 Pi_raw
        return Pi, parts, stats, grad_norm
    def _flatten_tensor_list(
        self, tensors: Sequence[Optional[tf.Tensor]], sizes: Sequence[int]
    ) -> tf.Tensor:
        flats: List[tf.Tensor] = []
        for tensor, size in zip(tensors, sizes):
            if tensor is None:
                flats.append(tf.zeros((size,), dtype=tf.float32))
            else:
                flats.append(tf.reshape(tf.cast(tensor, tf.float32), (-1,)))
        if not flats:
            return tf.zeros((0,), dtype=tf.float32)
        return tf.concat(flats, axis=0)

    def _assign_from_flat(
        self, flat_tensor: tf.Tensor, variables: Sequence[tf.Variable], sizes: Sequence[int]
    ):
        offset = 0
        for var, size in zip(variables, sizes):
            next_offset = offset + size
            slice_tensor = tf.reshape(flat_tensor[offset:next_offset], var.shape)
            var.assign(tf.cast(slice_tensor, var.dtype))
            offset = next_offset

    def _run_lbfgs_stage(self, total: TotalEnergy, show_progress: bool = False):
        if not self.cfg.lbfgs_enabled:
            return

        try:
            import tensorflow_probability as tfp
        except ImportError as exc:
            raise RuntimeError(
                "启用了 L-BFGS 精调，但当前环境未安装 tensorflow_probability。"
                "请先安装 tensorflow_probability 再重新运行。"
            ) from exc

        pbar = None
        if show_progress:
            lbfgs_kwargs = dict(
                total=max(1, int(self.cfg.lbfgs_max_iter)),
                desc="L-BFGS阶段 (2/2)",
                leave=True,
            )
            if self.cfg.train_bar_color:
                lbfgs_kwargs["colour"] = self.cfg.train_bar_color
            pbar = tqdm(**lbfgs_kwargs)

        train_vars = self._collect_trainable_variables()
        if not train_vars:
            raise RuntimeError("[lbfgs] 找不到可训练变量，无法执行 L-BFGS 精调。")

        sizes = []
        for var in train_vars:
            size = var.shape.num_elements()
            if size is None:
                raise ValueError(
                    f"[lbfgs] 变量 {var.name} 的形状包含未知维度，无法展开为一维向量。"
                )
            sizes.append(int(size))

        if self.cfg.lbfgs_reuse_last_batch and self._last_preload_case is not None:
            lbfgs_case = copy.deepcopy(self._last_preload_case)
        else:
            lbfgs_case = self._sample_preload_case()

        lbfgs_params = self._make_preload_params(lbfgs_case)
        order = lbfgs_case.get("order")
        if order is None:
            order_txt = "默认顺序"
        else:
            order_txt = "-".join(str(int(i) + 1) for i in order)

        print("[lbfgs] 开始 L-BFGS 精调阶段：")
        print(
            f"[lbfgs] 固定预紧力 P={[int(p) for p in lbfgs_case['P']]} N, 顺序={order_txt},"
            f" 最大迭代 {self.cfg.lbfgs_max_iter}, tol={self.cfg.lbfgs_tolerance}"
        )

        initial_position = self._flatten_tensor_list(train_vars, sizes)

        def _value_and_gradients(position):
            self._assign_from_flat(position, train_vars, sizes)
            with tf.GradientTape() as tape:
                tape.watch(train_vars)
                loss, Pi, parts, stats = self._compute_total_loss(
                    total, lbfgs_params, adaptive=False
                )
            grads = tape.gradient(loss, train_vars)
            grad_vec = self._flatten_tensor_list(grads, sizes)
            return loss, grad_vec

        results = tfp.optimizer.lbfgs_minimize(
            value_and_gradients_function=_value_and_gradients,
            initial_position=initial_position,
            tolerance=self.cfg.lbfgs_tolerance,
            max_iterations=self.cfg.lbfgs_max_iter,
            num_correction_pairs=self.cfg.lbfgs_history_size,
            parallel_iterations=1,
            linesearch_max_iterations=max(1, int(self.cfg.lbfgs_line_search)),
        )

        self._assign_from_flat(results.position, train_vars, sizes)

        status = "converged" if results.converged else "stopped"
        if results.failed:
            status = "failed"
        grad_norm = float(results.gradient_norm.numpy()) if results.gradient_norm is not None else float("nan")

        if pbar is not None:
            try:
                completed = int(results.num_iterations.numpy())
            except Exception:
                completed = 0
            pbar.update(max(0, min(self.cfg.lbfgs_max_iter, completed)))
            pbar.set_postfix_str(
                self._wrap_bar_text(
                    f"loss={float(results.objective_value.numpy()):.3e} grad={grad_norm:.3e}"
                )
            )
            pbar.close()

        print(
            f"[lbfgs] 完成：状态={status}, iters={int(results.num_iterations.numpy())}, "
            f"loss={float(results.objective_value.numpy()):.3e}, grad={grad_norm:.3e}"
        )


    # ----------------- 训练 -----------------
    def run(self):
        self.build()
        print(f"[trainer] 当前训练设备：{self.device_summary}")
        total = self._assemble_total()
        attach_ties_and_bcs_from_inp(
            total=total,
            asm=self.asm,
            inp_path=self.cfg.inp_path,
            cfg=self.cfg,
        )
        print("[dbg] Tie/BC 已挂载到 total")

        # ---- 初始化自适应损失权重状态 ----
        # 以 TotalConfig 里的 w_int / w_cn / ... 作为基准权重
        base_weights = {
            "E_int": self.cfg.total_cfg.w_int,
            "E_cn": self.cfg.total_cfg.w_cn,
            "E_ct": self.cfg.total_cfg.w_ct,
            "E_tie": self.cfg.total_cfg.w_tie,
            "W_pre": self.cfg.total_cfg.w_pre,
            "E_sigma": self.cfg.total_cfg.w_sigma,
            # 残差项默认权重为 0，需要的话再在 config 里改
            "R_fric_comp": 0.0,
            "R_contact_comp": 0.0,
        }

        adaptive_enabled = bool(getattr(self.cfg, "loss_adaptive_enabled", False))
        if adaptive_enabled:
            scheme = getattr(self.cfg.total_cfg, "adaptive_scheme", "contact_only")
            focus_terms = getattr(self.cfg, "loss_focus_terms", tuple())
            if focus_terms:
                scheme = "focus"
            self.loss_state = LossWeightState.from_config(
                base_weights=base_weights,
                adaptive_scheme=scheme,
                ema_decay=getattr(self.cfg, "loss_ema_decay", 0.95),
                min_factor=getattr(self.cfg, "loss_min_factor", 0.25),
                max_factor=getattr(self.cfg, "loss_max_factor", 4.0),
                gamma=getattr(self.cfg, "loss_gamma", 2.0),
                focus_terms=focus_terms,
                update_every=getattr(self.cfg, "loss_update_every", 1),
            )
        else:
            self.loss_state = None
        if self.cfg.lbfgs_enabled:
            train_desc = "Adam阶段 (1/2)"
        else:
            train_desc = "训练"
        train_pb_kwargs = dict(total=self.cfg.max_steps, desc=train_desc, leave=True)
        if self.cfg.train_bar_color:
            train_pb_kwargs["colour"] = self.cfg.train_bar_color
        with tqdm(**train_pb_kwargs) as p_train:
            for step in range(1, self.cfg.max_steps + 1):
                # 子进度条：本 step 的 4 个动作
                step_pb_kwargs = dict(total=4, leave=False)
                if self.cfg.step_bar_color:
                    step_pb_kwargs["colour"] = self.cfg.step_bar_color
                with tqdm(**step_pb_kwargs) as p_step:
                    # 1) 接触重采样
                    self._set_pbar_desc(p_step, f"step {step}: 接触重采样")
                    t0 = time.perf_counter()
                    contact_note = "跳过"
                    if self.contact is None:
                        contact_note = "跳过 (无接触体)"
                        self._contact_rar_cache = None
                    else:
                        should_resample = step == 1
                        if not should_resample and self.cfg.resample_contact_every > 0:
                            should_resample = (
                                (step - 1) % self.cfg.resample_contact_every == 0
                            )

                        if should_resample:
                            try:
                                cmap = resample_contact_map(
                                    self.asm,
                                    self._cp_specs,
                                    self.cfg.n_contact_points_per_pair,
                                    base_seed=self.cfg.contact_seed,
                                    step_index=step,
                                )
                                cat_uniform = cmap.concatenate()
                                cat, rar_note = self._maybe_apply_contact_rar(
                                    cat_uniform, step
                                )
                                self.contact.reset_for_new_batch()
                                self.contact.build_from_cat(
                                    cat, extra_weights=None, auto_orient=True
                                )
                                self._current_contact_cat = cat
                                contact_note = f"更新 {len(cmap)} 点"
                                if rar_note:
                                    contact_note += f" | {rar_note}"
                            except Exception as exc:
                                contact_note = "更新失败"
                                self._contact_rar_cache = None
                                print(f"[contact] 第 {step} 步接触重采样失败：{exc}")
                        else:
                            if self.cfg.resample_contact_every <= 0:
                                contact_note = "跳过 (沿用首步采样)"
                            else:
                                remaining = self.cfg.resample_contact_every - (
                                    (step - 1) % self.cfg.resample_contact_every
                                )
                                contact_note = f"跳过 (距下次还有 {remaining} 步)"
                    elapsed = time.perf_counter() - t0
                    self._step_stage_times.append(("resample", elapsed))
                    self._set_pbar_postfix(
                        p_step,
                        f"{contact_note} | {self._format_seconds(elapsed)}"
                    )
                    p_step.update(1)

                    # 2) 前向 + 反传（随机采样三螺栓预紧力）
                    self._set_pbar_desc(p_step, f"step {step}: 前向/反传")
                    t0 = time.perf_counter()
                    preload_case = self._sample_preload_case()
                    vol_note = ""
                    if self.elasticity is not None and hasattr(self.elasticity, "set_sample_indices"):
                        vol_indices, vol_note = self._maybe_apply_volume_rar(step)
                        self.elasticity.set_sample_indices(vol_indices)
                    Pi, parts, stats, grad_norm = self._train_step(total, preload_case)
                    P_np = preload_case["P"]
                    order_np = preload_case.get("order")
                    self._last_preload_case = copy.deepcopy(preload_case)
                    self._update_contact_rar_cache()
                    self._update_volume_rar_cache()
                    pi_val = float(Pi.numpy())
                    if self._pi_baseline is None:
                        self._pi_baseline = pi_val if pi_val != 0.0 else 1.0
                    if self._pi_ema is None:
                        self._pi_ema = pi_val
                    else:
                        ema_alpha = 0.1
                        self._pi_ema = (1 - ema_alpha) * self._pi_ema + ema_alpha * pi_val
                    rel_pi = pi_val / (self._pi_baseline or pi_val or 1.0)
                    rel_delta = None
                    if self._prev_pi is not None and self._prev_pi != 0.0:
                        rel_delta = (self._prev_pi - pi_val) / abs(self._prev_pi)
                    self._prev_pi = pi_val
                    elapsed = time.perf_counter() - t0
                    self._step_stage_times.append(("train", elapsed))
                    device = self._short_device_name(getattr(Pi, "device", None))
                    grad_val = float(grad_norm.numpy()) if hasattr(grad_norm, "numpy") else float(grad_norm)
                    rel_pct = rel_pi * 100.0 if rel_pi is not None else None
                    rel_txt = (
                        f"Πrel={rel_pct:.2f}%" if rel_pct is not None else "Πrel=--"
                    )
                    d_txt = (
                        f"ΔΠ={rel_delta * 100:+.1f}%"
                        if rel_delta is not None
                        else "ΔΠ=--"
                    )
                    ema_txt = f"Πema={self._pi_ema:.2e}" if self._pi_ema is not None else "Πema=--"
                    order_txt = ""
                    if order_np is not None:
                        order_txt = " order=" + "-".join(str(int(x) + 1) for x in order_np)
                    energy_summary = self._format_energy_summary(parts)
                    energy_txt = f" | {energy_summary}" if energy_summary else ""
                    if vol_note:
                        energy_txt += f" | {vol_note}"
                    train_note = (
                        f"P=[{int(P_np[0])},{int(P_np[1])},{int(P_np[2])}]"
                        f"{order_txt}{energy_txt} | Π={pi_val:.2e} {rel_txt} {d_txt} "
                        f"grad={grad_val:.2e} {ema_txt}"
                    )
                    if step == 1:
                        train_note += " | 首轮包含图追踪/缓存构建"
                    self._set_pbar_postfix(
                        p_step,
                        f"{train_note} | {self._format_seconds(elapsed)} | dev={device}"
                    )
                    p_step.update(1)

                    # 3) ALM 更新
                    self._set_pbar_desc(p_step, f"step {step}: ALM 更新")
                    t0 = time.perf_counter()
                    alm_note = "跳过"
                    if self.contact is None:
                        alm_note = "跳过 (无接触体)"
                    elif self.cfg.alm_update_every <= 0:
                        alm_note = "跳过 (已禁用)"
                    elif step % self.cfg.alm_update_every == 0:
                        params_for_update = self._make_preload_params(preload_case)
                        total.update_multipliers(self.model.u_fn, params=params_for_update)
                        alm_note = "已更新"
                    else:
                        remaining = self.cfg.alm_update_every - (
                            step % self.cfg.alm_update_every
                        )
                        alm_note = f"跳过 (距下次还有 {remaining} 步)"
                    elapsed = time.perf_counter() - t0
                    self._step_stage_times.append(("alm", elapsed))
                    self._set_pbar_postfix(
                        p_step,
                        f"{alm_note} | {self._format_seconds(elapsed)}"
                    )
                    p_step.update(1)

                    # 4) 日志 & ckpt
                    self._set_pbar_desc(p_step, f"step {step}: 日志/检查点")
                    t0 = time.perf_counter()
                    log_note = "跳过"
                    if self.cfg.log_every <= 0:
                        log_note = "跳过 (已禁用)"
                    else:
                        should_log = step == 1 or step % self.cfg.log_every == 0
                        if should_log:
                            postfix, log_note = self._format_train_log_postfix(
                                P_np,
                                Pi,
                                parts,
                                stats,
                                grad_val,
                                rel_pi,
                                rel_delta,
                                order_np,
                            )
                            if postfix:
                                p_train.set_postfix_str(postfix)

                            metric_name = self.cfg.save_best_on.lower()
                            metric_val = (
                                pi_val
                                if metric_name == "pi"
                                else float(parts["E_int"].numpy())
                            )
                            if metric_val < self.best_metric:
                                self.best_metric = metric_val
                                self.ckpt_manager.save(checkpoint_number=step)
                                log_note += " | 已保存"
                    if (
                        self.cfg.log_every > 0
                        and not (step == 1 or step % self.cfg.log_every == 0)
                    ):
                        remaining = self.cfg.log_every - (step % self.cfg.log_every)
                        log_note = f"跳过 (距下次还有 {remaining} 步)"
                    elapsed = time.perf_counter() - t0
                    self._step_stage_times.append(("log", elapsed))
                    self._set_pbar_postfix(
                        p_step,
                        f"{log_note} | {self._format_seconds(elapsed)}"
                    )
                    p_step.update(1)

                p_train.update(1)

                if step % max(1, self.cfg.log_every) == 0:
                    total_spent = sum(t for _, t in self._step_stage_times)
                    if total_spent > 0:
                        label_map = {
                            "resample": "采样",
                            "train": "前向/反传",
                            "alm": "ALM",
                            "log": "日志"
                        }
                        parts_txt = ", ".join(
                            f"{label_map.get(name, name)}:{t / total_spent * 100:.0f}%"
                            for name, t in self._step_stage_times
                        )
                        summary_note = (
                            f"step{step}耗时 {self._format_seconds(total_spent)} ({parts_txt})"
                        )
                        if step == 1:
                            summary_note += " | 首轮额外包括图追踪/初次缓存"
                        self._set_pbar_postfix(p_train, summary_note)
                    self._step_stage_times.clear()

            # 训练结束：再存一次
            self.ckpt_manager.save(checkpoint_number=self.cfg.max_steps)

        if self.cfg.lbfgs_enabled:
            self._run_lbfgs_stage(total, show_progress=True)

        self._visualize_after_training(n_samples=self.cfg.viz_samples_after_train)

    # ----------------- Checkpoint utilities -----------------
    def restore_checkpoint(self, checkpoint_path: Optional[str] = None,
                           expect_partial: bool = True) -> str:
        """Restore model/optimizer weights from a checkpoint.

        Args:
            checkpoint_path: Explicit checkpoint path. If ``None`` the latest
                checkpoint managed by :class:`tf.train.CheckpointManager` is
                used.
            expect_partial: Whether partial restores are acceptable (useful
                when optimizer slots are absent, e.g. when exporting models for
                inference only).

        Returns:
            The path of the checkpoint that was restored.

        Raises:
            FileNotFoundError: If no checkpoint can be located.
            RuntimeError: If ``build()`` has not been called yet.
        """

        if self.ckpt is None:
            raise RuntimeError("Trainer.restore_checkpoint() called before build().")

        ckpt_path = checkpoint_path
        if ckpt_path is None:
            if self.ckpt_manager is None:
                raise RuntimeError("Checkpoint manager not initialised; call build() first.")
            ckpt_path = self.ckpt_manager.latest_checkpoint

        if not ckpt_path:
            raise FileNotFoundError(
                "No checkpoint found. Train the model first or pass an explicit path."
            )

        status = self.ckpt.restore(ckpt_path)
        try:
            if expect_partial:
                status.expect_partial()
            else:
                status.assert_consumed()
        except AssertionError:
            # Fall back to the strongest available assertion; if it still
            # fails TensorFlow will raise, which is what we want.
            status.assert_existing_objects_matched()

        print(f"[trainer] Restored checkpoint -> {ckpt_path}")
        return ckpt_path

    def export_saved_model(self, export_dir: str) -> str:
        """Export the PINN displacement model as a TensorFlow SavedModel."""

        if self.model is None:
            raise RuntimeError("Trainer.export_saved_model() requires build()/restore().")

        n_bolts = max(1, len(self.cfg.preload_specs) or 3)

        module = _SavedModelModule(
            model=self.model,
            use_stages=bool(self.cfg.preload_use_stages),
            shift=float(self.cfg.model_cfg.preload_shift),
            scale=float(self.cfg.model_cfg.preload_scale),
            n_bolts=n_bolts,
        )
        serving_fn = module.run.get_concrete_function()
        tf.saved_model.save(module, export_dir, signatures={"serving_default": serving_fn})
        print(f"[trainer] SavedModel exported -> {export_dir}")
        return export_dir

    def generate_deflection_map(
        self,
        preload: Any,
        out_path: Optional[str] = None,
        title_prefix: Optional[str] = None,
        show: bool = False,
        data_out_path: Optional[str] = "auto",
        preload_order: Optional[Sequence[int]] = None,
    ) -> str:
        """Generate a mirror deflection contour for a user-specified preload.

        Args:
            preload: Iterable of three preload values (N) in the order
                ``[P1, P2, P3]``.
            out_path: Optional absolute/relative path to save the PNG. If
                omitted the file is stored under ``cfg.out_dir`` with a name
                derived from the preload values.
            title_prefix: Optional custom title prefix for the figure.
            show: If ``True`` call ``plt.show()`` instead of closing the
                figure (useful when running interactively).
            data_out_path: Text export path or ``"auto"``/``None`` for
                automatic/disabled displacement dumps.
            preload_order: Optional tightening sequence (either 0- or
                1-based) used when staged preload is enabled. When omitted,
                the natural 1-2-3 order is applied.

        Returns:
            The absolute/relative path where the image was written.
        """

        if self.asm is None or self.model is None:
            raise RuntimeError("Trainer.generate_deflection_map() requires build()/restore().")

        P = np.asarray(list(preload), dtype=np.float32).reshape(-1)
        if P.size != 3:
            raise ValueError("'preload' must contain exactly three values (for the three bolts).")

        case: Dict[str, np.ndarray] = {"P": P}
        if preload_order is not None and not self.cfg.preload_use_stages:
            print("[viz] preload_order 被忽略：配置中未启用分阶段加载。")
        if self.cfg.preload_use_stages:
            nb = P.size
            if preload_order is not None:
                order_arr = self._normalize_order(preload_order, nb)
            else:
                order_arr = np.arange(nb, dtype=np.int32)
            case["order"] = order_arr
            case.update(self._build_stage_case(P, order_arr))
            order_display = "-".join(str(int(o) + 1) for o in order_arr.tolist())
        else:
            order_arr = None
            order_display = None
        params_full = self._make_preload_params(case)
        params = self._extract_final_stage_params(params_full, keep_context=True)
        title = title_prefix or self.cfg.viz_title_prefix
        if order_display:
            title = f"{title}  (order={order_display})"

        if out_path is None:
            os.makedirs(self.cfg.out_dir, exist_ok=True)
            p_int = [int(round(float(x))) for x in P]
            ord_tag = f"_ord{order_display.replace('-', '')}" if order_display else ""
            out_path = os.path.join(
                self.cfg.out_dir,
                f"deflection_manual_P{p_int[0]}_{p_int[1]}_{p_int[2]}{ord_tag}.png",
            )

        resolved_data_path: Optional[str]
        if data_out_path is None:
            resolved_data_path = None
        else:
            key = str(data_out_path).strip().lower()
            if key == "auto":
                if self.cfg.viz_write_data and out_path:
                    resolved_data_path = os.path.splitext(out_path)[0] + ".txt"
                else:
                    resolved_data_path = None
            elif key in {"", "none"}:
                resolved_data_path = None
            else:
                resolved_data_path = data_out_path

        resolved_mesh_path: Optional[str]
        if self.cfg.viz_write_surface_mesh and out_path:
            resolved_mesh_path = "auto"
        else:
            resolved_mesh_path = None

        full_plot_enabled = bool(self.cfg.viz_plot_full_structure)
        full_struct_out = "auto" if (full_plot_enabled and out_path) else None
        full_struct_data = (
            "auto" if (full_plot_enabled and self.cfg.viz_write_full_structure_data and out_path) else None
        )

        diag_out: Dict[str, Any] = {} if self.cfg.viz_diagnose_blanks else None

        _, _, data_path = plot_mirror_deflection_by_name(
            self.asm,
            self.cfg.mirror_surface_name,
            self.model.u_fn,
            params,
            P_values=tuple(float(x) for x in P),
            out_path=out_path,
            render_surface=self.cfg.viz_surface_enabled,
            surface_source=self.cfg.viz_surface_source,
            title_prefix=title,
            units=self.cfg.viz_units,
            levels=self.cfg.viz_levels,
            symmetric=self.cfg.viz_symmetric,
            show=show,
            data_out_path=resolved_data_path,
            surface_mesh_out_path=resolved_mesh_path,
            plot_full_structure=full_plot_enabled,
            full_structure_out_path=full_struct_out,
            full_structure_data_out_path=full_struct_data,
            full_structure_part=self.cfg.viz_full_structure_part,
            style=self.cfg.viz_style,
            cmap=self.cfg.viz_colormap,
            draw_wireframe=self.cfg.viz_draw_wireframe,
            refine_subdivisions=self.cfg.viz_refine_subdivisions,
            refine_max_points=self.cfg.viz_refine_max_points,
            retriangulate_2d=self.cfg.viz_retriangulate_2d,
            eval_batch_size=self.cfg.viz_eval_batch_size,
            eval_scope=self.cfg.viz_eval_scope,
            diagnose_blanks=self.cfg.viz_diagnose_blanks,
            auto_fill_blanks=self.cfg.viz_auto_fill_blanks,
            # 强制启用可视化去除刚体位移，避免沿用 mirror_viz 的默认 False
            remove_rigid=True,
            diag_out=diag_out,
        )

        self.last_viz_diag = diag_out.get("blank_check") if diag_out is not None else None
        self.last_viz_data_path = data_path
        if self.cfg.viz_surface_enabled and data_path:
            print(f"[viz] displacement data -> {data_path}")
        if self.cfg.viz_surface_enabled and out_path:
            if order_display:
                print(f"[viz] saved -> {out_path}  (order={order_display})")
            else:
                print(f"[viz] saved -> {out_path}")
        return out_path

    # ----------------- 可视化（鲁棒多签名） -----------------
    def _call_viz(self, P: np.ndarray, params: Dict[str, tf.Tensor], out_path: str, title: str):
        bare = self.cfg.mirror_surface_name
        data_path = None
        if self.cfg.viz_write_data and out_path:
            data_path = os.path.splitext(out_path)[0] + ".txt"

        mesh_path = None
        if self.cfg.viz_write_surface_mesh and out_path:
            mesh_path = "auto"

        full_plot_enabled = bool(self.cfg.viz_plot_full_structure)
        full_struct_out = "auto" if (full_plot_enabled and out_path) else None
        full_struct_data = (
            "auto" if (full_plot_enabled and self.cfg.viz_write_full_structure_data and out_path) else None
        )

        diag_out: Dict[str, Any] = {} if self.cfg.viz_diagnose_blanks else None

        result = plot_mirror_deflection_by_name(
            self.asm,
            bare,
            self.model.u_fn,
            params,
            P_values=tuple(float(x) for x in P.reshape(-1)),
            out_path=out_path,
            render_surface=self.cfg.viz_surface_enabled,
            surface_source=self.cfg.viz_surface_source,
            title_prefix=title,
            units=self.cfg.viz_units,
            levels=self.cfg.viz_levels,
            symmetric=self.cfg.viz_symmetric,
            data_out_path=data_path,
            surface_mesh_out_path=mesh_path,
            plot_full_structure=full_plot_enabled,
            full_structure_out_path=full_struct_out,
            full_structure_data_out_path=full_struct_data,
            full_structure_part=self.cfg.viz_full_structure_part,
            style=self.cfg.viz_style,
            cmap=self.cfg.viz_colormap,
            draw_wireframe=self.cfg.viz_draw_wireframe,
            refine_subdivisions=self.cfg.viz_refine_subdivisions,
            refine_max_points=self.cfg.viz_refine_max_points,
            retriangulate_2d=self.cfg.viz_retriangulate_2d,
            eval_batch_size=self.cfg.viz_eval_batch_size,
            eval_scope=self.cfg.viz_eval_scope,
            diagnose_blanks=self.cfg.viz_diagnose_blanks,
            auto_fill_blanks=self.cfg.viz_auto_fill_blanks,
            # 强制启用可视化去除刚体位移，避免沿用 mirror_viz 的默认 False
            remove_rigid=True,
            diag_out=diag_out,
        )

        if diag_out is not None:
            self.last_viz_diag = diag_out.get("blank_check")
        return result

    def _fixed_viz_preload_cases(self) -> List[Dict[str, np.ndarray]]:
        """生成固定的 6 组预紧案例以避免可视化阶段的随机性."""

        nb = 3  # 现有镜面配置假定三颗螺栓

        def _make_case(P_list: Sequence[float], order: Sequence[int]) -> Dict[str, np.ndarray]:
            P_arr = np.asarray(P_list, dtype=np.float32).reshape(-1)
            if P_arr.size != nb:
                raise ValueError(f"固定可视化仅支持 {nb} 颗螺栓，收到 {P_arr.size} 维载荷。")
            case: Dict[str, np.ndarray] = {"P": P_arr}
            if not self.cfg.preload_use_stages:
                return case
            order_norm = self._normalize_order(order, nb)
            if order_norm is None:
                return case
            case["order"] = order_norm
            case.update(self._build_stage_case(P_arr, order_norm))
            return case

        cases: List[Dict[str, np.ndarray]] = []

        # 三组单螺栓 2000N，顺序固定为 1-2-3
        for P_single in ([2000.0, 0.0, 0.0], [0.0, 2000.0, 0.0], [0.0, 0.0, 2000.0]):
            cases.append(_make_case(P_single, order=[0, 1, 2]))

        # 三组 1500N 等幅，采用不同拧紧顺序
        for order in ([0, 1, 2], [0, 2, 1], [2, 1, 0]):
            cases.append(_make_case([1500.0, 1500.0, 1500.0], order=order))

        return cases

    def _visualize_after_training(self, n_samples: int = 5):
        if self.asm is None or self.model is None:
            return
        os.makedirs(self.cfg.out_dir, exist_ok=True)
        cases = self._fixed_viz_preload_cases()
        n_total = len(cases) if cases else n_samples
        print(
            f"[trainer] Generating {n_total} deflection maps for '{self.cfg.mirror_surface_name}' ..."
        )
        iter_cases = cases if cases else [self._sample_preload_case() for _ in range(n_samples)]
        for i, preload_case in enumerate(iter_cases):
            P = preload_case["P"]
            order_display = None
            if self.cfg.preload_use_stages and "order" in preload_case:
                order_display = "-".join(
                    str(int(o) + 1) for o in preload_case["order"].tolist()
                )
            title = f"{self.cfg.viz_title_prefix}  P=[{int(P[0])},{int(P[1])},{int(P[2])}]N"
            if order_display:
                title += f"  (order={order_display})"
            suffix = f"_{order_display.replace('-', '')}" if order_display else ""
            save_path = os.path.join(
                self.cfg.out_dir, f"deflection_{i+1:02d}{suffix}.png"
            )
            params_full = self._make_preload_params(preload_case)
            params_eval = self._extract_final_stage_params(params_full, keep_context=True)
            try:
                _, _, data_path = self._call_viz(P, params_eval, save_path, title)
                if self.cfg.viz_surface_enabled:
                    if not os.path.exists(save_path):
                        try:
                            import matplotlib.pyplot as plt
                            plt.savefig(save_path, dpi=200, bbox_inches="tight")
                            plt.close()
                        except Exception:
                            pass
                    if order_display:
                        print(f"[viz] saved -> {save_path}  (order={order_display})")
                    else:
                        print(f"[viz] saved -> {save_path}")
                    if data_path:
                        print(f"[viz] displacement data -> {data_path}")
            except TypeError as e:
                print("[viz] signature mismatch:", e)
            except Exception as e:
                print("[viz] error:", e)


class _SavedModelModule(tf.Module):
    """TensorFlow module exposing the PINN forward pass for SavedModel export."""

    @tf.autograph.experimental.do_not_convert
    def __init__(
        self,
        model: DisplacementModel,
        use_stages: bool,
        shift: float,
        scale: float = 1.0,
        n_bolts: int = 3,
    ):
        super().__init__(name="pinn_saved_model")
        # 1. 显式追踪子模块 (关键修复)
        # 将 DisplacementModel 的核心子层挂载到 self 上，确保 TF 能追踪到变量
        self.encoder = model.encoder
        self.field = model.field
        
        # 2. 保留原始模型的引用 (用于调用 u_fn)
        # 注意：直接用 self._model.u_fn 可能会导致追踪路径断裂
        # 我们需要确保 u_fn 使用的 encoder/field 就是上面挂载的这两个
        self._model = model

        self._use_stages = bool(use_stages)
        self._shift = tf.constant(shift, dtype=tf.float32)
        self._scale = tf.constant(scale, dtype=tf.float32)
        self._n_bolts = int(max(1, n_bolts))

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, 3], dtype=tf.float32, name="X"),
            tf.TensorSpec(shape=[None], dtype=tf.float32, name="P"),
            tf.TensorSpec(shape=[None], dtype=tf.int32, name="order"),
        ]
    )
    def run(self, X, P, order):
        # 准备参数
        params = self._prepare_params(P, order)
        
        # 调用模型的前向传播
        # 由于 self._model.encoder 就是 self.encoder，变量是共享且被追踪的
        return self._model.u_fn(X, params)

    def _prepare_params(self, P, order):
        # 确保 P 是 1D
        P = tf.reshape(P, (self._n_bolts,))
        
        # 如果不启用分阶段，直接返回 P
        if not self._use_stages:
            return {"P": P}
            
        # 归一化顺序
        order = self._normalize_order(order)
        
        # 构建阶段张量 (包含 P_hat 特征)
        stage_P, stage_feat = self._build_stage_tensors(P, order)
        
        # 返回最后一个阶段的数据
        return {"P": stage_P[-1], "P_hat": stage_feat[-1]}

    def _normalize_order(self, order):
        order = tf.reshape(order, (self._n_bolts,))
        default = tf.range(self._n_bolts, dtype=tf.int32)
        
        # 检查是否全部 >= 0
        cond = tf.reduce_all(order >= 0)
        order = tf.where(cond, order, default)
        
        # 检查是否需要从 1-based 转 0-based
        minv = tf.reduce_min(order)
        maxv = tf.reduce_max(order)

        def _one_based():
            return order - 1

        order = tf.cond(
            tf.logical_and(tf.greater_equal(minv, 1), tf.less_equal(maxv, self._n_bolts)),
            _one_based,
            lambda: order,
        )
        return order

    def _build_stage_tensors(self, P, order):
        stage_count = self._n_bolts
        cumulative = tf.zeros_like(P)
        mask = tf.zeros_like(P)
        
        # 使用 TensorArray 动态构建序列
        loads_ta = tf.TensorArray(tf.float32, size=stage_count)
        masks_ta = tf.TensorArray(tf.float32, size=stage_count)
        last_ta = tf.TensorArray(tf.float32, size=stage_count)

        def body(i, cum, mask_vec, loads, masks, lasts):
            # 获取当前步骤要拧的螺栓索引
            bolt = tf.gather(order, i)
            bolt = tf.clip_by_value(bolt, 0, self._n_bolts - 1)
            
            # 获取该螺栓的力
            load_val = tf.gather(P, bolt)
            idx = tf.reshape(bolt, (1, 1))
            
            # 更新累积载荷 (cumulative)
            cum = tf.tensor_scatter_nd_update(cum, idx, tf.reshape(load_val, (1,)))
            
            # 更新掩码 (mask)
            mask_vec = tf.tensor_scatter_nd_update(
                mask_vec, idx, tf.ones((1,), dtype=tf.float32)
            )
            
            # 记录到 Array
            loads = loads.write(i, cum)
            masks = masks.write(i, mask_vec)
            
            # 构建 last_active (当前操作的螺栓)
            last_vec = tf.zeros_like(P)
            last_vec = tf.tensor_scatter_nd_update(
                last_vec, idx, tf.ones((1,), dtype=tf.float32)
            )
            lasts = lasts.write(i, last_vec)
            
            return i + 1, cum, mask_vec, loads, masks, lasts

        _, cumulative, mask, loads_ta, masks_ta, last_ta = tf.while_loop(
            lambda i, *_: tf.less(i, stage_count),
            body,
            (0, cumulative, mask, loads_ta, masks_ta, last_ta),
        )

        stage_P = loads_ta.stack()
        stage_masks = masks_ta.stack()
        stage_last = last_ta.stack()

        # 构建 Rank 矩阵
        indices = tf.reshape(order, (-1, 1))
        ranks = tf.cast(tf.range(stage_count), tf.float32)
        rank_vec = tf.tensor_scatter_nd_update(
            tf.zeros((self._n_bolts,), tf.float32), indices, ranks
        )
        if stage_count > 1:
            rank_vec = rank_vec / tf.cast(stage_count - 1, tf.float32)
        else:
            rank_vec = tf.zeros_like(rank_vec)

        # 拼接最终特征 P_hat
        feats_ta = tf.TensorArray(tf.float32, size=stage_count)
        for i in range(stage_count):
            # 归一化 P
            norm = (stage_P[i] - self._shift) / self._scale
            # 拼接: [NormP, Mask, Last, Rank]
            feat = tf.concat([norm, stage_masks[i], stage_last[i], rank_vec], axis=0)
            feats_ta = feats_ta.write(i, feat)

        stage_feat = feats_ta.stack()
        return stage_P, stage_feat
