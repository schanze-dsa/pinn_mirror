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
from model.pinn_model import create_displacement_model, ModelConfig
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

    # 预紧
    preload_specs: List[Dict[str, str]] = field(default_factory=list)
    preload_n_points_each: int = 800

    # tie / 边界（如需）
    ties: List[Dict[str, Any]] = field(default_factory=list)
    bcs: List[Dict[str, Any]] = field(default_factory=list)

    # 预紧力范围（N）
    preload_min: float = 200.0
    preload_max: float = 1000.0
    preload_sequence: List[Any] = field(default_factory=list)
    preload_sequence_repeat: int = 1
    preload_sequence_shuffle: bool = False
    preload_sequence_jitter: float = 0.0

    # 预紧顺序（分步加载）
    preload_use_stages: bool = False
    preload_randomize_order: bool = True

    # 物理项/模型配置
    model_cfg: ModelConfig = field(default_factory=ModelConfig)
    elas_cfg: ElasticityConfig = field(
        default_factory=lambda: ElasticityConfig(coord_scale=1.0, chunk_size=1024, use_pfor=False)
    )
    contact_cfg: ContactOperatorConfig = field(default_factory=ContactOperatorConfig)
    preload_cfg: PreloadConfig = field(default_factory=PreloadConfig)
    total_cfg: TotalConfig = field(default_factory=lambda: TotalConfig(
        w_int=1.0, w_cn=1.0, w_ct=1.0, w_tie=1.0, w_bc=1.0, w_pre=1.0
    ))

    # 损失加权（自适应）
    loss_adaptive_enabled: bool = False
    loss_update_every: int = 1
    loss_ema_decay: float = 0.95
    loss_min_factor: float = 0.25
    loss_max_factor: float = 4.0
    loss_gamma: float = 2.0
    loss_focus_terms: Tuple[str, ...] = field(default_factory=tuple)

    # 训练超参
    max_steps: int = 1000
    lr: float = 1e-3
    grad_clip_norm: Optional[float] = 1.0
    log_every: int = 1
    alm_update_every: int = 10
    resample_contact_every: int = 10

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
    viz_samples_after_train: int = 5
    viz_title_prefix: str = "Mirror Deflection (trained PINN)"
    viz_style: str = "smooth"              # smooth Gouraud-shaded map by default
    viz_colormap: str = "turbo"             # Abaqus-like rainbow palette
    viz_levels: int = 24                    # used when style="contour"
    viz_symmetric: bool = True              # keep color limits symmetric around 0
    viz_units: str = "mm"
    viz_draw_wireframe: bool = False
    viz_write_data: bool = True             # export displacement samples next to figure
    viz_refine_subdivisions: int = 0        # >0 -> barycentric subdivisions per surface triangle
    viz_refine_max_points: int = 180_000    # guardrail against runaway refinement cost
    viz_eval_batch_size: int = 65_536       # batch PINN queries during visualization
    save_best_on: str = "Pi"   # or "E_int"


class Trainer:
    def __init__(self, cfg: TrainerConfig):
        self.cfg = cfg
        np.random.seed(cfg.seed)
        tf.random.set_seed(cfg.seed)

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
        self._train_vars: List[tf.Variable] = []

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

        # —— 体检/调试可读
        self.X_vol = None
        self.w_vol = None
        self.mat_id = None
        self.enum_names: List[str] = []
        self.id2props_map: Dict[int, Tuple[float, float]] = {}
        # 自适应损失权重的状态（在 run() 里初始化）
        self.loss_state: Optional[LossWeightState] = None


    # ----------------- 辅助工具 -----------------
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
            eint = (
                float(parts.get("E_int", tf.constant(0.0)).numpy())
                if "E_int" in parts
                else 0.0
            )
            en = 0.0
            if "E_cn" in parts:
                en = float(parts["E_cn"].numpy())
            elif "E_n" in parts:
                en = float(parts["E_n"].numpy())

            et = 0.0
            if "E_ct" in parts:
                et = float(parts["E_ct"].numpy())
            elif "E_t" in parts:
                et = float(parts["E_t"].numpy())


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
            if isinstance(stats, Mapping):
                val = stats.get("n_pen_ratio")
                if val is not None:
                    pen_ratio = float(val.numpy())
                val = stats.get("t_stick_ratio")
                if val is not None:
                    stick_ratio = float(val.numpy())
                val = stats.get("t_slip_ratio")
                if val is not None:
                    slip_ratio = float(val.numpy())
                val = stats.get("n_mean_gap")
                if val is not None:
                    mean_gap = float(val.numpy())

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
            postfix = (
                f"P=[{p1},{p2},{p3}]N{order_txt} Π={pin:.3e} Eint={eint:.3e} "
                f"En={en:.3e} Et={et:.3e} Wpre={wpre:.3e}{bolt_txt} "
                f"{rel_disp} {delta_disp} {grad_disp} {pen_disp} {stick_disp} {slip_disp} {gap_disp}"
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
        out = np.random.uniform(lo, hi, size=(3,)).astype(np.float32)
        self._last_preload_order = None
        return out

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
        return {
            "stages": np.stack(stage_loads).astype(np.float32),
            "stage_masks": np.stack(stage_masks).astype(np.float32),
            "stage_last": np.stack(stage_last).astype(np.float32),
            "stage_rank": rank.astype(np.float32),
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
        if order_np is None:
            order_np = np.arange(case["P"].shape[0], dtype=np.int32)
        order_tf = tf.convert_to_tensor(order_np, dtype=tf.int32)
        rank_tf = None
        if rank_np is not None:
            rank_tf = tf.convert_to_tensor(rank_np, dtype=tf.float32)
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
            if masks is not None:
                feat_parts.append(tf.convert_to_tensor(masks[idx], dtype=tf.float32))
            if lasts is not None:
                feat_parts.append(tf.convert_to_tensor(lasts[idx], dtype=tf.float32))
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
        params["stages"] = {
            "P": stage_tensor_P,
            "P_hat": stage_tensor_feat,
        }
        params["stage_order"] = order_tf
        if rank_tf is not None:
            params["stage_rank"] = rank_tf
        params["stage_count"] = tf.constant(stage_count, dtype=tf.int32)
        return params

    def _extract_final_stage_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if (
            self.cfg.preload_use_stages
            and isinstance(params, dict)
            and "stages" in params
        ):
            stages = params["stages"]
            if isinstance(stages, dict) and stages:
                last_P = stages["P"][-1]
                last_feat = stages["P_hat"][-1]
                return {"P": last_P, "P_hat": last_feat}
        return params

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

            # 7) 模型 & 优化器
            if cfg.mixed_precision:
                cfg.model_cfg.mixed_precision = cfg.mixed_precision
            self.model = create_displacement_model(cfg.model_cfg)
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
            mid = 0.5 * (float(cfg.preload_min) + float(cfg.preload_max))
            warmup_case = self._make_warmup_case()
            params = self._make_preload_params(warmup_case)
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

    def _train_step(self, total, preload_case: Dict[str, np.ndarray]):
        model = self.model
        opt = self.optimizer

        # 统一收集可训练变量
        train_vars = self._collect_trainable_variables()

        params = self._make_preload_params(preload_case)

        with tf.GradientTape() as tape:
            # 1) 先用 TotalEnergy 计算各个分量和“基线”Π_raw
            Pi_raw, parts, stats = total.energy(model.u_fn, params=params, tape=None)

            # 2) 根据当前残差更新自适应权重，并用 parts 重新组合总能量 Π
            if self.loss_state is not None:
                update_loss_weights(self.loss_state, parts, stats)
                Pi = combine_loss(parts, self.loss_state)   # 使用当前权重组合
            else:
                Pi = Pi_raw  # 没有自适应状态就退回原始 Π

            # 3) 加上 Keras 模型自带的正则项
            reg = tf.add_n(model.losses) if getattr(model, "losses", None) else 0.0

            loss = Pi + reg

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
            "E_bc": self.cfg.total_cfg.w_bc,
            "W_pre": self.cfg.total_cfg.w_pre,
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
        train_pb_kwargs = dict(total=self.cfg.max_steps, desc="Training", leave=True)
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
                                self.contact.reset_for_new_batch()
                                cat = cmap.concatenate()
                                self.contact.build_from_cat(
                                    cat, extra_weights=None, auto_orient=True
                                )
                                contact_note = f"更新 {len(cmap)} 点"
                            except Exception as exc:
                                contact_note = "更新失败"
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
                    Pi, parts, stats, grad_norm = self._train_step(total, preload_case)
                    P_np = preload_case["P"]
                    order_np = preload_case.get("order")
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
                    train_note = (
                        f"P=[{int(P_np[0])},{int(P_np[1])},{int(P_np[2])}]"
                        f"{order_txt} Π={pi_val:.2e} {rel_txt} {d_txt} "
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
        """Export the PINN displacement model as a TensorFlow SavedModel.

        The exported module exposes a callable with signature ``(X, P) -> u``
        where ``X`` is an ``(N, 3)`` tensor of spatial coordinates and ``P``
        is a ``(3,)`` tensor of bolt preload values (in Newtons). The return
        value ``u`` matches :meth:`model.u_fn` and has shape ``(N, 3)``.
        """

        if self.model is None:
            raise RuntimeError("Trainer.export_saved_model() requires build()/restore().")

        class _PINNModule(tf.Module):
            def __init__(self, model):
                super().__init__()
                self._model = model

            @tf.function(
                input_signature=[
                    tf.TensorSpec(shape=[None, 3], dtype=tf.float32, name="X"),
                    tf.TensorSpec(shape=[3], dtype=tf.float32, name="P"),
                ]
            )
            def __call__(self, X, P):
                params = {"P": tf.reshape(P, (3,))}
                return self._model.u_fn(X, params)

        module = _PINNModule(self.model)
        tf.saved_model.save(module, export_dir)
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
        params = self._extract_final_stage_params(params_full)
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

        _, _, data_path = plot_mirror_deflection_by_name(
            self.asm,
            self.cfg.mirror_surface_name,
            self.model.u_fn,
            params,
            P_values=tuple(float(x) for x in P),
            out_path=out_path,
            title_prefix=title,
            units=self.cfg.viz_units,
            levels=self.cfg.viz_levels,
            symmetric=self.cfg.viz_symmetric,
            show=show,
            data_out_path=resolved_data_path,
            style=self.cfg.viz_style,
            cmap=self.cfg.viz_colormap,
            draw_wireframe=self.cfg.viz_draw_wireframe,
            refine_subdivisions=self.cfg.viz_refine_subdivisions,
            refine_max_points=self.cfg.viz_refine_max_points,
            eval_batch_size=self.cfg.viz_eval_batch_size,
        )

        self.last_viz_data_path = data_path
        if data_path:
            print(f"[viz] displacement data -> {data_path}")
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

        return plot_mirror_deflection_by_name(
            self.asm,
            bare,
            self.model.u_fn,
            params,
            P_values=tuple(float(x) for x in P.reshape(-1)),
            out_path=out_path,
            title_prefix=title,
            units=self.cfg.viz_units,
            levels=self.cfg.viz_levels,
            symmetric=self.cfg.viz_symmetric,
            data_out_path=data_path,
            style=self.cfg.viz_style,
            cmap=self.cfg.viz_colormap,
            draw_wireframe=self.cfg.viz_draw_wireframe,
            refine_subdivisions=self.cfg.viz_refine_subdivisions,
            refine_max_points=self.cfg.viz_refine_max_points,
            eval_batch_size=self.cfg.viz_eval_batch_size,
        )

    def _visualize_after_training(self, n_samples: int = 5):
        if self.asm is None or self.model is None:
            return
        os.makedirs(self.cfg.out_dir, exist_ok=True)
        print(f"[trainer] Generating {n_samples} deflection maps for '{self.cfg.mirror_surface_name}' ...")
        for i in range(n_samples):
            preload_case = self._sample_preload_case()
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
            params_eval = self._extract_final_stage_params(params_full)
            try:
                _, _, data_path = self._call_viz(P, params_eval, save_path, title)
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
