# -*- coding: utf-8 -*-
"""
trainer.py — 主训练循环（精简日志 + 分阶段进度提示）。

该版本专注于保留关键构建/训练信息：
  - 初始化时报告是否启用 GPU。
  - 构建阶段仅输出必需的信息与接触汇总。
  - 单步训练进度条会标注当前阶段，便于观察训练流程。
"""

from __future__ import annotations
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm  # 仅用 tqdm.auto，适配 PyCharm/终端

# ---------- TF 显存与分配器 ----------
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

try:
    import colorama
    colorama.just_fix_windows_console()
except Exception:
    pass

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

        # —— 体检/调试可读
        self.X_vol = None
        self.w_vol = None
        self.mat_id = None
        self.enum_names: List[str] = []
        self.id2props_map: Dict[int, Tuple[float, float]] = {}

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

    # ----------------- 采样三螺栓预紧力 -----------------
    def _sample_P(self) -> np.ndarray:
        lo, hi = self.cfg.preload_min, self.cfg.preload_max
        return np.random.uniform(lo, hi, size=(3,)).astype(np.float32)

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

            # 3) 弹性项
            self.elasticity = ElasticityEnergy(
                X_vol=X_vol, w_vol=w_vol, mat_id=mat_id, matlib=self.id2props_map, cfg=cfg.elas_cfg
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

    @tf.function(jit_compile=False, reduce_retracing=True)
    def _train_step(self, total: TotalEnergy, P_tf: tf.Tensor):
        with tf.GradientTape() as tape:
            Pi, parts, stats = total.energy(self.model.u_fn, params={"P": P_tf})
            loss = Pi

        vars_ = self.model.encoder.trainable_variables + self.model.field.trainable_variables

        if isinstance(self.optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
            scaled_loss = self.optimizer.get_scaled_loss(loss)
            scaled_grads = tape.gradient(scaled_loss, vars_)
            grads = self.optimizer.get_unscaled_gradients(scaled_grads)
        else:
            grads = tape.gradient(loss, vars_)

        if self.cfg.grad_clip_norm:
            grads = [tf.clip_by_norm(g, self.cfg.grad_clip_norm) if g is not None else None for g in grads]
        grads_and_vars = [(g, v) for g, v in zip(grads, vars_) if g is not None]
        self.optimizer.apply_gradients(grads_and_vars)
        return Pi, parts, stats

    # ----------------- 训练 -----------------
    def run(self):
        self.build()
        print(f"[trainer] 当前训练设备：{self.device_summary}")
        total = self._assemble_total()

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
                    p_step.set_description_str(f"step {step}: 接触重采样")
                    t0 = time.perf_counter()
                    contact_note = "跳过"
                    if self.contact is not None and (
                        step == 1 or (
                            self.cfg.resample_contact_every > 0
                            and (step - 1) % self.cfg.resample_contact_every == 0
                        )
                    ):
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
                            self.contact.build_from_cat(cat, extra_weights=None, auto_orient=True)
                            contact_note = f"更新 {len(cmap)} 点"
                        except Exception as exc:
                            contact_note = "更新失败"
                            print(f"[contact] 第 {step} 步接触重采样失败：{exc}")
                    elapsed = time.perf_counter() - t0
                    self._step_stage_times.append(("resample", elapsed))
                    p_step.set_postfix_str(
                        f"{contact_note} | {self._format_seconds(elapsed)}"
                    )
                    p_step.update(1)

                    # 2) 前向 + 反传（随机采样三螺栓预紧力）
                    p_step.set_description_str(f"step {step}: 前向/反传")
                    t0 = time.perf_counter()
                    P_np = self._sample_P()
                    P_tf = tf.convert_to_tensor(P_np, dtype=tf.float32)
                    Pi, parts, stats, grad_norm = self._train_step(total, P_tf)
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
                    rel_txt = f"Πrel={rel_pi:.3f}"
                    d_txt = f"ΔΠ={(rel_delta * 100):.1f}%" if rel_delta is not None else "ΔΠ=--"
                    ema_txt = f"Πema={self._pi_ema:.2e}" if self._pi_ema is not None else "Πema=--"
                    train_note = (
                        f"P=[{int(P_np[0])},{int(P_np[1])},{int(P_np[2])}] "
                        f"Π={pi_val:.2e} {rel_txt} {d_txt} "
                        f"grad={grad_val:.2e} {ema_txt}"
                    )
                    p_step.set_postfix_str(
                        f"{train_note} | {self._format_seconds(elapsed)} | dev={device}"
                    )
                    p_step.update(1)

                    # 3) ALM 更新
                    p_step.set_description_str(f"step {step}: ALM 更新")
                    t0 = time.perf_counter()
                    alm_note = "跳过"
                    if (
                        self.contact is not None
                        and self.cfg.alm_update_every > 0
                        and step % self.cfg.alm_update_every == 0
                    ):
                        total.update_multipliers(self.model.u_fn, params={"P": P_tf})
                        alm_note = "已更新"
                    elapsed = time.perf_counter() - t0
                    self._step_stage_times.append(("alm", elapsed))
                    p_step.set_postfix_str(
                        f"{alm_note} | {self._format_seconds(elapsed)}"
                    )
                    p_step.update(1)

                    # 4) 日志 & ckpt
                    p_step.set_description_str(f"step {step}: 日志/检查点")
                    t0 = time.perf_counter()
                    log_note = "跳过"
                    if step % self.cfg.log_every == 0 or step == 1:
                        try:
                            p1, p2, p3 = [int(x) for x in P_np.tolist()]
                            pin = float(Pi.numpy())
                            eint = (
                                float(parts.get("E_int", tf.constant(0.0)).numpy())
                                if "E_int" in parts
                                else 0.0
                            )
                            en = (
                                float(parts.get("E_n", tf.constant(0.0)).numpy())
                                if "E_n" in parts
                                else 0.0
                            )
                            et = (
                                float(parts.get("E_t", tf.constant(0.0)).numpy())
                                if "E_t" in parts
                                else 0.0
                            )
                            wpre = (
                                float(parts.get("W_pre", tf.constant(0.0)).numpy())
                                if "W_pre" in parts
                                else 0.0
                            )

                            bolt_txt = ""
                            preload_stats = None
                            if isinstance(stats, dict):
                                preload_stats = stats.get("preload") or stats.get("preload_stats")
                            if isinstance(preload_stats, dict):
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
                            if isinstance(stats, dict):
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
                            rel_disp = f"Πrel={rel_pi:.3f}"
                            delta_disp = f"ΔΠ={(rel_delta * 100):.1f}%" if rel_delta is not None else "ΔΠ=--"
                            pen_disp = f"pen={pen_ratio * 100:.1f}%" if pen_ratio is not None else "pen=--"
                            stick_disp = f"stick={stick_ratio * 100:.1f}%" if stick_ratio is not None else "stick=--"
                            slip_disp = f"slip={slip_ratio * 100:.1f}%" if slip_ratio is not None else "slip=--"
                            gap_disp = f"⟨gap⟩={mean_gap:.2e}" if mean_gap is not None else "⟨gap⟩=--"

                            p_train.set_postfix_str(
                                f"P=[{p1},{p2},{p3}]N Π={pin:.3e} Eint={eint:.3e} "
                                f"En={en:.3e} Et={et:.3e} Wpre={wpre:.3e}{bolt_txt} "
                                f"{rel_disp} {delta_disp} {grad_disp} {pen_disp} {stick_disp} {slip_disp} {gap_disp}"
                            )
                            log_note = "已记录"
                        except Exception:
                            log_note = "记录异常"

                        metric_name = self.cfg.save_best_on.lower()
                        metric_val = pi_val if metric_name == "pi" else float(parts["E_int"].numpy())
                        if metric_val < self.best_metric:
                            self.best_metric = metric_val
                            self.ckpt_manager.save(checkpoint_number=step)
                            log_note += " | 已保存"
                    elapsed = time.perf_counter() - t0
                    self._step_stage_times.append(("log", elapsed))
                    p_step.set_postfix_str(
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
                        p_train.set_postfix_str(
                            f"step{step}耗时 {self._format_seconds(total_spent)} ({parts_txt})"
                        )
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

    def generate_deflection_map(self,
                                 preload: Any,
                                 out_path: Optional[str] = None,
                                 title_prefix: Optional[str] = None,
                                 show: bool = False) -> str:
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

        Returns:
            The absolute/relative path where the image was written.
        """

        if self.asm is None or self.model is None:
            raise RuntimeError("Trainer.generate_deflection_map() requires build()/restore().")

        P = np.asarray(list(preload), dtype=np.float32).reshape(-1)
        if P.size != 3:
            raise ValueError("'preload' must contain exactly three values (for the three bolts).")

        params = {"P": tf.convert_to_tensor(P, dtype=tf.float32)}
        title = title_prefix or self.cfg.viz_title_prefix

        if out_path is None:
            os.makedirs(self.cfg.out_dir, exist_ok=True)
            p_int = [int(round(float(x))) for x in P]
            out_path = os.path.join(
                self.cfg.out_dir,
                f"deflection_manual_P{p_int[0]}_{p_int[1]}_{p_int[2]}.png",
            )

        plot_mirror_deflection_by_name(
            self.asm,
            self.cfg.mirror_surface_name,
            self.model.u_fn,
            params,
            P_values=tuple(float(x) for x in P),
            out_path=out_path,
            title_prefix=title,
            show=show,
        )

        print(f"[viz] saved -> {out_path}")
        return out_path

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

    def generate_deflection_map(self,
                                 preload: Any,
                                 out_path: Optional[str] = None,
                                 title_prefix: Optional[str] = None,
                                 show: bool = False) -> str:
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

        Returns:
            The absolute/relative path where the image was written.
        """

        if self.asm is None or self.model is None:
            raise RuntimeError("Trainer.generate_deflection_map() requires build()/restore().")

        P = np.asarray(list(preload), dtype=np.float32).reshape(-1)
        if P.size != 3:
            raise ValueError("'preload' must contain exactly three values (for the three bolts).")

        params = {"P": tf.convert_to_tensor(P, dtype=tf.float32)}
        title = title_prefix or self.cfg.viz_title_prefix

        if out_path is None:
            os.makedirs(self.cfg.out_dir, exist_ok=True)
            p_int = [int(round(float(x))) for x in P]
            out_path = os.path.join(
                self.cfg.out_dir,
                f"deflection_manual_P{p_int[0]}_{p_int[1]}_{p_int[2]}.png",
            )

        plot_mirror_deflection_by_name(
            self.asm,
            self.cfg.mirror_surface_name,
            self.model.u_fn,
            params,
            P_values=tuple(float(x) for x in P),
            out_path=out_path,
            title_prefix=title,
            show=show,
        )

        print(f"[viz] saved -> {out_path}")
        return out_path

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

    def generate_deflection_map(self,
                                 preload: Any,
                                 out_path: Optional[str] = None,
                                 title_prefix: Optional[str] = None,
                                 show: bool = False) -> str:
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

        Returns:
            The absolute/relative path where the image was written.
        """

        if self.asm is None or self.model is None:
            raise RuntimeError("Trainer.generate_deflection_map() requires build()/restore().")

        P = np.asarray(list(preload), dtype=np.float32).reshape(-1)
        if P.size != 3:
            raise ValueError("'preload' must contain exactly three values (for the three bolts).")

        params = {"P": tf.convert_to_tensor(P, dtype=tf.float32)}
        title = title_prefix or self.cfg.viz_title_prefix

        if out_path is None:
            os.makedirs(self.cfg.out_dir, exist_ok=True)
            p_int = [int(round(float(x))) for x in P]
            out_path = os.path.join(
                self.cfg.out_dir,
                f"deflection_manual_P{p_int[0]}_{p_int[1]}_{p_int[2]}.png",
            )

        plot_mirror_deflection_by_name(
            self.asm,
            self.cfg.mirror_surface_name,
            self.model.u_fn,
            params,
            P_values=tuple(float(x) for x in P),
            out_path=out_path,
            title_prefix=title,
            show=show,
        )

        print(f"[viz] saved -> {out_path}")
        return out_path

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

    def generate_deflection_map(self,
                                 preload: Any,
                                 out_path: Optional[str] = None,
                                 title_prefix: Optional[str] = None,
                                 show: bool = False) -> str:
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

        Returns:
            The absolute/relative path where the image was written.
        """

        if self.asm is None or self.model is None:
            raise RuntimeError("Trainer.generate_deflection_map() requires build()/restore().")

        P = np.asarray(list(preload), dtype=np.float32).reshape(-1)
        if P.size != 3:
            raise ValueError("'preload' must contain exactly three values (for the three bolts).")

        params = {"P": tf.convert_to_tensor(P, dtype=tf.float32)}
        title = title_prefix or self.cfg.viz_title_prefix

        if out_path is None:
            os.makedirs(self.cfg.out_dir, exist_ok=True)
            p_int = [int(round(float(x))) for x in P]
            out_path = os.path.join(
                self.cfg.out_dir,
                f"deflection_manual_P{p_int[0]}_{p_int[1]}_{p_int[2]}.png",
            )

        plot_mirror_deflection_by_name(
            self.asm,
            self.cfg.mirror_surface_name,
            self.model.u_fn,
            params,
            P_values=tuple(float(x) for x in P),
            out_path=out_path,
            title_prefix=title,
            show=show,
        )

        print(f"[viz] saved -> {out_path}")
        return out_path

    # ----------------- 可视化（鲁棒多签名） -----------------
    def _call_viz(self, P: np.ndarray, out_path: str, title: str):
        bare = self.cfg.mirror_surface_name
        params = {"P": tf.convert_to_tensor(P.reshape(-1), dtype=tf.float32)}

        return plot_mirror_deflection_by_name(
            self.asm,
            bare,
            self.model.u_fn,
            params,
            P_values=tuple(float(x) for x in P.reshape(-1)),
            out_path=out_path,
            title_prefix=title,
        )

    def _visualize_after_training(self, n_samples: int = 5):
        if self.asm is None or self.model is None:
            return
        os.makedirs(self.cfg.out_dir, exist_ok=True)
        print(f"[trainer] Generating {n_samples} deflection maps for '{self.cfg.mirror_surface_name}' ...")
        for i in range(n_samples):
            P = self._sample_P()
            title = f"{self.cfg.viz_title_prefix}  P=[{int(P[0])},{int(P[1])},{int(P[2])}]N"
            save_path = os.path.join(self.cfg.out_dir, f"deflection_{i+1:02d}.png")
            try:
                self._call_viz(P, save_path, title)
                if not os.path.exists(save_path):
                    try:
                        import matplotlib.pyplot as plt
                        plt.savefig(save_path, dpi=200, bbox_inches="tight")
                        plt.close()
                    except Exception:
                        pass
                print(f"[viz] saved -> {save_path}")
            except TypeError as e:
                print("[viz] signature mismatch:", e)
            except Exception as e:
                print("[viz] error:", e)
