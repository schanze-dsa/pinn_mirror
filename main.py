#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py
-------
One-click runner for your DFEM/PINN project (PyCharm 直接运行即可).

本版包含：
- 启用 TF 显存分配器 cuda_malloc_async（需在 import TF 之前设置）
- 自动解析 INP & 表面 key（支持精确/模糊；含 bolt2 的 ASM::"bolt2 uo"）
- 与新版 surfaces.py / inp_parser.py 对齐（ELEMENT 表面可直接采样）
- 训练配置集中覆盖（降显存：节点前向分块、降低采样规模、混合精度）
- 训练配置由 config.yaml 驱动（未找到或缺失必填项会直接报错）
- 训练结束后在 outputs/ 生成随机 5 组镜面变形云图（文件名含三螺栓预紧力）
"""

# ====== 必须在导入 TensorFlow 之前设置 ======
import os
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")  # 可选：减少冗余日志
# ============================================

import sys
import argparse
import math
from datetime import datetime
from dataclasses import asdict
import yaml  # 新增：读取 config.yaml

# --- 确保 "src" 在 Python 路径中 ---
ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

CONFIG_PATH = os.path.join(ROOT, "config.yaml")

# ---------- SavedModel 默认输出路径 ----------
def _default_saved_model_dir(out_dir: str) -> str:
    """Return a timestamped SavedModel export directory under ``out_dir``."""

    base = os.path.abspath(out_dir or "outputs")
    os.makedirs(base, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(base, f"saved_model_{ts}")

# --- 项目内模块导入 ---
from train.trainer import TrainerConfig
from inp_io.inp_parser import load_inp
from mesh.contact_pairs import guess_surface_key


# ---------- 工具：读取 config.yaml（容错） ----------
def _load_yaml_config():
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"未找到 config.yaml（路径: {CONFIG_PATH}），请先准备配置文件后再运行。")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    print(f"[main] 成功读取 config.yaml。")
    return data


# ---------- 小工具：容错匹配表面 key ----------
def _auto_resolve_surface_keys(asm, key_or_hint: str) -> str:
    """
    支持“精确 key 或模糊片段”的自动匹配。
    - 若 key_or_hint 正好是 asm.surfaces 的键，直接返回；
    - 否则进行大小写不敏感的包含匹配；唯一匹配则返回该 key；否则抛出错误提示。
    """
    k = key_or_hint
    if k in asm.surfaces:
        return k
    g = guess_surface_key(asm, k)
    if g is not None:
        return g
    low = k.strip().lower()
    cands = [kk for kk, s in asm.surfaces.items()
             if low in kk.lower() or low in s.name.strip().lower()]
    if len(cands) == 1:
        return cands[0]
    elif len(cands) == 0:
        raise KeyError(f"找不到包含 '{k}' 的表面；请在 config.yaml 或 main.py 里把名字改得更准确一些。")
    else:
        msg = "匹配到多个表面：\n  " + "\n  ".join(cands) + "\n请改成更精确的名字。"
        raise KeyError(msg)


# ---------- 读取 INP + 组装 TrainerConfig（并返回 asm 以供审计打印） ----------
def _prepare_config_with_autoguess():
    # 0) 读取 config.yaml（若存在）
    cfg_yaml = _load_yaml_config()

    # 1) INP 路径
    inp_path = cfg_yaml.get("inp_path", "").strip()
    if not inp_path:
        raise ValueError("config.yaml 必须提供 inp_path。")
    if not os.path.exists(inp_path):
        raise FileNotFoundError(f"未找到 INP 文件：{inp_path}。请在 config.yaml 的 inp_path 中填写正确路径。")
    asm = load_inp(inp_path)

    # 2) 镜面表面名
    mirror_surface_name = cfg_yaml.get("mirror_surface_name", "").strip()
    if not mirror_surface_name:
        raise ValueError("config.yaml 必须提供 mirror_surface_name。")
    try:
        _ = _auto_resolve_surface_keys(asm, mirror_surface_name)
    except Exception as e:
        print("[main] 提示：镜面表面名自动匹配失败：", e)
        print("       继续使用你提供的名字（可视化时按该名字模糊匹配）。")

    # 3) 螺栓 up/down
    bolt_surfaces = []
    for b in cfg_yaml.get("bolts", []) or []:
        bolt_surfaces.append(
            {
                "name": b.get("name", ""),
                "up_key": b.get("up_surface_key", ""),
                "down_key": b.get("down_surface_key", ""),
            }
        )

    preload_specs = []
    for spec in bolt_surfaces:
        try:
            up_key = _auto_resolve_surface_keys(asm, spec["up_key"])
            dn_key = _auto_resolve_surface_keys(asm, spec["down_key"])
            preload_specs.append({"name": spec["name"], "up_key": up_key, "down_key": dn_key})
        except Exception as e:
            print(f"[main] 螺栓 '{spec['name']}' 的 up/down 自动匹配失败：{e}")
            print("       请在 config.yaml 的 bolts 中修正后再跑。")
            preload_specs.append(
                {"name": spec["name"], "up_key": spec["up_key"], "down_key": spec["down_key"]}
            )

    # 4) 接触对
    contact_pairs_cfg = cfg_yaml.get("contact_pairs", []) or []

    contact_pairs = []
    for p in contact_pairs_cfg:
        try:
            slave_key = _auto_resolve_surface_keys(asm, p["slave_key"])
            master_key = _auto_resolve_surface_keys(asm, p["master_key"])
            contact_pairs.append(
                {
                    "slave_key": slave_key,
                    "master_key": master_key,
                    "name": p.get("name", ""),
                    "interaction": p.get("interaction", ""),
                }
            )
        except Exception as e:
            print(f"[main] 接触对 '{p.get('name','')}' 自动匹配失败：{e}")
            print("       暂时跳过该接触对（可在 config.yaml 的 contact_pairs 中修正后再跑）。")

    # 5) 材料与 Part→材料映射
    mat_props = cfg_yaml.get("material_properties", {}) or {}
    if not isinstance(mat_props, dict) or not mat_props:
        raise ValueError("config.yaml 必须提供非空的 material_properties。")
    materials = {}
    for name, props in mat_props.items():
        E = props.get("E", None)
        nu = props.get("nu", None)
        if E is None or nu is None:
            continue
        materials[name] = (float(E), float(nu))
    if not materials:
        raise ValueError("material_properties 解析后为空，请检查配置内容。")

    part2mat = cfg_yaml.get("part2mat", {}) or {}
    if not part2mat:
        raise ValueError("config.yaml 必须提供非空的 part2mat。")

    # 6) 训练步数与采样设置：优先使用 config.yaml 中的 optimizer_config / elasticity_config
    optimizer_cfg = cfg_yaml.get("optimizer_config", {}) or {}
    elas_cfg_yaml = cfg_yaml.get("elasticity_config", {}) or {}

    train_steps = int(optimizer_cfg.get("epochs", TrainerConfig.max_steps))
    n_contact_points_per_pair = int(cfg_yaml.get("n_contact_points_per_pair", TrainerConfig.n_contact_points_per_pair))
    preload_face_points_each = int(cfg_yaml.get("preload_n_points_each", TrainerConfig.preload_n_points_each))
    preload_range = cfg_yaml.get("preload_range_n", (TrainerConfig.preload_min, TrainerConfig.preload_max))
    preload_min, preload_max = float(preload_range[0]), float(preload_range[1])

    # 7) 组装训练配置
    cfg = TrainerConfig(
        inp_path=inp_path,
        mirror_surface_name=mirror_surface_name,  # 可视化仍支持模糊匹配
        materials=materials,
        part2mat=part2mat,
        contact_pairs=contact_pairs,
        n_contact_points_per_pair=n_contact_points_per_pair,
        preload_specs=preload_specs,
        preload_n_points_each=preload_face_points_each,
        preload_min=preload_min,
        preload_max=preload_max,
        max_steps=train_steps,
        viz_samples_after_train=5,   # 随机 5 组，标题包含三螺栓预紧力
    )
    output_cfg = cfg_yaml.get("output_config", {}) or {}
    cfg.viz_use_shape_function_interp = bool(
        output_cfg.get("viz_use_shape_function_interp", cfg.viz_use_shape_function_interp)
    )
    if "viz_surface_source" in output_cfg:
        cfg.viz_surface_source = str(output_cfg["viz_surface_source"])
    if "viz_refine_subdivisions" in output_cfg:
        cfg.viz_refine_subdivisions = int(output_cfg["viz_refine_subdivisions"])
    cfg.adam_steps = cfg.max_steps

    cfg.lr = float(optimizer_cfg.get("learning_rate", cfg.lr))
    if "grad_clip_norm" in optimizer_cfg:
        cfg.grad_clip_norm = float(optimizer_cfg["grad_clip_norm"])
    if "log_every" in optimizer_cfg:
        cfg.log_every = int(optimizer_cfg["log_every"])

    lbfgs_cfg = optimizer_cfg.get("lbfgs", {}) or {}
    cfg.lbfgs_enabled = bool(optimizer_cfg.get("lbfgs_enabled", cfg.lbfgs_enabled))
    if lbfgs_cfg:
        cfg.lbfgs_enabled = bool(lbfgs_cfg.get("enabled", cfg.lbfgs_enabled))
        cfg.lbfgs_max_iter = int(lbfgs_cfg.get("max_iter", cfg.lbfgs_max_iter))
        cfg.lbfgs_tolerance = float(lbfgs_cfg.get("tolerance", cfg.lbfgs_tolerance))
        cfg.lbfgs_history_size = int(lbfgs_cfg.get("history_size", cfg.lbfgs_history_size))
        cfg.lbfgs_line_search = int(lbfgs_cfg.get("line_search", cfg.lbfgs_line_search))
        cfg.lbfgs_reuse_last_batch = bool(
            lbfgs_cfg.get("reuse_last_batch", cfg.lbfgs_reuse_last_batch)
        )

    # ===== 预紧分阶段 / 顺序设置 =====
    staging_cfg = cfg_yaml.get("preload_staging", {}) or {}

    # 顶层布尔开关优先，其次是 staging_cfg 内的 enabled
    use_stages_val = cfg_yaml.get("preload_use_stages", None)
    if use_stages_val is not None:
        cfg.preload_use_stages = bool(use_stages_val)
    if "enabled" in staging_cfg:
        cfg.preload_use_stages = bool(staging_cfg["enabled"])

    random_order_val = cfg_yaml.get("preload_randomize_order", None)
    if random_order_val is not None:
        cfg.preload_randomize_order = bool(random_order_val)
    if "randomize_order" in staging_cfg:
        cfg.preload_randomize_order = bool(staging_cfg["randomize_order"])

    if "repeat" in staging_cfg:
        cfg.preload_sequence_repeat = int(staging_cfg["repeat"])
    if "shuffle" in staging_cfg:
        cfg.preload_sequence_shuffle = bool(staging_cfg["shuffle"])
    if "jitter" in staging_cfg:
        cfg.preload_sequence_jitter = float(staging_cfg["jitter"])

    relax_top = cfg_yaml.get("preload_rank_relaxation", None)
    if relax_top is not None:
        cfg.preload_cfg.rank_relaxation = float(relax_top)
    if "relaxation" in staging_cfg:
        cfg.preload_cfg.rank_relaxation = float(staging_cfg["relaxation"])

    seq_overrides = cfg_yaml.get("preload_sequence", None)
    if seq_overrides:
        cfg.preload_sequence = list(seq_overrides)
    seq_from_staging = staging_cfg.get("sequence", None)
    if seq_from_staging:
        cfg.preload_sequence = list(seq_from_staging)

    if cfg.preload_sequence:
        cfg.preload_use_stages = True

    # ===== 损失加权配置（含自适应） =====
    loss_cfg_yaml = cfg_yaml.get("loss_config", {}) or {}
    base_weights_yaml = loss_cfg_yaml.get("base_weights", {}) or {}
    weight_key_map = {
        "w_int": ("w_int", "E_int"),
        "w_cn": ("w_cn", "E_cn"),
        "w_ct": ("w_ct", "E_ct"),
        "w_tie": ("w_tie", "E_tie"),
        "w_bc": ("w_bc", "E_bc"),
        "w_pre": ("w_pre", "W_pre"),
        "w_sigma": ("w_sigma", "E_sigma"),
    }
    for yaml_key, (attr, _) in weight_key_map.items():
        if yaml_key in base_weights_yaml:
            setattr(cfg.total_cfg, attr, float(base_weights_yaml[yaml_key]))

    adaptive_cfg = loss_cfg_yaml.get("adaptive", {}) or {}
    cfg.loss_adaptive_enabled = bool(
        adaptive_cfg.get("enabled", cfg.loss_adaptive_enabled)
    )
    cfg.loss_update_every = int(adaptive_cfg.get("update_every", cfg.loss_update_every))
    cfg.loss_ema_decay = float(adaptive_cfg.get("ema_decay", cfg.loss_ema_decay))
    if "min_weight" in adaptive_cfg:
        cfg.loss_min_factor = float(adaptive_cfg["min_weight"])
    if "max_weight" in adaptive_cfg:
        cfg.loss_max_factor = float(adaptive_cfg["max_weight"])
    temperature = float(adaptive_cfg.get("temperature", 0.0) or 0.0)
    if temperature > 0.0:
        cfg.loss_gamma = 1.0 / temperature
    else:
        cfg.loss_gamma = float(adaptive_cfg.get("gamma", cfg.loss_gamma))

    focus_terms_yaml = adaptive_cfg.get("focus_terms", []) or []
    focus_terms = []
    for item in focus_terms_yaml:
        key = str(item).strip()
        mapping = weight_key_map.get(key)
        if mapping is None:
            continue
        focus_terms.append(mapping[1])
    cfg.loss_focus_terms = tuple(focus_terms)
    cfg.total_cfg.adaptive_scheme = adaptive_cfg.get("scheme", cfg.total_cfg.adaptive_scheme)

    # 若启用分阶段加载但 focus_terms 未包含 W_pre，则自动加入以增强预紧信号
    if cfg.preload_use_stages and "W_pre" not in cfg.loss_focus_terms:
        cfg.loss_focus_terms = tuple(list(cfg.loss_focus_terms) + ["W_pre"])

    # 启用应力头时默认也纳入自适应关注项，避免固定权重过大导致梯度爆炸
    has_stress_head = getattr(cfg.model_cfg.field, "stress_out_dim", 0) > 0
    if has_stress_head and "E_sigma" not in cfg.loss_focus_terms:
        cfg.loss_focus_terms = tuple(list(cfg.loss_focus_terms) + ["E_sigma"])

    # 只要存在任意关注项，就切换为 focus 策略
    if cfg.loss_focus_terms:
        cfg.total_cfg.adaptive_scheme = "focus"

    cfg.resample_contact_every = int(
        cfg_yaml.get("resample_contact_every", cfg.resample_contact_every)
    )
    cfg.alm_update_every = int(cfg_yaml.get("alm_update_every", cfg.alm_update_every))


    # ===== 显存友好覆盖（建议先这样跑通，再逐步调回） =====
    # 1) 提升模型表达能力（更宽更深的位移网络 + 更大的条件编码器）
    cfg.model_cfg.encoder.width = 96
    cfg.model_cfg.encoder.depth = 3
    cfg.model_cfg.encoder.out_dim = 96
    cfg.model_cfg.field.width = 320
    cfg.model_cfg.field.depth = 9
    cfg.model_cfg.field.residual_skips = (3, 6, 8)

    # 2) DFEM 采样配置（不再设置 Jacobian 相关字段）
    #    - chunk_size: 节点前向/能量评估的分块大小（防止一次性吃满显存）
    #    - n_points_per_step: 每一步参与 DFEM 积分的子单元/积分点个数上限
    cfg.elas_cfg.chunk_size = int(elas_cfg_yaml.get("chunk_size", 0))
    cfg.elas_cfg.n_points_per_step = int(elas_cfg_yaml.get("n_points_per_step", 4096))
    cfg.elas_cfg.coord_scale = float(elas_cfg_yaml.get("coord_scale", 1.0))

    # 3) 接触/预紧采样：根据阶段数做显存友好的调整
    stage_multiplier = 1
    if cfg.preload_use_stages:
        stage_multiplier = max(1, len(cfg.preload_specs))
        if cfg.preload_sequence:
            for entry in cfg.preload_sequence:
                if isinstance(entry, dict):
                    order = entry.get("order") or entry.get("orders")
                    values = entry.get("values") or entry.get("P")
                    if order is not None:
                        stage_multiplier = max(stage_multiplier, len(order))
                    elif values is not None:
                        stage_multiplier = max(stage_multiplier, len(values))
                elif isinstance(entry, (list, tuple)):
                    stage_multiplier = max(stage_multiplier, len(entry))

    contact_target = cfg.n_contact_points_per_pair
    if stage_multiplier > 1:
        per_stage_contact = max(256, math.ceil(contact_target / stage_multiplier))
        approx_total_contact = per_stage_contact * stage_multiplier
        if per_stage_contact != contact_target:
            print(
                "[main] 分阶段预紧启用：将每对接触采样从 "
                f"{contact_target} 调整为每阶段 {per_stage_contact} (≈{approx_total_contact} 总点数)。"
            )
        # 分阶段计算仍会在同一梯度带内重复评估接触能，因此进一步限制总量
        contact_cap = 2048
        if per_stage_contact > contact_cap:
            per_stage_contact = contact_cap
            approx_total_contact = per_stage_contact * stage_multiplier
            print(
                "[main] 接触点上限触发：将每阶段采样压缩到 "
                f"{per_stage_contact} (≈{approx_total_contact} 总点数)。"
            )
        cfg.n_contact_points_per_pair = per_stage_contact

        preload_target = cfg.preload_n_points_each
        per_stage_preload = max(128, math.ceil(preload_target / stage_multiplier))
        approx_total_preload = per_stage_preload * stage_multiplier
        if per_stage_preload != preload_target:
            print(
                "[main] 分阶段预紧启用：将每个螺栓端面的采样从 "
                f"{preload_target} 调整为每阶段 {per_stage_preload} (≈{approx_total_preload} 总点数)。"
            )
        preload_cap = 1024
        if per_stage_preload > preload_cap:
            per_stage_preload = preload_cap
            approx_total_preload = per_stage_preload * stage_multiplier
            print(
                "[main] 预紧点上限触发：将每阶段端面采样压缩到 "
                f"{per_stage_preload} (≈{approx_total_preload} 总点数)。"
            )
        cfg.preload_n_points_each = per_stage_preload

        elas_target = cfg.elas_cfg.n_points_per_step
        per_stage_elas = max(1024, math.ceil(elas_target / stage_multiplier))
        if per_stage_elas != elas_target:
            print(
                "[main] 分阶段预紧启用：将 DFEM 每步积分点从 "
                f"{elas_target} 调整为每阶段 {per_stage_elas}。"
            )
            cfg.elas_cfg.n_points_per_step = per_stage_elas


    # 4) 混合精度（4080S 支持）
    cfg.mixed_precision = "mixed_float16"

    # 5) 根据预紧力范围自动调整归一化（映射到约 [-1, 1]）
    preload_lo, preload_hi = float(cfg.preload_min), float(cfg.preload_max)
    if preload_hi <= preload_lo:
        raise ValueError("预紧力范围 preload_range_n 的上限必须大于下限。")
    preload_mid = 0.5 * (preload_lo + preload_hi)
    preload_half_span = 0.5 * (preload_hi - preload_lo)
    cfg.model_cfg.preload_shift = preload_mid
    cfg.model_cfg.preload_scale = max(preload_half_span, 1e-3)
    # =================================================
    return cfg, asm


def _run_training(cfg, asm, export_saved_model: str = ""):
    from train.trainer import Trainer  # 再导一次确保路径就绪
    trainer = Trainer(cfg)
    trainer.run()

    export_dir = (export_saved_model or "").strip()
    if export_dir:
        export_dir = os.path.abspath(export_dir)
        os.makedirs(os.path.dirname(export_dir), exist_ok=True)
    else:
        export_dir = _default_saved_model_dir(cfg.out_dir)
        print(f"[main] 未提供 --export，将 SavedModel 写入: {export_dir}")
    trainer.export_saved_model(export_dir)

    print("\n✅ 训练完成！请到 'outputs/' 查看 5 张 “MIRROR up” 变形云图（文件名包含三颗预紧力数值）。")
    print("   如需修改 INP 路径、表面名或超参，请编辑 config.yaml。")


def _run_inference(cfg,
                   preload_values,
                   ckpt_path: str = "",
                   out_path: str = "",
                   data_out: str = "auto",
                   title_prefix: str = "",
                   show: bool = False,
                   preload_order=None,
                   export_saved_model: str = ""):
    from train.trainer import Trainer

    trainer = Trainer(cfg)
    trainer.build()
    restored = trainer.restore_checkpoint(ckpt_path or None)

    if export_saved_model:
        trainer.export_saved_model(export_saved_model)

    save_path = trainer.generate_deflection_map(
        preload_values,
        out_path=out_path or None,
        title_prefix=title_prefix or None,
        show=show,
        data_out_path=data_out,
        preload_order=preload_order,
    )

    print("\n✅ 推理完成！")
    print(f"   使用的检查点: {restored}")
    print(f"   生成的云图: {save_path}")
    if trainer.last_viz_data_path:
        print(f"   位移数据: {trainer.last_viz_data_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train the DFEM/PINN model or run inference with custom preloads."
    )
    parser.add_argument(
        "--mode", choices=["train", "infer"], default="train",
        help="train: 训练模型; infer: 使用已训模型生成云图"
    )
    parser.add_argument(
        "--preload", nargs=3, type=float, metavar=("P1", "P2", "P3"),
        help="三个螺栓的预紧力，单位 N (仅在 --mode infer 时使用)"
    )
    parser.add_argument(
        "--order", nargs=3, type=int, metavar=("B1", "B2", "B3"),
        help=(
            "分阶段推理时三颗螺栓的拧紧顺序；例如 '2 3 1' 表示第二颗先拧，"
            "再拧第三、第一颗。留空则默认按 1-2-3 的顺序加载。"
        ),
    )
    parser.add_argument(
        "--ckpt", default="",
        help="指定要恢复的检查点路径；默认使用 checkpoints/ 下最新的"
    )
    parser.add_argument(
        "--out", default="",
        help="保存推理云图的路径；默认写入 outputs/ 目录"
    )
    parser.add_argument(
        "--data", default="auto",
        help="云图对应的位移采样 txt 文件路径。使用 'auto' 表示与图片同名，"
             "使用 'none' 或空字符串表示不导出。"
    )
    parser.add_argument(
        "--title", default="",
        help="自定义云图标题前缀（默认沿用配置中的 viz_title_prefix）"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="推理时显示 matplotlib 窗口"
    )
    parser.add_argument(
        "--export", default="",
        help="将模型导出为 TensorFlow SavedModel 的目录"
    )

    args = parser.parse_args(argv)

    cfg, asm = _prepare_config_with_autoguess()

    if args.mode == "train":
        _run_training(cfg, asm, export_saved_model=args.export)
    else:
        if args.preload is None:
            parser.error("--mode infer 需要提供 --preload P1 P2 P3")
        _run_inference(
            cfg,
            preload_values=args.preload,
            ckpt_path=args.ckpt,
            out_path=args.out,
            data_out=args.data,
            title_prefix=args.title,
            show=args.show,
            preload_order=args.order,
            export_saved_model=args.export,
        )


if __name__ == "__main__":
    main()
