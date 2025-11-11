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
- 训练配置集中覆盖（降显存：CPU Jacobian 分块、降低采样规模、混合精度）
- 训练前“预训练审计打印”（镜面/螺栓/接触/绑定/超参等一并核对）
- 训练结束后在 outputs/ 生成随机 5 组镜面变形云图（文件名含三螺栓预紧力）
"""

# ====== 必须在导入 TensorFlow 之前设置 ======
import os
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")  # 可选：减少冗余日志
# ============================================

import sys
import argparse
from dataclasses import asdict

# --- 确保 "src" 在 Python 路径中 ---
ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# ================== USER SETTINGS ==================
# INP 文件路径（按你的实际路径）
INP_PATH = r"D:\\shuangfan\\shuangfan.inp"

# 镜面“上表面”的精确 key（避免歧义）
MIRROR_SURFACE_NAME = 'ASM::"MIRROR up"'

# 三个螺栓的上/下端面（使用你列出的精确 key；bolt2 的“上”是 ASM::"bolt2 uo"）
BOLT_SURFACES = [
    {"name": "bolt1", "up_key": 'ASM::"bolt1 up"',  "down_key": 'ASM::"bolt1 down"'},
    {"name": "bolt2", "up_key": 'ASM::"bolt2 uo"',  "down_key": 'ASM::"bolt2 down"'},
    {"name": "bolt3", "up_key": 'ASM::"bolt3 up"',  "down_key": 'ASM::"bolt3 down"'},
]

# 接触对（若暂不启用可留空；确认后再填写精确 key）
CONTACT_PAIRS = [
    # 示例（需要时再启用）：
    # {"slave_key": 'ASM::"bolt1 s"', "master_key": 'ASM::"MIRROR up"', "name": "b1_mirror"},
]

# 材料库（按实际材料调整 E, ν），单位 MPa
MATERIALS = {
    "mirror": (70000.0, 0.33),   # 例如铝合金镜坯
    "steel":  (210000.0, 0.30),  # 螺栓钢
}

# Part → 材料（使用你 INP 的 Part 名）
PART2MAT = {
    "mirror1": "mirror",
    "mirror2": "mirror",
    "bolt1": "steel",
    "bolt2": "steel",
    "bolt3": "steel",
    "auto":   "steel",  # 如非钢材，请对应修改
}

# 训练步数与采样设置（基础值；稍后会按“降显存覆盖”调整）
TRAIN_STEPS = 4000
CONTACT_POINTS_PER_PAIR = 6000
PRELOAD_FACE_POINTS_EACH = 800
# 三个螺栓随机预紧力范围（单位 N）
PRELOAD_RANGE_N = (200.0, 1000.0)
# ===================================================

# --- 项目内模块导入 ---
from train.trainer import TrainerConfig
from inp_io.inp_parser import load_inp
from mesh.contact_pairs import guess_surface_key


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
        raise KeyError(f"找不到包含 '{k}' 的表面；请在 main.py 里把名字改得更准确一些。")
    else:
        msg = "匹配到多个表面：\n  " + "\n  ".join(cands) + "\n请改成更精确的名字。"
        raise KeyError(msg)


# ---------- 审计打印：统计某个 ELEMENT 表面的元素面数量 ----------
def _count_surface_faces(asm, surface_key: str):
    """
    返回 (总面数, {S1:面数,...}, 示例若干元素ID)
    统计方法：对 SurfaceDef.items 中每个 (ELSET_NAME, S#) 调用 asm.expand_elset(elset)，
    将对应 elset 的元素个数记为该 S# 的面数（每个元素贡献一个对应面的四边形）。
    """
    if surface_key not in asm.surfaces:
        return 0, {}, []

    sdef = asm.surfaces[surface_key]
    per_face = {}
    samples = []
    total = 0
    for (elset_name, face) in sdef.items:
        face = str(face).upper().strip()
        try:
            eids = asm.expand_elset(elset_name)
        except Exception:
            continue
        cnt = len(eids)
        per_face[face] = per_face.get(face, 0) + cnt
        total += cnt
        # 收集少量样例
        for eid in eids[:5]:
            if len(samples) < 20:
                samples.append(int(eid))
    return total, per_face, samples


# ---------- 读取 INP + 组装 TrainerConfig（并返回 asm 以供审计打印） ----------
def _prepare_config_with_autoguess():
    # 1) 载入 INP
    if not os.path.exists(INP_PATH):
        raise FileNotFoundError(f"未找到 INP 文件：{INP_PATH}\n请在 main.py 顶部 INP_PATH 里填对路径。")
    asm = load_inp(INP_PATH)

    # 2) 镜面表面（已提供精确 key；若不是精确 key，这里会尝试模糊）
    try:
        _ = _auto_resolve_surface_keys(asm, MIRROR_SURFACE_NAME)
    except Exception as e:
        print("[main] 提示：镜面表面名自动匹配失败：", e)
        print("       继续使用你提供的名字（可视化时按该名字模糊匹配）。")

    # 3) 螺栓 up/down（使用你提供的精确 key；如果不是精确 key，这里也会尝试模糊）
    preload_specs = []
    for spec in BOLT_SURFACES:
        try:
            up_key = _auto_resolve_surface_keys(asm, spec["up_key"])
            dn_key = _auto_resolve_surface_keys(asm, spec["down_key"])
            preload_specs.append({"name": spec["name"], "up_key": up_key, "down_key": dn_key})
        except Exception as e:
            print(f"[main] 螺栓 '{spec['name']}' 的 up/down 自动匹配失败：{e}")
            print("       请在 main.py 的 BOLT_SURFACES 中把 up_key/down_key 改成更准确的名称。")
            preload_specs.append({"name": spec["name"], "up_key": spec["up_key"], "down_key": spec["down_key"]})

    # 4) 接触对（若填写 hint，这里解析；否则Trainer里会自动从INP猜测）
    contact_pairs = []
    for p in CONTACT_PAIRS:
        try:
            slave_key = _auto_resolve_surface_keys(asm, p["slave_key"])
            master_key = _auto_resolve_surface_keys(asm, p["master_key"])
            contact_pairs.append({"slave_key": slave_key, "master_key": master_key, "name": p.get("name", "")})
        except Exception as e:
            print(f"[main] 接触对 '{p.get('name','')}' 自动匹配失败：{e}")
            print("       暂时跳过该接触对（可在 main.py CONTACT_PAIRS 中修正后再跑）。")

    # 5) 组装训练配置
    cfg = TrainerConfig(
        inp_path=INP_PATH,
        mirror_surface_name=MIRROR_SURFACE_NAME,  # 可视化仍支持模糊匹配
        materials=MATERIALS,
        part2mat=PART2MAT,

        contact_pairs=contact_pairs,
        n_contact_points_per_pair=CONTACT_POINTS_PER_PAIR,

        preload_specs=preload_specs,
        preload_n_points_each=PRELOAD_FACE_POINTS_EACH,

        preload_min=PRELOAD_RANGE_N[0],
        preload_max=PRELOAD_RANGE_N[1],

        max_steps=TRAIN_STEPS,
        viz_samples_after_train=5,   # 随机 5 组，标题包含三螺栓预紧力
    )

    # ===== 显存友好覆盖（建议先这样跑通，再逐步调回） =====
    # 1) 提升模型表达能力（更宽更深的位移网络 + 更大的条件编码器）
    cfg.model_cfg.encoder.width = 96
    cfg.model_cfg.encoder.depth = 3
    cfg.model_cfg.encoder.out_dim = 96
    cfg.model_cfg.field.width = 320
    cfg.model_cfg.field.depth = 9
    cfg.model_cfg.field.residual_skips = (3, 6, 8)

    # 2) 把 Jacobian 前向+求导放在 CPU，并分块处理；关闭 pfor 降图复杂度
    cfg.elas_cfg.jac_chunk = 128             # 64/128/256 视显存调整
    cfg.elas_cfg.jac_device = "CPU"          # 关键：U+J 在 CPU，避免 GPU OOM
    cfg.elas_cfg.use_pfor = False            # 关闭 pfor

    # 3) 增大接触采样密度，并将重采样频率下调为每步刷新
    cfg.n_contact_points_per_pair = max(cfg.n_contact_points_per_pair, 6000)
    cfg.resample_contact_every = 1
    #    预紧端面采样使用高密度样本以放大不同预紧力的影响
    cfg.preload_n_points_each = max(cfg.preload_n_points_each, 800)

    # 4) 混合精度（4080S 支持）
    cfg.mixed_precision = "mixed_float16"

    # 5) 根据预紧力范围自动调整归一化（映射到约 [-1, 1]）
    preload_lo, preload_hi = float(cfg.preload_min), float(cfg.preload_max)
    if preload_hi <= preload_lo:
        raise ValueError("PRELOAD_RANGE_N 的上限必须大于下限。")
    preload_mid = 0.5 * (preload_lo + preload_hi)
    preload_half_span = 0.5 * (preload_hi - preload_lo)
    cfg.model_cfg.preload_shift = preload_mid
    cfg.model_cfg.preload_scale = max(preload_half_span, 1e-3)
    # =================================================

    return cfg, asm


# ---------- 预训练审计打印（你需要的“那种详单”） ----------
def _print_pretrain_audit(cfg: TrainerConfig, asm) -> None:
    print("\n======================================================================")
    print("[预训练审计] INP 解析摘要")
    print("======================================================================")
    try:
        s = asm.summary()
        print(f"[INP] parts={s.get('num_parts')}  surfaces={s.get('num_surfaces')}  "
              f"contact_pairs={s.get('num_contact_pairs')}  ties={s.get('num_ties')}")
    except Exception:
        pass

    # 镜面
    print("\n[镜面] Surface =", MIRROR_SURFACE_NAME)
    try:
        mirror_key = _auto_resolve_surface_keys(asm, MIRROR_SURFACE_NAME)
        total, per_face, samples = _count_surface_faces(asm, mirror_key)
        print(f"  - 解析到 key: {mirror_key}")
        if per_face:
            pfmt = ", ".join([f"{k}:{v}" for k, v in sorted(per_face.items())])
            print(f"  - 元素面统计: total={total} ({pfmt})")
        if samples:
            print(f"  - 示例元素ID(<=20): {samples}")
    except Exception as e:
        print("  - [WARN] 镜面表面统计失败：", e)

    # 螺栓 up/down
    print("\n[螺栓端面]")
    for spec in cfg.preload_specs:
        try:
            up_key, dn_key = spec["up_key"], spec["down_key"]
            t_u, pf_u, _ = _count_surface_faces(asm, up_key)
            t_d, pf_d, _ = _count_surface_faces(asm, dn_key)
            pfm_u = ", ".join([f"{k}:{v}" for k, v in sorted(pf_u.items())]) if pf_u else "N/A"
            pfm_d = ", ".join([f"{k}:{v}" for k, v in sorted(pf_d.items())]) if pf_d else "N/A"
            print(f"  - {spec['name']}:")
            print(f"      up   = {up_key}  faces={t_u} ({pfm_u})")
            print(f"      down = {dn_key}  faces={t_d} ({pfm_d})")
        except Exception as e:
            print(f"  - [WARN] {spec['name']} 统计失败：", e)

    # 接触/绑定
    print("\n[INP 接触/绑定]")
    try:
        # 交互库（若 parser 有 interactions 字段则读取 μ）
        interactions = getattr(asm, "interactions", None)
        mu_map = {}
        if isinstance(interactions, dict):
            for name, obj in interactions.items():
                mu = getattr(obj, "mu", None) or obj.get("mu") if isinstance(obj, dict) else None
                mu_map[name] = mu

        if asm.contact_pairs:
            print(f"[INP] Contact Pair 共 {len(asm.contact_pairs)} 组：")
            for idx, cp in enumerate(asm.contact_pairs, 1):
                inter = cp.interaction or ""
                mu = mu_map.get(inter, None)
                mu_str = f", μ={mu}" if mu is not None else ""
                print(f"  - #{idx}: master='{cp.master}', slave='{cp.slave}', interaction='{inter}'{mu_str}")
        else:
            print("[INP] 未在 *Contact Pair 中发现接触对（Trainer 将尝试自动识别）。")

        if asm.ties:
            print(f"[INP] Tie（绑定）共 {len(asm.ties)} 组：")
            for idx, t in enumerate(asm.ties, 1):
                print(f"  - #{idx}: master='{t.master}', slave='{t.slave}'")
        else:
            print("[INP] 未发现 Tie 绑定。")
    except Exception as e:
        print("[WARN] 接触/绑定审计失败：", e)

    # 训练关键配置核对
    print("\n[训练配置核对]")
    print(f"  - 训练步数 max_steps = {cfg.max_steps}")
    print(f"  - 接触采样 n_contact_points_per_pair = {cfg.n_contact_points_per_pair}")
    print(f"  - 预紧采样 preload_n_points_each = {cfg.preload_n_points_each}")
    print(f"  - 预紧力范围 N = {cfg.preload_min} ~ {cfg.preload_max}")
    print(f"  - 材料库（name -> (E, nu)）：{cfg.materials}")
    print(f"  - Part→材料：{cfg.part2mat}")
    print("======================================================================\n")


def _run_training(cfg, asm, export_saved_model: str = ""):
    # 训练前审计打印（你要的“那种详单”）
    _print_pretrain_audit(cfg, asm)

    from train.trainer import Trainer  # 再导一次确保路径就绪
    trainer = Trainer(cfg)
    trainer.run()

    if export_saved_model:
        trainer.export_saved_model(export_saved_model)

    print("\n✅ 训练完成！请到 'outputs/' 查看 5 张 “MIRROR up” 变形云图（文件名包含三颗预紧力数值）。")
    print("   如需修改 INP 路径、表面名或超参，直接改 main.py 顶部“USER SETTINGS”部分即可。")


def _run_inference(cfg,
                   preload_values,
                   ckpt_path: str = "",
                   out_path: str = "",
                   data_out: str = "auto",
                   title_prefix: str = "",
                   show: bool = False,
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
    )

    print("\n✅ 推理完成！")
    print(f"   使用的检查点: {restored}")
    print(f"   生成的云图: {save_path}")
    if trainer.last_viz_data_path:
        print(f"   位移数据: {trainer.last_viz_data_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the PINN or run inference with custom preloads.")
    parser.add_argument("--mode", choices=["train", "infer"], default="train", help="train: 训练模型; infer: 使用已训模型生成云图")
    parser.add_argument("--preload", nargs=3, type=float, metavar=("P1", "P2", "P3"),
                        help="三个螺栓的预紧力，单位 N (仅在 --mode infer 时使用)")
    parser.add_argument("--ckpt", default="", help="指定要恢复的检查点路径；默认使用 checkpoints/ 下最新的")
    parser.add_argument("--out", default="", help="保存推理云图的路径；默认写入 outputs/ 目录")
    parser.add_argument("--data", default="auto",
                        help="云图对应的位移采样 txt 文件路径。使用 'auto' 表示与图片同名，"
                             "使用 'none' 或空字符串表示不导出。")
    parser.add_argument("--title", default="", help="自定义云图标题前缀（默认沿用配置中的 viz_title_prefix）")
    parser.add_argument("--show", action="store_true", help="推理时显示 matplotlib 窗口")
    parser.add_argument("--export", default="", help="将模型导出为 TensorFlow SavedModel 的目录")

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
            export_saved_model=args.export,
        )


if __name__ == "__main__":
    main()