#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optuna_search.py
-----------------
使用 Optuna (TPE + Hyperband) 对关键超参进行自动搜索的小脚本。

搜索的超参包括：
- 学习率 lr（对收敛速度最敏感）
- RAR 抽样比例 rar_fraction（体积分点/接触共享）
- 傅里叶位置编码尺度缩放 sigma_scale（控制高频覆盖）

默认启用自适应 Loss 权重，并将可视化样本数设为 0 以加快试验。
"""

from __future__ import annotations

import argparse
import copy
import os
from typing import Optional

try:
    import optuna
    from optuna.pruners import HyperbandPruner
    from optuna.samplers import TPESampler
except Exception as exc:  # pragma: no cover - 依赖缺失时直接提示
    raise SystemExit(
        "未安装 optuna，请先安装：pip install optuna"
    ) from exc

from main import _prepare_config_with_autoguess
from train.trainer import Trainer


def _scale_fourier_sigmas(cfg, factor: float) -> None:
    """按比例缩放 Fourier 位置编码的 sigma/sigmas。"""

    fourier = cfg.model_cfg.field.fourier
    if getattr(fourier, "sigmas", None):
        fourier.sigmas = tuple(float(s) * factor for s in fourier.sigmas)
    else:
        fourier.sigma = float(fourier.sigma) * factor


def _objective(trial: optuna.trial.Trial, base_cfg, steps: int, log_every: int) -> float:
    cfg = copy.deepcopy(base_cfg)

    # 1) 采样超参
    cfg.lr = trial.suggest_float("lr", 3e-4, 3e-3, log=True)
    rar_fraction = trial.suggest_float("rar_fraction", 0.1, 0.9)
    sigma_scale = trial.suggest_float("sigma_scale", 0.5, 5.0, log=True)

    # 2) 应用到配置
    _scale_fourier_sigmas(cfg, sigma_scale)
    cfg.volume_rar_fraction = rar_fraction
    cfg.contact_rar_fraction = rar_fraction
    cfg.loss_adaptive_enabled = True
    cfg.max_steps = steps
    cfg.log_every = max(1, log_every)
    cfg.viz_samples_after_train = 0  # 搜索阶段不渲染云图

    # 将输出/ckpt 放到独立目录，避免污染正式训练
    cfg.out_dir = os.path.join(cfg.out_dir, "optuna")
    cfg.ckpt_dir = os.path.join(cfg.ckpt_dir, "optuna")

    trainer = Trainer(cfg)
    trainer.run()

    # 以 best_metric（按 save_best_on 选择）作为评估值，越小越好
    return float(getattr(trainer, "best_metric", float("inf")))


def _build_study(
    study_name: str,
    storage: Optional[str],
    seed: int,
    min_resource: int,
    max_resource: int,
) -> optuna.study.Study:
    sampler = TPESampler(seed=seed, multivariate=True)
    pruner = HyperbandPruner(min_resource=min_resource, max_resource=max_resource)
    return optuna.create_study(
        study_name=study_name or None,
        storage=storage or None,
        load_if_exists=bool(storage),
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="使用 Optuna 搜索 PINN 超参")
    parser.add_argument("--trials", type=int, default=10, help="搜索的 trial 数量")
    parser.add_argument("--timeout", type=int, default=None, help="搜索总时长（秒）")
    parser.add_argument("--steps", type=int, default=400, help="每个 trial 的训练步数")
    parser.add_argument("--log-every", type=int, default=20, help="日志/保存间隔步数")
    parser.add_argument("--study-name", type=str, default="pinn-optuna", help="Study 名称")
    parser.add_argument("--storage", type=str, default="", help="Optuna storage，例如 sqlite:///optuna.db")
    parser.add_argument("--seed", type=int, default=42, help="TPE 采样的随机种子")
    parser.add_argument("--min-resource", type=int, default=50, help="Hyperband 最小资源步数")

    args = parser.parse_args(argv)

    base_cfg, _ = _prepare_config_with_autoguess()
    study = _build_study(
        study_name=args.study_name,
        storage=args.storage,
        seed=args.seed,
        min_resource=max(1, args.min_resource),
        max_resource=max(args.min_resource, args.steps),
    )

    study.optimize(
        lambda trial: _objective(trial, base_cfg, args.steps, args.log_every),
        n_trials=args.trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )

    print("\n[optuna] 最优参数：", study.best_params)
    print("[optuna] 最优目标值：", study.best_value)


if __name__ == "__main__":
    main()

