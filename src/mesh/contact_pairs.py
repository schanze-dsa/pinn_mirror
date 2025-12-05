#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
contact_pairs.py
----------------
Build contact-pair sampling maps:
  - Sample points on the slave surface (area-proportional)
  - Project them to the closest points on the master surface
  - Return normals (from master tri), tangent basis t1/t2, and area weights

This module relies on:
  - src/io/inp_parser.AssemblyModel
  - src/mesh/surface_utils for surface triangulation, sampling, and projection

Output data structures are light-weight numpy arrays, ready for TF/JAX/PyTorch
wrappers in the contact operator (ALM with friction).

Author: you
"""

from __future__ import annotations
from mesh.surface_utils import resolve_surface_to_tris, compute_tri_geometry, sample_points_on_surface, project_points_onto_surface, build_contact_surfaces
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np

from inp_io.inp_parser import AssemblyModel, PartMesh
from mesh.surface_utils import (
    resolve_surface_to_tris,
    compute_tri_geometry,
    sample_points_on_surface,
    project_points_onto_surface,
    build_contact_surfaces,
)


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------

@dataclass
class ContactPairSpec:
    """User-level specification of a contact pair (surface keys in asm.surfaces)."""
    slave_key: str
    master_key: str
    name: str = ""  # optional alias for logging


@dataclass
class ContactPairData:
    """
    Concrete sampled data for one contact pair (size = n_pts).
    Arrays are np.float64 unless noted. Shapes:
      xs, xm, n, t1, t2 : (n, 3)
      w_area            : (n,)
      slave_tri_idx     : (n,) int64
      master_tri_idx    : (n,) int64
      dist              : (n,)  Euclidean distance from xs to xm
      pair_id           : (n,)  int64, an integer label for this pair (for grouping)
    """
    slave_part: str
    master_part: str
    xs: np.ndarray
    xm: np.ndarray
    n: np.ndarray
    t1: np.ndarray
    t2: np.ndarray
    w_area: np.ndarray
    slave_tri_idx: np.ndarray
    master_tri_idx: np.ndarray
    dist: np.ndarray
    pair_id: np.ndarray
    name: str = ""


@dataclass
class ContactMap:
    """
    A collection of ContactPairData for multiple pairs.
    Provides helpers to get concatenated arrays for batched computation.
    """
    pairs: List[ContactPairData]

    def concatenate(self) -> Dict[str, np.ndarray]:
        """Concatenate arrays from all pairs along the sample dimension."""
        if not self.pairs:
            return {}
        xs = np.concatenate([p.xs for p in self.pairs], axis=0)
        xm = np.concatenate([p.xm for p in self.pairs], axis=0)
        n  = np.concatenate([p.n  for p in self.pairs], axis=0)
        t1 = np.concatenate([p.t1 for p in self.pairs], axis=0)
        t2 = np.concatenate([p.t2 for p in self.pairs], axis=0)
        w  = np.concatenate([p.w_area for p in self.pairs], axis=0)
        sid = np.concatenate([p.slave_tri_idx for p in self.pairs], axis=0)
        mid = np.concatenate([p.master_tri_idx for p in self.pairs], axis=0)
        dist = np.concatenate([p.dist for p in self.pairs], axis=0)
        pid = np.concatenate([p.pair_id for p in self.pairs], axis=0)
        return dict(xs=xs, xm=xm, n=n, t1=t1, t2=t2, w_area=w,
                    slave_tri_idx=sid, master_tri_idx=mid, dist=dist, pair_id=pid)

    def __len__(self) -> int:
        return int(sum(p.xs.shape[0] for p in self.pairs))


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _orthonormal_tangent_basis(normals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given normals (N,3), compute two orthonormal tangent vectors t1,t2 for each.
    Strategy: pick an arbitrary reference axis not parallel to n; do cross products.

    Returns:
        t1, t2 : (N,3) each
    """
    N = normals.shape[0]
    n = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-16)

    # pick reference axis per normal (avoid near-parallel with [1,0,0])
    ref = np.tile(np.array([1.0, 0.0, 0.0]), (N, 1))
    mask = np.abs(n[:, 0]) > 0.9  # if too parallel to x-axis, use y-axis instead
    ref[mask] = np.array([0.0, 1.0, 0.0])

    # t1 = normalize(ref x n)
    t1 = np.cross(ref, n)
    t1 /= (np.linalg.norm(t1, axis=1, keepdims=True) + 1e-16)
    # t2 = n x t1
    t2 = np.cross(n, t1)
    t2 /= (np.linalg.norm(t2, axis=1, keepdims=True) + 1e-16)
    return t1, t2


def _compute_area_weights(tri_idx: np.ndarray, tri_areas: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Compute per-sample area weights ensuring unbiased Monte Carlo integration on the surface.

    We sampled triangles proportional to area, so a simple constant weight
    w = total_area / n_samples is unbiased. For slightly lower variance,
    we can do per-triangle counting: w_k = area(tri) / count(tri).
    """
    total_area = float(tri_areas.sum())
    if n_samples == 0 or total_area <= 0:
        return np.zeros((0,), dtype=np.float64)

    # Count how many times each triangle was sampled
    counts = np.bincount(tri_idx, minlength=tri_areas.shape[0]).astype(np.float64)
    w = tri_areas[tri_idx] / (counts[tri_idx] + 1e-16)  # area per sample inside that tri
    # Normalize to match total_area (optional, keeps sums comparable across pairs)
    scale = total_area / (w.sum() + 1e-16) * n_samples / n_samples
    return w * scale


# ---------------------------------------------------------------------
# Core builders
# ---------------------------------------------------------------------

def build_contact_pair_data(
    asm: AssemblyModel,
    slave_key: str,
    master_key: str,
    n_points: int,
    rng: Optional[np.random.Generator] = None,
    prefilter_k: int = 8,
    chunk: int = 4096,
    pair_id: int = 0,
    name: str = "",
) -> ContactPairData:
    """
    为一个接触对构建采样/投影数据。
    - 在从属面（slave surface）上采样 n_points
    - 投影到主面（master surface）（最近点）
    - 计算法向量（从主面）、切向基底和面积权重

    返回：
        ContactPairData（为摩擦接触能量准备）
    """
    if rng is None:
        rng = np.random.default_rng()

    # 解析两个接触面的部分和三角形表面
    part_s, ts_s, part_m, ts_m = build_contact_surfaces(asm, slave_key, master_key)

    # 在从属面上采样
    xs, tri_idx_s, bary_s, n_s_dummy = sample_points_on_surface(part_s, ts_s, n_points, rng=rng)

    # 投影到主面
    xm, n_m, tri_idx_m, dist = project_points_onto_surface(part_m, ts_m, xs, prefilter_k=prefilter_k, chunk=chunk)

    # 从主面法向量计算切向基底
    t1, t2 = _orthonormal_tangent_basis(n_m)

    # 使用从属面的三角形面积计算权重（因为在标准接触公式中，积分在从属面上进行）
    tri_areas_s, _, _ = compute_tri_geometry(part_s, ts_s)
    w = _compute_area_weights(tri_idx_s, tri_areas_s, n_points)

    return ContactPairData(
        slave_part=part_s.name,
        master_part=part_m.name,
        xs=xs, xm=xm, n=n_m, t1=t1, t2=t2, w_area=w,
        slave_tri_idx=tri_idx_s.astype(np.int64),
        master_tri_idx=tri_idx_m.astype(np.int64),
        dist=dist.astype(np.float64),
        pair_id=(np.full((n_points,), pair_id, dtype=np.int64)),
        name=name or f"{slave_key}__{master_key}",
    )

def build_contact_map(
    asm: AssemblyModel,
    specs: List[ContactPairSpec],
    n_points_per_pair: int,
    seed: Optional[int] = None,
    prefilter_k: int = 8,
    chunk: int = 4096,
) -> ContactMap:
    """
    Build ContactMap for a list of pairs. Each pair gets n_points_per_pair samples.
    """
    rng = np.random.default_rng(seed)
    pairs: List[ContactPairData] = []
    for pid, spec in enumerate(specs):
        data = build_contact_pair_data(
            asm=asm,
            slave_key=spec.slave_key,
            master_key=spec.master_key,
            n_points=n_points_per_pair,
            rng=rng,
            prefilter_k=prefilter_k,
            chunk=chunk,
            pair_id=pid,
            name=spec.name or f"pair_{pid}",
        )
        pairs.append(data)
    return ContactMap(pairs=pairs)


def resample_contact_map(
    asm: AssemblyModel,
    specs: List[ContactPairSpec],
    n_points_per_pair: int,
    base_seed: Optional[int],
    step_index: int,
    prefilter_k: int = 8,
    chunk: int = 4096,
) -> ContactMap:
    """
    Convenience function for training loops:
    - Each step call can resample contact points with a varying seed to avoid overfitting to fixed locations.
    """
    seed = None if base_seed is None else (base_seed + step_index * 9973)  # decorrelate seeds
    return build_contact_map(
        asm=asm,
        specs=specs,
        n_points_per_pair=n_points_per_pair,
        seed=seed,
        prefilter_k=prefilter_k,
        chunk=chunk,
    )


# ---------------------------------------------------------------------
# Optional utilities
# ---------------------------------------------------------------------

def guess_surface_key(asm: AssemblyModel, bare_name: str) -> Optional[str]:
    """
    Try to guess a full surface key in asm.surfaces from a bare name (case-insensitive substring match).
    Returns the first unique match or None if ambiguous/not found.
    """
    target_cs = bare_name.strip()
    target = target_cs.lower()

    # 0) 完全大小写敏感匹配（优先用用户提供的精确写法）
    cs_exact = [k for k, s in asm.surfaces.items()
                if s.name.strip() == target_cs or k.strip() == target_cs]
    if len(cs_exact) == 1:
        return cs_exact[0]

    # 1) 完全大小写不敏感匹配（若仍唯一则返回，否则视为歧义）
    exact = [k for k, s in asm.surfaces.items()
             if s.name.strip().lower() == target or k.strip().lower() == target]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        return None  # 仍然由上层提示冲突

    # 2) 片段匹配（若唯一则返回）
    matches = [k for k, s in asm.surfaces.items()
               if target in k.lower() or target in s.name.strip().lower()]
    if len(matches) == 1:
        return matches[0]
    return None  # ambiguous or not found; let the caller handle


# ---------------------------------------------------------------------
# Minimal self-test (optional; no command line required)
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Quick check when running this file alone; requires env INP_PATH or default path present.
    import os
    from inp_io.inp_parser import load_inp

    inp = os.environ.get("INP_PATH", "data/shuangfan.inp")
    if not os.path.exists(inp):
        print("[contact_pairs] INP not found. Set INP_PATH env or place file at data/shuangfan.inp.")
        exit(0)

    asm = load_inp(inp)

    # Heuristically pick two part-scope surfaces for a smoke test
    surface_keys = [k for k, v in asm.surfaces.items() if v.scope == "part"]
    if len(surface_keys) < 2:
        print("[contact_pairs] Not enough part-scope surfaces to test.")
        exit(0)

    # Use the first two as (slave, master)
    specs = [ContactPairSpec(slave_key=surface_keys[0], master_key=surface_keys[1], name="test_pair")]

    cmap = build_contact_map(asm, specs, n_points_per_pair=2000, seed=123)
    cat = cmap.concatenate()