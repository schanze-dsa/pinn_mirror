#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mirror_viz.py
-------------
Visualize deflection map of a mirror surface ("MIRROR up"):

Pipeline:
  1) Triangulate the named surface via surface_utils.resolve_surface_to_tris()
  2) Fit a best-fit plane using SVD over all surface vertices to get an orthonormal basis (e1,e2,n)
  3) Project surface nodes to 2D (u,v) in that plane
  4) Evaluate displacement field u(X; P) on unique surface nodes
  5) Take scalar deflection d = (u · n) along the global mirror normal
  6) Plot a smooth tripcolor (default) or tricontourf map in (u,v)-space; title includes (P1,P2,P3)

Notes:
- This module only handles visualization; it does not resample contact or modify physics.
- Units: assumed consistent with your model; the colorbar label can be configured via `units="mm"`.

Public API:
    fig, ax, data_path = plot_mirror_deflection(
        asm, surface_key, u_fn, params, P_values=(P1,P2,P3),
        out_path="outputs/mirror_P1_...png", title_prefix="Mirror Deflection",
        units="mm", levels=24, symmetric=True, show=False,
        data_out_path="auto", style="smooth", cmap="turbo"
    )

Author: you
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation


def _coerce_params_for_forward(params: dict) -> dict:
    """Ensure params contain a single-stage payload consumable by ``u_fn``."""

    if not isinstance(params, dict):
        return params
    if "stages" not in params:
        return params

    stages = params.get("stages")
    final = None
    if isinstance(stages, dict):
        P_seq = stages.get("P")
        Z_seq = stages.get("P_hat")
        if P_seq is not None and Z_seq is not None:
            final = {"P": P_seq[-1], "P_hat": Z_seq[-1]}
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

    for key in ("stage_order", "stage_rank", "stage_count"):
        if key in params and key not in final:
            final[key] = params[key]
    return final


def _eval_displacement_batched(u_fn, params, points: np.ndarray, batch_size: int) -> np.ndarray:
    """Evaluate ``u_fn`` on ``points`` in batches to control memory usage."""

    if points.size == 0:
        return np.zeros((0, 3), dtype=np.float64)

    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - tensorflow is required upstream
        raise RuntimeError("TensorFlow is required for evaluating the PINN model") from exc

    params = _coerce_params_for_forward(params)

    if batch_size is None or batch_size <= 0:
        batch_size = points.shape[0]

    outputs = []
    n = points.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = tf.convert_to_tensor(points[start:end], dtype=tf.float32)
        u_chunk = u_fn(chunk, params)
        outputs.append(np.asarray(u_chunk.numpy(), dtype=np.float64))
    return np.concatenate(outputs, axis=0)


def _with_new_stem(path: Path, new_stem: str) -> Path:
    """``Path.with_stem`` polyfill for Python <3.9.

    On older Python versions ``WindowsPath``/``PosixPath`` lack ``with_stem``;
    emulate it by replacing the filename while preserving the suffix and parent.
    """

    return path.with_name(new_stem + path.suffix)


def _eval_surface_or_assembly(
    u_fn,
    params,
    asm: AssemblyModel,
    surface_node_ids: np.ndarray,
    surface_points: np.ndarray,
    eval_batch_size: int,
    eval_scope: str,
    eval_subset: Optional[Tuple[np.ndarray, np.ndarray]] = None,
):
    """Evaluate displacements on the whole assembly, then slice out the surface.

    To align与“全结构求解、只输出表面”需求，如果装配提供了全局节点，则始终先
    对所有节点求解，再用表面节点索引提取对应位移；只有当装配缺少全局节点
    表时，才退回到仅对表面节点求解。

    Returns ``(u_surface, eval_meta)`` where ``u_surface`` aligns with ``surface_points``
    and ``eval_meta`` optionally carries assembly-wide displacements to reuse.
    """

    scope_key = (eval_scope or "surface").strip().lower()

    if not getattr(asm, "nodes", None):
        return _eval_displacement_batched(u_fn, params, surface_points, eval_batch_size), {}

    if scope_key not in {"all", "assembly", "global", "full"}:
        print(
            f"[viz] eval_scope='{scope_key}' 被强制为 'assembly' 以对全结构求解后再切表面"
        )

    if eval_subset is not None:
        global_nid, global_xyz = eval_subset
        scope_label = "part"
    else:
        global_nid = np.array(sorted(asm.nodes.keys()), dtype=np.int64)
        global_xyz = np.stack([asm.nodes[int(n)] for n in global_nid], axis=0).astype(np.float64)
        scope_label = "assembly"

    print(
        f"[viz] eval_scope={scope_label} -> querying {len(global_nid)} nodes "
        f"(surface nodes={len(surface_node_ids)})"
    )

    u_all = _eval_displacement_batched(u_fn, params, global_xyz, eval_batch_size)
    lookup = {int(n): u_all[i] for i, n in enumerate(global_nid)}
    u_surface = np.stack([lookup[int(n)] for n in surface_node_ids], axis=0)

    return u_surface, {
        "global_nodes": global_nid,
        "global_xyz": global_xyz,
        "u_all": u_all,
        "requested_scope": scope_key,
    }


def _refine_surface_samples(X3D: np.ndarray,
                            UV: np.ndarray,
                            tri_idx: np.ndarray,
                            subdivisions: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uniformly subdivide each triangle ``subdivisions`` times per edge.

    Returns refined ``(X_ref, UV_ref, tri_ref)`` suitable for plotting.
    """

    m = int(max(0, subdivisions))
    if m <= 0:
        return X3D, UV, tri_idx

    pts3d: List[np.ndarray] = []
    pts2d: List[np.ndarray] = []
    tris: List[List[int]] = []
    base_idx = 0

    for (i0, i1, i2) in tri_idx:
        X0, X1, X2 = X3D[[i0, i1, i2]]
        uv0, uv1, uv2 = UV[[i0, i1, i2]]
        local: Dict[Tuple[int, int], int] = {}

        for i in range(m + 1):
            for j in range(m + 1 - i):
                w1 = i / m
                w2 = j / m
                w0 = 1.0 - w1 - w2
                pts3d.append(w0 * X0 + w1 * X1 + w2 * X2)
                pts2d.append(w0 * uv0 + w1 * uv1 + w2 * uv2)
                local[(i, j)] = base_idx
                base_idx += 1

        for i in range(m):
            for j in range(m - i):
                a = local[(i, j)]
                b = local[(i + 1, j)]
                c = local[(i, j + 1)]
                tris.append([a, b, c])
                if i + j < m - 1:
                    d = local[(i + 1, j + 1)]
                    tris.append([b, d, c])

    return (
        np.asarray(pts3d, dtype=np.float64),
        np.asarray(pts2d, dtype=np.float64),
        np.asarray(tris, dtype=np.int32),
    )
from matplotlib import colors

from inp_io.inp_parser import AssemblyModel
from mesh.surface_utils import (
    TriSurface,
    compute_tri_geometry,
    resolve_surface_to_tris,
    triangulate_part_boundary,
    _fetch_xyz,
)


# -----------------------------
# Diagnostics for blank regions
# -----------------------------

@dataclass
class BlankRegionDiagnostics:
    requested_subdiv: int
    applied_subdiv: int
    nonfinite_deflection: int
    nonfinite_displacement: int
    nonfinite_uv: int
    tri_masked: int
    tri_dropped: int
    coverage_ratio_hull: float
    coverage_ratio_envelope: float
    n_boundary_loops: int
    envelope_area: float
    notes: List[str]
    boundary_loops: List[List[int]]

    def summary_lines(self) -> List[str]:
        lines = [
            f"[1] deflection NaN/Inf: {self.nonfinite_deflection}",
            f"[2] displacement/UV NaN/Inf: disp={self.nonfinite_displacement}, uv={self.nonfinite_uv}",
            f"[3] coverage (triangles / convex hull): {self.coverage_ratio_hull:.2%}",
            f"[3b] coverage (triangles / boundary envelope): {self.coverage_ratio_envelope:.2%}  loops={self.n_boundary_loops}",
            f"[4] refinement applied vs requested: {self.applied_subdiv} / {self.requested_subdiv}",
            f"[5] triangulation masked/dropped: mask={self.tri_masked}, drop={self.tri_dropped}",
            f"[6] additional notes: {'; '.join(self.notes) if self.notes else 'none'}",
        ]
        return lines

    @property
    def primary_cause(self) -> str:
        if self.nonfinite_deflection > 0:
            return "(1) 挠度中存在 NaN/Inf，绘图被掩码"
        if self.nonfinite_displacement > 0 or self.nonfinite_uv > 0:
            return "(2) 位移向量或投影坐标含 NaN/Inf，导致三角化异常"
        if self.coverage_ratio_envelope < 0.80:
            return "(3) 网格覆盖率低于 80%，几何存在孔洞/稀疏大三角形"
        if self.coverage_ratio_hull < 0.75:
            return "(3a) 凸包覆盖率低但边界覆盖正常，几何强凹或存在合法孔洞"
        if self.applied_subdiv < self.requested_subdiv:
            return "(4) 细分被 refine_max_points 裁剪，采样过于稀疏"
        if self.tri_masked > 0 or self.tri_dropped > 0:
            return "(5) 三角化阶段被掩码/丢弃，可能存在退化或重复点"
        return "(6) 未发现明显异常，请检查输入数据或导出文件"


def _convex_hull_area(points: np.ndarray) -> float:
    """Compute 2D convex hull area via monotonic chain (no extra deps)."""

    pts = np.unique(points, axis=0)
    if pts.shape[0] < 3:
        return 0.0

    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: List[np.ndarray] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = np.array(lower[:-1] + upper[:-1])
    if hull.shape[0] < 3:
        return 0.0
    area = 0.0
    for i in range(hull.shape[0]):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % hull.shape[0]]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def _triangle_area_sum(points: np.ndarray, tris: np.ndarray) -> float:
    if tris.size == 0:
        return 0.0
    p = points
    t = tris
    v1 = p[t[:, 1]] - p[t[:, 0]]
    v2 = p[t[:, 2]] - p[t[:, 0]]
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return float(np.sum(np.abs(cross)) * 0.5)


def _collect_boundary_loops(tris: np.ndarray) -> List[List[int]]:
    """Extract boundary loops (edges used exactly once) from a triangle list."""

    edge_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    for a, b, c in tris:
        edges = [(a, b), (b, c), (c, a)]
        for e0, e1 in edges:
            key = (e0, e1) if e0 <= e1 else (e1, e0)
            edge_counts[key] += 1

    boundary_edges = [e for e, cnt in edge_counts.items() if cnt == 1]
    if not boundary_edges:
        return []

    adj: Dict[int, List[int]] = defaultdict(list)
    for e0, e1 in boundary_edges:
        adj[e0].append(e1)
        adj[e1].append(e0)

    visited_edges = set()
    loops: List[List[int]] = []

    def next_neighbor(cur: int, prev: Optional[int]) -> Optional[int]:
        nbrs = adj[cur]
        if not nbrs:
            return None
        if prev is None:
            return nbrs[0]
        for n in nbrs:
            if n != prev:
                return n
        return nbrs[0]

    for edge in boundary_edges:
        if edge in visited_edges or (edge[1], edge[0]) in visited_edges:
            continue
        loop = []
        start, nxt = edge
        cur, prev = start, nxt
        loop.append(cur)
        while True:
            visited_edges.add((cur, prev))
            loop.append(prev)
            nxt2 = next_neighbor(prev, cur)
            if nxt2 is None:
                break
            cur, prev = prev, nxt2
            if prev == loop[0]:
                break
        loops.append(loop)
    return loops


def _loop_area(points: np.ndarray, loop: List[int]) -> float:
    if len(loop) < 3:
        return 0.0
    pts = points[np.asarray(loop, dtype=np.int64)]
    area = 0.0
    for i in range(len(loop)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(loop)]
        area += x1 * y2 - x2 * y1
    return 0.5 * area


def _diagnose_blank_regions(
    requested_subdiv: int,
    applied_subdiv: int,
    UV_plot: np.ndarray,
    tri_plot: np.ndarray,
    tri: Triangulation,
    u_plot: np.ndarray,
    d_plot: np.ndarray,
    tri_mask: Optional[np.ndarray],
) -> BlankRegionDiagnostics:

    nonfinite_deflection = int((~np.isfinite(d_plot)).sum())
    nonfinite_disp = int((~np.isfinite(u_plot)).sum())
    nonfinite_uv = int((~np.isfinite(UV_plot)).sum())

    tri_masked = int(np.count_nonzero(tri_mask)) if tri_mask is not None else 0
    tri_dropped = int(max(0, tri_plot.shape[0] - getattr(tri, "triangles", tri_plot).shape[0]))

    hull_area = _convex_hull_area(UV_plot)
    tri_area = _triangle_area_sum(UV_plot, tri_plot)
    coverage_ratio_hull = 0.0 if hull_area <= 0 else min(1.0, tri_area / hull_area)

    boundary_loops = _collect_boundary_loops(tri_plot)
    n_loops = len(boundary_loops)
    loop_areas = [abs(_loop_area(UV_plot, loop)) for loop in boundary_loops]
    envelope_area = float(sum(loop_areas)) if loop_areas else hull_area
    if loop_areas:
        # Treat the largest loop as the outer boundary; subtract smaller loops as holes when oriented oppositely
        max_idx = int(np.argmax(loop_areas))
        outer_area = loop_areas[max_idx]
        hole_area = sum(loop_areas[:max_idx] + loop_areas[max_idx + 1:])
        envelope_area = max(outer_area - hole_area, 0.0)

    coverage_ratio_env = 0.0 if envelope_area <= 0 else min(1.0, tri_area / envelope_area)

    notes: List[str] = []
    if hull_area <= 0:
        notes.append("投影凸包面积为 0，可能所有点共线或重复")
    if tri_area <= 0:
        notes.append("三角面积总和为 0，可能存在退化三角形")
    if n_loops > 1:
        notes.append(f"检测到 {n_loops} 条边界环，可能存在孔洞/不连通的补丁")
    if coverage_ratio_env >= 0.9 and coverage_ratio_hull < 0.75:
        notes.append("凸包面积显著大于边界包络，几何可能强凹但覆盖良好")

    return BlankRegionDiagnostics(
        requested_subdiv=requested_subdiv,
        applied_subdiv=applied_subdiv,
        nonfinite_deflection=nonfinite_deflection,
        nonfinite_displacement=nonfinite_disp,
        nonfinite_uv=nonfinite_uv,
        tri_masked=tri_masked,
        tri_dropped=tri_dropped,
        coverage_ratio_hull=coverage_ratio_hull,
        coverage_ratio_envelope=coverage_ratio_env,
        n_boundary_loops=n_loops,
        envelope_area=envelope_area,
        notes=notes,
        boundary_loops=boundary_loops,
    )


def _mask_tris_with_loops(tri: Triangulation, UV: np.ndarray, loops: List[List[int]]) -> np.ndarray:
    """Return a mask for triangles outside the outer boundary or inside holes."""

    if not loops:
        return np.zeros(tri.triangles.shape[0], dtype=bool)

    from matplotlib.path import Path

    areas = [abs(_loop_area(UV, loop)) for loop in loops]
    if not areas:
        return np.zeros(tri.triangles.shape[0], dtype=bool)

    outer_idx = int(np.argmax(areas))
    outer_loop = loops[outer_idx]
    hole_loops = [loop for i, loop in enumerate(loops) if i != outer_idx]

    centroids = UV[tri.triangles].mean(axis=1)
    mask = ~Path(UV[outer_loop]).contains_points(centroids)

    for loop in hole_loops:
        hole_path = Path(UV[loop])
        mask |= hole_path.contains_points(centroids)

    return mask


# -----------------------------
# Diagnostics for blank regions
# -----------------------------

@dataclass
class BlankRegionDiagnostics:
    requested_subdiv: int
    applied_subdiv: int
    nonfinite_deflection: int
    nonfinite_displacement: int
    nonfinite_uv: int
    tri_masked: int
    tri_dropped: int
    coverage_ratio_hull: float
    coverage_ratio_envelope: float
    n_boundary_loops: int
    envelope_area: float
    notes: List[str]
    boundary_loops: List[List[int]]

    def summary_lines(self) -> List[str]:
        lines = [
            f"[1] deflection NaN/Inf: {self.nonfinite_deflection}",
            f"[2] displacement/UV NaN/Inf: disp={self.nonfinite_displacement}, uv={self.nonfinite_uv}",
            f"[3] coverage (triangles / convex hull): {self.coverage_ratio_hull:.2%}",
            f"[3b] coverage (triangles / boundary envelope): {self.coverage_ratio_envelope:.2%}  loops={self.n_boundary_loops}",
            f"[4] refinement applied vs requested: {self.applied_subdiv} / {self.requested_subdiv}",
            f"[5] triangulation masked/dropped: mask={self.tri_masked}, drop={self.tri_dropped}",
            f"[6] additional notes: {'; '.join(self.notes) if self.notes else 'none'}",
        ]
        return lines

    @property
    def primary_cause(self) -> str:
        if self.nonfinite_deflection > 0:
            return "(1) 挠度中存在 NaN/Inf，绘图被掩码"
        if self.nonfinite_displacement > 0 or self.nonfinite_uv > 0:
            return "(2) 位移向量或投影坐标含 NaN/Inf，导致三角化异常"
        if self.coverage_ratio_envelope < 0.80:
            return "(3) 网格覆盖率低于 80%，几何存在孔洞/稀疏大三角形"
        if self.coverage_ratio_hull < 0.75:
            return "(3a) 凸包覆盖率低但边界覆盖正常，几何强凹或存在合法孔洞"
        if self.applied_subdiv < self.requested_subdiv:
            return "(4) 细分被 refine_max_points 裁剪，采样过于稀疏"
        if self.tri_masked > 0 or self.tri_dropped > 0:
            return "(5) 三角化阶段被掩码/丢弃，可能存在退化或重复点"
        return "(6) 未发现明显异常，请检查输入数据或导出文件"


def _convex_hull_area(points: np.ndarray) -> float:
    """Compute 2D convex hull area via monotonic chain (no extra deps)."""

    pts = np.unique(points, axis=0)
    if pts.shape[0] < 3:
        return 0.0

    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: List[np.ndarray] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = np.array(lower[:-1] + upper[:-1])
    if hull.shape[0] < 3:
        return 0.0
    area = 0.0
    for i in range(hull.shape[0]):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % hull.shape[0]]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def _triangle_area_sum(points: np.ndarray, tris: np.ndarray) -> float:
    if tris.size == 0:
        return 0.0
    p = points
    t = tris
    v1 = p[t[:, 1]] - p[t[:, 0]]
    v2 = p[t[:, 2]] - p[t[:, 0]]
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return float(np.sum(np.abs(cross)) * 0.5)


def _collect_boundary_loops(tris: np.ndarray) -> List[List[int]]:
    """Extract boundary loops (edges used exactly once) from a triangle list."""

    edge_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    for a, b, c in tris:
        edges = [(a, b), (b, c), (c, a)]
        for e0, e1 in edges:
            key = (e0, e1) if e0 <= e1 else (e1, e0)
            edge_counts[key] += 1

    boundary_edges = [e for e, cnt in edge_counts.items() if cnt == 1]
    if not boundary_edges:
        return []

    adj: Dict[int, List[int]] = defaultdict(list)
    for e0, e1 in boundary_edges:
        adj[e0].append(e1)
        adj[e1].append(e0)

    visited_edges = set()
    loops: List[List[int]] = []

    def next_neighbor(cur: int, prev: Optional[int]) -> Optional[int]:
        nbrs = adj[cur]
        if not nbrs:
            return None
        if prev is None:
            return nbrs[0]
        for n in nbrs:
            if n != prev:
                return n
        return nbrs[0]

    for edge in boundary_edges:
        if edge in visited_edges or (edge[1], edge[0]) in visited_edges:
            continue
        loop = []
        start, nxt = edge
        cur, prev = start, nxt
        loop.append(cur)
        while True:
            visited_edges.add((cur, prev))
            loop.append(prev)
            nxt2 = next_neighbor(prev, cur)
            if nxt2 is None:
                break
            cur, prev = prev, nxt2
            if prev == loop[0]:
                break
        loops.append(loop)
    return loops


def _loop_area(points: np.ndarray, loop: List[int]) -> float:
    if len(loop) < 3:
        return 0.0
    pts = points[np.asarray(loop, dtype=np.int64)]
    area = 0.0
    for i in range(len(loop)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(loop)]
        area += x1 * y2 - x2 * y1
    return 0.5 * area


def _diagnose_blank_regions(
    requested_subdiv: int,
    applied_subdiv: int,
    UV_plot: np.ndarray,
    tri_plot: np.ndarray,
    tri: Triangulation,
    u_plot: np.ndarray,
    d_plot: np.ndarray,
    tri_mask: Optional[np.ndarray],
) -> BlankRegionDiagnostics:

    nonfinite_deflection = int((~np.isfinite(d_plot)).sum())
    nonfinite_disp = int((~np.isfinite(u_plot)).sum())
    nonfinite_uv = int((~np.isfinite(UV_plot)).sum())

    tri_masked = int(np.count_nonzero(tri_mask)) if tri_mask is not None else 0
    tri_dropped = int(max(0, tri_plot.shape[0] - getattr(tri, "triangles", tri_plot).shape[0]))

    hull_area = _convex_hull_area(UV_plot)
    tri_area = _triangle_area_sum(UV_plot, tri_plot)
    coverage_ratio_hull = 0.0 if hull_area <= 0 else min(1.0, tri_area / hull_area)

    boundary_loops = _collect_boundary_loops(tri_plot)
    n_loops = len(boundary_loops)
    loop_areas = [abs(_loop_area(UV_plot, loop)) for loop in boundary_loops]
    envelope_area = float(sum(loop_areas)) if loop_areas else hull_area
    if loop_areas:
        # Treat the largest loop as the outer boundary; subtract smaller loops as holes when oriented oppositely
        max_idx = int(np.argmax(loop_areas))
        outer_area = loop_areas[max_idx]
        hole_area = sum(loop_areas[:max_idx] + loop_areas[max_idx + 1:])
        envelope_area = max(outer_area - hole_area, 0.0)

    coverage_ratio_env = 0.0 if envelope_area <= 0 else min(1.0, tri_area / envelope_area)

    notes: List[str] = []
    if hull_area <= 0:
        notes.append("投影凸包面积为 0，可能所有点共线或重复")
    if tri_area <= 0:
        notes.append("三角面积总和为 0，可能存在退化三角形")
    if n_loops > 1:
        notes.append(f"检测到 {n_loops} 条边界环，可能存在孔洞/不连通的补丁")
    if coverage_ratio_env >= 0.9 and coverage_ratio_hull < 0.75:
        notes.append("凸包面积显著大于边界包络，几何可能强凹但覆盖良好")

    return BlankRegionDiagnostics(
        requested_subdiv=requested_subdiv,
        applied_subdiv=applied_subdiv,
        nonfinite_deflection=nonfinite_deflection,
        nonfinite_displacement=nonfinite_disp,
        nonfinite_uv=nonfinite_uv,
        tri_masked=tri_masked,
        tri_dropped=tri_dropped,
        coverage_ratio_hull=coverage_ratio_hull,
        coverage_ratio_envelope=coverage_ratio_env,
        n_boundary_loops=n_loops,
        envelope_area=envelope_area,
        notes=notes,
        boundary_loops=boundary_loops,
    )


def _mask_tris_with_loops(tri: Triangulation, UV: np.ndarray, loops: List[List[int]]) -> np.ndarray:
    """Return a mask for triangles outside the outer boundary or inside holes."""

    if not loops:
        return np.zeros(tri.triangles.shape[0], dtype=bool)

    from matplotlib.path import Path

    areas = [abs(_loop_area(UV, loop)) for loop in loops]
    if not areas:
        return np.zeros(tri.triangles.shape[0], dtype=bool)

    outer_idx = int(np.argmax(areas))
    outer_loop = loops[outer_idx]
    hole_loops = [loop for i, loop in enumerate(loops) if i != outer_idx]

    centroids = UV[tri.triangles].mean(axis=1)
    mask = ~Path(UV[outer_loop]).contains_points(centroids)

    for loop in hole_loops:
        hole_path = Path(UV[loop])
        mask |= hole_path.contains_points(centroids)

    return mask


# -----------------------------
# Core helpers
# -----------------------------

def _fit_plane_basis(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a best-fit plane to 3D points X (N,3) by SVD.
    Returns:
        c : centroid (3,)
        e1, e2 : in-plane orthonormal basis (3,), (3,)
        n : unit normal (3,)  (right-handed: e1 x e2 = n)
    """
    c = X.mean(axis=0)
    A = X - c
    # SVD on covariance
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    # normal is the singular vector with smallest singular value -> last row of Vt
    n = Vt[-1, :]
    n = n / (np.linalg.norm(n) + 1e-16)
    # e1: choose the direction of largest variance (first row of Vt)
    e1 = Vt[0, :]
    e1 -= n * np.dot(e1, n)
    e1 = e1 / (np.linalg.norm(e1) + 1e-16)
    e2 = np.cross(n, e1)
    e2 = e2 / (np.linalg.norm(e2) + 1e-16)
    return c, e1, e2, n


def _unique_nodes_from_tris(ts: TriSurface) -> Tuple[np.ndarray, np.ndarray]:
    """
    From TriSurface.tri_node_ids (T,3), build:
        nid_unique : (Nu,) unique node ids
        tri_idx    : (T,3) triangulation indices into nid_unique
    """
    tri = ts.tri_node_ids.reshape(-1)
    nid_unique, inv = np.unique(tri, return_inverse=True)
    tri_idx = inv.reshape((-1, 3)).astype(np.int64)
    return nid_unique, tri_idx


def _export_surface_mesh(path: Path,
                         nid_unique: np.ndarray,
                         X3D: np.ndarray,
                         tri_idx: np.ndarray) -> None:
    """Write the reconstructed FE surface (nodes + triangles) to a PLY file."""

    path = path.with_suffix(path.suffix or ".ply")
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as fp:
        fp.write("ply\n")
        fp.write("format ascii 1.0\n")
        fp.write("comment reconstructed from INP surface for visualization audit\n")
        fp.write(f"element vertex {X3D.shape[0]}\n")
        fp.write("property float x\n")
        fp.write("property float y\n")
        fp.write("property float z\n")
        fp.write("property int node_id\n")
        fp.write(f"element face {tri_idx.shape[0]}\n")
        fp.write("property list uchar int vertex_indices\n")
        fp.write("end_header\n")

        for nid, xyz in zip(nid_unique, X3D):
            fp.write(f"{xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f} {int(nid)}\n")

        for tri in tri_idx:
            fp.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")


def _project_to_plane(X: np.ndarray, c: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    """
    Project 3D points X (N,3) to 2D coords (u,v) in plane basis {e1,e2} at origin c.
    """
    A = X - c[None, :]
    u = A @ e1
    v = A @ e2
    return np.stack([u, v], axis=1)  # (N,2)


# -----------------------------
# Main visualization
# -----------------------------

def plot_mirror_deflection(asm: AssemblyModel,
                           surface_key: str,
                           u_fn,
                           params: dict,
                           P_values: Optional[Tuple[float, float, float]] = None,
                           out_path: Optional[str] = None,
                           render_surface: bool = True,
                           title_prefix: str = "Mirror Deflection",
                           units: str = "mm",
                           levels: int = 24,
                           symmetric: bool = True,
                           show: bool = False,
                           data_out_path: Optional[str] = "auto",
                           surface_mesh_out_path: Optional[str] = "auto",
                           plot_full_structure: bool = False,
                           full_structure_out_path: Optional[str] = "auto",
                           full_structure_data_out_path: Optional[str] = None,
                           full_structure_part: Optional[str] = None,
                           surface_source: str = "part_top",
                           style: str = "smooth",
                           cmap: Optional[str] = None,
                           draw_wireframe: bool = False,
                           refine_subdivisions: int = 0,
                           refine_max_points: Optional[int] = None,
                           eval_batch_size: int = 65_536,
                           eval_scope: str = "assembly",
                           diagnose_blanks: bool = False,
                           auto_fill_blanks: bool = False,
                           diag_out: Optional[Dict[str, BlankRegionDiagnostics]] = None):
    """
    Visualize deflection along the global mirror normal of the given surface.

    Args:
        asm, surface_key : AssemblyModel and key in asm.surfaces (e.g., 'MIRROR_up' exact key)
        u_fn, params     : forward callable and params for your PINN (params should contain 'P' or 'P_hat')
        P_values         : (P1,P2,P3) in N, used in title. If None, will try params['P'].
        out_path         : if not None, save figure to this path
        title_prefix     : string prefix for the figure title
        units            : colorbar label for displacement units (e.g., "mm")
        levels           : number of contour levels
        symmetric        : if True, make color limits symmetric about 0 for diverging colormap
        show             : if True, call plt.show()
        data_out_path    : Path to write displacement samples. If "auto" and
                           ``out_path`` is provided, a ``.txt`` with the same
                           stem is written. Use ``None``/"none" to disable.
        surface_mesh_out_path : Path to write the reconstructed FE surface mesh
                           (triangles only, no displacement). If "auto" and
                           ``out_path`` is provided, a ``*_surface.ply`` next to
                           the figure is written. Use ``None``/"none" to disable.
        plot_full_structure  : If True and assembly nodes are available, also plot
                           a displacement magnitude map for the whole assembly
                           (scatter projected to the same best-fit plane).
                           full_structure_out_path : Path to write the full-structure displacement
                           plot. If "auto" and ``out_path`` is provided, writes
                           ``*_assembly.png`` next to the surface plot.
        full_structure_data_out_path : Optional path to write the assembly-wide
                           displacement samples (x,y,z,ux,uy,uz,|u|,u,v). If
                           None, no extra dataset is written; if "auto" and
                           ``out_path`` is provided, writes ``*_assembly.txt``.
        surface_source      : "surface" 使用 INP 表面；"part"/"part_top" 从目标零件
                            的外边界三角化得到表面，"part_top" 只保留主导法向一致的
                            外表面（便于提取镜面上表面环形云图）。
        style            : "smooth" to render a Gouraud-shaded tripcolor map
                            (Abaqus-like), "flat" for flat shading, or
                            "contour" to use tricontourf as in the legacy
                           implementation.
        cmap             : Optional matplotlib colormap name; defaults to
                           ``"turbo"`` for smooth/flat styles and
                           ``"coolwarm"`` for contour mode.
        draw_wireframe   : Whether to overlay triangle edges.
        refine_subdivisions : Uniform barycentric subdivisions per surface triangle.
        refine_max_points   : Optional guardrail limiting the total evaluation points.
        eval_batch_size     : Batch size when querying ``u_fn`` for visualization.
        eval_scope          : 参数保留向后兼容；只要装配提供全局节点，就会强制对
                              全部节点求解（"assembly"），再提取对应表面节点的结果。
                              仅当缺少全局节点表时才退回“只评估表面节点”。
        diagnose_blanks     : If True, run a one-click diagnosis to pinpoint blank-region causes.
        auto_fill_blanks    : If True and coverage is low, rebuild a 2D triangulation to fill holes
                              based on boundary loops (keeps NaN/Inf masking).
        diag_out            : Optional dict to receive ``{"blank_check": BlankRegionDiagnostics}``
                              for downstream logging.

    Returns:
        (fig, ax, data_path)
    """
    # 1) Triangulate surface & collect unique nodes
    ts = resolve_surface_to_tris(asm, surface_key, log_summary=False)
    source_key = (surface_source or "surface").strip().lower()
    part = asm.parts[ts.part_name]

    # 如果指定了从零件外边界重建表面，则使用覆盖更完整的三角网格（可选仅保留主导法向）
    if source_key in {"part", "part_top", "part_boundary"}:
        rebuilt = triangulate_part_boundary(part, ts.part_name, log_summary=False)
        if len(rebuilt) > 0:
            if source_key == "part_top":
                areas, normals, _ = compute_tri_geometry(part, rebuilt)
                if areas.size:
                    weighted = (areas[:, None] * normals).sum(axis=0)
                    norm = float(np.linalg.norm(weighted))
                    if norm > 0.0:
                        n_dom = weighted / norm
                        keep = (normals @ n_dom) > 0.2
                        if not np.any(keep):
                            keep = (normals @ n_dom) >= 0.0
                        if np.any(keep):
                            rebuilt = TriSurface(
                                name=rebuilt.name,
                                part_name=rebuilt.part_name,
                                tri_node_ids=rebuilt.tri_node_ids[np.asarray(keep)],
                                tri_elem_ids=rebuilt.tri_elem_ids[np.asarray(keep)],
                                tri_face_labels=list(
                                    np.asarray(rebuilt.tri_face_labels, dtype=object)[np.asarray(keep)]
                                ),
                            )
            ts = rebuilt
            part = asm.parts[ts.part_name]

    nid_unique, tri_idx = _unique_nodes_from_tris(ts)
    X3D = np.stack([part.nodes_xyz[int(n)] for n in nid_unique], axis=0).astype(np.float64)  # (Nu,3)

    # 2) Fit best-fit plane & project to 2D
    c, e1, e2, n = _fit_plane_basis(X3D)
    UV = _project_to_plane(X3D, c, e1, e2)  # (Nu,2)

    # 3) Evaluate displacement and take scalar deflection along normal
    eval_scope_key = (eval_scope or "surface").strip().lower()

    eval_subset = None
    target_part = full_structure_part or ts.part_name
    if target_part and getattr(asm, "parts", None):
        for name, part_obj in asm.parts.items():
            if target_part.lower() == name.lower():
                nid_subset = np.array(sorted(part_obj.nodes_xyz.keys()), dtype=np.int64)
                xyz_subset = (
                    np.stack([part_obj.nodes_xyz[int(n)] for n in nid_subset], axis=0)
                    .astype(np.float64)
                )
                eval_subset = (nid_subset, xyz_subset)
                break

    u_base, eval_meta = _eval_surface_or_assembly(
        u_fn,
        params,
        asm,
        nid_unique,
        X3D,
        eval_batch_size,
        eval_scope_key,
        eval_subset=eval_subset,
    )
    eval_scope_info = None
    if eval_meta:
        effective_mode = "assembly"
        eval_scope_info = {
            "mode": effective_mode,
            "requested": eval_meta.get("requested_scope", eval_scope_key),
            "global_node_count": int(eval_meta.get("global_nodes", np.array([])).shape[0]),
        }
    d_base = u_base @ n  # (Nu,) scalar deflection along global mirror normal

    # Optional barycentric refinement for smoother visualization
    applied_subdiv = max(0, int(refine_subdivisions or 0))
    max_pts = None if refine_max_points is None else int(refine_max_points)
    if applied_subdiv > 0 and max_pts is not None and max_pts > 0:
        per_tri = (applied_subdiv + 1) * (applied_subdiv + 2) // 2
        estimate = int(per_tri * tri_idx.shape[0])
        while estimate > max_pts and applied_subdiv > 0:
            applied_subdiv -= 1
            per_tri = (applied_subdiv + 1) * (applied_subdiv + 2) // 2
            estimate = int(per_tri * tri_idx.shape[0])
        if applied_subdiv < max(0, int(refine_subdivisions or 0)):
            print(
                "[viz] refine_subdivisions clipped to",
                applied_subdiv,
                f"to respect max_points={max_pts}.",
            )

    if applied_subdiv > 0:
        X_plot, UV_plot, tri_plot = _refine_surface_samples(X3D, UV, tri_idx, applied_subdiv)
        u_plot = _eval_displacement_batched(u_fn, params, X_plot, eval_batch_size)
        d_plot = u_plot @ n
    else:
        X_plot, UV_plot, tri_plot = X3D, UV, tri_idx
        u_plot, d_plot = u_base, d_base

    # Detect invalid predictions that will render as holes
    nonfinite_mask = ~np.isfinite(d_plot)
    tri_mask = None
    if np.any(nonfinite_mask):
        bad = int(nonfinite_mask.sum())
        frac = bad / float(d_plot.size)
        print(
            f"[viz] Warning: {bad}/{d_plot.size} deflection samples are NaN/Inf "
            f"({frac:.2%}); affected triangles will appear blank."
        )
        tri_mask = np.any(nonfinite_mask[tri_plot], axis=1)

    # 4) Triangulation in 2D
    tri = Triangulation(UV_plot[:, 0], UV_plot[:, 1], tri_plot)
    if tri_mask is not None and np.any(tri_mask):
        tri.set_mask(tri_mask)

    diag_result: Optional[BlankRegionDiagnostics] = None
    if render_surface and diagnose_blanks:
        diag_result = _diagnose_blank_regions(
            requested_subdiv=int(refine_subdivisions or 0),
            applied_subdiv=applied_subdiv,
            UV_plot=UV_plot,
            tri_plot=tri_plot,
            tri=tri,
            u_plot=u_plot,
            d_plot=d_plot,
            tri_mask=tri_mask,
        )

        coverage_threshold = 0.80
        if auto_fill_blanks and diag_result.coverage_ratio_envelope < coverage_threshold:
            tri = Triangulation(UV_plot[:, 0], UV_plot[:, 1])
            boundary_mask = _mask_tris_with_loops(tri, UV_plot, diag_result.boundary_loops)
            if np.any(boundary_mask):
                tri.set_mask(boundary_mask)

            if np.any(nonfinite_mask):
                invalid_mask = np.any(nonfinite_mask[tri.triangles], axis=1)
                if np.any(invalid_mask):
                    tri.set_mask(
                        invalid_mask if tri.mask is None else (tri.mask | invalid_mask)
                    )

            tri_plot = tri.triangles.astype(np.int32)
            tri_mask = tri.mask
            diag_result.notes.append("applied 2D re-triangulation to fill coverage gaps")
    if diag_out is not None:
        diag_out["blank_check"] = diag_result
        if eval_scope_info is not None:
            diag_out["eval_scope"] = eval_scope_info

    # 5) Draw surface map (optional)
    fig = ax = None
    if render_surface:
        fig, ax = plt.subplots(figsize=(7.8, 6.8), constrained_layout=True)

        style_key = (style or "smooth").strip().lower()
        if style_key not in {"smooth", "flat", "contour"}:
            style_key = "smooth"

        default_cmap = "turbo" if style_key in {"smooth", "flat"} else "coolwarm"
        cmap = cmap or default_cmap

        vmax = float(np.max(np.abs(d_plot))) + 1e-16 if symmetric else float(np.max(d_plot))
        vmin = -vmax if symmetric else float(np.min(d_plot))

        if style_key == "contour":
            contour_kwargs = {"levels": levels, "cmap": cmap}
            if symmetric:
                contour_kwargs.update(vmin=vmin, vmax=vmax)
            cs = ax.tricontourf(tri, d_plot, **contour_kwargs)
        else:
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            shading = "gouraud" if style_key == "smooth" else "flat"
            cs = ax.tripcolor(
                tri, d_plot, shading=shading, cmap=cmap, norm=norm, edgecolors="none"
            )

        if draw_wireframe:
            ax.triplot(tri, lw=0.35, color="#444444", alpha=0.45)

        cbar = fig.colorbar(cs, ax=ax, shrink=0.92, pad=0.02)
        cbar.set_label(f"Deflection along normal [{units}]")

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("u (best-fit plane)")
        ax.set_ylabel("v (best-fit plane)")

        if P_values is None:
            if isinstance(params, dict) and ("P" in params):
                P_values = tuple([float(x) for x in np.array(params["P"]).reshape(-1)])
        if P_values is not None and len(P_values) >= 3:
            P1, P2, P3 = P_values[0], P_values[1], P_values[2]
            title = f"{title_prefix}  |  P1={P1:.1f} N, P2={P2:.1f} N, P3={P3:.1f} N"
        else:
            title = title_prefix
        ax.set_title(title)

    # Optional assembly-wide displacement map (scatter)
    full_plot_path: Optional[Path] = None
    full_data_path: Optional[Path] = None
    if plot_full_structure and eval_meta:
        g_nid = eval_meta.get("global_nodes")
        g_xyz = eval_meta.get("global_xyz")
        u_all = eval_meta.get("u_all")
        if g_nid is not None and g_xyz is not None and u_all is not None:
            resolved_full = full_structure_out_path
            if isinstance(resolved_full, str):
                key = resolved_full.strip().lower()
                if key == "auto":
                    resolved_full = (
                        _with_new_stem(Path(out_path), Path(out_path).stem + "_assembly")
                        if out_path
                        else None
                    )
                elif key in {"", "none"}:
                    resolved_full = None
                else:
                    resolved_full = Path(resolved_full)
            if isinstance(resolved_full, Path):
                full_plot_path = resolved_full.with_suffix(".png")

            resolved_full_data = full_structure_data_out_path
            if isinstance(resolved_full_data, str):
                key = resolved_full_data.strip().lower()
                if key == "auto":
                    resolved_full_data = (
                        _with_new_stem(Path(out_path), Path(out_path).stem + "_assembly").with_suffix(".txt")
                        if out_path
                        else None
                    )
                elif key in {"", "none"}:
                    resolved_full_data = None
                else:
                    resolved_full_data = Path(resolved_full_data)

            uv_all = _project_to_plane(g_xyz, c, e1, e2)
            disp_mag = np.linalg.norm(u_all, axis=1)
            disp_max = float(np.max(disp_mag)) if disp_mag.size else 0.0
            norm_full = colors.Normalize(vmin=0.0, vmax=disp_max + 1e-16)
            fig_full, ax_full = plt.subplots(figsize=(7.8, 6.8), constrained_layout=True)
            sc = ax_full.scatter(
                uv_all[:, 0], uv_all[:, 1], c=disp_mag, cmap=cmap, norm=norm_full, s=6.0, alpha=0.9
            )
            cbar_full = fig_full.colorbar(sc, ax=ax_full, shrink=0.92, pad=0.02)
            cbar_full.set_label(f"Displacement magnitude [{units}]")
            ax_full.set_aspect("equal", adjustable="box")
            ax_full.set_xlabel("u (best-fit plane)")
            ax_full.set_ylabel("v (best-fit plane)")
            title_full = f"{title_prefix}  |  Assembly displacement magnitude"
            ax_full.set_title(title_full)

            if isinstance(resolved_full, Path):
                full_plot_path.parent.mkdir(parents=True, exist_ok=True)
                fig_full.savefig(full_plot_path, dpi=180)
                print(f"[viz] assembly displacement -> {full_plot_path}")
            if isinstance(resolved_full, Path) and show:
                plt.show()
            else:
                plt.close(fig_full)

            if isinstance(resolved_full_data, Path):
                full_data_path = resolved_full_data
                full_data_path.parent.mkdir(parents=True, exist_ok=True)
                header = [
                    "# Assembly-wide displacement samples exported by plot_mirror_deflection",
                    f"# units={units} surface={surface_key}",
                    f"# n_nodes={len(g_nid)}",  # type: ignore[arg-type]
                    "# columns: node_id x y z u_x u_y u_z |u| u_plane v_plane",
                ]
                with full_data_path.open("w", encoding="utf-8") as fp:
                    fp.write("\n".join(header) + "\n")
                    for idx, nid in enumerate(g_nid):
                        fp.write(
                            f"{int(nid):10d} "
                            f"{g_xyz[idx, 0]: .8f} {g_xyz[idx, 1]: .8f} {g_xyz[idx, 2]: .8f} "
                            f"{u_all[idx, 0]: .8f} {u_all[idx, 1]: .8f} {u_all[idx, 2]: .8f} "
                            f"{disp_mag[idx]: .8f} {uv_all[idx, 0]: .8f} {uv_all[idx, 1]: .8f}\n"
                        )

    # Save / show
    data_path: Optional[Path] = None
    if render_surface:
        resolved_data = data_out_path
        if isinstance(resolved_data, str):
            key = resolved_data.strip().lower()
            if key == "auto":
                resolved_data = Path(out_path).with_suffix(".txt") if out_path else None
            elif key in {"", "none"}:
                resolved_data = None
            else:
                resolved_data = Path(resolved_data)

        if isinstance(resolved_data, Path):
            data_path = resolved_data
            data_path.parent.mkdir(parents=True, exist_ok=True)
            header = [
                "# Mirror deflection samples exported by plot_mirror_deflection",
                f"# surface={surface_key} units={units}",
            ]
            if eval_scope_info is not None:
                header.append(
                    f"# eval_scope={eval_scope_info['mode']} global_node_count={eval_scope_info['global_node_count']}"
                )
            if P_values is not None:
                header.append(
                    "# preload=[" + ", ".join(f"{float(p):.6f}" for p in P_values[:3]) + "] N"
                )
            header.append(f"# refine_subdivisions_applied={applied_subdiv}")
            header.append("# note: exported samples correspond to original FE nodes.")
            header.append(
                "# columns: node_id x y z u_x u_y u_z deflection_normal u_plane v_plane"
            )
            with data_path.open("w", encoding="utf-8") as fp:
                fp.write("\n".join(header) + "\n")
                for idx, nid in enumerate(nid_unique):
                    fp.write(
                        f"{int(nid):10d} "
                        f"{X3D[idx, 0]: .8f} {X3D[idx, 1]: .8f} {X3D[idx, 2]: .8f} "
                        f"{u_base[idx, 0]: .8f} {u_base[idx, 1]: .8f} {u_base[idx, 2]: .8f} "
                        f"{d_base[idx]: .8f} {UV[idx, 0]: .8f} {UV[idx, 1]: .8f}\n"
                    )

        mesh_path: Optional[Path] = None
        resolved_mesh = surface_mesh_out_path
        if isinstance(resolved_mesh, str):
            mesh_key = resolved_mesh.strip().lower()
            if mesh_key == "auto":
                mesh_path = (
                    Path(out_path).with_name(Path(out_path).stem + "_surface.ply") if out_path else None
                )
            elif mesh_key in {"", "none"}:
                mesh_path = None
            else:
                mesh_path = Path(resolved_mesh)
        elif isinstance(resolved_mesh, Path):
            mesh_path = resolved_mesh

        if mesh_path is not None:
            _export_surface_mesh(mesh_path, nid_unique, X3D, tri_idx)
            print(f"[viz] surface mesh -> {mesh_path}")

        if out_path and fig is not None:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=180)

    if show and fig is not None:
        plt.show()
    else:
        plt.close(fig) if fig is not None else None
    return fig, ax, (str(data_path) if data_path is not None else None)


# -----------------------------
# Convenience wrapper
# -----------------------------

def plot_mirror_deflection_by_name(asm: AssemblyModel,
                                   mirror_surface_bare_name: str,
                                   *args, **kwargs):
    """
    Helper that searches for a surface key containing the bare name (case-insensitive),
    then calls plot_mirror_deflection. Useful when the exact key is long or namespaced.

    Example:
        plot_mirror_deflection_by_name(asm, "MIRROR up", u_fn, params, P_values=(300,500,700), out_path="out.png")
    """
    key = None
    low = mirror_surface_bare_name.strip().lower()
    for k, s in asm.surfaces.items():
        if low in k.lower() or low == s.name.strip().lower():
            key = k
            break
    if key is None:
        raise KeyError(f"[mirror_viz] Cannot find surface containing name '{mirror_surface_bare_name}'.")
    return plot_mirror_deflection(asm, key, *args, **kwargs)


# -----------------------------
# Minimal smoke test
# -----------------------------
if __name__ == "__main__":
    # This test assumes you have data/shuangfan.inp and a surface whose key contains "MIRROR" and "up".
    import os
    from inp_io.inp_parser import load_inp

    inp = os.environ.get("INP_PATH", "data/shuangfan.inp")
    if not os.path.exists(inp):
        print("[mirror_viz] Set INP_PATH or place your INP at data/shuangfan.inp to run the smoke test.")
        exit(0)

    asm = load_inp(inp)

    # dummy PINN forward: small bowl-shaped w-displacement
    import tensorflow as tf
    def u_fn(X, params=None):
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        uz = 1e-3 * (X[:, 0:1]**2 + X[:, 1:2]**2)
        return tf.concat([tf.zeros_like(uz), tf.zeros_like(uz), uz], axis=1)

    # preload for title
    P = (300.0, 500.0, 700.0)
    try:
        plot_mirror_deflection_by_name(
            asm, "MIRROR up",
            u_fn=u_fn,
            params={"P": tf.constant(P, dtype=tf.float32)},
            P_values=P,
            out_path="outputs/mirror_smoketest.png",
            title_prefix="Mirror Deflection (smoke test)",
            show=False
        )
        print("[mirror_viz] Saved outputs/mirror_smoketest.png")
    except Exception as e:
        print("[mirror_viz] Smoke test failed:", e)
