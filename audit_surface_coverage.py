#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit an Abaqus surface to check whether the triangulated mesh fully covers
its projected footprint.

Use this script to confirm whether a surface extracted from an INP file is
already incomplete (e.g., missing faces or overly sparse), which would explain
blank regions in downstream deflection plots. If the surface is complete here
but coverage remains low during training visualization, that indicates faces are
being dropped later in the pipeline.
"""
import argparse
import os
import sys
from typing import Dict, Tuple

import numpy as np

ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from inp_io.inp_parser import load_inp  # type: ignore
from mesh.surface_utils import (  # type: ignore
    _fetch_xyz,
    compute_tri_geometry,
    resolve_surface_to_tris,
)
from viz.mirror_viz import (  # type: ignore
    _collect_boundary_loops,
    _convex_hull_area,
    _fit_plane_basis,
    _loop_area,
    _project_to_plane,
    _triangle_area_sum,
    _unique_nodes_from_tris,
)


def _default_inp_path() -> str:
    cfg_path = os.path.join(ROOT, "config.yaml")
    if os.path.exists(cfg_path):
        try:
            import yaml  # type: ignore

            cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8")) or {}
            if isinstance(cfg, dict) and cfg.get("inp_path"):
                return str(cfg["inp_path"])
        except Exception:
            pass
    return ""


def _build_elem_type_index(asm) -> Dict[int, str]:
    idx: Dict[int, str] = {}
    for pname, part in asm.parts.items():
        for blk in part.element_blocks:
            et = (blk.elem_type or "").upper()
            for eid in blk.elem_ids:
                idx[int(eid)] = et
    return idx


def _coverage_stats(UV: np.ndarray, tri_idx: np.ndarray) -> Tuple[float, float, int]:
    tri_area = _triangle_area_sum(UV, tri_idx)
    hull_area = _convex_hull_area(UV)
    coverage_hull = 0.0 if hull_area <= 0 else min(1.0, tri_area / hull_area)

    loops = _collect_boundary_loops(tri_idx)
    loop_areas = [abs(_loop_area(UV, loop)) for loop in loops]
    envelope_area = float(sum(loop_areas)) if loop_areas else hull_area
    if loop_areas:
        max_idx = int(np.argmax(loop_areas))
        outer_area = loop_areas[max_idx]
        hole_area = sum(loop_areas[:max_idx] + loop_areas[max_idx + 1 :])
        envelope_area = max(outer_area - hole_area, 0.0)

    coverage_env = 0.0 if envelope_area <= 0 else min(1.0, tri_area / envelope_area)
    return coverage_hull, coverage_env, len(loops)


def audit_surface(inp_path: str, surface_key: str, coverage_threshold: float = 0.8) -> int:
    if not os.path.exists(inp_path):
        print(f"[error] INP file not found: {inp_path}")
        return 1

    print(f"[info] loading INP: {inp_path}")
    asm = load_inp(inp_path)

    try:
        ts = resolve_surface_to_tris(asm, surface_key)
    except Exception as e:
        print(f"[error] failed to resolve surface '{surface_key}': {e}")
        return 1

    # Unique nodes and projection
    nid_unique, tri_idx = _unique_nodes_from_tris(ts)
    XYZ = _fetch_xyz(asm, nid_unique)
    c, e1, e2, n = _fit_plane_basis(XYZ)
    UV = _project_to_plane(XYZ, c, e1, e2)

    coverage_hull, coverage_env, n_loops = _coverage_stats(UV, tri_idx)

    areas, _, _ = compute_tri_geometry(asm, ts)
    elem_type_idx = _build_elem_type_index(asm)
    elem_type_counts: Dict[str, int] = {}
    unknown_elems = 0
    for eid in ts.tri_elem_ids:
        et = elem_type_idx.get(int(eid))
        if et is None:
            unknown_elems += 1
            continue
        elem_type_counts[et] = elem_type_counts.get(et, 0) + 1

    print("\n=== Surface audit ===")
    print(f"surface key           : {surface_key}")
    print(f"triangles (count)     : {len(ts)}")
    print(f"unique nodes          : {len(nid_unique)}")
    print(f"triangle area stats   : min={areas.min():.4e}, mean={areas.mean():.4e}, max={areas.max():.4e}")
    print(f"element types used    : {', '.join([f'{k}: {v}' for k, v in sorted(elem_type_counts.items())]) or 'none'}")
    if unknown_elems:
        print(f"element ids w/o type  : {unknown_elems}")
    print(f"boundary loops        : {n_loops}")
    print(f"coverage vs hull      : {coverage_hull*100:6.2f}%")
    print(f"coverage vs envelope  : {coverage_env*100:6.2f}%")

    if coverage_env < coverage_threshold:
        print(
            f"[warn] coverage below threshold ({coverage_env*100:.2f}% < {coverage_threshold*100:.1f}%). "
            "Surface may be incomplete or too sparse."
        )
    else:
        print(f"[ok] coverage meets threshold ({coverage_env*100:.2f}% >= {coverage_threshold*100:.1f}%).")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Audit INP surface coverage before training.")
    parser.add_argument("--inp", dest="inp_path", type=str, default=None, help="Path to Abaqus INP file")
    parser.add_argument("--surface", dest="surface_key", type=str, default="MIRROR up", help="Surface name/key")
    parser.add_argument(
        "--coverage-threshold",
        dest="coverage_threshold",
        type=float,
        default=0.8,
        help="Warn if coverage vs boundary envelope falls below this fraction",
    )
    args = parser.parse_args()

    inp_path = args.inp_path or _default_inp_path()
    if not inp_path:
        print("[error] no INP path provided and config.yaml missing inp_path")
        return 1

    return audit_surface(inp_path, args.surface_key, args.coverage_threshold)


if __name__ == "__main__":
    raise SystemExit(main())
