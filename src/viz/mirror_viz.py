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
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib import colors

from inp_io.inp_parser import AssemblyModel
from mesh.surface_utils import TriSurface, resolve_surface_to_tris, _fetch_xyz


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
                           title_prefix: str = "Mirror Deflection",
                           units: str = "mm",
                           levels: int = 24,
                           symmetric: bool = True,
                           show: bool = False,
                           data_out_path: Optional[str] = "auto",
                           style: str = "smooth",
                           cmap: Optional[str] = None,
                           draw_wireframe: bool = False):
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
        style            : "smooth" to render a Gouraud-shaded tripcolor map
                           (Abaqus-like), "flat" for flat shading, or
                           "contour" to use tricontourf as in the legacy
                           implementation.
        cmap             : Optional matplotlib colormap name; defaults to
                           ``"turbo"`` for smooth/flat styles and
                           ``"coolwarm"`` for contour mode.
        draw_wireframe   : Whether to overlay triangle edges.

    Returns:
        (fig, ax, data_path)
    """
    # 1) Triangulate surface & collect unique nodes
    ts = resolve_surface_to_tris(asm, surface_key)
    part = asm.parts[ts.part_name]

    nid_unique, tri_idx = _unique_nodes_from_tris(ts)
    X3D = np.stack([part.nodes_xyz[int(n)] for n in nid_unique], axis=0).astype(np.float64)  # (Nu,3)

    # 2) Fit best-fit plane & project to 2D
    c, e1, e2, n = _fit_plane_basis(X3D)
    UV = _project_to_plane(X3D, c, e1, e2)  # (Nu,2)

    # 3) Evaluate displacement and take scalar deflection along normal
    #    NOTE: u_fn expects TF tensors; here we batch on all surface nodes.
    import tensorflow as tf
    Xtf = tf.convert_to_tensor(X3D, dtype=tf.float32)
    u = u_fn(Xtf, params)                          # (Nu,3) TF
    u_np = u.numpy().astype(np.float64)
    d = u_np @ n  # (Nu,) scalar deflection along global mirror normal

    # 4) Triangulation in 2D
    tri = Triangulation(UV[:, 0], UV[:, 1], tri_idx)

    # 5) Draw
    fig, ax = plt.subplots(figsize=(7.8, 6.8), constrained_layout=True)

    style_key = (style or "smooth").strip().lower()
    if style_key not in {"smooth", "flat", "contour"}:
        style_key = "smooth"

    default_cmap = "turbo" if style_key in {"smooth", "flat"} else "coolwarm"
    cmap = cmap or default_cmap

    vmax = float(np.max(np.abs(d))) + 1e-16 if symmetric else float(np.max(d))
    vmin = -vmax if symmetric else float(np.min(d))

    if style_key == "contour":
        contour_kwargs = {"levels": levels, "cmap": cmap}
        if symmetric:
            contour_kwargs.update(vmin=vmin, vmax=vmax)
        cs = ax.tricontourf(tri, d, **contour_kwargs)
    else:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        shading = "gouraud" if style_key == "smooth" else "flat"
        cs = ax.tripcolor(tri, d, shading=shading, cmap=cmap, norm=norm, edgecolors="none")

    if draw_wireframe:
        ax.triplot(tri, lw=0.35, color="#444444", alpha=0.45)

    cbar = fig.colorbar(cs, ax=ax, shrink=0.92, pad=0.02)
    cbar.set_label(f"Deflection along normal [{units}]")

    # Aspect & labels
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("u (best-fit plane)")
    ax.set_ylabel("v (best-fit plane)")

    # Title with preload values
    if P_values is None:
        if isinstance(params, dict) and ("P" in params):
            P_values = tuple([float(x) for x in np.array(params["P"]).reshape(-1)])
    if P_values is not None and len(P_values) >= 3:
        P1, P2, P3 = P_values[0], P_values[1], P_values[2]
        title = f"{title_prefix}  |  P1={P1:.1f} N, P2={P2:.1f} N, P3={P3:.1f} N"
    else:
        title = title_prefix
    ax.set_title(title)

    # Save / show
    data_path: Optional[Path] = None
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
        if P_values is not None:
            header.append(
                "# preload=[" + ", ".join(f"{float(p):.6f}" for p in P_values[:3]) + "] N"
            )
        header.append(
            "# columns: node_id x y z u_x u_y u_z deflection_normal u_plane v_plane"
        )
        with data_path.open("w", encoding="utf-8") as fp:
            fp.write("\n".join(header) + "\n")
            for idx, nid in enumerate(nid_unique):
                fp.write(
                    f"{int(nid):10d} "
                    f"{X3D[idx, 0]: .8f} {X3D[idx, 1]: .8f} {X3D[idx, 2]: .8f} "
                    f"{u_np[idx, 0]: .8f} {u_np[idx, 1]: .8f} {u_np[idx, 2]: .8f} "
                    f"{d[idx]: .8f} {UV[idx, 0]: .8f} {UV[idx, 1]: .8f}\n"
                )

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=180)
    if show:
        plt.show()
    else:
        plt.close(fig)
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
