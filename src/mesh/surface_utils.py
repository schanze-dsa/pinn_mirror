#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
surface_utils.py
----------------
Utilities to turn Abaqus element-based surfaces into triangulated surfaces,
and to sample/project points on them for contact modeling.

Key updates in this revision:
- Robust surface key normalization (quotes/space/case tolerant, auto "ASM::" prefix).
- Support both PART- and ASSEMBLY-scope *Surface, type=ELEMENT.
- Use AssemblyModel.expand_elset() for ELSET expansion (handles trailing _S# etc).
- Global element index (across all parts) so assembly-scope surfaces are resolved.
- Choose a reasonable part_name for TriSurface (owner if part-scope; otherwise the
  most frequent owning part of contributing elements); downstream helpers accept
  either PartMesh or AssemblyModel as coord provider.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Iterable
import numpy as np

# Parsed structures
from inp_io.inp_parser import AssemblyModel, PartMesh, SurfaceDef, SetDef, ElementBlock


# -----------------------------
# Surface key normalization
# -----------------------------

def _normalize_surface_key(asm: AssemblyModel, surface_key: str) -> str:
    """
    Try a variety of aliases for a user-provided surface key so we can find
    the actual key in asm.surfaces regardless of quotes/spacing/case or
    missing 'ASM::' prefix.
    """
    def dequote(s: str) -> str:
        return str(s).strip().strip('"').strip("'")

    raw = str(surface_key).strip()
    base = dequote(raw)
    base_nospace = base.replace(" ", "")

    candidates = [
        raw,
        base,
        f'"{base}"',
        f'ASM::"{base}"',
        f'ASM::{base}',
        base_nospace,
        f'"{base_nospace}"',
        f'ASM::"{base_nospace}"',
        f'ASM::{base_nospace}',
        base.lower(),
        base.upper(),
        f'ASM::"{base.lower()}"',
        f'ASM::"{base.upper()}"',
        f'ASM::{base.lower()}',
        f'ASM::{base.upper()}',
    ]
    seen, cands = set(), []
    for c in candidates:
        if c not in seen:
            seen.add(c); cands.append(c)

    for c in cands:
        if c in asm.surfaces:
            return c

    have = list(asm.surfaces.keys())[:20]
    raise KeyError(
        f"Surface '{surface_key}' not found in AssemblyModel.surfaces; "
        f"tried {cands[:6]} ...; have: {have}"
    )


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class TriSurface:
    """
    A triangulated surface living on a Part (or assembly).
    Triangles are stored as node-id triplets (not zero-based indices).
    """
    name: str
    part_name: str                    # if assembly-scope, best-guess major contributing part or "_ASM_"
    tri_node_ids: np.ndarray          # (T, 3) int64
    tri_elem_ids: np.ndarray          # (T,) int64
    tri_face_labels: List[str]        # length T (e.g., 'S2', ...)

    _areas: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _normals: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _centroids: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def __len__(self) -> int:
        return int(self.tri_node_ids.shape[0])


# -----------------------------
# Face maps
# -----------------------------

C3D8_FACES = {
    "S1": (1, 2, 3, 4),
    "S2": (5, 6, 7, 8),
    "S3": (1, 5, 8, 4),
    "S4": (2, 6, 7, 3),
    "S5": (1, 2, 6, 5),
    "S6": (4, 3, 7, 8),
}

C3D4_FACES = {
    "S1": (2, 3, 4),
    "S2": (1, 4, 3),
    "S3": (1, 2, 4),
    "S4": (1, 3, 2),
}

# 6-node wedge
C3D6_FACES = {
    "S1": (1, 2, 3),          # triangular faces (bottom)
    "S2": (4, 5, 6),          # top
    "S3": (1, 2, 5, 4),       # quads -> split into 2 tris
    "S4": (2, 3, 6, 5),
    "S5": (3, 1, 4, 6),
}

# 10-node tetra (corner + mid-edge nodes). We keep mid-nodes so coverage matches curved faces.
C3D10_FACES = {
    "S1": (1, 2, 3, 5, 6, 7),   # face opposite node 4
    "S2": (1, 4, 2, 8, 9, 5),
    "S3": (2, 4, 3, 9, 10, 6),
    "S4": (3, 4, 1, 10, 8, 7),
}

SUPPORTED_TYPES = {"C3D8", "C3D4", "C3D6", "C3D10"}


# -----------------------------
# ELSET expansion (fallback)
# -----------------------------

def _expand_elset_ids_fallback(setdef: SetDef) -> List[int]:
    """Simple expansion using SetDef.raw_lines; kept as a fallback."""
    out: List[int] = []
    for raw in setdef.raw_lines:
        toks = [t.strip() for t in raw.split(",") if t.strip()]
        ints = []
        ok = True
        for t in toks:
            try:
                ints.append(int(float(t)))
            except Exception:
                ok = False
                break
        if ok and len(ints) == 3 and ints[1] >= ints[0]:
            a, b, c = ints
            out.extend(list(range(a, b + 1, c)))
            continue
        for t in toks:
            try:
                out.append(int(float(t)))
            except Exception:
                pass
    return sorted(set(out))


# -----------------------------
# Main: resolve SurfaceDef -> TriSurface
# -----------------------------

def _emit_tris_from_face(face_nodes: Tuple[int, ...], tri_nodes: List[Tuple[int, int, int]]) -> int:
    """Triangulate faces with 3, 4, or 6 nodes (with mid-edge nodes)."""

    if len(face_nodes) == 3:
        tri_nodes.append((face_nodes[0], face_nodes[1], face_nodes[2]))
        return 1
    if len(face_nodes) == 4:
        tri_nodes.append((face_nodes[0], face_nodes[1], face_nodes[2]))
        tri_nodes.append((face_nodes[0], face_nodes[2], face_nodes[3]))
        return 2
    if len(face_nodes) == 6:
        a, b, c, ab, bc, ac = face_nodes
        tri_nodes.append((a, ab, ac))
        tri_nodes.append((ab, b, bc))
        tri_nodes.append((ac, bc, c))
        tri_nodes.append((ab, bc, ac))
        return 4
    return 0


def resolve_surface_to_tris(asm: AssemblyModel, surface_key: str, log_summary: bool = False) -> TriSurface:
    """
    Resolve a surface (part/assembly scope, ELEMENT) to a triangulated surface.
    """
    # 1) normalize key
    key = _normalize_surface_key(asm, surface_key)
    sdef: SurfaceDef = asm.surfaces[key]

    # 2) Build a global element index: eid -> (etype, conn, owner_part)
    elem_index: Dict[int, Tuple[str, List[int], str]] = {}
    for pname, part in asm.parts.items():
        for blk in part.element_blocks:
            et = (blk.elem_type or "").upper()
            for eid, conn in zip(blk.elem_ids, blk.connectivity):
                elem_index[int(eid)] = (et, conn, pname)

    tri_nodes: List[Tuple[int, int, int]] = []
    tri_eids: List[int] = []
    tri_labels: List[str] = []
    owner_votes: Dict[str, int] = {}

    skipped_types: Dict[str, int] = {}
    added_types: Dict[str, int] = {}
    missing_tokens: int = 0

    def _add_face_as_tris(eid: int, face_label: str, etype: str, conn: List[int], owner: str):
        et = etype.upper()
        if et not in SUPPORTED_TYPES:
            skipped_types[et] = skipped_types.get(et, 0) + 1
            return

        face: Optional[Tuple[int, ...]] = None
        if et == "C3D4":
            face = C3D4_FACES.get(face_label)
        elif et == "C3D8":
            face = C3D8_FACES.get(face_label)
        elif et == "C3D6":
            face = C3D6_FACES.get(face_label)
        elif et == "C3D10":
            face = C3D10_FACES.get(face_label)

        if not face:
            return
        node_ids = tuple(conn[i - 1] for i in face)
        n_tris = _emit_tris_from_face(node_ids, tri_nodes)
        if n_tris == 0:
            return
        tri_eids.extend([eid] * n_tris)
        tri_labels.extend([face_label] * n_tris)
        owner_votes[owner] = owner_votes.get(owner, 0) + 1
        added_types[et] = added_types.get(et, 0) + n_tris

    # 3) Iterate items
    for token, face in sdef.items:
        face_label = (face or "").upper()
        if face_label and not face_label.startswith("S"):
            # tolerate 's2' or weird cases
            face_label = f"S{''.join([ch for ch in face_label if ch.isdigit()])}" if any(ch.isdigit() for ch in face_label) else ""

        # direct element id?
        eid: Optional[int] = None
        try:
            eid = int(float(token))
        except Exception:
            eid = None

        if eid is not None:
            info = elem_index.get(eid)
            if not info:
                missing_tokens += 1
                continue
            etype, conn, owner = info
            _add_face_as_tris(eid, face_label, etype, conn, owner)
            continue

        # not an integer -> ELSET name
        eids: List[int] = []
        try:
            # preferred: parser's robust expander (handles trailing _S# and aliases)
            eids = list(asm.expand_elset(str(token)))
        except Exception:
            # fallback: direct lookup + local raw-line expander
            setdef = asm.elsets.get(str(token))
            if setdef:
                eids = _expand_elset_ids_fallback(setdef)
        if not eids:
            # try dequoted variant quickly
            t = str(token).strip().strip('"').strip("'")
            try:
                eids = list(asm.expand_elset(t))
            except Exception:
                pass

        for eid2 in eids:
            info = elem_index.get(int(eid2))
            if not info:
                missing_tokens += 1
                continue
            etype, conn, owner = info
            _add_face_as_tris(int(eid2), face_label, etype, conn, owner)

    if len(tri_nodes) == 0:
        raise ValueError(
            f"Surface '{surface_key}' resolved to 0 triangles. "
            f"Check its items/ELSETs/element types."
        )

    # 4) Decide part_name for TriSurface
    if sdef.scope == "part" and (sdef.owner in asm.parts):
        ts_part_name = sdef.owner
    else:
        # pick the most frequent owner among contributing elements
        ts_part_name = "_ASM_"
        if owner_votes:
            ts_part_name = max(owner_votes.items(), key=lambda kv: kv[1])[0]

    tri_node_ids = np.asarray(tri_nodes, dtype=np.int64)
    tri_elem_ids = np.asarray(tri_eids, dtype=np.int64)

    # emit a quick summary so users can see if unsupported element types caused coverage loss
    if log_summary and skipped_types:
        summary = ", ".join([f"{k}: {v}" for k, v in sorted(skipped_types.items())])
        print(f"[surface] '{sdef.name}' skipped unsupported element types -> {summary}")
    if log_summary and added_types:
        summary = ", ".join([f"{k}: {v} tris" for k, v in sorted(added_types.items())])
        print(f"[surface] '{sdef.name}' triangulated element faces -> {summary}")
    if log_summary and missing_tokens:
        print(
            f"[surface] '{sdef.name}' tokens not found in element index -> {missing_tokens} "
            "(check surface items/ELSET scope)"
        )

    # Optional: estimate how much of the owning part's boundary this surface covers
    def _enumerate_faces(etype: str, conn: Iterable[int]) -> List[Tuple[str, Tuple[int, ...]]]:
        et = etype.upper()
        face_map = None
        if et == "C3D4":
            face_map = C3D4_FACES
        elif et == "C3D8":
            face_map = C3D8_FACES
        elif et == "C3D6":
            face_map = C3D6_FACES
        elif et == "C3D10":
            face_map = C3D10_FACES
        if face_map is None:
            return []
        out: List[Tuple[str, Tuple[int, ...]]] = []
        for lbl, idxs in face_map.items():
            out.append((lbl, tuple(conn[i - 1] for i in idxs)))
        return out

    def _face_area(face_nodes: Tuple[int, ...], part_or_asm) -> float:
        if len(face_nodes) == 3:
            xyz = _fetch_xyz(part_or_asm, np.asarray(face_nodes))
            return 0.5 * np.linalg.norm(np.cross(xyz[1] - xyz[0], xyz[2] - xyz[0]))
        if len(face_nodes) == 4:
            a, b, c, d = face_nodes
            xyz = _fetch_xyz(part_or_asm, np.asarray([a, b, c, d]))
            area1 = 0.5 * np.linalg.norm(np.cross(xyz[1] - xyz[0], xyz[2] - xyz[0]))
            area2 = 0.5 * np.linalg.norm(np.cross(xyz[2] - xyz[0], xyz[3] - xyz[0]))
            return area1 + area2
        if len(face_nodes) == 6:
            a, b, c, ab, bc, ac = face_nodes
            xyz = _fetch_xyz(part_or_asm, np.asarray([a, b, c, ab, bc, ac]))
            # split into 4 tris consistent with _emit_tris_from_face
            def tri(i0, i1, i2):
                return 0.5 * np.linalg.norm(np.cross(xyz[i1] - xyz[i0], xyz[i2] - xyz[i0]))
            return tri(0, 3, 5) + tri(3, 1, 4) + tri(5, 4, 2) + tri(3, 4, 5)
        return 0.0

    if ts_part_name != "_ASM_" and ts_part_name in asm.parts:
        part = asm.parts[ts_part_name]
        boundary_faces: Dict[Tuple[int, ...], Tuple[str, str]] = {}
        face_hits: Dict[Tuple[int, ...], int] = {}
        # gather boundary faces (only supported types) by counting shared faces
        for blk in part.element_blocks:
            et = (blk.elem_type or "").upper()
            faces = []
            for conn in blk.connectivity:
                faces.extend(_enumerate_faces(et, conn))
            for lbl, nodes in faces:
                if not nodes:
                    continue
                key = tuple(sorted(nodes))
                boundary_faces[key] = (et, lbl)
                face_hits[key] = face_hits.get(key, 0) + 1

        # keep only faces that are unique (on the exterior)
        exterior_faces = {k: v for k, v in boundary_faces.items() if face_hits.get(k, 0) == 1}
        # total boundary area using supported faces
        boundary_area = 0.0
        for nodes in exterior_faces:
            boundary_area += _face_area(tuple(nodes), part)

        surface_area = compute_tri_geometry(part, TriSurface(
            name=sdef.name,
            part_name=ts_part_name,
            tri_node_ids=tri_node_ids,
            tri_elem_ids=tri_elem_ids,
            tri_face_labels=tri_labels,
        ))[0].sum()

        if log_summary and boundary_area > 0:
            coverage = 100.0 * surface_area / boundary_area
            print(
                f"[surface] '{sdef.name}' coverage vs outer boundary (part '{ts_part_name}', supported types): "
                f"{coverage:.2f}% (surface area={surface_area:.3f}, boundary area={boundary_area:.3f})"
            )
        elif log_summary:
            print(
                f"[surface] '{sdef.name}' boundary coverage check skipped (part '{ts_part_name}' has no supported faces)"
            )
    elif log_summary and ts_part_name == "_ASM_":
        print(
            f"[surface] '{sdef.name}' boundary coverage check skipped (assembly-scope mix of parts; run per-part check if needed)"
        )

    return TriSurface(
        name=sdef.name,
        part_name=ts_part_name,
        tri_node_ids=tri_node_ids,
        tri_elem_ids=tri_elem_ids,
        tri_face_labels=tri_labels,
    )


def triangulate_part_boundary(part, part_name: str, log_summary: bool = False) -> TriSurface:
    """Triangulate all exterior faces of a part into a TriSurface.

    This builds a mesh directly from supported element types (C3D4/6/8/10) and
    returns the exterior faces split into triangles, which can be used as a
    coverage-complete fallback when the INP surface set is incomplete.
    """

    tri_nodes: List[Tuple[int, int, int]] = []
    tri_eids: List[int] = []
    tri_labels: List[str] = []
    skipped: Dict[str, int] = {}
    added: Dict[str, int] = {}

    def _enumerate_faces(etype: str, conn: Iterable[int]) -> List[Tuple[str, Tuple[int, ...]]]:
        et = etype.upper()
        face_map = None
        if et == "C3D4":
            face_map = C3D4_FACES
        elif et == "C3D8":
            face_map = C3D8_FACES
        elif et == "C3D6":
            face_map = C3D6_FACES
        elif et == "C3D10":
            face_map = C3D10_FACES
        if face_map is None:
            return []
        out: List[Tuple[str, Tuple[int, ...]]] = []
        for lbl, idxs in face_map.items():
            out.append((lbl, tuple(conn[i - 1] for i in idxs)))
        return out

    face_hits: Dict[Tuple[int, ...], int] = {}
    face_payload: Dict[Tuple[int, ...], Tuple[int, str]] = {}

    for blk in part.element_blocks:
        et = (blk.elem_type or "").upper()
        faces = _enumerate_faces(et, [])  # seed to check support
        if not faces and et not in SUPPORTED_TYPES:
            skipped[et] = skipped.get(et, 0) + len(blk.connectivity)
            continue
        for eid, conn in zip(blk.elem_ids, blk.connectivity):
            for lbl, nodes in _enumerate_faces(et, conn):
                key = tuple(sorted(nodes))
                face_hits[key] = face_hits.get(key, 0) + 1
                face_payload[key] = (eid, lbl)

    exterior_faces = {k: v for k, v in face_payload.items() if face_hits.get(k, 0) == 1}

    for nodes, (eid, lbl) in exterior_faces.items():
        n_tris = _emit_tris_from_face(nodes, tri_nodes)
        if n_tris == 0:
            continue
        tri_eids.extend([eid] * n_tris)
        tri_labels.extend([lbl] * n_tris)
        added[(lbl, len(nodes))] = added.get((lbl, len(nodes)), 0) + n_tris

    if log_summary and skipped:
        summary = ", ".join([f"{k}: {v}" for k, v in sorted(skipped.items())])
        print(f"[surface] part '{part_name}' skipped unsupported element types -> {summary}")
    if log_summary and added:
        summary = ", ".join([f"{lbl}:{n}" for (lbl, _), n in sorted(added.items())])
        print(f"[surface] part '{part_name}' triangulated exterior faces -> {summary}")

    return TriSurface(
        name=f"{part_name}::boundary",
        part_name=part_name,
        tri_node_ids=np.asarray(tri_nodes, dtype=np.int64),
        tri_elem_ids=np.asarray(tri_eids, dtype=np.int64),
        tri_face_labels=tri_labels,
    )



# -----------------------------
# Geometry & sampling (coord provider can be PartMesh or AssemblyModel)
# -----------------------------

def _fetch_xyz(part_or_asm, node_ids: np.ndarray) -> np.ndarray:
    """
    Fetch (N,3) coords for node ids from either PartMesh.nodes_xyz or AssemblyModel.nodes.
    """
    if hasattr(part_or_asm, "nodes_xyz"):
        mapping = part_or_asm.nodes_xyz  # PartMesh
    else:
        mapping = part_or_asm.nodes      # AssemblyModel
    out = np.empty((node_ids.shape[0], 3), dtype=np.float64)
    for i, nid in enumerate(node_ids):
        out[i] = mapping[int(nid)]
    return out


def compute_tri_geometry(part_or_asm, ts: TriSurface) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-triangle (areas, normals(unit), centroids) and cache them.
    'part_or_asm' can be PartMesh or AssemblyModel.
    """
    if ts._areas is not None and ts._normals is not None and ts._centroids is not None:
        return ts._areas, ts._normals, ts._centroids

    tri_xyz = []
    for tri in ts.tri_node_ids:
        v = _fetch_xyz(part_or_asm, np.asarray(tri))
        tri_xyz.append(v)
    tri_xyz = np.asarray(tri_xyz)  # (T,3,3)

    e1 = tri_xyz[:, 1, :] - tri_xyz[:, 0, :]
    e2 = tri_xyz[:, 2, :] - tri_xyz[:, 0, :]
    cross = np.cross(e1, e2)
    areas = 0.5 * np.linalg.norm(cross, axis=1)

    normals = cross.copy()
    nz = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-16
    normals = normals / nz

    centroids = tri_xyz.mean(axis=1)

    ts._areas = areas
    ts._normals = normals
    ts._centroids = centroids
    return areas, normals, centroids


def sample_points_on_surface(part_or_asm, ts: TriSurface, n_points: int,
                             rng: Optional[np.random.Generator] = None
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Area-proportional sampling on triangulated surface.
    """
    if rng is None:
        rng = np.random.default_rng()

    areas, normals, centroids = compute_tri_geometry(part_or_asm, ts)
    probs = areas / (areas.sum() + 1e-16)

    tri_idx = rng.choice(len(ts), size=n_points, p=probs)

    r1 = np.sqrt(rng.random(n_points))
    r2 = rng.random(n_points)
    w0 = 1.0 - r1
    w1 = r1 * (1.0 - r2)
    w2 = r1 * r2
    bary = np.stack([w0, w1, w2], axis=1)

    X = np.empty((n_points, 3), dtype=np.float64)
    n = np.empty((n_points, 3), dtype=np.float64)
    for i, t in enumerate(tri_idx):
        tri_nodes = ts.tri_node_ids[int(t)]
        v = _fetch_xyz(part_or_asm, np.asarray(tri_nodes))
        X[i] = w0[i] * v[0] + w1[i] * v[1] + w2[i] * v[2]
        n[i] = normals[int(t)]
    return X, tri_idx.astype(np.int64), bary, n


def _closest_pt_on_triangle(P: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray):
    """
    Return (Q, dist2, bary) of exact closest point on triangle ABC to point P.
    """
    AB = B - A
    AC = C - A
    AP = P - A

    d1 = np.dot(AB, AP)
    d2 = np.dot(AC, AP)
    if d1 <= 0 and d2 <= 0:
        return A.copy(), np.sum((P - A) ** 2), np.array([1.0, 0.0, 0.0])

    BP = P - B
    d3 = np.dot(AB, BP)
    d4 = np.dot(AC, BP)
    if d3 >= 0 and d4 <= d3:
        return B.copy(), np.sum((P - B) ** 2), np.array([0.0, 1.0, 0.0])

    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        v = d1 / (d1 - d3)
        Q = A + v * AB
        return Q, np.sum((P - Q) ** 2), np.array([1 - v, v, 0.0])

    CP = P - C
    d5 = np.dot(AB, CP)
    d6 = np.dot(AC, CP)
    if d6 >= 0 and d5 <= d6:
        return C.copy(), np.sum((P - C) ** 2), np.array([0.0, 0.0, 1.0])

    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        w = d2 / (d2 - d6)
        Q = A + w * AC
        return Q, np.sum((P - Q) ** 2), np.array([1 - w, 0.0, w])

    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        Q = B + w * (C - B)
        return Q, np.sum((P - Q) ** 2), np.array([0.0, 1 - w, w])

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    u = 1.0 - v - w
    Q = u * A + v * B + w * C
    return Q, np.sum((P - Q) ** 2), np.array([u, v, w])


def project_points_onto_surface(part_or_asm, ts: TriSurface, Q: np.ndarray,
                                prefilter_k: int = 8, chunk: int = 4096
                                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Project query points Q (M,3) to closest points on the surface.
    """
    areas, normals, centroids = compute_tri_geometry(part_or_asm, ts)

    M = Q.shape[0]
    T = centroids.shape[0]
    Xp = np.empty((M, 3), dtype=np.float64)
    n = np.empty((M, 3), dtype=np.float64)
    idx = np.empty((M,), dtype=np.int64)
    dist = np.empty((M,), dtype=np.float64)

    tri_vertices = []
    for tri in ts.tri_node_ids:
        tri_vertices.append(_fetch_xyz(part_or_asm, np.asarray(tri)))
    tri_vertices = np.asarray(tri_vertices)  # (T,3,3)

    for s in range(0, M, chunk):
        e = min(M, s + chunk)
        Qc = Q[s:e]
        d2 = np.sum((Qc[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        k = min(prefilter_k, max(1, T - 1))
        cand_idx = np.argpartition(d2, kth=k, axis=1)[:, :k]

        for i, q in enumerate(Qc):
            best_d2 = float("inf")
            best_t = 0
            best_Q = None
            for t in cand_idx[i]:
                A, B, C = tri_vertices[int(t)]
                Qproj, d2t, _ = _closest_pt_on_triangle(q, A, B, C)
                if d2t < best_d2:
                    best_d2 = d2t
                    best_t = int(t)
                    best_Q = Qproj
            Xp[s + i] = best_Q
            n[s + i]  = normals[best_t]
            idx[s + i] = best_t
            dist[s + i] = np.sqrt(best_d2)

    return Xp, n, idx, dist


# -----------------------------
# Convenience
# -----------------------------

def _coord_provider_for_ts(asm: AssemblyModel, ts: TriSurface):
    """
    Return a coord provider that works with helpers above:
    - PartMesh if ts.part_name is a known part;
    - otherwise AssemblyModel (uses asm.nodes).
    """
    if ts.part_name in asm.parts:
        return asm.parts[ts.part_name]
    return asm  # assembly-level coords


def build_contact_surfaces(asm: AssemblyModel, slave_key: str, master_key: str
                           ) -> Tuple[object, TriSurface, object, TriSurface]:
    """
    Resolve two surfaces and return (slave_provider, slave_ts, master_provider, master_ts).
    Providers are either PartMesh or AssemblyModel; downstream helpers accept both.
    """
    ts_slave = resolve_surface_to_tris(asm, slave_key)
    ts_master = resolve_surface_to_tris(asm, master_key)
    part_slave = _coord_provider_for_ts(asm, ts_slave)
    part_master = _coord_provider_for_ts(asm, ts_master)
    return part_slave, ts_slave, part_master, ts_master


# -----------------------------
# Quick test (optional)
# -----------------------------

if __name__ == "__main__":
    # Minimal smoke test (requires inp_parser + a data path)
    from inp_io.inp_parser import load_inp
    import os
    inp = os.environ.get("INP_PATH", "data/shuangfan.inp")
    asm = load_inp(inp)

    # pick first surface
    key = next(iter(asm.surfaces.keys()), None)
    if key is None:
        print("No surfaces found.")
        exit(0)

    ts = resolve_surface_to_tris(asm, key)
    provider = _coord_provider_for_ts(asm, ts)
    areas, normals, centroids = compute_tri_geometry(provider, ts)
    print(f"[debug] Surface '{key}' -> {len(ts)} tris, area sum = {areas.sum():.3e}")

    X, tri_idx, bary, n = sample_points_on_surface(provider, ts, 8)
    print("[debug] Sampled points (first 3):\n", X[:3])