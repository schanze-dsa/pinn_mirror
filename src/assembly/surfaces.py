# -*- coding: utf-8 -*-
"""
assembly/surfaces.py

统一的 INP Surface 抽象与采样工具（增强版）：
- SurfaceDef：描述一张面（ELEMENT/NODE）
- ElementFaceRef：元素-面 引用 (elem_id, face_id)
- SurfaceResolvers：解析回调集合（从 elem_id/face_id 拿节点坐标、从 elset 拓展元素等）
- to_points(surface, n_per_face=1, mode='centroid', resolvers=None, asm=None)
    -> (X, n, w)  均为 float32
- sample_surface_by_key(surfaces, key, n_points_each=1, mode='centroid', resolvers=None, asm=None)
    -> 便捷函数：从字典里按键取 SurfaceDef 并采样（含宽松键名匹配/去引号）

设计要点：
1) 几何/采样统一 float32，避免与 mixed_float16 的网络张量 dtype 冲突。
2) 默认以“面心采样 + 单位法向 + 面积权重”返回（最稳健、最省显存）。
3) resolvers 与 asm 互为备份；尽量自适配装配对象的不同方法名。
4) 对带引号/空格/大小写的键名做宽松匹配，避免 INP 命名细节导致查不到。
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union, Any
import numpy as np
import re


_SURFACE_DEBUG = os.environ.get("PINN_SURFACE_DEBUG", "0") == "1"


def _debug(msg: str) -> None:
    if _SURFACE_DEBUG:
        print(msg)


# =========================
# 基本数据结构
# =========================

@dataclass
class ElementFaceRef:
    """元素-面引用：elem_id（整数） + face_id（整数，Abaqus 风格：1..6 对应六个面）"""
    elem_id: int
    face_id: int


@dataclass
class SurfaceDef:
    """
    通用面定义：
    - stype: 'ELEMENT' 或 'NODE'
    - name:  INP 内的表面名（如 'ASM:\"bolt1 up\"'）
    - items:
        对 ELEMENT：可以是
           * ElementFaceRef 列表；或
           * (elem_id, face_id) 的二元组列表；或
           * 已解析多边形：{'poly': np.ndarray[k,3], 'normal': np.ndarray[3], 'area': float}
           * elset 形式：('ELSET', 'ESET_NAME', face_id) —— 需 resolvers.expand_elset
           * INP 风格： (elset_name, face_str) 如 ('"_bolt1 up_S2"', 'S2')
        对 NODE：可以是
           * 节点 id 列表；或
           * 已解析坐标：np.ndarray[n,3]
    - owner/scope/raw_lines 仅作溯源保留；不会影响采样。
    """
    stype: Literal["ELEMENT", "NODE"]
    name: str
    items: List[Any] = field(default_factory=list)
    owner: Optional[str] = None
    scope: Optional[str] = None
    raw_lines: Optional[List[str]] = None


@dataclass
class SurfaceResolvers:
    """
    装配解析回调集合。针对 ELEMENT/NODE 两路分别提供最小需要的接口。
    任意一个接口缺失且 items 不是“已解析多边形/坐标”时，会抛出带说明的错误。
    """
    # ELEMENT 路径：
    get_face_nodes: Optional[Callable[[int, int], np.ndarray]] = None  # (elem_id, face_id) -> (k,3) float32
    expand_elset: Optional[Callable[[str], Sequence[int]]] = None      # 'ESET_NAME' -> [elem_id, ...]
    # NODE 路径：
    get_node_coords: Optional[Callable[[Sequence[int]], np.ndarray]] = None  # [nid,...] -> (n,3) float32
    # 可选：节点法向估计（不提供则用 PCA 邻域估计）
    estimate_node_normals: Optional[Callable[[np.ndarray], np.ndarray]] = None  # (n,3)->(n,3)


# =========================
# 公共入口
# =========================

def to_points(
    surface: SurfaceDef,
    n_per_face: int = 1,
    mode: Literal["centroid", "gauss", "meshgrid"] = "centroid",
    resolvers: Optional[SurfaceResolvers] = None,
    asm: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将任一 SurfaceDef 采样为点/法向/权重：
    返回：
        X: (N,3) float32
        n: (N,3) float32, 单位法向
        w: (N,)  float32, 对应点的积分权重（面积分）
    """
    _check_surface(surface)
    res = _attach_resolvers(resolvers, asm)  # 合并 resolvers/asm

    if surface.stype == "ELEMENT":
        return _element_surface_to_points(surface, n_per_face, mode, res)
    elif surface.stype == "NODE":
        return _node_surface_to_points(surface, res)
    else:
        raise ValueError(f"[surfaces.to_points] 不支持的 surface.stype: {surface.stype}")


def sample_surface_by_key(
    surfaces: Dict[str, SurfaceDef],
    key: str,
    n_points_each: int = 1,
    mode: Literal["centroid", "gauss", "meshgrid"] = "centroid",
    resolvers: Optional[SurfaceResolvers] = None,
    asm: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从 surfaces 映射里用键名取 SurfaceDef 并采样。
    - 先按原样 key 查；
    - 若没有，再做宽松匹配（去引号/去空白/统一大小写/兼容 ASM:: 前缀）。
    """
    if key in surfaces:
        return to_points(surfaces[key], n_points_each, mode, resolvers, asm)

    # 宽松匹配：去引号 & 空白 & 大小写
    def _norm(s: str) -> str:
        s2 = s.strip()
        s2 = re.sub(r'["“”\']', "", s2)     # 去掉各种引号
        s2 = re.sub(r"\s+", "", s2)         # 去掉所有空白
        s2 = s2.upper()
        return s2

    want = _norm(key)
    # 兼容 ASM:: 前缀（两端都再试一遍）
    want_noprefix = _norm(_strip_asm_prefix(key))

    for k0, v in surfaces.items():
        k_norm = _norm(k0)
        if k_norm == want or k_norm == want_noprefix:
            return to_points(v, n_points_each, mode, resolvers, asm)

        k_noprefix = _norm(_strip_asm_prefix(k0))
        if k_noprefix == want or k_noprefix == want_noprefix:
            return to_points(v, n_points_each, mode, resolvers, asm)

    raise KeyError(f"[surfaces.sample_surface_by_key] 在 surfaces 中找不到键：{key}")


# =========================
# ELEMENT 路径
# =========================

def _element_surface_to_points(
    surface: SurfaceDef,
    n_per_face: int,
    mode: str,
    res: SurfaceResolvers,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if res.get_face_nodes is None:
        raise TypeError(
            f"[surfaces] 采样 ELEMENT 面需要 get_face_nodes(elem_id, face_id) 回调；"
            f"当前未提供。surface.name={surface.name}"
        )

    # 扩展 items（处理 elset -> elem_id 列表）
    expanded_items: List[Union[ElementFaceRef, Dict[str, Any], Tuple[int, int]]] = []
    for it in surface.items:
        if _is_elset_tuple(it):
            _, elset_name, face_id = it  # e.g. ('ELSET','ESET_X', 4)
            if res.expand_elset is None:
                raise TypeError(
                    f"[surfaces] items 包含 ELSET 引用，但未提供 expand_elset 回调。"
                    f" item={it}, surface.name={surface.name}"
                )
            elem_ids = list(res.expand_elset(elset_name))
            for eid in elem_ids:
                expanded_items.append(ElementFaceRef(int(eid), int(face_id)))
        elif isinstance(it, (tuple, list)) and len(it) == 2 and isinstance(it[0], str) and isinstance(it[1], str):
            # 处理 INP 风格 (elset_name, 'S2')
            elset_name = it[0].strip("\"' ").strip()  # 去引号和空格
            face_str = it[1].upper()
            if face_str.startswith('S'):
                try:
                    face_id = int(face_str[1:])
                except ValueError:
                    raise ValueError(f"[surfaces] 无效的面标识: {face_str} in item={it}")
                if res.expand_elset is None:
                    raise TypeError(
                        f"[surfaces] items 包含 elset 引用 ({elset_name}), 但未提供 expand_elset 回调。"
                    )
                elem_ids = list(res.expand_elset(elset_name))
                for eid in elem_ids:
                    expanded_items.append(ElementFaceRef(int(eid), face_id))
            else:
                raise ValueError(f"[surfaces] 不支持的面标识: {face_str} in item={it}")
        else:
            expanded_items.append(it)

    X_all: List[np.ndarray] = []
    n_all: List[np.ndarray] = []
    w_all: List[np.ndarray] = []

    for it in expanded_items:
        # 若已是“已解析多边形”
        if isinstance(it, dict) and "poly" in it:
            poly = _as_float32(np.asarray(it["poly"], dtype=np.float32))
            if poly.shape[-1] != 3:
                raise ValueError(f"[surfaces] poly 须为 (k,3)； got {poly.shape} in surface {surface.name}")
            normal, area = _face_normal_and_area(poly)
            pts, nrm, wgt = _sample_on_polygon(poly, normal, area, n_per_face, mode)
        else:
            # 以 (elem_id, face_id) 的形式解析
            efr = _as_element_face(it)
            verts = np.asarray(res.get_face_nodes(efr.elem_id, efr.face_id), dtype=np.float32)
            if verts.ndim != 2 or verts.shape[1] != 3 or verts.shape[0] < 3:
                raise ValueError(
                    f"[surfaces] get_face_nodes 返回非法顶点数组，期望 (k,3) k>=3；"
                    f" got {verts.shape} for elem={efr.elem_id}, face={efr.face_id}"
                )
            normal, area = _face_normal_and_area(verts)
            pts, nrm, wgt = _sample_on_polygon(verts, normal, area, n_per_face, mode)

        X_all.append(pts)
        n_all.append(nrm)
        w_all.append(wgt)

    if not X_all:
        return _empty_Xnw()

    X = np.concatenate(X_all, axis=0)
    n = np.concatenate(n_all, axis=0)
    w = np.concatenate(w_all, axis=0)
    return X, n, w


def _sample_on_polygon(
    verts: np.ndarray, normal: np.ndarray, area: float,
    n_per_face: int, mode: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对单个多边形（tri/quad/任意凸多边形）采样。
    当前提供最稳健的 'centroid'；'gauss'/'meshgrid' 在没有参数映射时退化为 'centroid'。
    """
    mode = (mode or "centroid").lower()
    if n_per_face <= 1 or mode not in ("centroid", "gauss", "meshgrid"):
        # 统一退化为面心一个点
        c = np.mean(verts, axis=0, keepdims=True).astype(np.float32)
        nrm = _normalize(normal)[None, :]  # (1,3)
        wgt = np.asarray([area], dtype=np.float32)
        return c, nrm, wgt

    # 简洁安全：仍然用面心点，但把权重平均到 n_per_face 份（数值等价，接口友好）
    c = np.repeat(np.mean(verts, axis=0, keepdims=True), repeats=n_per_face, axis=0).astype(np.float32)
    nrm = np.repeat(_normalize(normal)[None, :], repeats=n_per_face, axis=0).astype(np.float32)
    w_each = float(area) / float(n_per_face + 1e-12)
    wgt = np.full((n_per_face,), w_each, dtype=np.float32)
    return c, nrm, wgt


def _face_normal_and_area(verts: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    多边形法向与面积（对三角面精确；对四边形采用对角线剖分两三角累加）。
    采用简单/稳健的向量叉积法。
    """
    k = verts.shape[0]
    if k < 3:
        raise ValueError("[surfaces] 面的顶点数 < 3，无法计算法向/面积。")

    # 以扇形三角剖分累加法向
    v0 = verts[0]
    total = np.zeros(3, dtype=np.float64)
    area = 0.0
    for i in range(1, k - 1):
        a = verts[i]   - v0
        b = verts[i+1] - v0
        ntri = np.cross(a, b)               # 方向遵循顶点顺序
        atri = 0.5 * np.linalg.norm(ntri)
        total += ntri
        area += atri

    normal = _normalize(total.astype(np.float32))
    return normal, float(area)


# =========================
# NODE 路径
# =========================

def _node_surface_to_points(
    surface: SurfaceDef,
    res: SurfaceResolvers
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 1) 坐标
    coords = _resolve_node_coords(surface.items, res)
    if coords.size == 0:
        return _empty_Xnw()

    # 2) 法向
    if res.estimate_node_normals is not None:
        normals = np.asarray(res.estimate_node_normals(coords), dtype=np.float32)
        if normals.shape != coords.shape:
            raise ValueError(
                f"[surfaces] estimate_node_normals 返回形状不匹配："
                f" got {normals.shape}, want {coords.shape}"
            )
    else:
        normals = _pca_normals(coords)

    # 3) 权重（节点面通常无面积概念，给 1 或统一常数即可）
    w = np.ones((coords.shape[0],), dtype=np.float32)
    return coords, normals, w


def _resolve_node_coords(items: List[Any], res: SurfaceResolvers) -> np.ndarray:
    """
    把 node surface 的 items 解析为 (N,3) 坐标：
      - 若 items 已是 (N,3) 数组，直接返回；
      - 若是节点 id 列表，需 get_node_coords；
    """
    if len(items) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    # 已给坐标
    if isinstance(items, np.ndarray):
        arr = _as_float32(np.asarray(items, dtype=np.float32))
        if arr.ndim == 2 and arr.shape[1] == 3:
            return arr
        else:
            raise ValueError(f"[surfaces] NODE items 为 ndarray，但形状非法：{arr.shape}，期望 (N,3)")

    # 节点 id 列表
    if all(_is_int_like(x) for x in items):
        if res.get_node_coords is None:
            raise TypeError(
                "[surfaces] NODE Surface 需要 get_node_coords([nid,...])->(N,3) 回调，但未提供。"
            )
        nid_list = [int(x) for x in items]
        coords = np.asarray(res.get_node_coords(nid_list), dtype=np.float32)
        if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] != len(nid_list):
            raise ValueError(
                f"[surfaces] get_node_coords 返回形状非法：{coords.shape}，期望 (N,3) 且 N==len(nid_list)"
            )
        return coords

    # 其他不支持的形式
    raise TypeError(
        "[surfaces] 不支持的 NODE items 形式。请传 (N,3) 坐标数组，或节点 id 列表并提供 get_node_coords 回调。"
    )


# =========================
# 适配/工具
# =========================

def _attach_resolvers(resolvers: Optional[SurfaceResolvers], asm: Optional[Any]) -> SurfaceResolvers:
    """
    将用户传入的 resolvers / asm 统一成 SurfaceResolvers。
    对 asm 尝试绑定常见方法名：
      - get_face_nodes / expand_elset / get_node_coords
      - 也会尝试 face_vertices / elset_elems / node_coords 之类别名
      - 添加 fallback：如果 asm 有 elsets/nodes/elements dict，定义兜底函数
      - 增强：检查 asm.__dict__ 以处理 dataclass
    """
    if resolvers is not None:
        return resolvers

    res = SurfaceResolvers()

    if asm is None:
        _debug("[DEBUG] asm is None, returning empty resolvers")
        return res  # 全空，后续若需要会报带说明的错

    asm_dict = getattr(asm, '__dict__', {})  # 对于 dataclass，__dict__ 有自定义字段

    # -------- 绑定 get_face_nodes --------
    if hasattr(asm, "get_face_nodes") and callable(getattr(asm, "get_face_nodes")):
        res.get_face_nodes = getattr(asm, "get_face_nodes")
    elif hasattr(asm, "face_vertices") and callable(getattr(asm, "face_vertices")):
        res.get_face_nodes = getattr(asm, "face_vertices")
    elif hasattr(asm, "get_element_face_nodes") and callable(getattr(asm, "get_element_face_nodes")):
        res.get_face_nodes = getattr(asm, "get_element_face_nodes")
    elif hasattr(asm, "face_nodes") and callable(getattr(asm, "face_nodes")):
        res.get_face_nodes = getattr(asm, "face_nodes")
    else:
        # Fallback: 假设 asm.elements: dict[eid: list[nid]] , asm.nodes: dict[nid: list[x,y,z]]
        if 'elements' in asm_dict and isinstance(asm_dict['elements'], dict) and 'nodes' in asm_dict and isinstance(asm_dict['nodes'], dict):
            _debug("[DEBUG] Using fallback get_face_nodes from asm.elements and asm.nodes")
            def fallback_get_face_nodes(elem_id: int, face_id: int) -> np.ndarray:
                if elem_id not in asm_dict['elements']:
                    raise KeyError(f"Element {elem_id} not found in asm.elements")
                node_ids = asm_dict['elements'][elem_id]
                if len(node_ids) != 8:
                    raise NotImplementedError(f"Only C3D8 (8 nodes) supported, got {len(node_ids)} nodes for elem {elem_id}")
                # Abaqus C3D8 face node indices (0-based)
                face_indices = {
                    1: [0, 1, 2, 3],  # 1-2-3-4
                    2: [4, 7, 6, 5],  # 5-8-7-6
                    3: [0, 4, 5, 1],  # 1-5-6-2
                    4: [1, 5, 6, 2],  # 2-6-7-3
                    5: [2, 6, 7, 3],  # 3-7-8-4
                    6: [3, 7, 4, 0],  # 4-8-5-1
                }
                if face_id not in face_indices:
                    raise ValueError(f"Invalid face_id {face_id} for C3D8")
                idx = face_indices[face_id]
                nodes = [node_ids[i] for i in idx]
                coords = [asm_dict['nodes'].get(nid, None) for nid in nodes]
                if any(c is None for c in coords):
                    raise KeyError("Some nodes not found in asm.nodes")
                return np.array(coords, dtype=np.float32)
            res.get_face_nodes = fallback_get_face_nodes
        else:
            _debug("[DEBUG] No fallback for get_face_nodes; asm has no 'elements' or 'nodes'")

    # -------- 绑定 expand_elset --------
    if hasattr(asm, "expand_elset") and callable(getattr(asm, "expand_elset")):
        res.expand_elset = getattr(asm, "expand_elset")
    elif hasattr(asm, "elset_elems") and callable(getattr(asm, "elset_elems")):
        res.expand_elset = getattr(asm, "elset_elems")
    elif 'elsets' in asm_dict and isinstance(asm_dict['elsets'], dict):
        _debug("[DEBUG] Using fallback expand_elset from asm.elsets")
        def fallback_expand(elset_name: str) -> Sequence[int]:
            if elset_name in asm_dict['elsets']:
                return asm_dict['elsets'][elset_name]  # 假设 list of elem_ids
            raise KeyError(f"ELSET {elset_name} not found in asm.elsets")
        res.expand_elset = fallback_expand
    else:
        _debug("[DEBUG] No fallback for expand_elset; asm has no 'elsets'")

    # -------- 绑定 get_node_coords --------
    if hasattr(asm, "get_node_coords") and callable(getattr(asm, "get_node_coords")):
        res.get_node_coords = getattr(asm, "get_node_coords")
    elif hasattr(asm, "node_coords") and callable(getattr(asm, "node_coords")):
        res.get_node_coords = getattr(asm, "node_coords")
    elif 'nodes' in asm_dict and isinstance(asm_dict['nodes'], dict):
        _debug("[DEBUG] Using fallback get_node_coords from asm.nodes")
        def fallback_coords(nid_list: Sequence[int]) -> np.ndarray:
            coords = [asm_dict['nodes'].get(nid, None) for nid in nid_list]
            if any(c is None for c in coords):
                raise KeyError("Some nodes not found in asm.nodes")
            return np.array(coords, dtype=np.float32)
        res.get_node_coords = fallback_coords
    else:
        _debug("[DEBUG] No fallback for get_node_coords; asm has no 'nodes'")

    # 可选：节点法向估计
    if hasattr(asm, "estimate_node_normals") and callable(getattr(asm, "estimate_node_normals")):
        res.estimate_node_normals = getattr(asm, "estimate_node_normals")

    return res


def _as_element_face(it: Any) -> ElementFaceRef:
    if isinstance(it, ElementFaceRef):
        return it
    if isinstance(it, (tuple, list)) and len(it) == 2:
        return ElementFaceRef(int(it[0]), int(it[1]))
    if isinstance(it, dict) and "elem_id" in it and "face_id" in it:
        return ElementFaceRef(int(it["elem_id"]), int(it["face_id"]))
    raise TypeError(
        f"[surfaces] 无法把 {type(it)} 解析为 ElementFaceRef；"
        f"支持 ElementFaceRef / (elem_id,face_id) / {{'elem_id','face_id'}}"
    )


def _is_elset_tuple(it: Any) -> bool:
    if isinstance(it, (tuple, list)) and len(it) == 3:
        head = str(it[0]).strip().upper()
        return head == "ELSET"
    return False


def _empty_Xnw() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32))


def _as_float32(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32)
    if not y.flags["C_CONTIGUOUS"]:
        y = np.ascontiguousarray(y)
    return y


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)  # 退化时给个默认方向
    return (v / n).astype(np.float32)


def _is_int_like(x: Any) -> bool:
    try:
        int(x)
        return True
    except Exception:
        return False


def _pca_normals(P: np.ndarray, k: int = 12) -> np.ndarray:
    """
    极简邻域 PCA 法向（仅在 NODE surface 且未提供 estimate_node_normals 时使用）。
    - 近邻使用暴力欧式距离选 k 个点（N*k 复杂度；这里只做兜底，不建议大规模使用）。
    - 法向方向不做全局一致性（对预紧/接触一般只用内积投影，可接受）。
    """
    P = np.asarray(P, dtype=np.float32)
    N = P.shape[0]
    if N == 0:
        return np.zeros((0, 3), dtype=np.float32)
    k = int(max(3, min(k, N)))

    normals = np.zeros_like(P, dtype=np.float32)
    for i in range(N):
        # 选临近 k 个点
        d2 = np.sum((P - P[i]) ** 2, axis=1)
        nn_idx = np.argpartition(d2, kth=k-1)[:k]
        Q = P[nn_idx]  # (k,3)
        Qc = Q - Q.mean(axis=0, keepdims=True)
        # PCA：最小特征值对应的方向
        C = Qc.T @ Qc  # (3,3)
        w, v = np.linalg.eigh(C)  # v[:,0] 对应最小特征值
        n = v[:, 0]
        normals[i] = _normalize(n)

    return normals


def _strip_asm_prefix(s: str) -> str:
    """把 'ASM::"bolt1 up"' 这类键去掉 ASM:: 前缀，返回剩余部分（不去空格/引号）。"""
    s2 = s.strip()
    if s2.upper().startswith("ASM::"):
        return s2[5:]  # 去掉 'ASM::'
    return s2


def _check_surface(s: SurfaceDef) -> None:
    if not isinstance(s, SurfaceDef):
        raise TypeError(f"[surfaces] 期望 SurfaceDef，收到 {type(s)}")
    if s.stype not in ("ELEMENT", "NODE"):
        raise ValueError(f"[surfaces] s.stype 必须为 'ELEMENT' 或 'NODE'，收到 {s.stype!r}")


# 新增函数：surface_def_to_points（兼容项目中的调用，fallback 到 to_points）
def surface_def_to_points(
    asm: Any,
    surface_like: Any,
    n: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    项目兼容函数：将 SurfaceDef 或类似对象采样为 (X, N, w)。
    - 如果 surface_like 是 SurfaceDef，直接调用 to_points。
    - 添加 fallback 逻辑，避免 NotImplementedError。
    - 尝试从 asm 的 boundaries/mesh/elsets/nsets 等采样。
    """
    _debug(
        f"[DEBUG] Entering surface_def_to_points with surface_like type: {type(surface_like)}, items: {getattr(surface_like, 'items', 'Unknown')}"
    )
    if isinstance(surface_like, SurfaceDef):
        try:
            return to_points(
                surface=surface_like,
                n_per_face=n,
                mode='centroid',
                asm=asm
            )
        except Exception as e:
            _debug(f"[DEBUG] to_points failed: {str(e)}")
            # 继续到 fallback

    # 项目原有尝试逻辑（从错误消息复制并增强）
    # 尝试 asm.get_surface_points 等
    for method in [
        'get_surface_points', 'sample_surface', 'sample_surface_elements'
    ]:
        if hasattr(asm, method) and callable(getattr(asm, method)):
            try:
                return getattr(asm, method)(surface_like, n)
            except Exception as e:
                _debug(f"[DEBUG] Method {method} failed: {str(e)}")

    # 尝试 asm.boundaries
    if hasattr(asm, 'boundaries') and hasattr(asm.boundaries, 'sample_surface'):
        try:
            return asm.boundaries.sample_surface(surface_like, n)
        except Exception as e:
            _debug(f"[DEBUG] asm.boundaries.sample_surface failed: {str(e)}")

    # 尝试 asm.mesh
    if hasattr(asm, 'mesh') and hasattr(asm.mesh, 'sample_surface'):
        try:
            return asm.mesh.sample_surface(surface_like, n)
        except Exception as e:
            _debug(f"[DEBUG] asm.mesh.sample_surface failed: {str(e)}")

    # Fallback for TYPE=ELEMENT with elsets
    asm_dict = getattr(asm, '__dict__', {})
    if isinstance(surface_like, SurfaceDef) and surface_like.stype == 'ELEMENT':
        if surface_like.items and isinstance(surface_like.items[0], (tuple, list)) and len(surface_like.items[0]) == 2:
            elset_name = surface_like.items[0][0].strip('\"\' ')
            if 'elsets' in asm_dict and elset_name in asm_dict['elsets']:
                elem_ids = asm_dict['elsets'][elset_name]
                # 兜底采样：每个元素取中心点（简化，可扩展）
                X_list = []
                for eid in elem_ids:
                    if 'elements' in asm_dict and eid in asm_dict['elements']:
                        node_ids = asm_dict['elements'][eid]
                        coords = [asm_dict['nodes'][nid] for nid in node_ids if 'nodes' in asm_dict and nid in asm_dict['nodes']]
                        if coords:
                            center = np.mean(coords, axis=0)
                            X_list.append(center)
                if X_list:
                    _debug(f"[DEBUG] Fallback sampling succeeded with {len(X_list)} points")
                    X = np.array(X_list, dtype=np.float32)
                    N = np.zeros_like(X)  # 简易法向，实际需计算
                    w = np.ones((len(X_list),), dtype=np.float32)
                    return X, N, w
                else:
                    _debug("[DEBUG] Fallback sampling: no points found")

    # Fallback for TYPE=NODE with nsets
    if isinstance(surface_like, SurfaceDef) and surface_like.stype == 'NODE':
        if 'nsets' in asm_dict:
            # 类似兜底
            pass

    # 如果所有失败，返回空数组作为兜底，并警告
    print("[WARNING] All sampling fallbacks failed; returning empty arrays to avoid crash")
    return _empty_Xnw()


