# -*- coding: utf-8 -*-
"""
attach_ties_bcs.py — 挂载 Tie / Boundary（优先复用已解析的 asm 对象，Py38+）

功能层次（按优先级）：
1) 原生路径：若工程存在 TiePenalty/BoundaryPenalty，且 asm 暴露
   get_triangulated_surface / project_points_onto_surface / get_nset_node_ids / get_node_coords，
   则使用这些接口构造真实约束；
2) 降级路径：若缺少上述接口/类，则从 asm.surfaces / nsets / elsets / elements
   中恢复节点坐标；slave 面上采样 n_tie_points 个点，映射到 master（最近邻）；Boundary
   直接取 nset 坐标；构造 SimpleTie/SimpleBC 作为占位对象，保证流程可运行；
3) 兼容仅含 Abaqus 风格的 *Surface 引用（surface 只有 items/stype），支持：
   - type=NODE：items 为 NSET 名；
   - type=ELEMENT：items 为 ELSET(+面号 S1..S6)，内置 C3D8/C3D20 角点面映射，其它类型回退为全节点。

对外主入口：
    attach_ties_and_bcs_from_inp(total, asm, inp_path, cfg)
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Set
import numpy as np

# ====== 可选导入：若工程内有真实罚项实现，则优先使用 ======
try:
    from physics.tie_constraints import TiePenalty  # type: ignore
except Exception:
    TiePenalty = None  # type: ignore

try:
    from physics.boundary_conditions import BoundaryPenalty, BoundaryConfig  # type: ignore
except Exception:
    BoundaryPenalty = None  # type: ignore
    BoundaryConfig = None  # type: ignore


# ============================================================
# 1) 名称解析 & 几何辅助（原接口优先，缺失则降级）
#    ——含 SetDef 展开、items/stype 解析 + dict→array 规范化
# ============================================================
def _resolve_key_in_dict(d: Dict[str, Any], key: str) -> str:
    """在字典中宽松匹配 key：去引号/前后空格/ASM:: 前缀，大小写不敏感"""
    if key in d:
        return key
    cand = [
        key,
        key.strip(),
        key.strip().strip('"'),
        key.strip().strip("'"),
        key.replace("ASM::", "").strip(),
        f'ASM::{key}',
        f'ASM::"{key}"',
        f'"{key}"',
    ]
    for c in cand:
        if c in d:
            return c
    low = {k.lower(): k for k in d.keys()}
    for c in cand:
        k = low.get(c.lower())
        if k is not None:
            return k
    raise KeyError(f"找不到键：{key}；可用键示例：{list(d.keys())[:8]}")


def _resolve_surface_key(asm, key: str) -> str:
    if not hasattr(asm, "surfaces"):
        raise AttributeError("asm 缺少 .surfaces。")
    return _resolve_key_in_dict(asm.surfaces, key)  # type: ignore


def _flatten_to_int_ids(obj) -> np.ndarray:
    """
    把各种“集合表示”（SetDef/对象/字典/嵌套list/ndarray/字符串等）统一展开为去重后的 int64 数组。
    字符串中会提取所有数字（支持 '1,2,3' / '1 2 3' / 'S1'→忽略字母，只取数字）。
    """
    res: List[int] = []
    stack = [obj]
    seen: Set[int] = set()
    while stack:
        cur = stack.pop()
        key = id(cur)
        if key in seen:
            continue
        seen.add(key)

        if isinstance(cur, (int, np.integer)):
            res.append(int(cur)); continue
        if isinstance(cur, np.ndarray):
            stack.extend(cur.reshape(-1).tolist()); continue
        if isinstance(cur, (list, tuple, set)):
            stack.extend(list(cur)); continue
        if isinstance(cur, str):
            nums = re.findall(r"-?\d+", cur)
            res.extend(int(n) for n in nums); continue
        if isinstance(cur, dict):
            for k in ("ids", "items", "members", "node_ids", "nodes", "elements", "data"):
                if k in cur:
                    stack.append(cur[k])
            stack.extend(cur.values()); continue
        got_attr = False
        for attr in ("ids", "items", "members", "node_ids", "nodes", "elements", "data"):
            if hasattr(cur, attr):
                stack.append(getattr(cur, attr)); got_attr = True
        if got_attr: continue
        if hasattr(cur, "id"):
            try:
                res.append(int(getattr(cur, "id"))); continue
            except Exception:
                pass
        continue

    if not res:
        return np.zeros((0,), dtype=np.int64)
    return np.array(sorted(set(res)), dtype=np.int64)


def _as_array3(x) -> np.ndarray:
    """
    将节点/点集统一成 (N,3) 的 float32 数组：
    - dict {id: (x,y,z)} -> 按 values() 转为数组
    - list/tuple/np.ndarray -> 直接转 np.float32
    """
    if isinstance(x, dict):
        arr = np.array(list(x.values()), dtype=np.float32)
    else:
        arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1 and arr.size % 3 == 0:
        arr = arr.reshape(-1, 3)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"_as_array3: expect (N,3), got shape={arr.shape}")
    return arr


def _coords_for_node_ids(asm, node_ids) -> np.ndarray:
    nodes = getattr(asm, "nodes", None)
    if nodes is None:
        raise AttributeError("asm 缺少 .nodes，无法通过节点ID取坐标。")
    ids = _flatten_to_int_ids(node_ids)
    if ids.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    if isinstance(nodes, dict):
        X = np.array([nodes[int(i)] for i in ids], dtype=np.float32)
        return X

    nodes = np.asarray(nodes)
    max_id = int(ids.max(initial=0))
    if nodes.shape[0] > max_id:  # 0-based
        return nodes[ids].astype(np.float32)
    else:  # 1-based
        return nodes[ids - 1].astype(np.float32)


def _to_upper(s: Optional[str]) -> str:
    return (s or "").upper()


def _guess_stype(surface_obj) -> str:
    st = getattr(surface_obj, "stype", None)
    if st:
        return str(st)
    raw = getattr(surface_obj, "raw_lines", None)
    txt = "\n".join(raw) if isinstance(raw, (list, tuple)) else str(raw or "")
    m = re.search(r"type\s*=\s*(NODE|ELEMENT)", txt, flags=re.I)
    return m.group(1) if m else ""


def _parse_items(surface_obj) -> List[Tuple[str, str, Optional[str]]]:
    """
    解析 surface_obj.items → 统一三元组：
      ('NSET',  set_name, None)
      ('ELSET', set_name, face_label or None)  # face: 'S1'..'S6'
    """
    items = getattr(surface_obj, "items", None)
    out: List[Tuple[str, str, Optional[str]]] = []
    if items is None:
        return out

    def _push(kind, name, face=None):
        if name is None: return
        out.append((kind, str(name), (str(face).strip() if face is not None else None)))

    for it in items:
        if isinstance(it, str):
            toks = [t.strip() for t in it.split(",") if t.strip()]
            if len(toks) == 1: _push("AUTO", toks[0], None)
            elif len(toks) >= 2: _push("AUTO", toks[0], toks[1])
        elif isinstance(it, (list, tuple)):
            if len(it) == 1: _push("AUTO", str(it[0]), None)
            elif len(it) >= 2: _push("AUTO", str(it[0]), str(it[1]))
        elif isinstance(it, dict):
            name = it.get("name") or it.get("set") or it.get("elset")
            face = it.get("face") or it.get("s") or it.get("side")
            kind = it.get("kind") or it.get("type")
            _push((str(kind).upper() if kind else "AUTO"), name, face)
        else:
            s = str(it)
            toks = [t.strip() for t in s.split(",") if t.strip()]
            if toks: _push("AUTO", toks[0], toks[1] if len(toks) > 1 else None)

    st = _to_upper(_guess_stype(surface_obj))
    fixed: List[Tuple[str, str, Optional[str]]] = []
    for kind, name, face in out:
        if kind == "AUTO":
            kind = "NSET" if "NODE" in st else "ELSET"
        fixed.append((kind, name, face))
    return fixed


def _pick_face_nodes(conn: np.ndarray, face: Optional[str]) -> np.ndarray:
    """
    根据面号从立体单元连接表取面节点：
    - 支持 C3D8/C3D20(R) 的角点面映射；
    - 其它未知类型或 face 为空时退化为“全节点”；
    - 壳单元(3/4/6/8节点)通常已是面，face 为空时直接返回。
    """
    conn = np.asarray(conn).reshape(-1)
    try:
        conn = conn.astype(np.int64)  # 有时会是 float
    except Exception:
        pass

    n = conn.size
    lab = _to_upper(face)

    if n in (3, 4, 6, 8) and lab == "":
        return conn

    if n == 8:  # C3D8
        face_map = {
            "S1": [0, 1, 2, 3], "S2": [4, 5, 6, 7],
            "S3": [0, 4, 5, 1], "S4": [1, 5, 6, 2],
            "S5": [2, 6, 7, 3], "S6": [3, 7, 4, 0],
        }
        if lab in face_map:
            return conn[face_map[lab]]

    if n >= 20:  # C3D20(R)：取角点 0..7 映射
        corner = conn[:8]
        face_map = {
            "S1": [0, 1, 2, 3], "S2": [4, 5, 6, 7],
            "S3": [0, 4, 5, 1], "S4": [1, 5, 6, 2],
            "S5": [2, 6, 7, 3], "S6": [3, 7, 4, 0],
        }
        if lab in face_map:
            return corner[face_map[lab]]

    return conn


def _elem_conn(e: Any) -> np.ndarray:
    """尽量从元素对象/字典/数组还原连接表（节点ID序列）。"""
    if isinstance(e, (list, tuple, np.ndarray)):
        arr = np.asarray(e)
        if arr.dtype == object:
            return _flatten_to_int_ids(arr.tolist())
        try:
            return arr.astype(np.int64)
        except Exception:
            return _flatten_to_int_ids(arr)
    if isinstance(e, dict):
        for k in ("conn", "nodes", "node_ids", "connectivity", "indices", "data"):
            if k in e:
                return _elem_conn(e[k])
        return _flatten_to_int_ids(e)
    for k in ("conn", "nodes", "node_ids", "connectivity", "indices", "data"):
        if hasattr(e, k):
            return _elem_conn(getattr(e, k))
    return _flatten_to_int_ids(str(e))


def _extract_surface_VF(asm, surface_obj) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    扩展版提取 (V, F)：
    - 先尝试直接字段 V/verts/vertices/X/nodes + F/faces/tris/triangles；
    - 若无坐标，则解析 items/stype，从 nsets/elsets/elements 反推节点坐标；
    - 返回 (V, None) 也可工作（从顶点采样）。
    """
    def _get(obj, k):
        if isinstance(obj, dict):
            return obj.get(k, None)
        return getattr(obj, k, None)

    # A) 直接字段
    V = None
    for k in ("V", "verts", "vertices", "X", "nodes"):
        V = _get(surface_obj, k)
        if V is not None:
            V = np.asarray(V, dtype=np.float32); break

    F = None
    for k in ("F", "faces", "tris", "triangles", "elements", "elems"):
        F = _get(surface_obj, k)
        if F is not None:
            F = np.asarray(F, dtype=np.int32); break

    if V is not None:
        if F is not None and F.ndim == 2 and F.shape[1] == 4:
            t1 = F[:, [0, 1, 2]]; t2 = F[:, [0, 2, 3]]
            F = np.vstack([t1, t2]).astype(np.int32)
        return V.astype(np.float32), (None if F is None else F.astype(np.int32))

    # B) items/stype 解析：NODE/ELSET 引用
    triples = _parse_items(surface_obj)
    if triples:
        elsets = getattr(asm, "elsets", {})
        elements = getattr(asm, "elements", None)
        nsets = getattr(asm, "nsets", {})

        node_ids: Set[int] = set()
        for kind, name, face in triples:
            if kind == "NSET":
                if isinstance(nsets, dict) and nsets:
                    try:
                        key = _resolve_key_in_dict(nsets, name)
                        node_ids_arr = _flatten_to_int_ids(nsets[key])
                        node_ids.update(int(i) for i in node_ids_arr)
                    except Exception:
                        pass
            else:  # ELSET
                if not (isinstance(elsets, dict) and elsets and elements is not None):
                    continue
                try:
                    key = _resolve_key_in_dict(elsets, name)
                except Exception:
                    continue
                elem_ids = _flatten_to_int_ids(elsets[key])

                if isinstance(elements, dict):
                    for eid in elem_ids:
                        e = elements.get(int(eid))
                        if e is None: continue
                        conn = _elem_conn(e)
                        node_ids.update(int(i) for i in _pick_face_nodes(conn, face))
                else:
                    arr = np.asarray(elements, dtype=object)
                    if elem_ids.size:
                        max_eid = int(elem_ids.max())
                        zero_based = (len(arr) > max_eid)
                    else:
                        zero_based = True
                    for eid in elem_ids:
                        idx = int(eid) if zero_based else int(eid) - 1
                        if idx < 0 or idx >= len(arr): continue
                        conn = _elem_conn(arr[idx])
                        node_ids.update(int(i) for i in _pick_face_nodes(conn, face))

        if not node_ids:
            raise KeyError("无法从 surface.items 解析到任何节点，请检查 NSET/ELSET 名称或 elements/sets 是否已填充。")

        V = _coords_for_node_ids(asm, np.array(sorted(list(node_ids)), dtype=np.int64))
        return V.astype(np.float32), None

    # C) 字符串：可能是另一个 surface 名或 nset 名
    if isinstance(surface_obj, str):
        name = surface_obj.strip().strip('"').strip("'")
        try:
            skey = _resolve_surface_key(asm, name)
            return _extract_surface_VF(asm, asm.surfaces[skey])
        except Exception:
            pass
        nsets = getattr(asm, "nsets", {})
        if isinstance(nsets, dict) and nsets:
            try:
                key = _resolve_key_in_dict(nsets, name)
                V = _coords_for_node_ids(asm, nsets[key])
                return V.astype(np.float32), None
            except Exception:
                pass

    # D) 表面对象就是节点ID序列
    if isinstance(surface_obj, (list, tuple, np.ndarray)) and len(surface_obj) > 0:
        arr = np.asarray(surface_obj)
        if np.issubdtype(arr.dtype, np.integer):
            V = _coords_for_node_ids(asm, arr)
            return V.astype(np.float32), None

    # E) 还是拿不到：报错并列出可用字段
    keys = list(surface_obj.keys()) if isinstance(surface_obj, dict) else [
        k for k in dir(surface_obj) if not str(k).startswith("_")
    ]
    raise KeyError(
        "表面对象中未找到坐标信息。已尝试 V/verts/vertices/X/nodes/node_ids/nset/elset/items；实际可用字段示例：{}".format(
            keys[:12]
        )
    )


def _tri_area(V, F):
    a = V[F[:, 0]]; b = V[F[:, 1]]; c = V[F[:, 2]]
    return np.linalg.norm(np.cross(b - a, c - a), axis=1) * 0.5


def _sample_points_on_surface(V, F, n_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """面积权采样；无 F 则从顶点均匀采样。"""
    if F is None or len(F) == 0:
        idx = np.random.choice(V.shape[0], size=n_points, replace=(V.shape[0] < n_points))
        xs = V[idx]; w = np.ones((xs.shape[0],), dtype=np.float32)
        return xs.astype(np.float32), w
    area = _tri_area(V, F)
    prob = area / max(area.sum(), 1e-12)
    tri_idx = np.random.choice(F.shape[0], size=n_points, p=prob)
    r = np.random.rand(n_points, 2)
    b0 = 1 - np.sqrt(r[:, 0])
    b1 = np.sqrt(r[:, 0]) * (1 - r[:, 1])
    b2 = np.sqrt(r[:, 0]) * r[:, 1]
    tri = F[tri_idx]
    xs = (V[tri[:, 0]] * b0[:, None] + V[tri[:, 1]] * b1[:, None] + V[tri[:, 2]] * b2[:, None])
    w = np.ones((xs.shape[0],), dtype=np.float32)
    return xs.astype(np.float32), w


def _nearest_on_vertices(xs, V_master) -> Tuple[np.ndarray, np.ndarray]:
    """numpy 最近邻到 master 顶点；用于降级实现。"""
    d2 = ((xs[:, None, :] - V_master[None, :, :]) ** 2).sum(axis=2)  # [Ns, Nm]
    j = np.argmin(d2, axis=1)
    xm = V_master[j]
    w = np.ones((xs.shape[0],), dtype=np.float32)
    return xm.astype(np.float32), w


# ============================================================
# 1.1) 从已解析的 asm 对象提取 Tie/Boundary 配置
# ============================================================
def _extract_ties_from_asm(asm) -> List[Dict[str, Any]]:
    ties_cfg: List[Dict[str, Any]] = []
    for t in getattr(asm, "ties", []) or []:
        master = getattr(t, "master", None)
        slave = getattr(t, "slave", None)
        if not master or not slave:
            continue
        name = getattr(t, "name", None) or f"TIE@{master}->{slave}"
        ties_cfg.append({"name": name, "master": master, "slave": slave})
    return ties_cfg


def _parse_boundary_entry(raw_entry: Any) -> Dict[str, Any]:
    raw = getattr(raw_entry, "raw", raw_entry)
    row = [t.strip() for t in str(raw).split(",") if t.strip()]
    setn = row[0] if row else ""
    typ = "BOUNDARY"
    d1 = d2 = None

    if len(row) >= 2 and row[1].upper().startswith("ENCASTRE"):
        typ = "ENCASTRE"
        d1, d2 = 1, 6
    else:
        def _to_int(x):
            try:
                return int(float(x))
            except Exception:
                return None

        d1 = _to_int(row[1]) if len(row) >= 2 else None
        d2 = _to_int(row[2]) if len(row) >= 3 else d1

    return {"set": setn, "type": typ, "dof1": d1, "dof2": d2, "raw": raw}


def _extract_bcs_from_asm(asm) -> List[Dict[str, Any]]:
    bcs_cfg: List[Dict[str, Any]] = []
    for b in getattr(asm, "boundaries", []) or []:
        bcs_cfg.append(_parse_boundary_entry(b))
    return bcs_cfg


def build_surface_correspondence(asm, slave_key: str, master_key: str, n_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成 Tie 点对应：(xs on slave, xm on master, w)。
    原接口优先（get_triangulated_surface + project_points_onto_surface），否则降级到采样+最近邻。
    """
    # 原接口优先
    try:
        if hasattr(asm, "get_triangulated_surface") and hasattr(asm, "project_points_onto_surface"):
            V_s, F_s = asm.get_triangulated_surface(slave_key)   # type: ignore
            V_m, F_m = asm.get_triangulated_surface(master_key)  # type: ignore
            xs, _ = _sample_points_on_surface(np.asarray(V_s, np.float32),
                                              None if F_s is None else np.asarray(F_s, np.int32),
                                              n_points)
            xm = _as_array3(asm.project_points_onto_surface(xs, master_key))  # ★ 规范化
            w = np.ones((xs.shape[0],), dtype=np.float32)
            return xs.astype(np.float32), xm, w
    except Exception:
        pass

    # 降级路径：从 surfaces/nsets 等推导 V、F
    skey = _resolve_surface_key(asm, slave_key)
    mkey = _resolve_surface_key(asm, master_key)
    surf_s = asm.surfaces[skey]
    surf_m = asm.surfaces[mkey]
    V_s, F_s = _extract_surface_VF(asm, surf_s)
    V_m, F_m = _extract_surface_VF(asm, surf_m)

    xs, ws = _sample_points_on_surface(V_s, F_s, n_points)
    xm, wm = _nearest_on_vertices(xs, V_m)
    w = (ws * wm).astype(np.float32)
    return xs.astype(np.float32), xm.astype(np.float32), w


def get_nset_coords(asm, nset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """优先原接口（get_nset_node_ids/get_node_coords），否则从 asm.nsets/nodes 还原。"""
    if hasattr(asm, "get_nset_node_ids") and hasattr(asm, "get_node_coords"):
        try:
            node_ids = asm.get_nset_node_ids(nset_name)  # type: ignore
            X = _as_array3(asm.get_node_coords(node_ids))   # ★ 规范化为 (N,3)
            w = np.ones((X.shape[0],), dtype=np.float32)
            return X, w
        except Exception:
            pass

    nsets = getattr(asm, "nsets", None)
    if isinstance(nsets, dict) and nset_name in nsets:
        X = _coords_for_node_ids(asm, nsets[nset_name])
        w = np.ones((X.shape[0],), dtype=np.float32)
        return X.astype(np.float32), w

    # 找不到就返回空
    return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)


# ============================================================
# 3) 占位对象（当真实罚项类不可用时）
# ============================================================
class SimpleTie(object):
    def __init__(self, name: str, master: str, slave: str, xs=None, xm=None, w=None):
        self.name = name
        self.master = master
        self.slave = slave
        self.xs = xs
        self.xm = xm
        self.w = w

    def __repr__(self):
        return "SimpleTie(name={!r}, master={!r}, slave={!r}, xs={}, xm={})".format(
            self.name, self.master, self.slave,
            None if self.xs is None else self.xs.shape,
            None if self.xm is None else self.xm.shape
        )


class SimpleBC(object):
    def __init__(self, setname: str, dof1: Optional[int], dof2: Optional[int], kind: str = "ENCASTRE", X=None):
        self.set = setname
        self.dof1 = dof1
        self.dof2 = dof2
        self.kind = kind
        self.X = X

    def __repr__(self):
        return "SimpleBC(set={!r}, dof1={}, dof2={}, kind={!r}, X={})".format(
            self.set, self.dof1, self.dof2, self.kind,
            None if self.X is None else self.X.shape
        )


# ============================================================
# 4) 对外主函数：解析 + 几何构造 + 挂载
# ============================================================
def attach_ties_and_bcs_from_inp(total, asm, inp_path: str, cfg) -> None:
    """
    从已解析的 asm 对象中提取 Tie/Boundary 并挂到 total.attach(...)。
    - 若检测到真实罚项类/接口，优先构造真实算子；
    - 否则构造 SimpleTie/SimpleBC 以保证流程可运行。
    """
    ties_cfg = _extract_ties_from_asm(asm)
    bcs_cfg = _extract_bcs_from_asm(asm)

    n_tie_points = int(getattr(cfg, "n_tie_points", 2000))
    tie_alpha = float(getattr(cfg, "tie_alpha", 1.0e3))
    bc_alpha = float(getattr(cfg, "bc_alpha", 1.0e4))
    bc_mu = float(getattr(cfg, "bc_mu", 1.0e3))
    bc_mode = str(getattr(cfg, "bc_mode", "penalty")).lower()

    ties_out: List[Any] = []
    bcs_out: List[Any] = []

    # ---- Ties ----
    for t in ties_cfg:
        master = t["master"]; slave = t["slave"]
        name = t.get("name", f"TIE@{master}->{slave}")

        xs, xm, w = build_surface_correspondence(asm, slave, master, n_tie_points)
        xs = _as_array3(xs); xm = _as_array3(xm)  # ★ 兜底

        if TiePenalty is not None:
            try:
                tie = TiePenalty(alpha=tie_alpha)  # 若你的类构造不同，可在此适配
                if hasattr(tie, "build_from_points"):
                    tie.build_from_points(xs, xm, w)
                elif hasattr(tie, "build"):
                    tie.build(xs, xm, w)
                else:
                    tie.xs = xs; tie.xm = xm; tie.w = w
                tie.name = name
                ties_out.append(tie)
                continue
            except Exception:
                pass  # 回退到 SimpleTie

        ties_out.append(SimpleTie(name=name, master=master, slave=slave, xs=xs, xm=xm, w=w))

    # ---- Boundary / ENCASTRE ----
    for b in bcs_cfg:
        setn = b.get("set", "?")
        typ = b.get("type", "BOUNDARY")
        d1 = b.get("dof1")
        d2 = b.get("dof2") if b.get("dof2") is not None else d1

        X, _ = get_nset_coords(asm, setn)
        if isinstance(X, (list, tuple, np.ndarray, dict)):
            X = _as_array3(X) if len(np.asarray(X).shape) != 0 else X  # ★ 兜底

        if BoundaryPenalty is not None and isinstance(X, np.ndarray) and X.shape[0] > 0:
            try:
                bc_cfg = BoundaryConfig(alpha=bc_alpha, mode=bc_mode, mu=bc_mu)
                bc = BoundaryPenalty(cfg=bc_cfg, dof1=d1, dof2=d2, kind=typ)
                if hasattr(bc, "build"):
                    bc.build(X)
                else:
                    bc.X = X; bc.dof1 = d1; bc.dof2 = d2; bc.kind = typ
                bcs_out.append(bc)
                continue
            except Exception:
                pass  # 回退

        bcs_out.append(SimpleBC(setname=setn, dof1=d1, dof2=d2, kind=typ, X=X if isinstance(X, np.ndarray) and X.shape[0] else None))

    # 一次性挂入
    total.attach(ties=ties_out, bcs=bcs_out)
