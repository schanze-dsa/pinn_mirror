# ================================================================
# dfem_utils.py
# DFEM utility functions for elasticity_energy.py (Jacobian-free)
#
# 提供：
#   1) tetra_B_and_volume()     - 计算四面体的形函数梯度 B、体积 vol
#   2) build_dfem_subcells()    - 从 AssemblyModel 生成 DFEM 所需全部张量
#
# 作者：ChatGPT（2025）
# ================================================================

import numpy as np


# ---------------------------------------------------------------
# 1. 四面体形函数梯度 + 体积（常梯度）
# ---------------------------------------------------------------
def tetra_B_and_volume(X4):
    """
    输入：
        X4: (4, 3) NumPy 数组，每行为一个顶点的坐标
    输出：
        B:   (6, 12) Voigt 形式的常梯度矩阵
        vol: 四面体体积（正）
    """

    x1, x2, x3, x4 = X4

    # 四面体体积（6 倍体积的行列式式子）
    M = np.vstack([x2 - x1, x3 - x1, x4 - x1]).T
    vol = abs(np.linalg.det(M)) / 6.0
    if vol <= 1e-16:
        raise ValueError("四面体体积过小，可能退化。")

    # 形函数梯度：通过解线性系统得到常梯度 (∂Ni/∂x, ∂Ni/∂y, ∂Ni/∂z)
    V = np.array(
        [
            [1.0, *x1],
            [1.0, *x2],
            [1.0, *x3],
            [1.0, *x4],
        ],
        dtype=np.float64,
    )

    try:
        coeffs = np.linalg.solve(V, np.eye(4))  # shape (4,4)
    except np.linalg.LinAlgError as exc:
        raise ValueError("四面体几何退化，无法求解形函数梯度。") from exc

    grads = coeffs[1:, :].T  # (4,3)

    # Voigt 形式 B (6, 12)
    B = np.zeros((6, 12), dtype=np.float32)
    for i, g in enumerate(grads):
        ix = 3 * i
        B[0, ix + 0] = g[0]
        B[1, ix + 1] = g[1]
        B[2, ix + 2] = g[2]
        B[3, ix + 0] = g[1]
        B[3, ix + 1] = g[0]
        B[4, ix + 1] = g[2]
        B[4, ix + 2] = g[1]
        B[5, ix + 0] = g[2]
        B[5, ix + 2] = g[0]

    return B.astype(np.float32), float(vol)


# ---------------------------------------------------------------
# 2. DFEM 子单元构建（核心）
# ---------------------------------------------------------------
def build_dfem_subcells(asm, part2mat, materials):
    """
    输入：
        asm: AssemblyModel (INP 解析结果)
        part2mat:  字典 {part_name: material_name}
        materials: 字典 {material_name: (E, nu)}

    输出：
        dict:
            X_nodes : (Nnode, 3)
            B       : (Nsub, 6, 12)
            w       : (Nsub,)
            lam     : (Nsub,)
            mu      : (Nsub,)
            dof_idx : (Nsub, 12)
    """

    if not getattr(asm, "nodes", None):
        raise ValueError("AssemblyModel 缺少节点坐标信息 (asm.nodes 为空)。")

    # 节点在 AssemblyModel 中以 dict[nid] = (x,y,z) 存储，需转换为 0-based 顺序数组
    node_ids = sorted(int(nid) for nid in asm.nodes.keys())
    id2idx = {nid: i for i, nid in enumerate(node_ids)}
    X_nodes = np.asarray([asm.nodes[nid] for nid in node_ids], dtype=np.float32)

    B_list = []
    vol_list = []
    lam_list = []
    mu_list = []
    dof_idx_list = []

    # 遍历所有 Part（CD3D8/C3D4 都被拆成四面体）
    for part_name, part in asm.parts.items():
        mat_name = part2mat.get(part_name, None)
        if mat_name is None:
            raise KeyError(f"Part '{part_name}' 无材料映射。请检查 part2mat。")

        if mat_name not in materials:
            raise KeyError(f"材料 '{mat_name}' 未在 materials 字典中注册。")

        mat_props = materials[mat_name]
        if isinstance(mat_props, (tuple, list)) and len(mat_props) >= 2:
            E, nu = float(mat_props[0]), float(mat_props[1])
        elif isinstance(mat_props, dict):
            try:
                E = float(mat_props["E"])
                nu = float(mat_props["nu"])
            except KeyError as exc:
                raise KeyError(f"材料 '{mat_name}' 需要提供 E 与 nu。") from exc
        else:
            raise TypeError(
                f"材料 '{mat_name}' 的属性需为 (E, nu) 或包含 E/nu 的字典，实际为 {type(mat_props)}。"
            )
        lam = (E * nu) / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))

        for blk in getattr(part, "element_blocks", []):
            etype = (blk.elem_type or "").upper()
            if etype not in {"C3D4", "C3D8"}:
                # 目前仅支持常用的四面体/六面体，其他类型如 C3D10/C3D20 需额外实现拆分
                continue

            for conn_raw in blk.connectivity:
                conn = [int(n) for n in conn_raw]

                if etype == "C3D8":
                    if len(conn) < 8:
                        raise ValueError(f"C3D8 单元期望 8 个节点，实际 {len(conn)}。")
                    tet_conns = [
                        [conn[0], conn[1], conn[3], conn[4]],
                        [conn[1], conn[2], conn[3], conn[6]],
                        [conn[1], conn[5], conn[4], conn[6]],
                        [conn[3], conn[4], conn[7], conn[6]],
                    ]
                else:  # C3D4
                    if len(conn) < 4:
                        raise ValueError(f"C3D4 单元期望至少 4 个节点，实际 {len(conn)}。")
                    tet_conns = [conn[:4]]

                for tet in tet_conns:
                    try:
                        idxs = [id2idx[int(nid)] for nid in tet]
                    except KeyError as exc:
                        raise KeyError(f"四面体节点 {tet} 不在 AssemblyModel.nodes 中。") from exc

                    X4 = X_nodes[idxs, :]
                    B, vol = tetra_B_and_volume(X4)

                    B_list.append(B)
                    vol_list.append(vol)
                    lam_list.append(lam)
                    mu_list.append(mu)

                    dof_idx = []
                    for idx in idxs:
                        base = 3 * idx
                        dof_idx.extend([base + 0, base + 1, base + 2])
                    dof_idx_list.append(dof_idx)

    # 汇总为 NumPy 数组，供 ElasticityEnergy 缓存为 TensorFlow 张量
    if not B_list:
        raise ValueError("未从装配中提取到任何 DFEM 子单元，请检查材料映射与单元类型。")

    return dict(
        X_nodes=X_nodes,
        B=np.asarray(B_list, dtype=np.float32),
        w=np.asarray(vol_list, dtype=np.float32),
        lam=np.asarray(lam_list, dtype=np.float32),
        mu=np.asarray(mu_list, dtype=np.float32),
        dof_idx=np.asarray(dof_idx_list, dtype=np.int32),
    )
