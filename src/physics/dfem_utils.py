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
import tensorflow as tf


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

    # 形函数梯度（常梯度四面体的标准公式）
    # N1 = a1 + b1*x + c1*y + d1*z, 但梯度直接由几何决定
    def grad_N(v1, v2, v3):
        return np.cross(v2 - v1, v3 - v1) / (6.0 * vol)

    g1 = grad_N(x2, x3, x4)
    g2 = grad_N(x3, x4, x1)
    g3 = grad_N(x4, x1, x2)
    g4 = grad_N(x1, x2, x3)

    grads = [g1, g2, g3, g4]

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
            X_nodes_tf : (Nnode, 3)
            B_tf       : (Nsub, 6, 12)
            vol_tf     : (Nsub,)
            w_tf       : (Nsub,)
            lam_tf     : (Nsub,)
            mu_tf      : (Nsub,)
            dof_idx_tf : (Nsub, 12)
    """

    X_nodes = asm.nodes  # shape = (Nnode, 3)
    X_nodes_tf = tf.constant(X_nodes, dtype=tf.float32)

    B_list = []
    vol_list = []
    lam_list = []
    mu_list = []
    dof_idx_list = []

    # 遍历所有 Part（CD3D8/C3D4 都被拆成四面体）
    for part in asm.parts.values():
        mat_name = part2mat.get(part.name, None)
        if mat_name is None:
            raise KeyError(f"Part '{part.name}' 无材料映射。请检查 part2mat。")

        if mat_name not in materials:
            raise KeyError(f"材料 '{mat_name}' 未在 materials 字典中注册。")

        E, nu = materials[mat_name]
        lam = (E * nu) / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))

        for elem in part.elems:
            conn = elem.conn  # element 的节点列表 (例如 C3D8 是 8 点)

            # 把 C3D8 拆成 4 个四面体；若本来就是 C3D4 则只生成 1 个
            if len(conn) == 8:
                tet_conn = [
                    [conn[0], conn[1], conn[3], conn[4]],
                    [conn[1], conn[2], conn[3], conn[6]],
                    [conn[1], conn[5], conn[4], conn[6]],
                    [conn[3], conn[4], conn[7], conn[6]],
                ]
            elif len(conn) == 4:
                tet_conn = [conn]
            else:
                raise ValueError(f"不支持的单元节点数: {len(conn)}")

            # 每个四面体都构造 B、vol、局部 DOF 索引
            for tet in tet_conn:
                X4 = X_nodes[tet, :]
                B, vol = tetra_B_and_volume(X4)

                B_list.append(B)
                vol_list.append(vol)
                lam_list.append(lam)
                mu_list.append(mu)

                # 对四点 → 每点 3 个 DOF → 共 12 个自由度
                dof_idx = []
                for nid in tet:
                    dof_idx.extend([3 * nid + 0, 3 * nid + 1, 3 * nid + 2])
                dof_idx_list.append(dof_idx)

    # 转成 TensorFlow 张量
    B_tf = tf.constant(np.array(B_list, dtype=np.float32))              # (Nsub, 6, 12)
    vol_tf = tf.constant(np.array(vol_list, dtype=np.float32))          # (Nsub,)
    w_tf = vol_tf                                                        # 权重 = 小单元体积
    lam_tf = tf.constant(np.array(lam_list, dtype=np.float32))          # (Nsub,)
    mu_tf = tf.constant(np.array(mu_list, dtype=np.float32))            # (Nsub,)
    dof_idx_tf = tf.constant(np.array(dof_idx_list, dtype=np.int32))    # (Nsub, 12)

    return dict(
        X_nodes_tf=X_nodes_tf,
        B_tf=B_tf,
        vol_tf=vol_tf,
        w_tf=w_tf,
        lam_tf=lam_tf,
        mu_tf=mu_tf,
        dof_idx_tf=dof_idx_tf,
    )
