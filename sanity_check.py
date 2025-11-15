# -*- coding: utf-8 -*-
"""
sanity_check.py — 训练前体检（扩展：始终检测并打印 *Tie / *Boundary / ENCASTRE）

做了哪些事（仅本文件内改动，其他模块不动）：
  1) 解析 INP：始终打印 *Tie（绑定）与 *Boundary（固定/位移/ENCASTRE）统计与前若干条明细
  2) 若存在 train.attach_ties_bcs.attach_ties_and_bcs_from_inp，则以最小替身 Total 跑一遍“构造+挂载”验证
  3) 支持 --inp 指定 INP；不传则按如下优先级选择：
     args.inp  ->  TrainerConfig.inp_path  ->  D:\shuangfan\shuangfan.inp  ->  <项目默认 data/shuangfan.inp>
"""

import os
import sys
import argparse
import re
from types import SimpleNamespace
from typing import List, Dict, Any

# ---------------- PATH：确保 src/ 可导入 ----------------
ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# tqdm 可选
try:
    from tqdm.auto import tqdm
except Exception:
    class _DummyPB:
        def __init__(self, total=None, desc=None, leave=True, **_): print(desc or "")
        def update(self, n=1): pass
        def set_description_str(self, *_): pass
        def set_postfix_str(self, *_): pass
        def __enter__(self): return self
        def __exit__(self, *exc): pass
    tqdm = _DummyPB  # type: ignore

# ---------------- 项目内模块（若缺失也能优雅退化） ----------------
# 仅为取默认 inp_path；拿不到也不影响本脚本主体
try:
    from train.trainer import TrainerConfig  # type: ignore
except Exception:
    TrainerConfig = None  # type: ignore

# INP -> Assembly
from inp_io.inp_parser import load_inp  # type: ignore

# 可选：构造并挂载 Tie/BC 的工具（若你已新建 attach_ties_bcs.py）
try:
    from train.attach_ties_bcs import attach_ties_and_bcs_from_inp  # type: ignore
    _HAS_ATTACH = True
except Exception:
    attach_ties_and_bcs_from_inp = None  # type: ignore
    _HAS_ATTACH = False


# ============================================================
#  1) 纯文本解析：*Tie / *Boundary
# ============================================================
def _clean(s: str) -> str:
    return s.strip().rstrip(",")


def parse_inp_tie_boundary(inp_path: str) -> Dict[str, Any]:
    """
    从 INP 解析 *Tie 和 *Boundary（含 ENCASTRE）
    返回：
      {
        "ties": [{"name": "...", "master": "...", "slave": "..."}, ...],
        "bcs":  [{"set": "NSET_X", "type": "ENCASTRE"/"BOUNDARY", "dof1": 1, "dof2": 6, "raw": "..."} ...]
      }
    """
    if not os.path.isfile(inp_path):
        raise FileNotFoundError(f"INP 不存在: {inp_path}")

    with open(inp_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip() for ln in f]

    ties: List[Dict[str, Any]] = []
    bcs : List[Dict[str, Any]] = []

    re_tie_hdr  = re.compile(r"^\*Tie\b", re.I)
    re_tie_name = re.compile(r"\bname\s*=\s*([^,]+)", re.I)
    re_bnd_hdr  = re.compile(r"^\*Boundary\b", re.I)

    i, n = 0, len(lines)
    while i < n:
        s  = lines[i].strip()
        su = s.upper()

        # ---- *Tie
        if re_tie_hdr.match(su):
            name = None
            m = re_tie_name.search(s)
            if m:
                name = _clean(m.group(1))
            j = i + 1
            toks: List[str] = []
            while j < n and lines[j].strip() and not lines[j].lstrip().startswith("*"):
                toks += [_clean(t) for t in lines[j].split(",") if t.strip()]
                j += 1
                if len(toks) >= 2:
                    break
            if len(toks) >= 2:
                master, slave = toks[0], toks[1]
                ties.append({
                    "name": name or f"TIE@{master}->{slave}",
                    "master": master, "slave": slave
                })
            i = j
            continue

        # ---- *Boundary
        if re_bnd_hdr.match(su):
            i += 1
            while i < n and lines[i].strip() and not lines[i].lstrip().startswith("*"):
                row = [t.strip() for t in lines[i].split(",")]
                if not row:
                    i += 1
                    continue
                if len(row) >= 2 and row[1].upper().startswith("ENCASTRE"):
                    bcs.append({"set": row[0], "type": "ENCASTRE", "dof1": 1, "dof2": 6, "raw": lines[i]})
                else:
                    # 形如：NSET, dof1, dof2, value
                    def _to_int(x):
                        try:
                            return int(float(x))
                        except Exception:
                            return None
                    d1 = _to_int(row[1]) if len(row) >= 2 else None
                    d2 = _to_int(row[2]) if len(row) >= 3 else d1
                    bcs.append({"set": row[0], "type": "BOUNDARY", "dof1": d1, "dof2": d2, "raw": lines[i]})
                i += 1
            continue

        i += 1

    return {"ties": ties, "bcs": bcs}


# ============================================================
#  2) 运行“附加流程”验证（若工具存在）
# ============================================================
class _StubTotal:
    """最小 TotalEnergy 替身：只收集 attach 进来的 ties/bcs。"""
    def __init__(self):
        self.ties: List[Any] = []
        self.bcs : List[Any] = []
    def attach(self, ties=None, bcs=None, **_):
        if ties: self.ties.extend(ties)
        if bcs : self.bcs.extend(bcs)


def try_attach_flow(asm, inp_path: str, cfg_like) -> Dict[str, int]:
    """
    若项目中存在 attach_ties_bcs 工具，则跑一遍“解析+构造+挂载”流程；
    返回 {'ties': X, 'bcs': Y}。
    """
    if not _HAS_ATTACH or attach_ties_and_bcs_from_inp is None:
        raise ImportError("未找到 train.attach_ties_bcs.attach_ties_and_bcs_from_inp")
    stub = _StubTotal()
    attach_ties_and_bcs_from_inp(
        total=stub,
        asm=asm,
        inp_path=inp_path,
        cfg=cfg_like,
    )
    return {"ties": len(stub.ties), "bcs": len(stub.bcs)}


# ============================================================
#  3) 主流程
# ============================================================
def main():
    # ---- 参数
    ap = argparse.ArgumentParser(description="Sanity Check — *Tie / *Boundary / ENCASTRE 检测与挂载验证")
    ap.add_argument("--inp", type=str, default=None, help="INP 路径（不传则自动选择）")
    args = ap.parse_args()

    # ---- 选 INP：args -> TrainerConfig.inp_path -> 你的固定路径 -> data/shuangfan.inp
    default_user = r"D:\shuangfan\shuangfan.inp"
    default_proj = os.path.join(ROOT, "data", "shuangfan.inp")
    if TrainerConfig is not None:
        cfg0 = TrainerConfig()
        candidates = [args.inp, getattr(cfg0, "inp_path", None), default_user, default_proj]
        cfg_like = cfg0
        # 确保 attach 流程需要的键存在
        if not hasattr(cfg_like, "n_tie_points"): cfg_like.n_tie_points = 2000
        if not hasattr(cfg_like, "tie_alpha"):    cfg_like.tie_alpha    = 1.0e3
        if not hasattr(cfg_like, "bc_alpha"):     cfg_like.bc_alpha     = 1.0e4
    else:
        candidates = [args.inp, default_user, default_proj]
        cfg_like = SimpleNamespace(
            n_tie_points=2000, tie_alpha=1.0e3, bc_alpha=1.0e4
        )

    inp_path = next((p for p in candidates if p and os.path.isfile(p)), candidates[-1])
    if not inp_path or not os.path.isfile(inp_path):
        print(f"[ERR] 没找到可用的 INP。候选：{candidates}")
        sys.exit(2)

    print("=" * 70)
    print("[1/3] 文本解析 INP：*Tie / *Boundary / ENCASTRE")
    print("=" * 70)
    try:
        parsed = parse_inp_tie_boundary(inp_path)
        ties = parsed["ties"]
        bcs  = parsed["bcs"]
        print(f"[INP] 解析统计：ties={len(ties)}, boundary={len(bcs)}")
        if ties:
            print("  · Tie 前 5 条：")
            for i, t in enumerate(ties[:5], start=1):
                print(f"    - #{i}: name='{t.get('name')}', master='{t.get('master')}', slave='{t.get('slave')}'")
        else:
            print("  · 未发现 *Tie 语句。")
        if bcs:
            enc_cnt = sum(1 for b in bcs if b.get("type") == "ENCASTRE"
                          or (b.get("dof1") == 1 and b.get("dof2") == 6))
            print(f"  · Boundary 前 5 条（其中 ENCASTRE≈{enc_cnt}）：")
            for i, b in enumerate(bcs[:5], start=1):
                d1, d2 = b.get("dof1"), b.get("dof2")
                span = f"DOF {d1}..{d2}" if d1 is not None else "DOF ?"
                print(f"    - #{i}: set='{b.get('set')}', type='{b.get('type')}', {span}")
        else:
            print("  · 未发现 *Boundary 语句。")
    except Exception as e:
        print(f"[ERR] INP 解析失败：{e}")
        sys.exit(2)

    print("\n" + "=" * 70)
    print("[2/3] 读取 Assembly：load_inp(...)")
    print("=" * 70)
    try:
        asm = load_inp(inp_path)
        nsurf = len(getattr(asm, "surfaces", {}))
        nelst = len(getattr(asm, "elsets", {}))
        print(f"[ASM] surfaces={nsurf}, elsets={nelst}")
    except Exception as e:
        print(f"[ERR] 读取 INP -> Assembly 失败：{e}")
        print("      无法执行后续“构造+挂载”验证。")
        sys.exit(2)

    print("\n" + "=" * 70)
    print("[3/3] （可选）运行构造+挂载验证：attach_ties_and_bcs_from_inp(...)")
    print("=" * 70)
    if not _HAS_ATTACH:
        print("[warn] 未找到 train.attach_ties_bcs.attach_ties_and_bcs_from_inp，跳过本步骤。")
        print("       如需连同构造/挂载一起验证，请添加该文件。")
        print("\n✅ 体检完成（已打印 Tie/Boundary 解析统计）。")
        sys.exit(0)

    try:
        counts = try_attach_flow(asm=asm, inp_path=inp_path, cfg_like=cfg_like)
        print(f"[ok] 构造+挂载成功：ties={counts['ties']}, encastre_bcs={counts['bcs']}")
        if counts["ties"] == 0:
            print("[warn] 代码层面未挂载到任何 Tie；请检查 *Tie 名称与 master/slave 表面键是否与 asm.surfaces 一致。")
        if counts["bcs"] == 0:
            print("[warn] 代码层面未挂载到任何 ENCASTRE/固定边界；请检查 *Boundary/*Nset 与节点集展开是否成功。")
        print("\n✅ 体检完成。")
        sys.exit(0)
    except Exception as e:
        print(f"[ERR] 构造/挂载流程失败：{e}")
        print("      建议：")
        print("        1) attach_ties_bcs.py 中使用的装配接口函数名与工程一致；")
        print("        2) cfg_like.n_tie_points / tie_alpha / bc_alpha 存在且数值合理；")
        print("        3) 表面/节点集名称大小写/引号/空格与 INP 完全一致。")
        sys.exit(3)


if __name__ == "__main__":
    main()
