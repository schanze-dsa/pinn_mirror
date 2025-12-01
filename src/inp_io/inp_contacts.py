# -*- coding: utf-8 -*-
"""
inp_contacts.py
----------------
Abaqus INP 接触信息解析器（轻依赖版）。
识别以下指令并转为结构化数据：
  *SURFACE
  *CONTACT PAIR
  *TIE
  *SURFACE INTERACTION   (只记录名称与参数，不深入力学细节)
也会统计 *CONTACT 的存在次数（用于 sanity 展示）。

主要公开函数：
  - parse_inp_contacts(path) -> InpContacts
  - discover_contact_pairs(path) -> List[Dict[str, Any]]
  - quick_summary(path) -> str

集成到 Trainer 的典型用法（在 trainer.build() 里）：
  from io.inp_contacts import discover_contact_pairs
  if not cfg.contact_pairs:
      cfg.contact_pairs = discover_contact_pairs(cfg.inp_path)  # [{'type': 'contact_pair'|'tie', 'master':..., 'slave':..., 'interaction':...}, ...]

注意：
  - 本解析器只做“名称层面”的解析，不去展开具体几何/单元面。
  - 若你的 INP 使用了 *CONTACT / *CONTACT INCLUSIONS / *CONTACT PROPERTY 等更复杂新语法，
    仍可通过统计与弱解析给出有用的提示，但不会做完整语义还原。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any


# =========================
# 数据结构
# =========================

@dataclass
class SurfaceDef:
    """Abaqus *SURFACE 定义（名称与少量元数据）"""
    name: str
    stype: Optional[str] = None     # TYPE=ELEMENT / NODE 等
    internal_lines: List[str] = field(default_factory=list)  # 面内条目（未做几何展开）


@dataclass
class ContactPairDef:
    """来自 *CONTACT PAIR 的接触对定义"""
    master: str
    slave: str
    interaction: Optional[str] = None
    options: Dict[str, str] = field(default_factory=dict)     # 额外的关键字参数
    raw_line: Optional[str] = None                             # 原始行（便于溯源）

    def as_trainer_dict(self) -> Dict[str, Any]:
        return {
            "type": "contact_pair",
            "master": self.master,
            "slave": self.slave,
            "interaction": self.interaction
        }


@dataclass
class TieDef:
    """来自 *TIE 的约束对（把它当作一种‘接触耦合’）"""
    name: Optional[str]
    master: str
    slave: str
    options: Dict[str, str] = field(default_factory=dict)
    raw_line: Optional[str] = None

    def as_trainer_dict(self) -> Dict[str, Any]:
        return {
            "type": "tie",
            "master": self.master,
            "slave": self.slave,
            "interaction": self.name or None  # 用 name 字段承载标识
        }


@dataclass
class SurfaceInteractionDef:
    """记录 *SURFACE INTERACTION 的名称与参数（不展开具体接触性质）"""
    name: Optional[str]
    options: Dict[str, str] = field(default_factory=dict)


@dataclass
class InpContacts:
    """聚合解析结果"""
    surfaces: Dict[str, SurfaceDef] = field(default_factory=dict)
    contact_pairs: List[ContactPairDef] = field(default_factory=list)
    ties: List[TieDef] = field(default_factory=list)
    surface_interactions: Dict[str, SurfaceInteractionDef] = field(default_factory=dict)
    keyword_counts: Dict[str, int] = field(default_factory=dict)  # 例如 {'CONTACT': 13, 'CONTACT_PAIR': 8, ...}


# =========================
# 工具：读取 + 预处理
# =========================

_COMMENT_LINE = re.compile(r'^\s*\*\*')  # Abaqus 注释行以 ** 开头
_STAR_LINE = re.compile(r'^\s*\*', re.IGNORECASE)

def _read_text(path: str) -> str:
    """读取 INP 文本，尝试多种编码以避免报错。"""
    for enc in ("utf-8", "gbk", "latin-1"):
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                return f.read()
        except Exception:
            continue
    # 最后一次宽松读取
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _normalize_line(s: str) -> str:
    """去除换行两端空白"""
    return s.strip()


def _split_keyvals(opt_str: str) -> Dict[str, str]:
    """
    解析形如 "INTERACTION=Int-1, ADJUST=YES" 的参数列表
    返回 dict，键值均做 strip 与大写键。
    """
    out: Dict[str, str] = {}
    if not opt_str:
        return out
    parts = [p.strip() for p in opt_str.split(",") if p.strip()]
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            out[k.strip().upper()] = v.strip()
        else:
            out[p.strip().upper()] = "YES"
    return out


# =========================
# 解析器主体
# =========================

def parse_inp_contacts(path: str) -> InpContacts:
    """
    粗解析 Abaqus INP 中的接触相关信息，返回 InpContacts。
    解析策略保守：不深入几何，仅抽取名称层面的 master/slave/interaction。
    """
    text = _read_text(path)
    lines = text.splitlines()

    res = InpContacts()

    # 统计关键词次数（供 sanity 展示）
    def _count_kw(kw: str) -> int:
        # 仅统计行首 *KW（忽略参数）
        pat = re.compile(rf"^\s*\*{re.escape(kw)}\b", re.IGNORECASE)
        return sum(1 for ln in lines if pat.search(ln))
    for kw in ("CONTACT", "CONTACT PAIR", "TIE", "SURFACE", "SURFACE INTERACTION", "CONTACT PROPERTY"):
        res.keyword_counts[kw.replace(" ", "_")] = _count_kw(kw)

    i = 0
    n = len(lines)

    # 扫描全文件
    while i < n:
        raw = lines[i]
        if _COMMENT_LINE.match(raw) or not raw.strip():
            i += 1
            continue

        # 命令行：以 * 开头
        if _STAR_LINE.match(raw):
            # 取关键字和参数串
            first = raw.strip().lstrip("*")
            # 关键字名（逗号或行尾之前）
            if "," in first:
                kw, opt = first.split(",", 1)
                kw = kw.strip().upper()
                opt = opt.strip()
            else:
                kw = first.strip().upper()
                opt = ""

            # 统一化
            kw = kw.replace("_", " ")

            # ---------- 解析 *SURFACE ----------
            if kw == "SURFACE":
                opts = _split_keyvals(opt)
                name = opts.get("NAME") or opts.get("SURFACE")  # 有些写法
                stype = opts.get("TYPE")
                # 收集到下一条 * 命令前的内容
                body: List[str] = []
                j = i + 1
                while j < n and not _STAR_LINE.match(lines[j]):
                    ln = lines[j]
                    if not _COMMENT_LINE.match(ln) and ln.strip():
                        body.append(_normalize_line(ln))
                    j += 1
                if name:
                    # 名称大小写保持原样，但内部索引用统一小写去重
                    key = name.lower()
                    res.surfaces[key] = SurfaceDef(name=name, stype=stype, internal_lines=body)
                i = j
                continue

            # ---------- 解析 *SURFACE INTERACTION ----------
            if kw == "SURFACE INTERACTION":
                opts = _split_keyvals(opt)
                name = opts.get("NAME")
                key = (name or f"__INT_{i}__").lower()
                res.surface_interactions[key] = SurfaceInteractionDef(name=name, options=opts)
                # 跳过其可能的参数/表格（直到下一个 * 行）
                j = i + 1
                while j < n and not _STAR_LINE.match(lines[j]):
                    j += 1
                i = j
                continue

            # ---------- 解析 *CONTACT PAIR ----------
            if kw == "CONTACT PAIR":
                opts = _split_keyvals(opt)
                interaction = opts.get("INTERACTION")
                # 其 body 每行形如： master_surface , slave_surface
                j = i + 1
                while j < n and not _STAR_LINE.match(lines[j]):
                    ln = lines[j].strip()
                    if ln and not _COMMENT_LINE.match(ln):
                        # 去掉尾注释/多余空白
                        # 有时可能写“Master,Slave”，或 “Master , Slave”
                        parts = [p.strip() for p in ln.split(",") if p.strip()]
                        if len(parts) >= 2:
                            master = parts[0]
                            slave = parts[1]
                            cpd = ContactPairDef(
                                master=master,
                                slave=slave,
                                interaction=interaction,
                                options=opts.copy(),
                                raw_line=ln
                            )
                            res.contact_pairs.append(cpd)
                    j += 1
                i = j
                continue

            # ---------- 解析 *TIE ----------
            if kw == "TIE":
                opts = _split_keyvals(opt)
                tname = opts.get("NAME")
                # body：一行“master, slave”
                j = i + 1
                while j < n and not _STAR_LINE.match(lines[j]):
                    ln = lines[j].strip()
                    if ln and not _COMMENT_LINE.match(ln):
                        parts = [p.strip() for p in ln.split(",") if p.strip()]
                        if len(parts) >= 2:
                            master = parts[0]
                            slave = parts[1]
                            td = TieDef(
                                name=tname,
                                master=master,
                                slave=slave,
                                options=opts.copy(),
                                raw_line=ln
                            )
                            res.ties.append(td)
                    j += 1
                i = j
                continue

            # ---------- 其他命令：跳过其 body ----------
            j = i + 1
            while j < n and not _STAR_LINE.match(lines[j]):
                j += 1
            i = j
            continue

        # 非命令/非注释普通行（通常不应出现在顶层），跳过
        i += 1

    return res


# =========================
# 输出给 Trainer / Sanity 的接口
# =========================

def discover_contact_pairs(inppath: str) -> List[Dict[str, Any]]:
    """
    从 INP 里查找接触对与 tie，合并为 Trainer 可直接使用的简化结构列表：
      [{'type': 'contact_pair'|'tie', 'master': 'SurfA', 'slave': 'SurfB', 'interaction': 'Int-1'|None}, ...]
    """
    info = parse_inp_contacts(inppath)

    # 去重（master, slave, type, interaction）层面
    seen: set[Tuple[str, str, str, Optional[str]]] = set()
    out: List[Dict[str, Any]] = []

    for c in info.contact_pairs:
        key = (c.master, c.slave, "contact_pair", c.interaction)
        if key not in seen:
            out.append(c.as_trainer_dict())
            seen.add(key)

    for t in info.ties:
        key = (t.master, t.slave, "tie", t.name or None)
        if key not in seen:
            out.append(t.as_trainer_dict())
            seen.add(key)

    return out


def quick_summary(inppath: str) -> str:
    """
    生成一段可打印的摘要，便于快速检查接触/绑定配置。
    """
    info = parse_inp_contacts(inppath)

    lines: List[str] = []
    kc = info.keyword_counts

    lines.append("[contact] 关键字计数：")
    lines.append(f"  *CONTACT           : {kc.get('CONTACT', 0)}")
    lines.append(f"  *CONTACT PAIR      : {kc.get('CONTACT_PAIR', 0)}")
    lines.append(f"  *TIE               : {kc.get('TIE', 0)}")
    lines.append(f"  *SURFACE           : {kc.get('SURFACE', 0)}")
    lines.append(f"  *SURFACE INTERACTION: {kc.get('SURFACE_INTERACTION', 0)}")
    lines.append(f"  *CONTACT PROPERTY  : {kc.get('CONTACT_PROPERTY', 0)}")

    lines.append("")
    lines.append("[contact] SURFACE 列表（仅显示名称与 TYPE）：")
    if info.surfaces:
        for s in sorted(info.surfaces.values(), key=lambda x: x.name.lower()):
            lines.append(f"  - {s.name} (TYPE={s.stype or 'N/A'})")
    else:
        lines.append("  (无)")

    lines.append("")
    lines.append("[contact] CONTACT PAIR：")
    if info.contact_pairs:
        for i, c in enumerate(info.contact_pairs):
            lines.append(f"  - [{i}] master={c.master} , slave={c.slave} , interaction={c.interaction or 'None'}")
    else:
        lines.append("  (无)")

    lines.append("")
    lines.append("[contact] TIE：")
    if info.ties:
        for i, t in enumerate(info.ties):
            lines.append(f"  - [{i}] name={t.name or 'None'} , master={t.master} , slave={t.slave}")
    else:
        lines.append("  (无)")

    lines.append("")
    lines.append("[contact] 供 Trainer 使用的接触对（合并 + 去重）：")
    pairs = discover_contact_pairs(inppath)
    if pairs:
        for i, p in enumerate(pairs):
            lines.append(f"  - [{i}] type={p['type']}, master={p['master']}, slave={p['slave']}, interaction={p.get('interaction')}")
    else:
        lines.append("  (无)")

    return "\n".join(lines)


# =========================
# 自测入口（可选）
# =========================

if __name__ == "__main__":
    # 允许直接运行进行一个快速的终端预览
    inp = os.environ.get("INP_PATH", "").strip()
    if not inp:
        print("用法：在环境变量 INP_PATH 中指定 .inp 路径，然后运行本文件。")
    elif not os.path.isfile(inp):
        print(f"[ERR] INP 不存在：{inp}")
    else:
        print(quick_summary(inp))
