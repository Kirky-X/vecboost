#!/usr/bin/env python3
"""merge_delta_spec.py — 把 delta spec 确定性地合并进 main spec。

替代 archive --sync 此前由 LLM 子 agent 执行的合并（见 references/archive.md）。
合并是确定性结构操作（违反"确定性逻辑禁止交给模型"，故改为显式代码）。

Spec 文件格式（Markdown）：

    # Spec — <capability>

    > Delta spec for change `<name>`. ...   （仅 delta 有此行）

    ## Requirements

    ### R-<cap>-NNN: <标题>
    <正文>

    **验收标准：**
    - <条目>

    ## Constraints
    <自由文本/列表>

    ## Out of Scope
    <自由文本/列表>

合并语义（纯代码，无 LLM）：

  Requirements（按 R-<cap>-NNN 键结构化）：
    - ADD    : R-ID 仅在 delta                → 追加
    - MODIFY : R-ID 同时存在                  → delta 的标题+正文替换 main
    - DELETE : delta 标题文本 == `~~DELETE~~` → 丢弃该 R-ID
    - KEEP   : R-ID 仅在 main                 → 原样保留
    输出顺序：所有 R-ID 按数字后缀稳定排序（确定性规范序）。

  Constraints / Out of Scope（按行的自由文本）：
    - 按精确（去空白）行做并集；main 顺序在前，delta 独有行追加在后。
    - 要"修改"某行约束：直接编辑 main spec（合并不做语义近似）。

幂等性：对同一 (main, delta) 合并两次产生完全相同的字节。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REQ_HEADER_RE = re.compile(r"^###\s+(R-[A-Za-z0-9][A-Za-z0-9-]*?-\d+)\s*:\s*(.*)$")
SECTION_RE = re.compile(r"^##\s+(.+?)\s*$")
TITLE_RE = re.compile(r"^#\s+(.+?)\s*$")

DELETE_MARKER = "~~DELETE~~"


class Spec:
    def __init__(self) -> None:
        self.title: str = ""  # 整行，如 "# Spec — references-consistency"
        self.intro: list[str] = []  # 标题后、首个 ## 前的行
        self.requirements: dict[
            str, dict
        ] = {}  # R-ID -> {"title": str, "body": [lines]}
        self.sections: dict[str, list[str]] = {}  # 段名 -> 行（非 Requirements 段）
        self.section_order: list[str] = []


def _strip_blank_ends(lines: list[str]) -> list[str]:
    out = list(lines)
    while out and out[-1].strip() == "":
        out.pop()
    while out and out[0].strip() == "":
        out.pop(0)
    return out


def parse_spec(path: Path) -> Spec:
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    spec = Spec()

    n = len(lines)
    i = 0
    while i < n and lines[i].strip() == "":
        i += 1
    if i < n:
        m = TITLE_RE.match(lines[i])
        if m:
            spec.title = lines[i].rstrip()
            i += 1

    intro_start = i
    while i < n and not SECTION_RE.match(lines[i]):
        i += 1
    spec.intro = _strip_blank_ends(lines[intro_start:i])

    cur_section: str | None = None
    cur_body: list[str] = []
    in_req = False
    cur_req_id: str | None = None
    cur_req_title: str | None = None
    cur_req_body: list[str] = []

    def flush_section() -> None:
        nonlocal cur_section, cur_body
        if cur_section is not None:
            spec.sections[cur_section] = _strip_blank_ends(cur_body)
            spec.section_order.append(cur_section)
        cur_body = []

    def flush_req() -> None:
        nonlocal cur_req_id, cur_req_title, cur_req_body
        if cur_req_id is not None:
            spec.requirements[cur_req_id] = {
                "title": (cur_req_title or "").rstrip(),
                "body": _strip_blank_ends(cur_req_body),
            }
        cur_req_id = None
        cur_req_title = None
        cur_req_body = []

    while i < n:
        line = lines[i]
        m_sec = SECTION_RE.match(line)
        if m_sec:
            if in_req:
                flush_req()
            flush_section()
            cur_section = m_sec.group(1).strip()
            in_req = cur_section.lower() == "requirements"
            cur_body = []
            i += 1
            continue
        if in_req:
            m_req = REQ_HEADER_RE.match(line)
            if m_req:
                flush_req()
                cur_req_id = m_req.group(1)
                cur_req_title = m_req.group(2)
                cur_req_body = []
                i += 1
                continue
            if cur_req_id is not None:
                cur_req_body.append(line.rstrip())
                i += 1
                continue
            cur_body.append(line.rstrip())
            i += 1
            continue
        if cur_section is not None:
            cur_body.append(line.rstrip())
        i += 1

    if in_req:
        flush_req()
    flush_section()
    return spec


def _capability_from_title(title: str) -> str:
    m = TITLE_RE.match(title) if title else None
    if not m:
        return "capability"
    rest = m.group(1)
    if rest.lower().startswith("spec"):
        rest = re.sub(r"^spec\s*[—\-:]\s*", "", rest, flags=re.IGNORECASE)
    return rest.strip() or "capability"


def _sort_req_ids(ids):
    def key(rid):
        m = re.match(r"^R-(.+)-(\d+)$", rid)
        if m:
            return (m.group(1).lower(), int(m.group(2)))
        return (rid.lower(), 0)

    return sorted(ids, key=key)


def merge(main: Spec, delta: Spec, main_existed: bool) -> Spec:
    result = Spec()

    if main_existed:
        result.title = main.title
        result.intro = list(main.intro)
    else:
        result.title = delta.title or "# Spec — capability"
        cap = _capability_from_title(result.title)
        result.intro = [f"> Main spec for capability `{cap}`."]

    result.section_order = list(main.section_order)
    for s in delta.section_order:
        if s not in result.section_order:
            result.section_order.append(s)

    merged_reqs: dict[str, dict] = {}
    for rid in set(main.requirements) | set(delta.requirements):
        in_main = rid in main.requirements
        in_delta = rid in delta.requirements
        if in_delta:
            d = delta.requirements[rid]
            if d["title"].strip() == DELETE_MARKER:
                continue  # DELETE
            merged_reqs[rid] = {
                "title": d["title"],
                "body": list(d["body"]),
            }  # MODIFY / ADD
        elif in_main:
            merged_reqs[rid] = {
                "title": main.requirements[rid]["title"],
                "body": list(main.requirements[rid]["body"]),
            }  # KEEP
    result.requirements = merged_reqs

    result.sections = {}
    for s in result.section_order:
        if s.lower() == "requirements":
            result.sections[s] = list(main.sections.get(s, []))
            continue
        main_body = main.sections.get(s, [])
        delta_body = delta.sections.get(s, [])
        merged = list(main_body)
        seen = {ln.strip() for ln in main_body if ln.strip()}
        for ln in delta_body:
            if ln.strip() and ln.strip() not in seen:
                merged.append(ln)
                seen.add(ln.strip())
        result.sections[s] = merged
    return result


def emit(spec: Spec) -> str:
    blocks: list[str] = []
    if spec.title:
        blocks.append(spec.title)
    intro = [ln for ln in spec.intro if ln.strip()]
    if intro:
        blocks.append("\n".join(intro))

    for sec in spec.section_order:
        if sec.lower() == "requirements":
            req_intro = [ln for ln in spec.sections.get(sec, []) if ln.strip()]
            req_blocks: list[str] = []
            if req_intro:
                req_blocks.append("\n".join(req_intro))
            for rid in _sort_req_ids(spec.requirements.keys()):
                d = spec.requirements[rid]
                hdr = f"### {rid}: {d['title']}".rstrip()
                body = "\n".join(d["body"]).strip("\n")
                req_blocks.append(hdr + "\n\n" + body if body else hdr)
            section_text = "\n\n".join(req_blocks)
            blocks.append(
                f"## {sec}\n\n" + section_text if section_text else f"## {sec}"
            )
        else:
            body = "\n".join(spec.sections.get(sec, [])).strip("\n")
            blocks.append(f"## {sec}\n\n" + body if body else f"## {sec}")

    return "\n\n".join(blocks) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="把 delta spec 确定性地合并进 main spec（替代 LLM 合并）。",
    )
    ap.add_argument("--main", required=True, help="main spec 路径（可不存在）")
    ap.add_argument("--delta", required=True, help="delta spec 路径（必须存在）")
    ap.add_argument("--out", help="输出路径（默认写回 --main）")
    ap.add_argument("--dry-run", action="store_true", help="只打印合并结果，不写文件")
    args = ap.parse_args(argv)

    main_path = Path(args.main)
    delta_path = Path(args.delta)
    if not delta_path.is_file():
        print(f"error: delta spec 未找到: {delta_path}", file=sys.stderr)
        return 1

    main_existed = main_path.is_file()
    main_spec = parse_spec(main_path) if main_existed else Spec()
    delta_spec = parse_spec(delta_path)

    result = merge(main_spec, delta_spec, main_existed)
    out_text = emit(result)

    if args.dry_run:
        sys.stdout.write(out_text)
        if not out_text.endswith("\n"):
            sys.stdout.write("\n")
        return 0

    out_path = Path(args.out) if args.out else main_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out_text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
