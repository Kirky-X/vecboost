#!/usr/bin/env python3
# Copyright (c) 2025-2026 Kirky.X🌠
# SPDX-License-Identifier: Apache-2.0

"""i18n 一致性核验（默认英文 + 硬编码文案回流拦截）。

扫描范围：生产码 `src/` 与 `transcribe-poc/src/`（`examples/` `tests/` `benches/`
属演示与断言面，不在本门禁管辖内）。规则：

1. R1 默认英文：`src/i18n/locales/en/*.ftl` 不得出现任何 CJK 字符（含注释）。
2. R2 键齐性：en 与 zh 的键集合必须完全相等（逐文件比对，不编译即可拦）。
3. R3 引用有效性：代码里静态 `tr("k")` / `tr_with_args("k", ..)` / `tr_locale("k", ..)`
   的 `k`，以及 `#[forge(i18n_key = "k")]` 协议面键（CLI --help / MCP），均须存在于 FTL
   （缺键会在运行时静默回退打印键名）。
4. R4 禁硬编码中文：生产码字符串字面量（注释与 `#[cfg(test)]` 之后除外）禁 CJK；
   开发者可见的 panic/assert 文案与测试夹具须进 allowlist 并写明理由。
5. R5 棘轮：输出宏（`print*/eprintln!` 与 `log::info!/warn!/error!`）里的句子型
   英文字面量站点数必须逐文件 ≤ 基线（`scripts/i18n_baseline.json`）。
   超额 = 新增未国际化文案；低于基线 = 基线未同步下调（同样报错，防基线虚挂）。

`log::debug!/trace!` 不在 R5 计量内（热路径日志按频次分层原则保留英文原文）。
豁免一律按「文件 + 行内片段」锚定，不用裸行号（行号随改动漂移）。

输出：违规清单（stdout）。退出码 0=通过，1=存在违规。
附加 `--list`：按文件逐条打印 R4 中文站点与 R5 未国际化文案站点（供分类与审计，不改变退出码语义）。
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOCALES = ROOT / "src" / "i18n" / "locales"
PROD_DIRS = (ROOT / "src", ROOT / "transcribe-poc" / "src")
ALLOWLIST_PATH = ROOT / "scripts" / "i18n_allowlist.json"
BASELINE_PATH = ROOT / "scripts" / "i18n_baseline.json"

CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")
# 输出宏：stdout/stderr 打印 + 面向用户的日志级别（debug/trace 不计）
OUT_MACROS = ("println!", "println !", "print!", "eprintln!", "eprint!",
              "log::info!", "log::warn!", "log::error!",
              "info!", "warn!", "error!")
TR_CALL = re.compile(r"(?<![A-Za-z_])tr(?:_with_args|_locale)?\s*\(")
TR_KEY = re.compile(
    r'(?<![A-Za-z_])tr(?:_with_args|_locale)?\s*\(\s*"([a-z0-9][a-z0-9-]*)"'
)
# `#[forge(..., i18n_key = "k")]`：协议面（CLI --help / MCP）走 sdforge 注册表，
# 不经 `tr()` 调用面，故单独纳入 R3 校验，否则属性里的错键不会在任何单测报错。
I18N_KEY_ATTR = re.compile(r'i18n_key\s*=\s*"([a-z0-9][a-z0-9-]*)"')
FTL_KEY = re.compile(r"^([a-z][a-z0-9-]*)\s*=", re.M)
# 句子型文本：剔除 `{ ... }` 插值段后仍含字母+空格 —— 排除机器 token
#（WALL_S=、[PASS]、JSON key、纯占位串等）
SENTENCE = re.compile(r"[A-Za-z].*[ ]|[ ][A-Za-z]")
PLACEHOLDER = re.compile(r"\{[^{}]*\}")


def read(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf8", errors="replace")


def prod_files() -> list[pathlib.Path]:
    """生产码文件：`src/` 与 `transcribe-poc/src/`，剔除整文件级测试模块
    （`**/tests.rs`、`**/tests/**` —— 它们由 `#[cfg(test)] mod tests;` 挂载，
    文件内部没有 `#[cfg(test)]` 标记，按文件粒度排除）。"""
    out: list[pathlib.Path] = []
    for d in PROD_DIRS:
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*.rs")):
            if "target" in p.parts:
                continue
            if p.name == "tests.rs" or "tests" in p.parent.parts:
                continue
            out.append(p)
    return out


def mask_comments(text: str) -> str:
    """把注释与字符串字面量之外的注释部分替换成空格，保留换行以维持行号。"""
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == '"' or (c == "r" and i + 1 < n and text[i + 1] in '"#'):
            i = skip_string(text, i)  # 字符串本体不动，只保证注释判定不被误吃
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                out[i] = " "
                i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            depth = 1
            while i < n and depth:
                if text[i] == "\n":
                    i += 1
                    continue
                if text[i] == "/" and i + 1 < n and text[i + 1] == "*":
                    depth += 1
                    out[i] = " "
                    i += 1
                elif text[i] == "*" and i + 1 < n and text[i + 1] == "/":
                    depth -= 1
                    out[i] = " "
                    i += 1
                out[i] = " "
                i += 1
            continue
        i += 1
    return "".join(out)


def skip_string(text: str, i: int) -> int:
    """从下标 i（指向 `"` 或 `r` 起始）跳到字符串结束后的位置。"""
    n = len(text)
    if text[i] == "r":
        j = i + 1
        hashes = 0
        while j < n and text[j] == "#":
            hashes += 1
            j += 1
        if j >= n or text[j] != '"':
            return i + 1
        close = '"' + "#" * hashes
        end = text.find(close, j + 1)
        return n if end < 0 else end + len(close)
    j = i + 1
    while j < n:
        if text[j] == "\\":
            j += 2
            continue
        if text[j] == '"':
            return j + 1
        j += 1
    return n


def iter_literals(text: str):
    """产出 (line_no, literal) —— 仅字符串字面量，跳过注释。"""
    i, n, line = 0, len(text), 1
    while i < n:
        c = text[i]
        if c == "\n":
            line += 1
            i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            i = skip_block_comment(text, i)
            continue
        if c == '"' or (c == "r" and i + 1 < n and text[i + 1] in '"#'):
            start_line = line
            end = skip_string(text, i)
            body = text[i:end]
            body = re.sub(r"^r\#*\"", "", body)
            body = re.sub(r"\"\#*$", "", body)
            body = body.strip('"')
            yield start_line, body
            line += text[i:end].count("\n")
            i = end
            continue
        i += 1


def skip_block_comment(text: str, i: int) -> int:
    n = len(text)
    depth = 1
    j = i + 2
    while j < n and depth:
        if text[j] == "/" and j + 1 < n and text[j + 1] == "*":
            depth += 1
            j += 2
            continue
        if text[j] == "*" and j + 1 < n and text[j + 1] == "/":
            depth -= 1
            j += 2
            continue
        j += 1
    return j


def production_region(text: str) -> str:
    """截断到首个 `#[cfg(test)]` —— 之后按项目惯例属测试区。"""
    cut = text.find("#[cfg(test)]")
    return text if cut < 0 else text[:cut]


def balance_end(text: str, open_idx: int) -> int:
    """返回与 text[open_idx]=='(' 匹配的 ')' 下标，找不到返回 -1。"""
    depth = 0
    i = open_idx
    n = len(text)
    while i < n:
        c = text[i]
        if c == '"' or (c == "r" and i + 1 < n and text[i + 1] in '"#'):
            i = skip_string(text, i)
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def load_allowlist() -> dict:
    if not ALLOWLIST_PATH.is_file():
        return {"r4_cjk": [], "r5_untranslated": []}
    data = json.loads(read(ALLOWLIST_PATH))
    data.setdefault("r4_cjk", [])
    data.setdefault("r5_untranslated", [])
    return data


def exempt(entries: list[dict], rel: str, line_text: str) -> str | None:
    for e in entries:
        if e.get("file") != rel:
            continue
        needle = e.get("match", "")
        if needle and needle in line_text:
            return e.get("reason", "")
    return None


def ftl_keys(path: pathlib.Path) -> set[str]:
    return {m.group(1) for m in FTL_KEY.finditer(read(path))}


def main() -> int:
    listing = "--list" in sys.argv
    violations: list[str] = []
    inventory: list[str] = []
    allow = load_allowlist()

    # --- R1 默认英文：en FTL 零 CJK ---
    en_text = "".join(read(p) for p in sorted((LOCALES / "en").glob("*.ftl")))
    for line_no, line in enumerate(en_text.splitlines(), 1):
        if CJK.search(line):
            violations.append(f"R1 en FTL 含中文（默认语言必须严格英文）: {line.strip()[:90]}")

    # --- R2 键齐性：逐文件比对 en/zh ---
    for name in ("messages.ftl", "errors.ftl"):
        en_keys = ftl_keys(LOCALES / "en" / name)
        zh_keys = ftl_keys(LOCALES / "zh" / name)
        for missing in sorted(en_keys - zh_keys):
            violations.append(f"R2 {name}: 键 {missing} 缺 zh 侧译文")
        for extra in sorted(zh_keys - en_keys):
            violations.append(f"R2 {name}: 键 {extra} 缺 en 侧译文")

    all_keys = ftl_keys(LOCALES / "en" / "messages.ftl") | ftl_keys(LOCALES / "en" / "errors.ftl")
    baseline: dict[str, int] = {}
    if BASELINE_PATH.is_file():
        baseline = json.loads(read(BASELINE_PATH)).get("files", {})
    actual_counts: dict[str, int] = {}
    reasons: dict[str, str] = {}
    for rel, v in baseline.items():
        reasons[rel] = v.get("reason", "") if isinstance(v, dict) else ""
    baseline_counts = {k: (v["count"] if isinstance(v, dict) else v) for k, v in baseline.items()}
    for path in prod_files():
        rel = str(path.relative_to(ROOT))
        text = read(path)
        prod = production_region(text)
        lines = prod.splitlines()

        # --- R3 静态 tr 引用键必须存在 ---
        for line_no, line in enumerate(lines, 1):
            if line.strip().startswith("//"):
                continue
            for key in TR_KEY.findall(line) + I18N_KEY_ATTR.findall(line):
                if key not in all_keys:
                    violations.append(f"R3 {rel}:{line_no} 引用缺键 {key}（运行时回退打印键名）")

        # --- R4 生产码字符串字面量禁 CJK ---
        for line_no, literal in iter_literals(prod):
            if not CJK.search(literal):
                continue
            line_text = lines[line_no - 1] if line_no <= len(lines) else literal
            reason = exempt(allow["r4_cjk"], rel, line_text)
            if listing:
                mark = f"[豁免:{reason}]" if reason else "[违规]"
                inventory.append(
                    f"R4 {rel}:{line_no} {mark} {literal.strip()[:80]}"
                )
            if reason is None and not listing:
                violations.append(
                    f"R4 {rel}:{line_no} 硬编码中文字面量未国际化: {literal.strip()[:70]!r}"
                    "（豁免须进 scripts/i18n_allowlist.json 并写明理由）"
                )

        # --- R5 输出宏句子型英文字面量站点计数（棘轮）---
        masked = mask_comments(prod)
        hits = 0
        for m in re.finditer(r"\b(" + "|".join(map(re.escape, OUT_MACROS)) + r")\s*\(", masked):
            end = balance_end(masked, m.end() - 1)
            if end < 0:
                continue
            body = masked[m.end():end]
            if TR_CALL.search(body):
                continue
            sentence = next(
                (
                    lit
                    for _, lit in iter_literals(body)
                    if SENTENCE.search(PLACEHOLDER.sub("", lit))
                ),
                None,
            )
            if sentence is None:
                continue
            line_no = masked[:m.start()].count("\n") + 1
            line_text = lines[line_no - 1] if line_no <= len(lines) else sentence
            reason = exempt(allow["r5_untranslated"], rel, line_text)
            if reason is not None:
                if listing:
                    inventory.append(f"R5 {rel}:{line_no} [豁免:{reason}] {sentence.strip()[:80]}")
                continue
            if listing:
                inventory.append(f"R5 {rel}:{line_no} [待国际化] {sentence.strip()[:80]}")
                continue
            hits += 1
        if hits:
            actual_counts[rel] = hits
            limit = baseline_counts.get(rel)
            if limit is None:
                violations.append(
                    f"R5 {rel} 出现 {hits} 处未国际化输出文案，基线未登记该文件"
                    "（新面需先在 scripts/i18n_baseline.json 记账）"
                )
            elif hits > limit:
                violations.append(f"R5 {rel} 未国际化输出文案 {hits} > 基线 {limit}（禁止回流）")
            elif hits < limit:
                reason = reasons.get(rel, "")
                if reason:
                    # 已按频次分层/机器输出定性的常驻面：基线即上限，不下调
                    pass
                else:
                    violations.append(
                        f"R5 {rel} 已降至 {hits} < 基线 {limit}，请把基线同步下调（防虚挂）"
                    )

    stale = sorted(set(baseline_counts) - set(actual_counts))
    for rel in stale:
        violations.append(
            f"R5 基线残留 {rel}（现值 0 < 基线 {baseline_counts[rel]}），请下调或删除该条目"
        )

    if listing:
        for item in inventory:
            print(item)
        print(f"-- 站点合计：{len(inventory)}（R4 {sum(1 for x in inventory if x.startswith('R4'))}"
              f" / R5 {sum(1 for x in inventory if x.startswith('R5'))}）")
        return 0

    if violations:
        print(f"i18n 一致性核验失败：{len(violations)} 项")
        for v in violations:
            print(f"  - {v}")
        return 1

    print(
        f"i18n 一致性核验通过：en/zh 各 {len(all_keys)} 键齐平；"
        f"生产码 {len(prod_files())} 文件无硬编码中文与未登记文案"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
