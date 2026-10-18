#!/usr/bin/env python3
# Copyright (c) 2025-2026 Kirky.X🌠
# SPDX-License-Identifier: Apache-2.0

"""文档-代码一致性核验。

系统性提取文档中可验证声明并与代码核对：
1. HTTP 端点：文档声明的 /api/1/* 与 /v1/*、/health、/metrics 路径必须存在于 #[forge] 注册
2. gRPC 方法：文档声明的 vecboost.* 必须与 #[forge(grpc_method=...)] 一致
3. 环境变量：文档声明的 VECBOOST_* 必须能映射到 VecboostConfig 字段、被 Rust 代码
   或 scripts/ 运维脚本显式消费
4. CLI 子命令：文档示例的子命令必须在 CLI 注册中
5. 文档锚点：`](#frag)` 与 `](path#frag)` 必须解析得到目标的 GitHub heading id，
   且标题不得含 U+FE0F 变体选择符（会隐形残留进 id）

输出：差异清单（stdout）。退出码 0=一致，1=存在差异。
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = (
    [ROOT / "README.md", ROOT / "README_EN.md"]
    + sorted((ROOT / "docs").rglob("*.md"))
)
SRC = ROOT / "src"

# ---------------------------------------------------------------- 代码侧事实

def src_text() -> str:
    # src/ 为主（forge 路由/gRPC/CLI 事实源）；benches/ 一并纳入以核验基准类
    # 环境变量（如 VECBOOST_BENCH_MODEL 仅被 benches 消费）。
    # tests/ 纳入以核验测试专用环境变量（如 VECBOOST_GGUF_MODEL）。
    return "\n".join(
        p.read_text(errors="replace")
        for d in (ROOT / "src", ROOT / "benches", ROOT / "tests")
        for p in d.rglob("*.rs")
    )


def code_routes(src: str) -> set[str]:
    """从 #[forge(path = "...", method = ...)] 与 no_prefix 提取注册路由。"""
    routes: set[str] = set()
    for m in re.finditer(r"#\[\s*forge\s*\((.*?)\)\s*\]", src, re.S):
        attrs = m.group(1)
        pm = re.search(r'path\s*=\s*"([^"]+)"', attrs)
        if not pm:
            continue
        path = pm.group(1)
        no_prefix = "no_prefix" in attrs and "true" in attrs
        routes.add(path if no_prefix else f"/api/1{path}")
    # 手写例外
    if 'route("/metrics"' in src:
        routes.add("/metrics")
    return routes


def code_grpc_methods(src: str) -> set[str]:
    return set(re.findall(r'grpc_method\s*=\s*"(vecboost\.[a-z_]+)"', src))


def code_env_vars(src: str) -> set[str]:
    """代码显式消费的 VECBOOST_* 环境变量。

    Rust 面取字符串字面量；scripts/ 运维脚本（如 build/run-memcap.sh 的
    cgroup 上限旋钮）以 shell 裸引用消费，一并纳入。
    """
    consumed = set(re.findall(r'"(VECBOOST_[A-Z0-9_]+)"', src))
    scripts = ROOT / "scripts"
    if scripts.is_dir():
        for p in scripts.iterdir():
            if p.suffix in (".sh", ".py"):
                consumed |= set(
                    re.findall(r"(VECBOOST_[A-Z0-9_]+)", p.read_text(errors="replace"))
                )
    return consumed


def code_cli_subcommands(src: str) -> set[str]:
    # 全量 CLI 面：所有 forge(cli = true) 的 name（旧版白名单只覆盖 5 个旧命令，
    # "代码事实 cli=N" 打印值与真实 CLI 面脱节）
    return set(re.findall(r'name\s*=\s*"([a-z_]+)"\s*,\s*version\s*=\s*1\s*,\s*cli\s*=\s*true', src))


# confers env 映射可用性：VECBOOST_<SECTION>_<FIELD>，FIELD 不含下划线时可达
CONFIG_SECTIONS: dict[str, set[str]] = {
    "server": {"host", "port", "timeout", "grpc", "workers", "body", "request", "cors"},
    "model": {"repo", "revision", "path", "gpu", "size", "dimension", "length", "batch", "quantized", "max", "resident"},
    "embedding": set(),
    "rerank": set(),
    "monitoring": set(),
    "auth": set(),
    "rate": set(),
    "audit": set(),
    "logging": set(),
    "semantic": set(),
    "device": set(),
    "database": set(),
    "pipeline": set(),
    "memory": set(),
}
# 上表中 section/field 为前缀匹配（如 server.grpc* / model.repo*），subset 字段可放宽；
# 字段名含下划线（如 max_batch_size）无法通过 confers env 映射，属文档禁用区。

EXPLICIT_ENV_OK = {
    "VECBOOST_JWT_SECRET", "VECBOOST_ADMIN_PASSWORD", "VECBOOST_ENCRYPTION_KEY",
    "VECBOOST_REQUIRE_ENCRYPTION", "VECBOOST_KEY_STORAGE_TYPE", "VECBOOST_KEY_FILE_PATH",
    "VECBOOST_LANG", "HF_ENDPOINT", "VECBOOST_DATABASE_PASSWORD", "VECBOOST_MODEL_API_KEY",
    "VECBOOST_LOG_LEVEL", "VECBOOST_ALLOW_INSECURE",
}


def env_var_resolvable(var: str, src: str) -> bool:
    if var in code_env_vars(src):
        return True
    if var in EXPLICIT_ENV_OK:
        return var in src
    body = var[len("VECBOOST_"):].lower()
    parts = body.split("_")
    if len(parts) < 2:
        return False
    section = parts[0]
    field = "_".join(parts[1:])
    if section not in CONFIG_SECTIONS:
        return False
    fields = CONFIG_SECTIONS[section]
    if not fields:  # 整段放行（下划线字段除外）
        return "_" not in field
    return field.split("_")[0] in fields


# ---------------------------------------------------------------- 文档锚点

# slug 规则经浏览器直取 GitHub 渲染结果校准（对 30 条实测 heading id 做到 30/30
# 复现）：转小写；删除标点与 emoji 基础码位；保留字母数字/CJK/连字符/下划线；
# 空格转 "-"（故 emoji 开头的标题 id 带前导连字符）；U+FE0F 变体选择符**不删除**，
# 会残留在 id 首位——因此标题必须避开带 VS16 的 emoji，否则锚点只能写隐形字符。
VS16 = "\ufe0f"


def gh_slug(heading: str) -> str:
    body = re.sub(r"[`*_]", "", heading).lower()
    return "".join(c for c in body if c.isalnum() or c in " -_" or c == VS16).replace(" ", "-")


def strip_code(text: str) -> str:
    """挖掉围栏代码块与行内 code span（保留行结构）。

    文档里常有 `](#frag)`、path#frag 这类**示例写法**，以及 ```bash 块里以
    `#` 开头的 shell 注释；不挖掉会被当成真链接/真标题误报。
    """
    out: list[str] = []
    fence = False
    for line in text.splitlines():
        if re.match(r"^\s*(```|~~~)", line):
            fence = not fence
            out.append("")
            continue
        if fence:
            out.append("")
        else:
            out.append(re.sub(r"`[^`]*`", "", line))
    return "\n".join(out)


def heading_lines(text: str) -> list[tuple[int, str]]:
    return [
        (n, m.group(1))
        for n, line in enumerate(text.splitlines(), 1)
        for m in [re.match(r"^#{1,6}\s+(.*?)\s*$", line)]
        if m
    ]


def anchor_targets(path: pathlib.Path) -> set[str]:
    """一页里所有可作为 #frag 目标的 id：自动 slug（含重名的 -1/-2 后缀）+ 显式 id。"""
    text = strip_code(path.read_text(errors="replace"))
    targets: set[str] = set(re.findall(r'<a(?: id| name)="([^"]+)"', text))
    seen: dict[str, int] = {}
    for _, h in heading_lines(text):
        s = gh_slug(h)
        seen[s] = seen.get(s, 0) + 1
        # GitHub 对重名标题只给第二个及以后加 -N，首个仍为裸 slug
        targets.add(s if seen[s] == 1 else f"{s}-{seen[s] - 1}")
    return targets


def resolve_link(from_file: pathlib.Path, ref: str) -> pathlib.Path | None:
    """跨文件链接按相对当前文件、再相对仓库根依次解析。"""
    for cand in ((from_file.parent / ref), (ROOT / ref)):
        p = cand.resolve()
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------- 核验

def check_anchors(docs: list[pathlib.Path]) -> tuple[list[str], int]:
    """锚点可达性与 slug 卫生校验，返回（差异清单, 已校验锚点数）。"""
    issues: list[str] = []
    checked = 0
    cache: dict[pathlib.Path, set[str]] = {}
    for p in docs:
        rel = p.relative_to(ROOT)
        text = strip_code(p.read_text(errors="replace"))
        for _, h in heading_lines(text):
            if VS16 in h:
                issues.append(
                    f"[anchor] {rel} 标题「{h[:32]}」含 U+FE0F 变体选择符，"
                    "会隐形残留进 GitHub heading id，须换成无 VS16 的 emoji"
                )
        for n, line in enumerate(text.splitlines(), 1):
            frags = ["#" + f for f in re.findall(r"\]\(#([^)\s]+)\)", line)]
            frags += re.findall(r"\]\(([^)\s]+#[^)\s]+)\)", line)
            for t in frags:
                path_part, _, frag = t.partition("#")
                if path_part.startswith(("http://", "https://", "mailto:")):
                    continue
                checked += 1
                if path_part:
                    q = resolve_link(p, path_part)
                    if q is None:
                        issues.append(f"[anchor] {rel}:{n} 链接 {path_part} 目标文件不存在")
                        continue
                    pool = cache.setdefault(q, anchor_targets(q))
                else:
                    pool = cache.setdefault(p, anchor_targets(p))
                if frag not in pool:
                    near = sorted(x for x in pool if x.lstrip("-") == frag.lstrip("-"))
                    hint = f"GitHub 实际 id 应为 {near[0]}" if near else "目标文件中无匹配标题"
                    issues.append(f"[anchor] {rel}:{n} 锚点 #{frag} 不可达——{hint}")
    return issues, checked


def main() -> int:
    src = src_text()
    docs_text = "\n".join(p.read_text(errors="replace") for p in DOCS if p.exists())
    issues: list[str] = []

    # 1. HTTP 端点
    routes = code_routes(src)
    doc_eps = set(re.findall(r"`/(?:api/1|v1)/[a-z0-9/_-]*`", docs_text))
    doc_eps = {e.strip("`") for e in doc_eps}
    doc_eps = {e for e in doc_eps if not e.endswith("/api/1/") and len(e) > len("/api/1/")}
    for ep in sorted(doc_eps):
        if ep not in routes:
            issues.append(f"[endpoint] 文档声明 {ep} 在代码路由中不存在（已注册: {sorted(routes)}）")

    # 2. gRPC 方法
    methods = code_grpc_methods(src)
    doc_methods = set(re.findall(r"`(vecboost\.[a-z_]+)`", docs_text))
    doc_methods -= {"vecboost.db", "vecboost.log", "vecboost.git", "vecboost.png"}
    for gm in sorted(doc_methods):
        if gm not in methods:
            issues.append(f"[grpc] 文档声明 {gm} 未在代码中注册（已注册: {sorted(methods)}）")

    # 3. 环境变量（下划线结尾的匹配是 `VECBOOST_BUILD_*` 族前缀写法的截断，非真实变量名）
    env_vars = {
        v for v in re.findall(r"VECBOOST_[A-Z0-9_]+", docs_text) if not v.endswith("_")
    }
    for var in sorted(env_vars):
        if not env_var_resolvable(var, src):
            issues.append(f"[env] 文档声明 {var} 无代码消费且无法经 confers 映射到 VecboostConfig 字段")

    # 4. CLI 子命令
    cli_cmds = code_cli_subcommands(src)
    doc_cli = set(re.findall(r"vecboost\s+([a-z_]+)\b", docs_text))
    doc_cli &= set(re.findall(r"`?(embed|embed_batch|compute_similarity|search|rerank|decisions|translate|detect|transcribe)`?", docs_text))
    for cmd in sorted(doc_cli):
        if cmd not in cli_cmds:
            issues.append(f"[cli] 文档声明子命令 {cmd} 未注册")

    # 5. 文档锚点与 slug 卫生
    anchor_issues, anchors_checked = check_anchors([p for p in DOCS if p.exists()])
    issues += anchor_issues

    print(
        f"代码事实: routes={len(routes)} grpc={len(methods)} cli={len(cli_cmds)} "
        f"anchors={anchors_checked}"
    )
    if issues:
        print(f"\n发现 {len(issues)} 处不一致:")
        for i in issues:
            print(f"  - {i}")
        return 1
    print("文档一致性核验通过：无差异")
    return 0


if __name__ == "__main__":
    sys.exit(main())
