#!/usr/bin/env python3
"""文档-代码一致性核验（specmark change: exhaustive-scenario-testing / T011）。

系统性提取文档中可验证声明并与代码核对：
1. HTTP 端点：文档声明的 /api/1/* 与 /v1/*、/health、/metrics 路径必须存在于 #[forge] 注册
2. gRPC 方法：文档声明的 vecboost.* 必须与 #[forge(grpc_method=...)] 一致
3. 环境变量：文档声明的 VECBOOST_* 必须能映射到 AppConfig 字段或被代码显式消费
4. CLI 子命令：文档示例的子命令必须在 CLI 注册中

输出：差异清单（stdout）。退出码 0=一致，1=存在差异。
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = [
    ROOT / "README.md",
    ROOT / "README_EN.md",
    ROOT / "docs" / "API_REFERENCE.md",
    ROOT / "docs" / "USER_GUIDE.md",
]
SRC = ROOT / "src"

# ---------------------------------------------------------------- 代码侧事实

def src_text() -> str:
    # src/ 为主（forge 路由/gRPC/CLI 事实源）；benches/ 一并纳入以核验基准类
    # 环境变量（如 VECBOOST_BENCH_MODEL 仅被 benches 消费）。
    return "\n".join(
        p.read_text(errors="replace")
        for d in (ROOT / "src", ROOT / "benches")
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
    """代码显式消费的 VECBOOST_* 环境变量。"""
    return set(re.findall(r'"(VECBOOST_[A-Z0-9_]+)"', src))


def code_cli_subcommands(src: str) -> set[str]:
    return set(re.findall(r'name\s*=\s*"(embed|embed_batch|compute_similarity|rerank|search)"\s*,\s*version', src))


# confers env 映射可用性：VECBOOST_<SECTION>_<FIELD>，FIELD 不含下划线时可达
CONFIG_SECTIONS: dict[str, set[str]] = {
    "server": {"host", "port", "timeout", "grpc", "workers"},
    "model": {"repo", "revision", "path", "gpu", "size", "dimension", "length"},
    "embedding": set(),
    "rerank": set(),
    "monitoring": set(),
    "auth": set(),
    "rate": set(),
    "audit": set(),
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


# ---------------------------------------------------------------- 核验

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

    # 3. 环境变量
    env_vars = set(re.findall(r"VECBOOST_[A-Z0-9_]+", docs_text))
    for var in sorted(env_vars):
        if not env_var_resolvable(var, src):
            issues.append(f"[env] 文档声明 {var} 无代码消费且无法经 confers 映射到 AppConfig 字段")

    # 4. CLI 子命令
    cli_cmds = code_cli_subcommands(src)
    doc_cli = set(re.findall(r"vecboost\s+(embed_batch|compute_similarity|embed|rerank|search)\b", docs_text))
    for cmd in sorted(doc_cli):
        if cmd not in cli_cmds:
            issues.append(f"[cli] 文档声明子命令 {cmd} 未注册")

    print(f"代码事实: routes={len(routes)} grpc={len(methods)} cli={len(cli_cmds)}")
    if issues:
        print(f"\n发现 {len(issues)} 处不一致:")
        for i in issues:
            print(f"  - {i}")
        return 1
    print("文档一致性核验通过：无差异")
    return 0


if __name__ == "__main__":
    sys.exit(main())
