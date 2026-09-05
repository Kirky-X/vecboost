#!/usr/bin/env bash
# run-scenario-tests.sh — full-scenario-testing 场景测试编排器
#
# 用法:
#   bash scripts/run-scenario-tests.sh [--skip-build] [--skip-cargo-test]
#
# 流程: 构建 → 模式探针(CLI/MCP/library) → 逐套件 pytest → cargo test 回归 → 汇总
# 产物: tests/scenario/run/modes/*（模式探针输出）+ 各配置档 server.log
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/target/debug/vecboost"
MODES="$ROOT/tests/scenario/run/modes"
PY=${PYTHON:-python3}
SKIP_BUILD=0; SKIP_CARGO=0
for a in "$@"; do case "$a" in --skip-build) SKIP_BUILD=1;; --skip-cargo-test) SKIP_CARGO=1;; esac; done

cd "$ROOT"
mkdir -p "$MODES"

echo "=== [1/4] 构建 ==="
if [ "$SKIP_BUILD" -eq 0 ]; then
  cargo build --features "http,grpc,cli,auth,db,mcp" 2>&1 | tail -2 || exit 1
fi
[ -x "$BIN" ] || { echo "二进制不存在: $BIN"; exit 1; }

echo "=== [2/4] 模式探针 (CLI / MCP / library) ==="
BASE_DIR="$ROOT/tests/scenario/run/base"
mkdir -p "$BASE_DIR/config"
if [ ! -f "$BASE_DIR/config/config.toml" ]; then
  # base 配置档（与 conftest 一致的最小配置）
  cat > "$BASE_DIR/config/config.toml" <<EOF
[server]
host = "127.0.0.1"
port = 9101

[model]
model_path = "$ROOT/models/BAAI-bge-small-en-v1.5"
use_gpu = false
expected_dimension = 384

[embedding]
cache_enabled = true

[rate_limit]
enabled = true
ip_whitelist = ["127.0.0.1", "::1"]

[auth]
enabled = false

[database]
url = "sqlite::memory:"
EOF
fi

# CLI: embed 与 rerank 子命令（sdforge CLI 签名为 --req <JSON DTO>）
( cd "$BASE_DIR" && timeout 90 "$BIN" embed --req '{"text":"CLI 模式嵌入测试"}' \
    > "$MODES/cli_embed.json" 2>"$MODES/cli_embed.err" )
echo $? > "$MODES/cli_embed.meta"
( cd "$BASE_DIR" && timeout 90 "$BIN" rerank \
    --req '{"query":"什么是机器学习","documents":["机器学习是人工智能的分支","今天天气不错","深度学习用梯度下降训练模型"],"top_k":3}' \
    > "$MODES/cli_rerank.json" 2>"$MODES/cli_rerank.err" )
echo $? > "$MODES/cli_rerank.meta"

# MCP: stdio JSON-RPC（initialize → tools/list → tools/call embed_text）
# 注意：工具入参为 {"req": <DTO>} 包装（与 CLI --req 同构）
( cd "$BASE_DIR" && printf '%s\n' \
    '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"scenario","version":"0"}}}' \
    '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
    '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
    '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"embed_text","arguments":{"req":{"text":"mcp 测试"}}}}' \
    | timeout 90 "$BIN" --mcp > "$MODES/mcp_out.jsonl" 2>"$MODES/mcp.err" )
echo $? > "$MODES/mcp.meta"
# 提取 JSON-RPC 响应（跳过混入 stdout 的日志行）
$PY - <<'PYEOF'
import json, pathlib
modes = pathlib.Path("tests/scenario/run/modes")
out = {}
for line in (modes / "mcp_out.jsonl").read_text(errors="replace").splitlines():
    line = line.strip()
    if not line.startswith("{"):
        continue
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        continue
    if obj.get("id") == 2:
        out["tools_list"] = obj
    if obj.get("id") == 3:
        out["tools_call"] = obj
(modes / "mcp_tools_list.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
PYEOF

# 嵌入式 library 示例
timeout 600 cargo run -q -p vecboost-examples --bin library_usage \
    > "$MODES/library.log" 2>&1
echo $? > "$MODES/library.meta"

echo "=== [3/4] 场景套件 (pytest) ==="
SUITES="test_embed_normal test_embed_abnormal test_rerank test_model_zh test_auth test_security test_server_modes test_config_device"
declare -A RESULTS
overall=0
for s in $SUITES; do
  echo "--- $s ---"
  timeout 1200 $PY -m pytest "tests/scenario/$s.py" -v --tb=short 2>&1 | tail -4
  rc=${PIPESTATUS[0]}
  RESULTS[$s]=$rc
  [ $rc -ne 0 ] && overall=1
done

echo "=== [4/4] 现有测试回归 (cargo test) ==="
if [ "$SKIP_CARGO" -eq 0 ]; then
  cargo test --features "http,cli,auth,db" 2>&1 | tail -5
  echo "cargo_test_exit=$?" | tee "$MODES/cargo_test.meta"
fi

echo ""
echo "=== 汇总 ==="
for s in $SUITES; do
  rc=${RESULTS[$s]:-"NA"}
  case $rc in 0) st=PASS;; *) st="EXIT=$rc";; esac
  printf '%-28s %s\n' "$s" "$st"
done
exit $overall
