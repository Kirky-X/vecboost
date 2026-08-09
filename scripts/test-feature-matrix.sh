#!/usr/bin/env bash
# VecBoost 特性组合测试矩阵
#
# 遍历所有关键特性组合，验证编译 + 测试通过。
# 用法: bash scripts/test-feature-matrix.sh
#
# 退出码: 0 = 全部通过, 1 = 有失败

set -euo pipefail

PASS=0
FAIL=0
SKIP=0

run_check() {
    local name="$1"
    shift
    printf "%-50s " "$name"
    if "$@" >/dev/null 2>&1; then
        echo "PASS"
        ((PASS++))
    else
        echo "FAIL"
        ((FAIL++))
    fi
}

run_test() {
    local name="$1"
    shift
    printf "%-50s " "$name"
    if "$@" >/dev/null 2>&1; then
        echo "PASS"
        ((PASS++))
    else
        echo "FAIL"
        ((FAIL++))
    fi
}

echo "============================================"
echo " VecBoost Feature Matrix Test"
echo "============================================"
echo ""

# --- 编译验证 (cargo check) ---
echo "--- 编译验证 ---"

run_check "no-default-features" \
    cargo check --no-default-features

run_check "schema-only" \
    cargo check --no-default-features --features schema

run_check "auth-only (no http)" \
    cargo check --no-default-features --features auth

run_check "onnx-only" \
    cargo check --no-default-features --features onnx

run_check "db-only" \
    cargo check --no-default-features --features db

run_check "grpc-only" \
    cargo check --no-default-features --features grpc

run_check "cli-only" \
    cargo check --no-default-features --features cli

run_check "mcp-only" \
    cargo check --no-default-features --features mcp

echo ""
echo "--- 组合编译验证 ---"

run_check "http+auth" \
    cargo check --features "http,auth"

run_check "http+grpc" \
    cargo check --features "http,grpc"

run_check "http+cli" \
    cargo check --features "http,cli"

run_check "http+mcp" \
    cargo check --features "http,mcp"

run_check "http+db" \
    cargo check --features "http,db"

run_check "http+onnx" \
    cargo check --features "http,onnx"

run_check "auth+onnx (no http)" \
    cargo check --no-default-features --features "auth,onnx"

run_check "full (http+auth+onnx+db+grpc+cli+mcp)" \
    cargo check --features "http,auth,onnx,db,grpc,cli,mcp"

echo ""
echo "--- 测试验证 ---"

run_test "http (default)" \
    cargo test --features http --lib

run_test "http+auth" \
    cargo test --features "http,auth" --lib

run_test "http+onnx" \
    cargo test --features "http,onnx" --lib

run_test "http+db" \
    cargo test --features "http,db" --lib

run_test "http+auth+onnx" \
    cargo test --features "http,auth,onnx" --lib

run_test "full (http+auth+onnx+db+grpc+cli+mcp)" \
    cargo test --features "http,auth,onnx,db,grpc,cli,mcp" --lib

run_test "no-default-features (lib)" \
    cargo test --no-default-features --lib

echo ""
echo "--- 集成测试 ---"

run_test "integration (http)" \
    cargo test --features http --test integration

echo ""
echo "============================================"
echo " Results: $PASS passed, $FAIL failed, $SKIP skipped"
echo "============================================"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
exit 0
