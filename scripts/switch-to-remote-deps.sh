#!/usr/bin/env bash
# switch-to-remote-deps.sh — 将 6 个生态库依赖从本地 path 切换回远端 version
#
# 用于发布前恢复:确保发布版本使用远端 crates.io 依赖,
# 而非本地 path 依赖。
#
# 用法: ./scripts/switch-to-remote-deps.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

print_header "切换到远端 version 依赖"

CARGO_TOML="$PROJECT_ROOT/Cargo.toml"

# 6 个生态库及其远端版本号(与 Cargo.toml 保持同步)
declare -A LIBS=(
    ["confers"]="0.4"
    ["inklog"]="0.1"
    ["oxcache"]="0.3"
    ["limiteron"]="0.2"
    ["dbnexus"]="0.4"
    ["sdforge"]="0.4"
)

switched=0
skipped=0
failed=0

for lib in "${!LIBS[@]}"; do
    version="${LIBS[$lib]}"

    # 检查是否已经是 version 依赖
    if grep -q "^${lib} = { version = \"${version}\"" "$CARGO_TOML"; then
        print_warning "$lib: 已经是 version 依赖,跳过"
        skipped=$((skipped + 1))
        continue
    fi

    # 检查 path 依赖是否存在
    if grep -q "^${lib} = { path = \"\.\./libraries/${lib}\"" "$CARGO_TOML"; then
        # 执行替换:path = "../libraries/xxx" → version = "x.x"
        sed -i "s|^${lib} = { path = \"\.\./libraries/${lib}\"|${lib} = { version = \"${version}\"|" "$CARGO_TOML"
        print_success "$lib: path → version \"${version}\""
        switched=$((switched + 1))
    else
        print_error "$lib: 未找到 path 依赖行"
        failed=$((failed + 1))
    fi
done

echo ""
print_info "切换汇总: 切换=${switched}, 跳过=${skipped}, 失败=${failed}"

if [ "$failed" -gt 0 ]; then
    print_error "有 ${failed} 个库切换失败"
    exit 1
fi

# 验证无 path 依赖残留
print_info "验证无 path 依赖残留..."
if grep -q 'path = "\.\./libraries/' "$CARGO_TOML"; then
    print_error "仍有 path 依赖残留:"
    grep -n 'path = "\.\./libraries/' "$CARGO_TOML"
    exit 1
fi

print_info "验证编译..."
cd "$PROJECT_ROOT"
if cargo check --features http,auth,db,oxcache,limiteron,config,inklog 2>&1 | tail -5; then
    print_success "切换完成,编译通过"
else
    print_error "编译失败"
    exit 1
fi
