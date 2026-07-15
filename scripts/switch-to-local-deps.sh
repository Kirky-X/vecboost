#!/usr/bin/env bash
# switch-to-local-deps.sh — 将 6 个生态库依赖从远端 version 切换为本地 path
#
# 用于本地回归测试:当修改了 libraries/ 下的生态库源码后,
# 运行此脚本切换 Cargo.toml 为 path 依赖,验证主 crate 与本地库的兼容性。
#
# 用法: ./scripts/switch-to-local-deps.sh
# 恢复: ./scripts/switch-to-remote-deps.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

print_header "切换到本地 path 依赖"

CARGO_TOML="$PROJECT_ROOT/Cargo.toml"

# 6 个生态库及其当前远端版本号(与 Cargo.toml 保持同步)
declare -A LIBS=(
    ["confers"]="0.4"
    ["inklog"]="0.1"
    ["oxcache"]="0.3"
    ["limiteron"]="0.2"
    ["dbnexus"]="0.4"
    ["sdforge"]="0.4"
)

# 备份 Cargo.toml
BACKUP="$CARGO_TOML.bak"
cp "$CARGO_TOML" "$BACKUP"
print_info "已备份 Cargo.toml → Cargo.toml.bak"

switched=0
skipped=0
failed=0

for lib in "${!LIBS[@]}"; do
    version="${LIBS[$lib]}"
    local_path="$PROJECT_ROOT/../libraries/$lib"

    # 检查是否已经是 path 依赖
    if grep -q "^${lib} = { path = \"\.\./libraries/${lib}\"" "$CARGO_TOML"; then
        print_warning "$lib: 已经是 path 依赖,跳过"
        skipped=$((skipped + 1))
        continue
    fi

    # 检查 version 依赖是否存在
    if grep -q "^${lib} = { version = \"${version}\"" "$CARGO_TOML"; then
        # 检查本地路径是否存在
        if [ ! -d "$local_path" ]; then
            print_error "$lib: 本地路径 $local_path 不存在,跳过"
            failed=$((failed + 1))
            continue
        fi

        # 执行替换:version = "x.x" → path = "../libraries/xxx"
        sed -i "s|^${lib} = { version = \"${version}\"|${lib} = { path = \"../libraries/${lib}\"|" "$CARGO_TOML"
        print_success "$lib: version \"${version}\" → path \"../libraries/${lib}\""
        switched=$((switched + 1))
    else
        print_error "$lib: 未找到 version = \"${version}\" 依赖行"
        failed=$((failed + 1))
    fi
done

echo ""
print_info "切换汇总: 切换=${switched}, 跳过=${skipped}, 失败=${failed}"

if [ "$failed" -gt 0 ]; then
    print_error "有 ${failed} 个库切换失败,已保留备份 Cargo.toml.bak"
    exit 1
fi

print_info "验证编译..."
cd "$PROJECT_ROOT"
if cargo check --features http,auth,db,oxcache,limiteron,config,inklog 2>&1 | tail -5; then
    print_success "切换完成,编译通过"
else
    print_error "编译失败,可执行: cp Cargo.toml.bak Cargo.toml 回滚"
    exit 1
fi
