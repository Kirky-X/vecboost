#!/usr/bin/env bash
# pre-commit.sh — VecBoost 统一的 pre-commit / CI 检查工具
#
# 合并了原 pre-commit-check.sh、check_copyright.sh、install-pre-commit.sh
# 与 test-feature-matrix.sh，提供单一入口与子命令分发。
#
# 子命令：
#   check      运行 fmt + clippy + check + build（pre-commit 钩子使用）
#   copyright   检查所有 .rs 文件的版权头
#   install     安装 pre-commit Git 钩子
#   uninstall   移除 pre-commit Git 钩子
#   run         手动运行 check
#   matrix      跨 feature 组合矩阵测试（原 test-feature-matrix.sh）
#
# 用法：
#   ./pre-commit.sh check
#   ./pre-commit.sh copyright
#   ./pre-commit.sh install [--uninstall] [--run]
#   ./pre-commit.sh matrix [--quick] [--dry-run]

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

# =============================================================================
# Feature 选择（按平台，与原 pre-commit-check.sh 一致）
# =============================================================================
get_features() {
    if [ -n "${VECBOOST_PRECOMMIT_FEATURES:-}" ]; then
        echo "$VECBOOST_PRECOMMIT_FEATURES"
        return
    fi
    case "$(uname -s)" in
        Linux)  echo "onnx,grpc" ;;
        Darwin) echo "metal,onnx,grpc,redis" ;;
        MINGW*|CYGWIN*|MSYS*) echo "onnx,grpc,redis" ;;
        *) echo "grpc" ;;
    esac
}

# =============================================================================
# check：fmt + clippy + check + build
# =============================================================================
cmd_check() {
    local FEATURES
    FEATURES=$(get_features)
    print_header "VecBoost Pre-commit Checks"
    echo -e "${YELLOW}Platform:${NC} $(uname -s)"
    echo -e "${YELLOW}Project:${NC} $PROJECT_ROOT"
    echo -e "${YELLOW}Features:${NC} $FEATURES"
    echo ""

    local PASSED=0 FAILED=0

    if run_timed "cargo fmt --all -- --check" "cargo fmt --all -- --check" 2; then
        print_result 0 "cargo fmt --check"; PASSED=$((PASSED + 1))
    else
        print_result 1 "cargo fmt --check" "Format check failed"; FAILED=$((FAILED + 1))
    fi
    echo ""

    local clippy_cmd="cargo clippy --features $FEATURES --lib --tests -- -D warnings -A dead_code -A unused -A private_interfaces -A clippy::style -A clippy::complexity -A clippy::perf"
    echo -e "  ${BLUE}Command:${NC} $clippy_cmd"
    if run_timed "cargo clippy" "$clippy_cmd" 15; then
        print_result 0 "cargo clippy"; PASSED=$((PASSED + 1))
    else
        print_result 1 "cargo clippy" "Clippy lint failed"; FAILED=$((FAILED + 1))
    fi
    echo ""

    if run_timed "cargo check --features $FEATURES" "cargo check --features $FEATURES" 10; then
        print_result 0 "cargo check"; PASSED=$((PASSED + 1))
    else
        print_result 1 "cargo check" "Compilation check failed"; FAILED=$((FAILED + 1))
    fi
    echo ""

    if run_timed "cargo build --features $FEATURES" "cargo build --features $FEATURES --quiet" 10; then
        print_result 0 "cargo build"; PASSED=$((PASSED + 1))
    else
        print_result 1 "cargo build" "Build failed"; FAILED=$((FAILED + 1))
    fi
    echo ""

    print_header "Check Summary"
    echo -e "  ${GREEN}✅ Passed:${NC}  $PASSED"
    echo -e "  ${RED}❌ Failed:${NC}  $FAILED"
    echo ""

    if [ "$FAILED" -gt 0 ]; then
        echo -e "${RED}════════════════════════════════════════════════════════════${NC}"
        echo -e "${RED}  ❌ PRE-COMMIT CHECKS FAILED${NC}"
        echo -e "${RED}════════════════════════════════════════════════════════════${NC}"
        exit 1
    fi
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  ✅ ALL PRE-COMMIT CHECKS PASSED${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
}

# =============================================================================
# copyright：检查 .rs 文件版权头（原 check_copyright.sh）
# =============================================================================
cmd_copyright() {
    print_header "Copyright Header Check"
    local violations=0 violated_files=""

    local files=()
    while IFS= read -r f; do files+=("$f"); done < <({
        find src tests examples -name "*.rs" -type f 2>/dev/null
        [ -f build.rs ] && echo "build.rs"
    } | sort)

    for file in "${files[@]}"; do
        local header
        header=$(head -5 "$file" 2>/dev/null || echo "")
        if ! echo "$header" | grep -qE "Copyright \(c\) 2025(-2026)? Kirky\.X"; then
            violated_files="${violated_files}\n  - $file (missing/incorrect copyright)"
            violations=$((violations + 1))
            continue
        fi
        if ! echo "$header" | grep -q "MIT License"; then
            violated_files="${violated_files}\n  - $file (missing MIT License)"
            violations=$((violations + 1))
            continue
        fi
    done

    if [ "$violations" -gt 0 ]; then
        print_error "Copyright Header Check FAILED"
        echo "Found $violations file(s) with incorrect copyright headers:"
        echo -e "$violated_files"
        echo ""
        echo "Expected header format:"
        echo "  // Copyright (c) 2025-2026 Kirky.X"
        echo "  //"
        echo "  // Licensed under the MIT License"
        echo "  // See LICENSE file in the project root for full license information."
        exit 1
    fi

    print_success "Copyright Header Check PASSED"
    echo "All ${#files[@]} .rs files have correct copyright headers."
    exit 0
}

# =============================================================================
# install / uninstall / run
# =============================================================================
check_prerequisites() {
    print_info "Checking prerequisites..."
    local missing=()
    if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
        missing+=("Python 3")
    fi
    if ! command -v cargo &> /dev/null; then
        missing+=("Rust/Cargo")
    fi
    if ! command -v git &> /dev/null; then
        missing+=("Git")
    fi
    if [ ${#missing[@]} -ne 0 ]; then
        print_error "Missing prerequisites: ${missing[*]}"
        echo "Please install the missing tools and try again."
        exit 1
    fi
    print_success "All prerequisites met"
}

cmd_install() {
    print_info "Installing pre-commit hooks..."
    check_prerequisites
    print_info "Ensuring pre-commit is installed..."
    python3 -m pip install --upgrade pre-commit 2>/dev/null || \
        python3 -m pip install pre-commit 2>/dev/null || \
        pip install pre-commit 2>/dev/null || {
            print_warning "Could not install pre-commit via pip"
            pip3 install pre-commit
        }
    print_info "Installing Git hooks..."
    pre-commit install
    if pre-commit install --hook-type commit-msg 2>/dev/null; then
        print_success "Commit-msg hook installed"
    fi
    print_info "Updating hooks to latest versions..."
    pre-commit autoupdate
    print_success "Pre-commit hooks installed successfully!"
    echo ""
    echo "The following checks will now run automatically before each commit:"
    echo "  1. cargo fmt - Code formatting check"
    echo "  2. cargo clippy - Code quality check (strict)"
    echo "  3. cargo check - Compilation check"
}

cmd_uninstall() {
    print_info "Uninstalling pre-commit hooks..."
    if command -v pre-commit &> /dev/null; then
        pre-commit clean
        pre-commit uninstall 2>/dev/null || true
        print_success "Pre-commit hooks removed"
    else
        print_warning "pre-commit not found, hooks may not be installed"
    fi
}

cmd_run() {
    print_info "Running pre-commit checks..."
    echo ""
    cmd_check
}

# =============================================================================
# matrix：跨 feature 组合的矩阵测试（原 test-feature-matrix.sh）
# =============================================================================
declare -a COMBINATIONS=(
    "default|"
    "grpc|grpc"
    "ecosystem|auth,redis,db,inklog,limiteron,oxcache"
    "cuda-network|cuda,grpc,auth,redis"
)

# 尊重 --dry-run：打印命令但不执行；否则走 run_timed（带超时）
matrix_run() {
    local label="$1" cmd="$2" timeout_minutes="${3:-15}"
    if [ "${DRY_RUN:-0}" -eq 1 ]; then
        echo -e "  ${YELLOW}[DRY-RUN]${NC} $cmd"
        return 0
    fi
    run_timed "$label" "$cmd" "$timeout_minutes"
}

cmd_matrix() {
    local QUICK_MODE=false
    DRY_RUN=0
    for arg in "$@"; do
        case $arg in
            --quick) QUICK_MODE=true ;;
            --dry-run) DRY_RUN=1 ;;
            --help|-h)
                echo "Usage: $0 matrix [--quick] [--dry-run]"
                echo ""
                echo "Options:"
                echo "  --quick    Skip build step, only run fmt+clippy+test"
                echo "  --dry-run  Show commands without executing"
                echo "  --help     Show this help message"
                return 0
                ;;
            *) echo "Unknown option: $arg"; return 1 ;;
        esac
    done

    local PASSED=0 FAILED=0 SKIPPED=0
    print_header "VecBoost Feature Matrix Tests"
    echo -e "${YELLOW}Mode:${NC} $([ "$QUICK_MODE" = true ] && echo 'quick (no build)' || echo 'full')"
    echo -e "${YELLOW}Dry Run:${NC} $([ "$DRY_RUN" = 1 ] && echo 'yes' || echo 'no')"
    echo -e "${YELLOW}Combinations:${NC} ${#COMBINATIONS[@]}"
    echo ""

    local RESULTS=()
    for entry in "${COMBINATIONS[@]}"; do
        IFS='|' read -r name features <<< "$entry"
        print_header "Combination: $name"
        echo -e "${YELLOW}Features:${NC} ${features:-<default>}"
        echo ""

        local combo_passed=0 combo_failed=0

        echo -e "${YELLOW}━━━ Step 1: Format Check ━━━${NC}"
        if matrix_run "cargo fmt --check" "cargo fmt --all -- --check" 2; then
            print_result 0 "cargo fmt --check"; combo_passed=$((combo_passed + 1))
        else
            print_result 1 "cargo fmt --check" "Format check failed"; combo_failed=$((combo_failed + 1))
        fi
        echo ""

        echo -e "${YELLOW}━━━ Step 2: Clippy Lint ━━━${NC}"
        local clippy_cmd
        if [ -n "$features" ]; then
            clippy_cmd="cargo clippy --features $features --all-targets -- -D warnings"
        else
            clippy_cmd="cargo clippy --all-targets -- -D warnings"
        fi
        echo -e "  ${BLUE}Command:${NC} $clippy_cmd"
        if matrix_run "cargo clippy" "$clippy_cmd" 15; then
            print_result 0 "cargo clippy"; combo_passed=$((combo_passed + 1))
        else
            print_result 1 "cargo clippy" "Clippy lint failed"; combo_failed=$((combo_failed + 1))
        fi
        echo ""

        if [ "$QUICK_MODE" = false ]; then
            echo -e "${YELLOW}━━━ Step 3: Build ━━━${NC}"
            local build_cmd
            if [ -n "$features" ]; then build_cmd="cargo build --features $features"; else build_cmd="cargo build"; fi
            echo -e "  ${BLUE}Command:${NC} $build_cmd"
            if matrix_run "cargo build" "$build_cmd" 20; then
                print_result 0 "cargo build"; combo_passed=$((combo_passed + 1))
            else
                print_result 1 "cargo build" "Build failed"; combo_failed=$((combo_failed + 1))
            fi
            echo ""
        else
            echo -e "${YELLOW}━━━ Step 3: Build (SKIPPED) ━━━${NC}"
            SKIPPED=$((SKIPPED + 1))
            echo ""
        fi

        echo -e "${YELLOW}━━━ Step 4: Tests ━━━${NC}"
        local test_cmd
        if [ -n "$features" ]; then test_cmd="cargo test --features $features --lib"; else test_cmd="cargo test --lib"; fi
        echo -e "  ${BLUE}Command:${NC} $test_cmd"
        if matrix_run "cargo test --lib" "$test_cmd" 30; then
            print_result 0 "cargo test --lib"; combo_passed=$((combo_passed + 1))
        else
            print_result 1 "cargo test --lib" "Tests failed"; combo_failed=$((combo_failed + 1))
        fi
        echo ""

        if [ "$combo_failed" -eq 0 ]; then
            echo -e "${GREEN}✅ Combination '$name' PASSED ($combo_passed checks)${NC}"
            RESULTS+=("PASS|$name|$combo_passed|$combo_failed")
        else
            echo -e "${RED}❌ Combination '$name' FAILED ($combo_passed passed, $combo_failed failed)${NC}"
            RESULTS+=("FAIL|$name|$combo_passed|$combo_failed")
        fi
    done

    print_header "Feature Matrix Summary"
    echo -e "${YELLOW}Combination Results:${NC}"
    echo "─────────────────────────────────────────"
    for result in "${RESULTS[@]}"; do
        IFS='|' read -r status name passed failed <<< "$result"
        if [ "$status" = "PASS" ]; then
            echo -e "  ${GREEN}✅${NC} $name (${passed} checks passed)"
        else
            echo -e "  ${RED}❌${NC} $name (${passed} passed, ${failed} failed)"
        fi
    done
    echo "─────────────────────────────────────────"
    echo ""
    echo -e "  ${GREEN}✅ Total Passed:${NC}  $PASSED"
    echo -e "  ${RED}❌ Total Failed:${NC}  $FAILED"
    [ "$SKIPPED" -gt 0 ] && echo -e "  ${YELLOW}⚠️  Total Skipped:${NC} $SKIPPED"
    echo ""

    if [ "$FAILED" -gt 0 ]; then
        echo -e "${RED}════════════════════════════════════════════════════════════${NC}"
        echo -e "${RED}  ❌ FEATURE MATRIX TESTS FAILED${NC}"
        echo -e "${RED}════════════════════════════════════════════════════════════${NC}"
        exit 1
    fi
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  ✅ ALL FEATURE MATRIX TESTS PASSED${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
}

# =============================================================================
# Help / Main
# =============================================================================
show_help() {
    echo ""
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║            VecBoost Pre-commit / CI Tool                  ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  check       Run fmt + clippy + check + build (pre-commit hook)"
    echo "  copyright    Verify copyright headers in all .rs files"
    echo "  install      Install pre-commit Git hooks"
    echo "  uninstall    Remove pre-commit Git hooks"
    echo "  run          Run checks manually"
    echo "  matrix       Cross feature-combination matrix tests"
    echo "  --help, -h   Show this help"
    echo ""
}

main() {
    case "${1:-}" in
        check) cmd_check ;;
        copyright) cmd_copyright ;;
        install) cmd_install ;;
        uninstall) cmd_uninstall ;;
        run) cmd_run ;;
        matrix) cmd_matrix "${@:2}" ;;
        --help|-h|"") show_help ;;
        *) print_error "Unknown command: $1"; show_help; exit 1 ;;
    esac
}

main "$@"
