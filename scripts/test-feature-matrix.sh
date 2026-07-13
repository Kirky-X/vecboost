#!/bin/bash
#
# VecBoost Feature Matrix Test Script
# =====================================
# Runs cargo test across 4 feature combinations to verify
# feature flag isolation and compatibility.
#
# Combinations:
#   1. default           - Default features (http)
#   2. grpc              - gRPC server support
#   3. ecosystem         - auth,redis,db,inklog,limiteron,oxcache
#   4. cuda-network      - cuda,grpc,auth,redis
#
# Usage:
#   ./scripts/test-feature-matrix.sh           # Run all combinations
#   ./scripts/test-feature-matrix.sh --quick   # Skip build, only fmt+clippy+test
#   ./scripts/test-feature-matrix.sh --dry-run # Show commands without executing
#
# Exit codes:
#   0 - All combinations passed
#   1 - One or more combinations failed
#

# Note: Not using set -e because we handle errors manually
# and ((var++)) returns non-zero when var is 0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Counters
PASSED=0
FAILED=0
SKIPPED=0

# Parse arguments
QUICK_MODE=false
DRY_RUN=false
for arg in "$@"; do
    case $arg in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--quick] [--dry-run]"
            echo ""
            echo "Options:"
            echo "  --quick    Skip build step, only run fmt+clippy+test"
            echo "  --dry-run  Show commands without executing"
            echo "  --help     Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            exit 1
            ;;
    esac
done

# Feature combinations (must match .github/workflows/feature-matrix.yml)
declare -a COMBINATIONS=(
    "default|"
    "grpc|grpc"
    "ecosystem|auth,redis,db,inklog,limiteron,oxcache"
    "cuda-network|cuda,grpc,auth,redis"
)

# Print section header
print_header() {
    echo ""
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo ""
}

# Print check result
print_result() {
    local status=$1
    local check_name=$2
    local message=$3

    if [ "$status" -eq 0 ]; then
        echo -e "  ${GREEN}✅ PASS${NC}: $check_name"
        ((PASSED++))
    else
        echo -e "  ${RED}❌ FAIL${NC}: $check_name"
        if [ -n "$message" ]; then
            echo -e "     ${YELLOW}→${NC} $message"
        fi
        ((FAILED++))
    fi
}

# Run a command, optionally in dry-run mode
run_cmd() {
    local cmd="$1"
    local timeout_minutes="${2:-15}"

    if [ "$DRY_RUN" = true ]; then
        echo -e "  ${YELLOW}[DRY-RUN]${NC} $cmd"
        return 0
    fi

    if timeout "${timeout_minutes}m" bash -c "$cmd" > /dev/null 2>&1; then
        return 0
    else
        local exit_code=$?
        if [ $exit_code -eq 124 ]; then
            echo -e "     ${YELLOW}→${NC} Timeout after ${timeout_minutes} minutes"
        fi
        return 1
    fi
}

# =============================================================================
# MAIN
# =============================================================================

print_header "VecBoost Feature Matrix Tests"

echo -e "${YELLOW}Project:${NC} $PROJECT_ROOT"
echo -e "${YELLOW}Mode:${NC} $([ "$QUICK_MODE" = true ] && echo 'quick (no build)' || echo 'full')"
echo -e "${YELLOW}Dry Run:${NC} $([ "$DRY_RUN" = true ] && echo 'yes' || echo 'no')"
echo -e "${YELLOW}Combinations:${NC} ${#COMBINATIONS[@]}"
echo ""

# Track results per combination
declare -a RESULTS

# =============================================================================
# Run each feature combination
# =============================================================================
for entry in "${COMBINATIONS[@]}"; do
    IFS='|' read -r name features <<< "$entry"

    print_header "Combination: $name"
    echo -e "${YELLOW}Features:${NC} ${features:-<default>}"
    echo ""

    combo_passed=0
    combo_failed=0

    # Step 1: cargo fmt --check
    echo -e "${YELLOW}━━━ Step 1: Format Check ━━━${NC}"
    if run_cmd "cargo fmt --all -- --check" 2; then
        print_result 0 "cargo fmt --check"
        ((combo_passed++))
    else
        print_result 1 "cargo fmt --check" "Format check failed"
        ((combo_failed++))
    fi
    echo ""

    # Step 2: cargo clippy
    echo -e "${YELLOW}━━━ Step 2: Clippy Lint ━━━${NC}"
    if [ -n "$features" ]; then
        clippy_cmd="cargo clippy --features $features --all-targets -- -D warnings"
    else
        clippy_cmd="cargo clippy --all-targets -- -D warnings"
    fi
    echo -e "  ${BLUE}Command:${NC} $clippy_cmd"
    if run_cmd "$clippy_cmd" 15; then
        print_result 0 "cargo clippy"
        ((combo_passed++))
    else
        print_result 1 "cargo clippy" "Clippy lint failed"
        ((combo_failed++))
    fi
    echo ""

    # Step 3: cargo build (skip in quick mode)
    if [ "$QUICK_MODE" = false ]; then
        echo -e "${YELLOW}━━━ Step 3: Build ━━━${NC}"
        if [ -n "$features" ]; then
            build_cmd="cargo build --features $features"
        else
            build_cmd="cargo build"
        fi
        echo -e "  ${BLUE}Command:${NC} $build_cmd"
        if run_cmd "$build_cmd" 20; then
            print_result 0 "cargo build"
            ((combo_passed++))
        else
            print_result 1 "cargo build" "Build failed"
            ((combo_failed++))
        fi
        echo ""
    else
        echo -e "${YELLOW}━━━ Step 3: Build (SKIPPED) ━━━${NC}"
        ((SKIPPED++))
        echo ""
    fi

    # Step 4: cargo test
    echo -e "${YELLOW}━━━ Step 4: Tests ━━━${NC}"
    if [ -n "$features" ]; then
        test_cmd="cargo test --features $features --lib"
    else
        test_cmd="cargo test --lib"
    fi
    echo -e "  ${BLUE}Command:${NC} $test_cmd"
    if run_cmd "$test_cmd" 30; then
        print_result 0 "cargo test --lib"
        ((combo_passed++))
    else
        print_result 1 "cargo test --lib" "Tests failed"
        ((combo_failed++))
    fi
    echo ""

    # Combination summary
    if [ $combo_failed -eq 0 ]; then
        echo -e "${GREEN}✅ Combination '$name' PASSED ($combo_passed checks)${NC}"
        RESULTS+=("PASS|$name|$combo_passed|$combo_failed")
    else
        echo -e "${RED}❌ Combination '$name' FAILED ($combo_passed passed, $combo_failed failed)${NC}"
        RESULTS+=("FAIL|$name|$combo_passed|$combo_failed")
    fi
done

# =============================================================================
# SUMMARY
# =============================================================================
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
if [ $SKIPPED -gt 0 ]; then
    echo -e "  ${YELLOW}⚠️  Total Skipped:${NC} $SKIPPED"
fi
echo ""

if [ $FAILED -gt 0 ]; then
    echo -e "${RED}════════════════════════════════════════════════════════════${NC}"
    echo -e "${RED}  ❌ FEATURE MATRIX TESTS FAILED${NC}"
    echo -e "${RED}════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "  ${YELLOW}Failed combinations:${NC}"
    for result in "${RESULTS[@]}"; do
        IFS='|' read -r status name passed failed <<< "$result"
        if [ "$status" = "FAIL" ]; then
            echo -e "    ${RED}•${NC} $name"
        fi
    done
    echo ""
    exit 1
else
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  ✅ ALL FEATURE MATRIX TESTS PASSED${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
    echo ""
    exit 0
fi
