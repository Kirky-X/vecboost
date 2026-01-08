#!/bin/bash
#
# VecBoost Pre-commit Check Script
# =================================
# This script runs all pre-commit checks for the VecBoost Rust project.
# It must pass all checks before a commit is allowed.
#
# Checks performed:
# 1. cargo fmt - Code formatting check
# 2. cargo clippy - Code quality check (strict, all warnings as errors)
# 3. cargo check - Compilation check (all features)
#
# Exit codes:
# 0 - All checks passed
# 1 - One or more checks failed
#

set -e  # Exit on error

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

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Get number of CPU cores
get_cpu_cores() {
    if command_exists nproc; then
        nproc
    elif command_exists sysctl; then
        sysctl -n hw.ncpu 2>/dev/null || echo 4
    else
        echo 4
    fi
}

# Run a check with timeout
run_check() {
    local check_name="$1"
    local command="$2"
    local timeout_minutes="${3:-10}"

    echo -e "  ${BLUE}Running:${NC} $check_name"
    echo -e "  ${BLUE}Command:${NC} $command"
    echo ""

    # Run with timeout
    if timeout "${timeout_minutes}m" bash -c "$command" > /dev/null 2>&1; then
        print_result 0 "$check_name" "All checks passed"
        return 0
    else
        local exit_code=$?
        if [ $exit_code -eq 124 ]; then
            print_result 1 "$check_name" "Timeout after ${timeout_minutes} minutes"
        else
            # Show actual error output
            timeout "${timeout_minutes}m" bash -c "$command" 2>&1 | tail -20
            echo ""
            print_result 1 "$check_name" "Check failed with exit code $exit_code"
        fi
        return 1
    fi
}

# Get features based on platform - only use features that work on this platform
get_features() {
    local platform=$(uname -s)
    local features=""

    case "$platform" in
        Linux)
            # Linux: cuda, onnx, grpc (skip redis - has platform-specific deps)
            features="cuda,onnx,grpc"
            ;;
        Darwin)
            # macOS: metal, onnx, grpc, redis (all work on macOS)
            features="metal,onnx,grpc,redis"
            ;;
        MINGW*|CYGWIN*|MSYS*)
            # Windows: onnx, grpc, redis
            features="onnx,grpc,redis"
            ;;
        *)
            # Default: basic features
            features="grpc"
            ;;
    esac

    echo "$features"
}

# =============================================================================
# MAIN CHECKS
# =============================================================================

print_header "VecBoost Pre-commit Checks"

# Determine platform and features
PLATFORM=$(uname -s)
FEATURES=$(get_features)

echo -e "${YELLOW}Platform:${NC} $PLATFORM"
echo -e "${YELLOW}Project:${NC} $PROJECT_ROOT"
echo -e "${YELLOW}CPU Cores:${NC} $(get_cpu_cores)"
echo -e "${YELLOW}Features:${NC} $FEATURES"
echo ""

# -----------------------------------------------------------------------------
# Check 1: cargo fmt
# -----------------------------------------------------------------------------
echo ""
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}  Check 1: Code Formatting (cargo fmt)${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

if run_check "cargo fmt --all -- --check" "cargo fmt --all -- --check" 2; then
    :
fi

echo ""

# -----------------------------------------------------------------------------
# Check 2: cargo clippy
# -----------------------------------------------------------------------------
echo ""
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}  Check 2: Code Quality (cargo clippy)${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  ${YELLOW}Note:${NC} Running with --features $FEATURES (platform-specific)"

if ! command_exists cargo; then
    echo -e "  ${YELLOW}⚠️  SKIP:${NC} cargo not found, skipping clippy check"
    ((SKIPPED++))
else
    if rustup component list --installed 2>/dev/null | grep -q clippy; then
        if run_check "cargo clippy --features $FEATURES --all-targets -- -D warnings" \
                    "cargo clippy --features $FEATURES --all-targets -- -D warnings" 15; then
            :
        fi
    else
        echo -e "  ${YELLOW}⚠️  SKIP:${NC} clippy component not installed"
        echo "  Install with: rustup component add clippy"
        ((SKIPPED++))
    fi
fi

echo ""

# -----------------------------------------------------------------------------
# Check 3: cargo check
# -----------------------------------------------------------------------------
echo ""
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}  Check 3: Compilation (cargo check)${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  ${YELLOW}Note:${NC} Running with --features $FEATURES (platform-specific)"

if run_check "cargo check --features $FEATURES" "cargo check --features $FEATURES" 10; then
    :
fi

echo ""

# -----------------------------------------------------------------------------
# Check 4: Build workspace (optional quick check)
# -----------------------------------------------------------------------------
echo ""
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}  Check 4: Build Check (cargo build)${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

if run_check "cargo build --features $FEATURES" "cargo build --features $FEATURES --quiet" 10; then
    :
fi

echo ""

# =============================================================================
# SUMMARY
# =============================================================================

print_header "Check Summary"

echo -e "  ${GREEN}✅ Passed:${NC}  $PASSED"
echo -e "  ${RED}❌ Failed:${NC}  $FAILED"
if [ $SKIPPED -gt 0 ]; then
    echo -e "  ${YELLOW}⚠️  Skipped:${NC} $SKIPPED"
fi
echo ""

if [ $FAILED -gt 0 ]; then
    echo -e "${RED}════════════════════════════════════════════════════════════${NC}"
    echo -e "${RED}  ❌ PRE-COMMIT CHECKS FAILED${NC}"
    echo -e "${RED}════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "  ${YELLOW}Please fix the above issues before committing.${NC}"
    echo ""
    echo "  Common fixes:"
    echo "    • Format code:      cargo fmt --all"
    echo "    • Fix clippy:       cargo clippy --all-features --fix"
    echo "    • Check errors:     cargo check --all-features"
    echo ""
    exit 1
else
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  ✅ ALL PRE-COMMIT CHECKS PASSED${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "  ${GREEN}You may now commit your changes.${NC}"
    echo ""
    exit 0
fi
