#!/usr/bin/env bash
# common.sh — VecBoost 脚本公共库
#
# 被 scripts/ 下的其它脚本 source，提供颜色输出、日志函数、超时命令封装
# 与项目根目录解析，以消除各脚本间的重复样板代码。

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 项目根目录（scripts/ 的父目录，调用方如需可覆盖）
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

print_info()    { echo -e "${BLUE}ℹ️  $1${NC}"; }
print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error()   { echo -e "${RED}❌ $1${NC}"; }

print_header() {
    echo ""
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo ""
}

print_result() {
    local status=$1 name=$2 message="${3:-}"
    if [ "$status" -eq 0 ]; then
        echo -e "  ${GREEN}✅ PASS${NC}: $name"
        return 0
    fi
    echo -e "  ${RED}❌ FAIL${NC}: $name"
    [ -n "$message" ] && echo -e "     ${YELLOW}→${NC} $message"
    return 1
}

# run_timed <name> <cmd> [timeout_minutes]
# 以 timeout 封装执行命令（静默输出）；DRY_RUN=1 时仅打印不执行。
# 返回 0（成功）/ 1（失败或超时）。
run_timed() {
    local name="$1" cmd="$2" timeout_minutes="${3:-10}"
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo -e "  ${YELLOW}[DRY-RUN]${NC} $cmd"
        return 0
    fi
    if timeout "${timeout_minutes}m" bash -c "$cmd" >/dev/null 2>&1; then
        return 0
    fi
    local ec=$?
    if [ "$ec" -eq 124 ]; then
        echo -e "     ${YELLOW}→${NC} Timeout after ${timeout_minutes} minutes"
    fi
    return 1
}
