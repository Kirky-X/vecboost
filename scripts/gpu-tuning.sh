#!/usr/bin/env bash
# GPU 部署调优脚本
# 基于鲲鹏 GPU 应用优化白皮书「硬件优化手段」和「操作系统优化」章节
#
# 用法: sudo bash scripts/gpu-tuning.sh [--check|--apply]
#   --check  仅检测当前状态（默认）
#   --apply  应用推荐配置（需 root 权限）

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

MODE="${1:---check}"

info()  { echo -e "${GREEN}[✓]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
error() { echo -e "${RED}[✗]${NC} $*"; }

# ─── 前置检查 ───────────────────────────────────────────────
check_nvidia_driver() {
    if ! command -v nvidia-smi &>/dev/null; then
        error "nvidia-smi 未找到。请确认 NVIDIA 驱动已安装。"
        exit 1
    fi
    info "nvidia-smi 已找到: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo '查询失败')"
}

# ─── 1. GPU 持久模式 ────────────────────────────────────────
check_persistence_mode() {
    echo ""
    echo "═══ 1. GPU 持久模式 (Persistence Mode) ═══"
    echo "  鲲鹏建议: 开启持久模式避免 GPU 低负载休眠后唤醒延迟"
    echo ""

    local mode
    mode=$(nvidia-smi --query-gpu=persistence_mode --format=csv,noheader,nounits 2>/dev/null || echo "unknown")

    if [[ "$mode" == "1" || "$mode" == *"Enabled"* ]]; then
        info "持久模式已开启"
    else
        warn "持久模式未开启 (当前: $mode)"
        if [[ "$MODE" == "--apply" ]]; then
            echo "  → 正在开启持久模式..."
            nvidia-smi -pm 1 && info "持久模式已开启" || error "开启失败"
        else
            echo "  → 修复: sudo nvidia-smi -pm 1"
        fi
    fi
}

# ─── 2. 透明大页 (Transparent Huge Pages) ──────────────────
check_transparent_hugepage() {
    echo ""
    echo "═══ 2. 透明大页 (Transparent Huge Pages) ═══"
    echo "  鲲鹏建议: 开启 THP 减少 CPU↔GPU 数据传输的 TLB 消耗"
    echo ""

    local thp_file="/sys/kernel/mm/transparent_hugepage/enabled"
    if [[ ! -f "$thp_file" ]]; then
        warn "透明大页文件不存在 ($thp_file)，可能非 Linux 或内核不支持"
        return
    fi

    local content
    content=$(cat "$thp_file")

    if echo "$content" | grep -q '\[always\]'; then
        info "透明大页已开启 (always)"
    elif echo "$content" | grep -q '\[madvise\]'; then
        warn "透明大页为 madvise 模式（部分开启）"
        if [[ "$MODE" == "--apply" ]]; then
            echo "  → 正在设置为 always..."
            echo always > "$thp_file" && info "透明大页已设为 always" || error "设置失败（需 root 权限）"
        else
            echo "  → 修复: echo always | sudo tee $thp_file"
        fi
    else
        warn "透明大页未开启"
        if [[ "$MODE" == "--apply" ]]; then
            echo "  → 正在设置为 always..."
            echo always > "$thp_file" && info "透明大页已设为 always" || error "设置失败（需 root 权限）"
        else
            echo "  → 修复: echo always | sudo tee $thp_file"
        fi
    fi
}

# ─── 3. GPU 时钟频率 ────────────────────────────────────────
check_clock_frequency() {
    echo ""
    echo "═══ 3. GPU 时钟频率 ═══"
    echo "  鲲鹏建议: 锁定 GPU 时钟频率消除频率波动导致的性能抖动"
    echo ""

    local clocks
    clocks=$(nvidia-smi --query-gpu=clocks.max.graphics,clocks.max.memory,clocks.current.graphics,clocks.current.memory \
        --format=csv,noheader 2>/dev/null || echo "unknown")

    if [[ "$clocks" != "unknown" ]]; then
        info "GPU 时钟频率信息:"
        echo "  $clocks"
        echo ""
        echo "  → 锁定频率命令: sudo nvidia-smi -ac <mem_clock>,<graphics_clock>"
        echo "  → 恢复默认: sudo nvidia-smi -rac"
    else
        warn "无法查询 GPU 时钟频率"
    fi
}

# ─── 4. ECC 内存状态 ────────────────────────────────────────
check_ecc_status() {
    echo ""
    echo "═══ 4. ECC 内存状态 ═══"
    echo "  鲲鹏建议: 生产环境开启 ECC 防止内存位翻转导致计算错误"
    echo ""

    local ecc
    ecc=$(nvidia-smi --query-gpu=ecc.mode.current --format=csv,noheader,nounits 2>/dev/null || echo "unknown")

    if [[ "$ecc" == "1" || "$ecc" == *"Enabled"* ]]; then
        info "ECC 内存已开启"
    elif [[ "$ecc" == "unknown" ]]; then
        warn "无法查询 ECC 状态（GPU 可能不支持 ECC）"
    else
        warn "ECC 内存未开启 (当前: $ecc)"
        if [[ "$MODE" == "--apply" ]]; then
            echo "  → 正在开启 ECC（需要重启生效）..."
            nvidia-smi -e 1 && warn "ECC 已设置为开启，需要重启 GPU 生效" || error "开启失败"
        else
            echo "  → 修复: sudo nvidia-smi -e 1（需重启）"
        fi
    fi
}

# ─── 5. GPU 计算模式 ────────────────────────────────────────
check_compute_mode() {
    echo ""
    echo "═══ 5. GPU 计算模式 ═══"
    echo "  鲲鹏建议: 设置 Exclusive_Process 避免多进程竞争 GPU 资源"
    echo ""

    local mode
    mode=$(nvidia-smi --query-gpu=compute_mode --format=csv,noheader 2>/dev/null || echo "unknown")

    if echo "$mode" | grep -qi "exclusive"; then
        info "GPU 计算模式: Exclusive（最优）"
    elif echo "$mode" | grep -qi "default"; then
        warn "GPU 计算模式: Default（多进程共享）"
        if [[ "$MODE" == "--apply" ]]; then
            echo "  → 正在设置为 Exclusive_Process..."
            nvidia-smi -c EXCLUSIVE_PROCESS && info "已设置为 Exclusive_Process" || error "设置失败"
        else
            echo "  → 修复: sudo nvidia-smi -c EXCLUSIVE_PROCESS"
        fi
    else
        warn "GPU 计算模式: $mode"
    fi
}

# ─── 6. GPU 信息总览 ────────────────────────────────────────
print_gpu_summary() {
    echo ""
    echo "═══ GPU 信息总览 ═══"
    nvidia-smi --query-gpu=name,memory.total,driver_version,cuda_version,temperature.gpu,power.draw \
        --format=csv 2>/dev/null || warn "无法获取 GPU 信息总览"
}

# ─── 主流程 ──────────────────────────────────────────────────
main() {
    echo "╔══════════════════════════════════════════════╗"
    echo "║     VecBoost GPU 调优检测工具 v1.0          ║"
    echo "║     基于鲲鹏 GPU 应用优化白皮书              ║"
    echo "╚══════════════════════════════════════════════╝"
    echo ""
    echo "模式: $MODE"
    echo ""

    check_nvidia_driver
    print_gpu_summary
    check_persistence_mode
    check_transparent_hugepage
    check_clock_frequency
    check_ecc_status
    check_compute_mode

    echo ""
    echo "═══ 检测完成 ═══"
    if [[ "$MODE" == "--check" ]]; then
        echo "提示: 使用 'sudo bash scripts/gpu-tuning.sh --apply' 应用推荐配置"
    fi
}

main
