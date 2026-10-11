#!/usr/bin/env bash
# run-memcap.sh — 通用进程内存上限封装（wsl2-oom-hardening-r2 D2）
#
# 以 systemd --user scope 包裹任意命令（主要用途：常驻推理服务）——服务异常
# 膨胀时先被自身 cgroup 上限击杀（自爆保整机），而不是等整机逼近极限由内核
# OOM 或 mem-guard 兜底。构建场景用 scripts/build-memcap.sh。
#
# 用法：scripts/run-memcap.sh <command> [args...]
#   例：scripts/run-memcap.sh target/release/vecboost config-e2e-transcribe.toml
# 旋钮（env）：
#   VECBOOST_RUN_MEMMAX  硬上限（MemoryMax），默认 40G
#   VECBOOST_RUN_MEMHIGH 软上限（MemoryHigh，先节流），默认 36G
#   VECBOOST_RUN_SWAPMAX swap 上限（MemorySwapMax），默认 4G
#   总预算 = MemoryMax + MemorySwapMax，超总预算即 cgroup OOM 击杀；
#   SwapMax 必须同设，否则超限页换出 swap 即绕过上限（r1 实测教训）
set -euo pipefail

memmax="${VECBOOST_RUN_MEMMAX:-40G}"
memhigh="${VECBOOST_RUN_MEMHIGH:-36G}"
swapmax="${VECBOOST_RUN_SWAPMAX:-4G}"

exec systemd-run --user --scope --collect \
    --unit="vecboost-run-$(date +%s%N)" \
    -p "MemoryMax=${memmax}" \
    -p "MemoryHigh=${memhigh}" \
    -p "MemorySwapMax=${swapmax}" \
    "$@"
