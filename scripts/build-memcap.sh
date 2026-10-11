#!/usr/bin/env bash
# build-memcap.sh — 构建内存上限封装（wsl2-oom-hardening D3）
#
# 以 systemd --user scope 包裹 cargo：超上限时 cgroup 杀掉 rustc/ld.mold =
# 构建显性失败，而非与常驻推理服务叠加把 RAM+swap 打穿（2026-10-11 OOM 事故
# 触发源之一即 release LTO 构建与 45G 服务并发）。
#
# 用法：scripts/build-memcap.sh build --release ...
# 旋钮（env）：
#   VECBOOST_BUILD_MEMMAX 硬上限（cgroup MemoryMax），默认 26G
#   VECBOOST_BUILD_MEMHIGH 软上限（MemoryHigh，先节流），默认 20G
#   VECBOOST_BUILD_JOBS   CARGO_BUILD_JOBS，默认 8
set -euo pipefail

memmax="${VECBOOST_BUILD_MEMMAX:-26G}"
memhigh="${VECBOOST_BUILD_MEMHIGH:-20G}"
jobs="${VECBOOST_BUILD_JOBS:-8}"

exec systemd-run --user --scope --collect \
    --unit="vecboost-build-$(date +%s%N)" \
    -p "MemoryMax=${memmax}" \
    -p "MemoryHigh=${memhigh}" \
    env CARGO_BUILD_JOBS="$jobs" cargo "$@"
