#!/usr/bin/env bash
# mem-guard.sh — 用户级 earlyoom（wsl2-oom-hardening D4）
#
# MemAvailable 低于阈值时，在目标列表（comm 名匹配、仅当前用户进程）内按 RSS
# 选最大者 SIGTERM；同一 pid 10s 后仍存活升级 SIGKILL。无匹配目标仅告警，
# 绝不波及列表外进程。日志落 $MEM_GUARD_LOG（默认 ~/.local/state/）。
#
# 旋钮（env）：
#   MEM_GUARD_MIN_MB     阈值（MB），默认 max(2048, 5%×MemTotal)
#   MEM_GUARD_TARGETS    逗号分隔 comm 名，默认 vecboost
#   MEM_GUARD_INTERVAL_S 轮询间隔秒，默认 5
#   MEM_GUARD_LOG        日志路径
#
# 免 root：配合 scripts/mem-guard-install.sh 以 systemd --user 托管。
set -u

interval="${MEM_GUARD_INTERVAL_S:-5}"
targets="${MEM_GUARD_TARGETS:-vecboost}"
log="${MEM_GUARD_LOG:-$HOME/.local/state/vecboost-mem-guard.log}"
mkdir -p "$(dirname "$log")" || exit 1

# 阈值缺省：5% MemTotal，下限 2048MB
default_min_mb="$(awk '/^MemTotal:/{mb=int($2/1024); r=int(mb*0.05); if (r<2048) r=2048; print r}' /proc/meminfo)"
min_mb="${MEM_GUARD_MIN_MB:-$default_min_mb}"

logline() { printf '%s %s\n' "$(date -Is)" "$*" >>"$log"; }
trap 'logline "guard exited pid=$$"' EXIT
trap 'logline "guard stopped by signal pid=$$"; exit 0' INT TERM

logline "guard started: pid=$$ min_mb=$min_mb targets=$targets interval=${interval}s"
declare -A termed_at

while :; do
    avail_mb="$(awk '/^MemAvailable:/{print int($2/1024)}' /proc/meminfo)"
    if [ -n "$avail_mb" ] && [ "$avail_mb" -lt "$min_mb" ]; then
        # 目标内最大 RSS（仅当前用户进程；ps --sort=-rss 降序首中即最大，
        # comm 随选一并带出——进程死后二次 ps 取不到）
        hit="$(ps -u "$(id -u)" -o rss=,pid=,comm= --sort=-rss 2>/dev/null |
            awk -v t="$targets" '{
                n = split(t, a, ",")
                for (i = 1; i <= n; i++) if ($3 == a[i]) { print $2 ":" $3; exit }
            }')"
        pid="${hit%%:*}"
        comm="${hit#*:}"
        # 清理过期 term 记录：pid 可能被系统回收复用，不能让旧态影响新进程
        now="$(date +%s)"
        for p in "${!termed_at[@]}"; do
            [ "$now" -ge $((termed_at[$p] + 15)) ] && unset 'termed_at[$p]'
        done
        if [ -z "${pid:-}" ]; then
            logline "warn: MemAvailable=${avail_mb}MB below ${min_mb}MB but no target in [$targets]"
        elif [ -n "${termed_at[$pid]:-}" ] && [ "$now" -ge $((termed_at[$pid] + 10)) ]; then
            kill -KILL "$pid" 2>/dev/null &&
                logline "kill -KILL pid=$pid comm=$comm (still alive 10s after term) avail=${avail_mb}MB"
            unset 'termed_at[$pid]'
        elif [ -z "${termed_at[$pid]:-}" ]; then
            rss_kb="$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ')"
            kill -TERM "$pid" 2>/dev/null &&
                logline "kill -TERM pid=$pid comm=$comm rss=${rss_kb}KB avail=${avail_mb}MB below=${min_mb}MB"
            termed_at[$pid]="$now"
        fi
    fi
    sleep "$interval"
done
