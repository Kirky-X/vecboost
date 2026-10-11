#!/usr/bin/env bash
# mem-guard-install.sh — 安装用户级 mem-guard systemd 单元（wsl2-oom-hardening D4）
# 免 root：写 ~/.config/systemd/user/vecboost-mem-guard.service + enable --now。
# 幂等：重复执行覆盖自家单元文件，不触碰他者。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
guard="${script_dir}/mem-guard.sh"
unit_dir="${HOME}/.config/systemd/user"
unit="${unit_dir}/vecboost-mem-guard.service"

[ -x "$guard" ] || { chmod +x "$guard"; }
mkdir -p "$unit_dir"

cat >"$unit" <<EOF
[Unit]
Description=vecboost user-level earlyoom guard (MemAvailable threshold killer)
Documentation=file://${script_dir}/../docs/MEMORY_SAFETY.md

[Service]
ExecStart=${guard}
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now vecboost-mem-guard.service
status="$(systemctl --user is-active vecboost-mem-guard.service || true)"
echo "installed: $unit"
echo "status: $status"
[ "$status" = "active" ] || {
    echo "guard not active — inspect: journalctl --user -u vecboost-mem-guard" >&2
    exit 1
}
