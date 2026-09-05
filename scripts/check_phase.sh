#!/usr/bin/env bash
# check_phase.sh — 阶段完成确定性判定器。
#
# 为 specmark 自动链的阶段转换提供确定性判断（规则 3：确定性逻辑禁止交给模型）。
# agent 在阶段完成前调用此脚本获取客观状态，而非自行读文件判断。
#
# 子命令：
#   artifacts <change>     检查产物完整性（proposal.md / design.md / tasks.md / specs/）
#   tasks <change>         检查任务完成状态（- [ ] vs - [x] 计数）
#   converge-readiness <change>  检查是否可进入 converge（所有原始任务 - [x]）
#   archive-readiness <change>   检查是否可归档（所有 phase 任务 - [x]）
#   complexity <change>    评估变更复杂度（短程/长程判定）
#
# 退出码：0 = 条件满足/检查通过；1 = 条件不满足；2 = 输入错误。
# JSON 输出到 stdout，人类可读摘要到 stderr。

set -euo pipefail

SCRIPT_REAL="$(readlink -f "$0")"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_REAL")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

err()  { printf '[ERROR] %s\n' "$*" >&2; }
info() { printf '[INFO] %s\n' "$*" >&2; }

usage() {
  cat <<'EOF'
Usage: check_phase.sh <subcommand> <change-name>

Subcommands:
  artifacts <name>          检查产物完整性
  tasks <name>              检查任务完成状态
  converge-readiness <name> 检查是否可进入 converge
  archive-readiness <name>  检查是否可归档
  complexity <name>         评估变更复杂度（短程/长程）

Exit codes: 0 = condition met; 1 = condition not met; 2 = input error.
EOF
}

[[ $# -ge 2 ]] || { usage; exit 2; }
SUBCMD="$1"
CHANGE_NAME="$2"

CHANGE_DIR="$ROOT/specmark/changes/$CHANGE_NAME"

# kebab-case 校验
if [[ ! "$CHANGE_NAME" =~ ^[a-z0-9][a-z0-9-]*$ ]]; then
  err "非法 change 名（须 kebab-case）: $CHANGE_NAME"
  exit 2
fi

[[ -d "$CHANGE_DIR" ]] || { err "change 目录不存在: $CHANGE_DIR"; exit 2; }

# ---------- 子命令: artifacts ----------
cmd_artifacts() {
  local has_proposal=0 has_design=0 has_tasks=0 has_specs=0
  local spec_count=0

  [[ -f "$CHANGE_DIR/proposal.md" ]] && has_proposal=1
  [[ -f "$CHANGE_DIR/design.md" ]]   && has_design=1
  [[ -f "$CHANGE_DIR/tasks.md" ]]    && has_tasks=1

  if [[ -d "$CHANGE_DIR/specs" ]]; then
    spec_count=$(find "$CHANGE_DIR/specs" -name "spec.md" 2>/dev/null | wc -l)
    [[ $spec_count -gt 0 ]] && has_specs=1
  fi

  local all_present=1
  [[ $has_proposal -eq 1 && $has_design -eq 1 && $has_tasks -eq 1 ]] || all_present=0

  cat <<EOF
{
  "change": "$CHANGE_NAME",
  "proposal": $has_proposal,
  "design": $has_design,
  "tasks": $has_tasks,
  "specs": $has_specs,
  "spec_count": $spec_count,
  "all_present": $all_present
}
EOF

  if [[ $all_present -eq 1 ]]; then
    info "✓ 产物完整 (proposal + design + tasks$([ $has_specs -eq 1 ] && echo " + $spec_count specs"))"
  else
    local missing=()
    [[ $has_proposal -eq 0 ]] && missing+=("proposal.md")
    [[ $has_design -eq 0 ]]   && missing+=("design.md")
    [[ $has_tasks -eq 0 ]]    && missing+=("tasks.md")
    info "✗ 缺失产物: ${missing[*]}"
  fi

  return $(( 1 - all_present ))
}

# ---------- 子命令: tasks ----------
cmd_tasks() {
  local tasks_file="$CHANGE_DIR/tasks.md"
  if [[ ! -f "$tasks_file" ]]; then
    err "tasks.md 不存在"
    echo '{"change":"'"$CHANGE_NAME"'","error":"tasks.md not found"}'
    return 2
  fi

  local total=0 completed=0 remaining=0
  local convergence_total=0 convergence_completed=0
  local in_convergence=0

  while IFS= read -r line; do
    # 检测是否进入 Convergence phase
    if [[ "$line" =~ ^##\ Phase\ [0-9]+:\ Convergence ]]; then
      in_convergence=1
      continue
    fi

    # 匹配任务行
    if [[ "$line" =~ ^-\ \[([ xX])\] ]]; then
      local checked="${BASH_REMATCH[1]}"
      total=$((total + 1))
      if [[ "$checked" == "x" || "$checked" == "X" ]]; then
        completed=$((completed + 1))
      else
        remaining=$((remaining + 1))
      fi

      if [[ $in_convergence -eq 1 ]]; then
        convergence_total=$((convergence_total + 1))
        if [[ "$checked" == "x" || "$checked" == "X" ]]; then
          convergence_completed=$((convergence_completed + 1))
        fi
      fi
    fi
  done < "$tasks_file"

  local original_total=$((total - convergence_total))
  local original_completed=$((completed - convergence_completed))
  local all_original_done=0
  [[ $original_completed -ge $original_total && $original_total -gt 0 ]] && all_original_done=1
  local all_done=0
  [[ $completed -ge $total && $total -gt 0 ]] && all_done=1

  cat <<EOF
{
  "change": "$CHANGE_NAME",
  "total": $total,
  "completed": $completed,
  "remaining": $remaining,
  "original_total": $original_total,
  "original_completed": $original_completed,
  "convergence_total": $convergence_total,
  "convergence_completed": $convergence_completed,
  "all_original_done": $all_original_done,
  "all_done": $all_done
}
EOF

  info "任务进度: $completed/$total 完成 (原始: $original_completed/$original_total, 收敛: $convergence_completed/$convergence_total)"
  if [[ $all_done -eq 1 ]]; then
    info "✓ 所有任务完成"
  elif [[ $all_original_done -eq 1 ]]; then
    info "✓ 原始任务全部完成，收敛任务仍有 $((convergence_total - convergence_completed)) 个"
  else
    info "✗ 仍有 $remaining 个任务未完成"
  fi

  [[ $all_done -eq 1 ]] && return 0
  return 1
}

# ---------- 子命令: converge-readiness ----------
cmd_converge_readiness() {
  local tasks_file="$CHANGE_DIR/tasks.md"
  if [[ ! -f "$tasks_file" ]]; then
    err "tasks.md 不存在"
    echo '{"change":"'"$CHANGE_NAME"'","ready":false,"reason":"tasks.md not found"}'
    return 2
  fi

  # 检查原始任务（Convergence 节之前的任务）是否全部 - [x]
  local in_convergence=0
  local original_open=0
  local original_total=0

  while IFS= read -r line; do
    if [[ "$line" =~ ^##\ Phase\ [0-9]+:\ Convergence ]]; then
      in_convergence=1
      continue
    fi
    if [[ $in_convergence -eq 0 && "$line" =~ ^-\ \[([ xX])\] ]]; then
      original_total=$((original_total + 1))
      if [[ "${BASH_REMATCH[1]}" != "x" && "${BASH_REMATCH[1]}" != "X" ]]; then
        original_open=$((original_open + 1))
      fi
    fi
  done < "$tasks_file"

  if [[ $original_total -eq 0 ]]; then
    echo '{"change":"'"$CHANGE_NAME"'","ready":false,"reason":"no tasks found"}'
    info "✗ 未找到任何任务"
    return 1
  fi

  if [[ $original_open -gt 0 ]]; then
    echo "{\"change\":\"$CHANGE_NAME\",\"ready\":false,\"reason\":\"$original_open original tasks still open\"}"
    info "✗ 仍有 $original_open 个原始任务未完成，不能进入 converge"
    return 1
  fi

  echo "{\"change\":\"$CHANGE_NAME\",\"ready\":true,\"original_total\":$original_total}"
  info "✓ 所有原始任务完成，可进入 converge"
  return 0
}

# ---------- 子命令: archive-readiness ----------
cmd_archive_readiness() {
  local tasks_file="$CHANGE_DIR/tasks.md"
  local has_proposal=0 has_design=0

  [[ -f "$CHANGE_DIR/proposal.md" ]] && has_proposal=1
  [[ -f "$CHANGE_DIR/design.md" ]]   && has_design=1

  if [[ ! -f "$tasks_file" ]]; then
    echo "{\"change\":\"$CHANGE_NAME\",\"ready\":false,\"reason\":\"tasks.md not found\"}"
    info "✗ tasks.md 不存在"
    return 1
  fi

  # 检查所有任务（含 convergence phase）是否全部 - [x]
  local total=0 completed=0
  while IFS= read -r line; do
    if [[ "$line" =~ ^-\ \[([ xX])\] ]]; then
      total=$((total + 1))
      if [[ "${BASH_REMATCH[1]}" == "x" || "${BASH_REMATCH[1]}" == "X" ]]; then
        completed=$((completed + 1))
      fi
    fi
  done < "$tasks_file"

  local remaining=$((total - completed))

  if [[ $remaining -gt 0 ]]; then
    echo "{\"change\":\"$CHANGE_NAME\",\"ready\":false,\"remaining\":$remaining,\"total\":$total}"
    info "✗ 仍有 $remaining/$total 个任务未完成，不能归档"
    return 1
  fi

  if [[ $has_proposal -eq 0 || $has_design -eq 0 ]]; then
    echo "{\"change\":\"$CHANGE_NAME\",\"ready\":false,\"reason\":\"missing artifacts\"}"
    info "✗ 缺失产物文件"
    return 1
  fi

  echo "{\"change\":\"$CHANGE_NAME\",\"ready\":true,\"total\":$total}"
  info "✓ 所有任务完成且产物完整，可归档"
  return 0
}

# ---------- 子命令: complexity ----------
cmd_complexity() {
  local tasks_file="$CHANGE_DIR/tasks.md"
  local task_count=0 module_count=0 has_multi_domain=0
  local domain="code"  # 默认 domain

  # 从 proposal.md 读取 domain 声明
  if [[ -f "$CHANGE_DIR/proposal.md" ]]; then
    local domain_match
    domain_match=$(grep -oP '<!--\s*domain:\s*\K[a-z]+' "$CHANGE_DIR/proposal.md" 2>/dev/null || true)
    [[ -n "$domain_match" ]] && domain="$domain_match"
  fi

  if [[ -f "$tasks_file" ]]; then
    # 计算任务数
    task_count=$(grep -c '^\- \[[ xX]\]' "$tasks_file" 2>/dev/null || echo 0)

    # 按 domain 计算唯一模块数
    if [[ "$domain" == "code" ]]; then
      # code 域：用源码路径正则提取，取第一级目录
      local modules
      modules=$(grep -oP '(?:src/|lib/|app/)?[a-zA-Z0-9_/-]+\.[a-zA-Z]+' "$tasks_file" 2>/dev/null \
        | sed 's|/[^/]*$||' | sort -u | wc -l)
      module_count=$modules
    else
      # 非 code 域：用 `→ ` 后的交付物标识按目录去重计数
      local modules
      modules=$(grep -oP '→\s*\K\S+' "$tasks_file" 2>/dev/null \
        | sed 's|/[^/]*$||' | sort -u | wc -l)
      module_count=$modules
    fi
  fi

  # 检查 proposal.md 是否涉及多个能力域
  if [[ -f "$CHANGE_DIR/proposal.md" ]]; then
    local scope_lines
    scope_lines=$(sed -n '/^## Scope/,/^## /p' "$CHANGE_DIR/proposal.md" 2>/dev/null | wc -l)
    [[ $scope_lines -gt 10 ]] && has_multi_domain=1
  fi

  # 长程判定（与 propose.md 步骤 4c 一致）
  local is_long=0
  [[ $task_count -ge 5 ]] && is_long=1
  [[ $module_count -ge 3 ]] && is_long=1
  [[ $has_multi_domain -eq 1 ]] && is_long=1

  local complexity="short"
  [[ $is_long -eq 1 ]] && complexity="long"

  cat <<EOF
{
  "change": "$CHANGE_NAME",
  "complexity": "$complexity",
  "domain": "$domain",
  "task_count": $task_count,
  "module_count": $module_count,
  "multi_domain": $has_multi_domain,
  "criteria": {
    "tasks_ge_5": $([ $task_count -ge 5 ] && echo true || echo false),
    "modules_ge_3": $([ $module_count -ge 3 ] && echo true || echo false),
    "multi_domain": $([ $has_multi_domain -eq 1 ] && echo true || echo false)
  }
}
EOF

  info "复杂度: $complexity (domain: $domain, 任务: $task_count, 模块: $module_count, 多域: $has_multi_domain)"
  return 0
}

# ---------- 主分发 ----------
case "$SUBCMD" in
  artifacts)           cmd_artifacts ;;
  tasks)               cmd_tasks ;;
  converge-readiness)  cmd_converge_readiness ;;
  archive-readiness)   cmd_archive_readiness ;;
  complexity)          cmd_complexity ;;
  -h|--help)           usage; exit 0 ;;
  *)                   err "未知子命令: $SUBCMD"; usage; exit 2 ;;
esac
