#!/usr/bin/env bash
# Copyright (c) 2025-2026 Kirky.X🌠
# SPDX-License-Identifier: Apache-2.0
#
# link-assets.sh — worktree 测试资产准备（E2E 修复，2026-10-16）
#
# 背景：场景测试与真机测试依赖大体积资产（models/ 3rdparty/ assets/，均
# gitignore 不入库）。在 git worktree 中开发时这些目录不存在，相关测试按
# 守卫打印 SKIP（不假红）；需要真跑时用本脚本把主工作区的资产以符号链接
# 方式就位到当前 checkout。
#
# 用法:
#   bash scripts/link-assets.sh [主工作区路径]
#   缺省自动探测：git rev-parse --git-common-dir 的父目录即主工作区。
#
# 清除: rm models 3rdparty assets
# 注意: 符号链接不被 .gitignore 的 `models/`（尾斜杠，仅匹配目录）模式
#       匹配——git status 会把它显示为 untracked（?? models），不要 git add。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MAIN="${1:-}"

if [ -z "$MAIN" ]; then
  # git common dir 指向主仓库的 .git 目录，其父目录即主工作区
  COMMON="$(git -C "$ROOT" rev-parse --git-common-dir)"
  if [ "$COMMON" = ".git" ]; then
    MAIN="$ROOT"
  else
    MAIN="$(cd "$ROOT" && cd "$(dirname "$COMMON")" && pwd)"
  fi
fi

if [ "$MAIN" = "$ROOT" ]; then
  echo "当前即主工作区（$ROOT），无需链接。"
  exit 0
fi

echo "主工作区: $MAIN"
linked=0
for d in models 3rdparty assets; do
  if [ -L "$ROOT/$d" ]; then
    echo "跳过 $d：符号链接已存在"
    continue
  fi
  if [ -e "$ROOT/$d" ]; then
    echo "跳过 $d：已存在实体目录（非符号链接，不覆盖）"
    continue
  fi
  if [ ! -e "$MAIN/$d" ]; then
    echo "跳过 $d：主工作区无此资产（$MAIN/$d）"
    continue
  fi
  ln -sfn "$MAIN/$d" "$ROOT/$d"
  echo "链接 $d -> $MAIN/$d"
  linked=$((linked + 1))
done

if [ "$linked" -eq 0 ]; then
  echo "未新增任何链接。"
else
  echo "完成（$linked 个）。清除: rm models 3rdparty assets"
fi
