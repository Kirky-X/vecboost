#!/usr/bin/env bash
# archive_change.sh — flock 保护 + 只读哨兵 + 可选 sync + commit SHA 锚定的归档操作。
#
# 这是 archive 子命令（references/archive.md）的确定性执行器：
#   - 获取 change 级独占 flock（防并发损坏）
#   - 维护 specmark/archive/.readonly 哨兵（只读历史标记）
#   - 拒绝覆盖既有归档条目（只读强制）
#   - 可选 --sync：对每个 delta spec 调 merge_delta_spec.py（确定性合并，替代 LLM）
#   - 移动 specmark/changes/<name> → specmark/archive/<date>-<name>
#   - 写 meta.json：锚定归档时的 git commit SHA + ISO 日期
#
# 用法：
#   archive_change.sh <change-name> [--sync] [--date YYYY-MM-DD]
#
# 退出码：0 成功；1 输入/状态错误；2 锁竞争失败。

set -euo pipefail

SCRIPT_REAL="$(readlink -f "$0")"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_REAL")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

err()  { printf '[ERROR] %s\n' "$*" >&2; }
info() { printf '[INFO] %s\n' "$*" >&2; }

usage() {
  cat <<EOF
Usage: archive_change.sh <change-name> [--sync] [--date YYYY-MM-DD] [-h]

Acquires an exclusive per-change flock, enforces archive read-only sentinel,
moves specmark/changes/<name> to specmark/archive/<date>-<name>, optionally
syncs delta specs into specmark/specs/ (via merge_delta_spec.py), and writes
meta.json anchoring the archive to the current git commit SHA.

Options:
  --sync            Run merge_delta_spec.py for each delta spec under <name>/specs/.
  --date YYYY-MM-DD Override archive date stamp (default: today UTC).
  -h, --help        Show this help.
EOF
}

CHANGE_NAME=""
SYNC=0
DATE_OVERRIDE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sync) SYNC=1 ;;
    --date)
      shift
      [[ $# -gt 0 ]] || { err "--date 需要参数"; exit 1; }
      DATE_OVERRIDE="$1"
      ;;
    -h|--help) usage; exit 0 ;;
    -*) err "未知选项: $1"; usage; exit 1 ;;
    *)
      [[ -z "$CHANGE_NAME" ]] && CHANGE_NAME="$1" || { err "多余参数: $1"; exit 1; }
      ;;
  esac
  shift
done

[[ -n "$CHANGE_NAME" ]] || { err "需要 change 名"; usage; exit 1; }

# kebab-case 校验，防路径穿越
if [[ ! "$CHANGE_NAME" =~ ^[a-z0-9][a-z0-9-]*$ ]]; then
  err "非法 change 名（须 kebab-case）: $CHANGE_NAME"
  exit 1
fi

CHANGE_DIR="$ROOT/specmark/changes/$CHANGE_NAME"
ARCHIVE_ROOT="$ROOT/specmark/archive"
SPECS_ROOT="$ROOT/specmark/specs"
LOCKS_DIR="$ROOT/specmark/.locks"

[[ -d "$CHANGE_DIR" ]] || { err "change 目录不存在: $CHANGE_DIR"; exit 1; }

DATE="${DATE_OVERRIDE:-$(date -u +%Y-%m-%d)}"
if [[ ! "$DATE" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
  err "非法日期: $DATE"; exit 1
fi

TARGET="$ARCHIVE_ROOT/$DATE-$CHANGE_NAME"

mkdir -p "$LOCKS_DIR" "$ARCHIVE_ROOT"

# 只读哨兵：归档树只读历史标记
SENTINEL="$ARCHIVE_ROOT/.readonly"
if [[ ! -f "$SENTINEL" ]]; then
  printf 'specmark archive — read-only history. 仅追加新归档条目，禁止修改/删除既有条目。\n' > "$SENTINEL"
fi

# 只读强制：拒绝覆盖既有归档（这是写入哨兵后唯一允许的写——追加新条目）
if [[ -e "$TARGET" ]]; then
  err "归档目标已存在: $TARGET"
  err "只读强制：拒绝覆盖。请重命名既有归档或换 --date。"
  exit 1
fi

LOCKFILE="$LOCKS_DIR/$CHANGE_NAME.lock"
LOCK_HELD=0

# 独占 flock（fd 9），最多等 10 秒
exec 9>"$LOCKFILE"
if ! flock -w 10 9; then
  err "无法获取锁（其他进程持有）: $LOCKFILE"
  exit 2
fi
LOCK_HELD=1

cleanup() {
  if [[ "$LOCK_HELD" == "1" ]]; then
    flock -u 9 2>/dev/null || true
    LOCK_HELD=0
  fi
}
trap cleanup EXIT

# 锁内二次检查（race-safe）
if [[ -e "$TARGET" ]]; then
  err "归档目标已存在（锁后）: $TARGET"
  exit 1
fi
if [[ ! -d "$CHANGE_DIR" ]]; then
  err "change 目录在锁内消失: $CHANGE_DIR"
  exit 1
fi

# 捕获 commit SHA（best-effort；无 git 则 null）
COMMIT_JSON="null"
if git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  SHA="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || true)"
  if [[ -n "$SHA" ]]; then
    COMMIT_JSON="\"$SHA\""
  fi
fi

# 可选 sync：移动前 delta spec 仍在 change 目录
SYNC_RESULT="未启用 --sync"
if [[ $SYNC -eq 1 ]]; then
  shopt -s nullglob
  DELTA_SPECS=("$CHANGE_DIR"/specs/*/spec.md)
  shopt -u nullglob
  if [[ ${#DELTA_SPECS[@]} -gt 0 ]]; then
    for delta in "${DELTA_SPECS[@]}"; do
      cap="$(basename "$(dirname "$delta")")"
      main_spec="$SPECS_ROOT/$cap/spec.md"
      # kebab-case 校验 capability，防路径穿越
      if [[ ! "$cap" =~ ^[a-z0-9][a-z0-9-]*$ ]]; then
        err "非法 capability 目录名: $cap"
        exit 1
      fi
      info "sync delta → $main_spec"
      python3 "$SCRIPT_DIR/merge_delta_spec.py" \
        --main "$main_spec" --delta "$delta" --out "$main_spec" \
        || { err "delta 合并失败: $delta"; exit 1; }
    done
    SYNC_RESULT="已同步 ${#DELTA_SPECS[@]} 个 spec"
  else
    SYNC_RESULT="--sync 但无 delta spec"
  fi
fi

# 执行移动（原子）
mv "$CHANGE_DIR" "$TARGET"

# 写 meta.json（锚定 commit SHA + 日期）
META="$TARGET/meta.json"
SYNC_JSON=$([[ $SYNC -eq 1 ]] && echo true || echo false)
cat > "$META" <<EOF
{
  "change": "$CHANGE_NAME",
  "archived_at": "$DATE",
  "commit_sha": $COMMIT_JSON,
  "synced": $SYNC_JSON
}
EOF

# 摘要输出（供 archive 子命令展示）
printf '## 归档完成\n\n'
printf '**变更：** %s\n' "$CHANGE_NAME"
printf '**归档到：** specmark/archive/%s-%s/\n' "$DATE" "$CHANGE_NAME"
printf '**Commit SHA：** %s\n' "${COMMIT_JSON//\"/}"
printf '**Sync：** %s\n' "$SYNC_RESULT"
printf '**锁：** 已释放 %s\n' "$LOCKFILE"
