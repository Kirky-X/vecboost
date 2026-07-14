#!/usr/bin/env bash
# archive.sh — flock 保护 + 只读哨兵 + 可选 sync + commit SHA 锚定的归档操作。
#
# 由 archive_change.sh 与 merge_delta_spec.py 合并而来：delta spec 的确定性
# 合并逻辑已内嵌为 merge_delta_spec()（Python heredoc），不再依赖外部 .py 文件。
#
# 用法：
#   archive.sh <change-name> [--sync] [--date YYYY-MM-DD]
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
Usage: archive.sh <change-name> [--sync] [--date YYYY-MM-DD] [-h]

Acquires an exclusive per-change flock, enforces archive read-only sentinel,
moves specmark/changes/<name> to specmark/archive/<date>-<name>, optionally
syncs delta specs into specmark/specs/ (via inlined merge_delta_spec), and writes
meta.json anchoring the archive to the current git commit SHA.

Options:
  --sync            Run merge_delta_spec for each delta spec under <name>/specs/.
  --date YYYY-MM-DD Override archive date stamp (default: today UTC).
  -h, --help        Show this help.
EOF
}

# =============================================================================
# merge_delta_spec — 把 delta spec 确定性地合并进 main spec（替代 LLM 合并）。
# 参数透传给内嵌 Python：--main / --delta / --out / --dry-run
# =============================================================================
merge_delta_spec() {
  python3 - "$@" <<'PYEOF'
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REQ_HEADER_RE = re.compile(r"^###\s+(R-[A-Za-z0-9][A-Za-z0-9-]*?-\d+)\s*:\s*(.*)$")
SECTION_RE = re.compile(r"^##\s+(.+?)\s*$")
TITLE_RE = re.compile(r"^#\s+(.+?)\s*$")

DELETE_MARKER = "~~DELETE~~"


class Spec:
    def __init__(self) -> None:
        self.title: str = ""  # 整行，如 "# Spec — references-consistency"
        self.intro: list[str] = []  # 标题后、首个 ## 前的行
        self.requirements: dict[
            str, dict
        ] = {}  # R-ID -> {"title": str, "body": [lines]}
        self.sections: dict[str, list[str]] = {}  # 段名 -> 行（非 Requirements 段）
        self.section_order: list[str] = []


def _strip_blank_ends(lines: list[str]) -> list[str]:
    out = list(lines)
    while out and out[-1].strip() == "":
        out.pop()
    while out and out[0].strip() == "":
        out.pop(0)
    return out


def parse_spec(path: Path) -> Spec:
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    spec = Spec()

    n = len(lines)
    i = 0
    while i < n and lines[i].strip() == "":
        i += 1
    if i < n:
        m = TITLE_RE.match(lines[i])
        if m:
            spec.title = lines[i].rstrip()
            i += 1

    intro_start = i
    while i < n and not SECTION_RE.match(lines[i]):
        i += 1
    spec.intro = _strip_blank_ends(lines[intro_start:i])

    cur_section: str | None = None
    cur_body: list[str] = []
    in_req = False
    cur_req_id: str | None = None
    cur_req_title: str | None = None
    cur_req_body: list[str] = []

    def flush_section() -> None:
        nonlocal cur_section, cur_body
        if cur_section is not None:
            spec.sections[cur_section] = _strip_blank_ends(cur_body)
            spec.section_order.append(cur_section)
        cur_body = []

    def flush_req() -> None:
        nonlocal cur_req_id, cur_req_title, cur_req_body
        if cur_req_id is not None:
            spec.requirements[cur_req_id] = {
                "title": (cur_req_title or "").rstrip(),
                "body": _strip_blank_ends(cur_req_body),
            }
        cur_req_id = None
        cur_req_title = None
        cur_req_body = []

    while i < n:
        line = lines[i]
        m_sec = SECTION_RE.match(line)
        if m_sec:
            if in_req:
                flush_req()
            flush_section()
            cur_section = m_sec.group(1).strip()
            in_req = cur_section.lower() == "requirements"
            cur_body = []
            i += 1
            continue
        if in_req:
            m_req = REQ_HEADER_RE.match(line)
            if m_req:
                flush_req()
                cur_req_id = m_req.group(1)
                cur_req_title = m_req.group(2)
                cur_req_body = []
                i += 1
                continue
            if cur_req_id is not None:
                cur_req_body.append(line.rstrip())
                i += 1
                continue
            cur_body.append(line.rstrip())
            i += 1
            continue
        if cur_section is not None:
            cur_body.append(line.rstrip())
        i += 1

    if in_req:
        flush_req()
    flush_section()
    return spec


def _capability_from_title(title: str) -> str:
    m = TITLE_RE.match(title) if title else None
    if not m:
        return "capability"
    rest = m.group(1)
    if rest.lower().startswith("spec"):
        rest = re.sub(r"^spec\s*[—\-:]\s*", "", rest, flags=re.IGNORECASE)
    return rest.strip() or "capability"


def _sort_req_ids(ids):
    def key(rid):
        m = re.match(r"^R-(.+)-(\d+)$", rid)
        if m:
            return (m.group(1).lower(), int(m.group(2)))
        return (rid.lower(), 0)

    return sorted(ids, key=key)


def merge(main: Spec, delta: Spec, main_existed: bool) -> Spec:
    result = Spec()

    if main_existed:
        result.title = main.title
        result.intro = list(main.intro)
    else:
        result.title = delta.title or "# Spec — capability"
        cap = _capability_from_title(result.title)
        result.intro = [f"> Main spec for capability `{cap}`."]

    result.section_order = list(main.section_order)
    for s in delta.section_order:
        if s not in result.section_order:
            result.section_order.append(s)

    merged_reqs: dict[str, dict] = {}
    for rid in set(main.requirements) | set(delta.requirements):
        in_main = rid in main.requirements
        in_delta = rid in delta.requirements
        if in_delta:
            d = delta.requirements[rid]
            if d["title"].strip() == DELETE_MARKER:
                continue  # DELETE
            merged_reqs[rid] = {
                "title": d["title"],
                "body": list(d["body"]),
            }  # MODIFY / ADD
        elif in_main:
            merged_reqs[rid] = {
                "title": main.requirements[rid]["title"],
                "body": list(main.requirements[rid]["body"]),
            }  # KEEP
    result.requirements = merged_reqs

    result.sections = {}
    for s in result.section_order:
        if s.lower() == "requirements":
            result.sections[s] = list(main.sections.get(s, []))
            continue
        main_body = main.sections.get(s, [])
        delta_body = delta.sections.get(s, [])
        merged = list(main_body)
        seen = {ln.strip() for ln in main_body if ln.strip()}
        for ln in delta_body:
            if ln.strip() and ln.strip() not in seen:
                merged.append(ln)
                seen.add(ln.strip())
        result.sections[s] = merged
    return result


def emit(spec: Spec) -> str:
    blocks: list[str] = []
    if spec.title:
        blocks.append(spec.title)
    intro = [ln for ln in spec.intro if ln.strip()]
    if intro:
        blocks.append("\n".join(intro))

    for sec in spec.section_order:
        if sec.lower() == "requirements":
            req_intro = [ln for ln in spec.sections.get(sec, []) if ln.strip()]
            req_blocks: list[str] = []
            if req_intro:
                req_blocks.append("\n".join(req_intro))
            for rid in _sort_req_ids(spec.requirements.keys()):
                d = spec.requirements[rid]
                hdr = f"### {rid}: {d['title']}".rstrip()
                body = "\n".join(d["body"]).strip("\n")
                req_blocks.append(hdr + "\n\n" + body if body else hdr)
            section_text = "\n\n".join(req_blocks)
            blocks.append(
                f"## {sec}\n\n" + section_text if section_text else f"## {sec}"
            )
        else:
            body = "\n".join(spec.sections.get(sec, [])).strip("\n")
            blocks.append(f"## {sec}\n\n" + body if body else f"## {sec}")

    return "\n\n".join(blocks) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="把 delta spec 确定性地合并进 main spec（替代 LLM 合并）。",
    )
    ap.add_argument("--main", required=True, help="main spec 路径（可不存在）")
    ap.add_argument("--delta", required=True, help="delta spec 路径（必须存在）")
    ap.add_argument("--out", help="输出路径（默认写回 --main）")
    ap.add_argument("--dry-run", action="store_true", help="只打印合并结果，不写文件")
    args = ap.parse_args(argv)

    main_path = Path(args.main)
    delta_path = Path(args.delta)
    if not delta_path.is_file():
        print(f"error: delta spec 未找到: {delta_path}", file=sys.stderr)
        return 1

    main_existed = main_path.is_file()
    main_spec = parse_spec(main_path) if main_existed else Spec()
    delta_spec = parse_spec(delta_path)

    result = merge(main_spec, delta_spec, main_existed)
    out_text = emit(result)

    if args.dry_run:
        sys.stdout.write(out_text)
        if not out_text.endswith("\n"):
            sys.stdout.write("\n")
        return 0

    out_path = Path(args.out) if args.out else main_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out_text, encoding="utf-8")
    return 0


sys.exit(main(sys.argv[1:]))
PYEOF
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

# kebab-case 校验（允许点号用于版本号），防路径穿越
if [[ ! "$CHANGE_NAME" =~ ^[a-z0-9][a-z0-9.-]*$ ]]; then
  err "非法 change 名（须 kebab-case，可含版本号点号）: $CHANGE_NAME"
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
      merge_delta_spec --main "$main_spec" --delta "$delta" --out "$main_spec" \
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
