#!/usr/bin/env python3
"""validate_manifest.py — 实验manifest 校验器（T029，仅标准库）。

校验 docs/experiments/README.md 定义的必填字段、runs≥3、中位数一致性、
evidence sha256。用法：

    python3 scripts/validate_manifest.py <manifest.json>

退出码：0 = 校验通过；1 = 校验失败（列出全部问题）。
校验通过 ≠ 结论成立：ABBA 顺序与单变量纪律由评审人核对。
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path

REQUIRED_TOP = ["hypothesis", "change", "hardware", "baseline", "trial", "quality", "evidence"]
REQUIRED_CHANGE = ["commit", "variable"]
REQUIRED_RUNS = 3
REQUIRED_QUALITY = ["kind", "passed"]
REQUIRED_EVIDENCE = ["baseline_log", "baseline_sha256", "trial_log", "trial_sha256"]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(manifest_path: Path) -> list[str]:
    errors: list[str] = []
    try:
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return [f"manifest 不可读: {e}"]

    for key in REQUIRED_TOP:
        if key not in m:
            errors.append(f"缺少必填字段: {key}")
    if errors:
        return errors

    for key in REQUIRED_CHANGE:
        if key not in m.get("change", {}):
            errors.append(f"缺少 change.{key}（单变量纪律要求记录被测 commit 与唯一变量）")
    if not str(m.get("hardware", {}).get("cpu", "")).strip():
        errors.append("hardware.cpu 缺失——性能数字必须标注主机")

    for side in ("baseline", "trial"):
        block = m.get(side, {})
        runs = block.get("runs")
        if not isinstance(runs, list) or len(runs) < REQUIRED_RUNS:
            errors.append(f"{side}.runs 须为 ≥{REQUIRED_RUNS} 次的数值数组")
            continue
        if any(not isinstance(r, (int, float)) for r in runs):
            errors.append(f"{side}.runs 含非数值项")
            continue
        recomputed = statistics.median(runs)
        declared = block.get("median")
        if not isinstance(declared, (int, float)):
            errors.append(f"{side}.median 缺失或非数值")
        elif abs(declared - recomputed) > 1e-9:
            errors.append(f"{side}.median 声明 {declared} 与重算 {recomputed} 不一致")

    quality = m.get("quality", {})
    for key in REQUIRED_QUALITY:
        if key not in quality:
            errors.append(f"缺少 quality.{key}")
    if quality.get("kind") == "byte_identical" and quality.get("byte_identical") is not True:
        errors.append("无损优化 (kind=byte_identical) 要求 quality.byte_identical = true")

    evidence = m.get("evidence", {})
    for key in REQUIRED_EVIDENCE:
        if key not in evidence:
            errors.append(f"缺少 evidence.{key}")
    if not errors:
        for log_key, hash_key in (("baseline_log", "baseline_sha256"), ("trial_log", "trial_sha256")):
            log_path = Path(evidence[log_key])
            if not log_path.is_file():
                errors.append(f"evidence.{log_key} 文件不存在: {log_path}")
                continue
            actual = sha256_of(log_path)
            declared = str(evidence[hash_key]).lower()
            if actual != declared:
                errors.append(f"evidence.{hash_key} 不匹配：声明 {declared}，实际 {actual}")

    if not str(m.get("conclusion", "")).strip():
        errors.append("缺少 conclusion（接受/拒绝 + 依据；负结果同样要记录）")
    return errors


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    path = Path(sys.argv[1])
    errors = check(path)
    if errors:
        print(f"FAIL {path}")
        for e in errors:
            print(f"  - {e}")
        return 1
    print(f"PASS {path}（字段/runs/中位数/sha256 校验通过；ABBA 与单变量请评审人核对）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
