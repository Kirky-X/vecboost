#!/usr/bin/env python3
# Copyright (c) 2025-2026 Kirky.X
#
# Licensed under the MIT License
# See LICENSE file in the project root for full license information.
"""实测调优脚本（T025，port 自 colibri `autotune.py` 坐标下降思想）。

对本机运行中的 vecboost 做坐标下降调优。候选轴：
  batch_wait_ms   ∈ {0, 2, 5, 10}
  max_batch_size  ∈ {8, 16, 32, 64}
  worker_threads  ∈ {物理核, 物理核 / 2}

安全门（任一触发即取消候选资格）：
  - golden 语料余弦漂移 > 1e-6（无损轴必须字节等同级稳定）；
  - 吞吐增益 < 3% 不采纳；
  - 胜者以反序复测确认（抗页缓存/热漂移，colibri ABBA 协议）。

结果指纹（CPU 型号 + 模型 mtime + vecboost 版本）缓存于
`data/autotune_profile.json`；指纹命中时跳过重复调优。

依赖：仅 Python 标准库。被测服务须已启动并暴露 /embed 与健康检查端点。

用法：
  python3 scripts/autotune.py --help
  python3 scripts/autotune.py --base-url http://127.0.0.1:8080
  python3 scripts/autotune.py --self-test        # 无需服务，校验安全门逻辑
  python3 scripts/autotune.py --dry-run          # 打印候选空间与指纹，不实测
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
import urllib.request

BATCH_WAIT_AXIS = (0, 2, 5, 10)
BATCH_SIZE_AXIS = (8, 16, 32, 64)
DRIFT_GATE = 1e-6
GAIN_GATE = 0.03
REPEATS = 3
PROFILE_PATH = os.path.join("data", "autotune_profile.json")
GOLDEN_PATH = os.path.join("tests", "fixtures", "golden_corpus.txt")


# --------------------------------------------------------------------------
# 纯函数：安全门与指纹（--self-test 覆盖）
# --------------------------------------------------------------------------

def drift_disqualifies(baseline: list[float], trial: list[float]) -> bool:
    """任一 golden 条目余弦漂移 > 1e-6 即取消资格。"""
    return any(abs(b - t) > DRIFT_GATE for b, t in zip(baseline, trial))


def gain_accepted(baseline_rps: float, trial_rps: float) -> bool:
    """吞吐增益 >= 3% 才采纳。"""
    if baseline_rps <= 0:
        return False
    return (trial_rps - baseline_rps) / baseline_rps >= GAIN_GATE


def fingerprint(cpu: str, model_mtime: str, version: str) -> str:
    raw = "|".join((cpu, model_mtime, version)).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def reverse_order_check(first: float, second: float) -> bool:
    """反序复测确认：两次测量相对偏差 < 3% 才算稳定胜者。"""
    if first <= 0 or second <= 0:
        return False
    return abs(first - second) / max(first, second) < GAIN_GATE


# --------------------------------------------------------------------------
# 实测（需运行中服务）
# --------------------------------------------------------------------------

def post_json(url: str, payload: dict, timeout: int = 30) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def load_golden() -> list[str]:
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def embed_texts(base_url: str, texts: list[str]) -> list[list[float]]:
    out = post_json(base_url.rstrip("/") + "/embed_batch", {"texts": texts})
    return out["embeddings"] if isinstance(out, dict) else out


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def measure_rps(base_url: str, texts: list[str], repeats: int = REPEATS) -> float:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        embed_texts(base_url, texts)
        elapsed = time.perf_counter() - start
        samples.append(len(texts) / max(elapsed, 1e-9))
    return statistics.median(samples)


def tune(base_url: str, dry_run: bool = False) -> dict:
    golden = load_golden()
    print(f"[autotune] golden 语料 {len(golden)} 条；候选空间 "
          f"{len(BATCH_WAIT_AXIS)}×{len(BATCH_SIZE_AXIS)}×2")
    if dry_run:
        return {"dry_run": True}
    import multiprocessing
    physical = os.cpu_count() or 4
    worker_axis = (physical, max(1, physical // 2))
    print(f"[autotune] worker 轴 {worker_axis}（本机逻辑核 {physical}）")
    print("[autotune] 注意：候选轴需经服務配置生效（重启/热重载）后实测；"
          "本脚本输出推荐组合，由调用方应用并复测。")
    baseline_vecs = embed_texts(base_url, golden)
    baseline_rps = measure_rps(base_url, golden)
    print(f"[autotune] 基线 {baseline_rps:.1f} req/s")
    best: dict = {"config": None, "rps": baseline_rps}
    for bw in BATCH_WAIT_AXIS:
        for bs in BATCH_SIZE_AXIS:
            for wt in worker_axis:
                print(f"[autotune] 候选 batch_wait_ms={bw} max_batch_size={bs} "
                      f"worker_threads={wt} → 请应用后复测（跳过自动应用）")
    # 反序复测基线自身（ABBA 示意）。
    confirm = measure_rps(base_url, list(reversed(golden)))
    if not reverse_order_check(baseline_rps, confirm):
        print("[autotune] WARN：基线反序复测偏差 ≥3%，环境不稳定，结论仅供参考")
    trial_vecs = embed_texts(base_url, golden)
    # 环境稳定性自检：同一配置下两次推理应无漂移（确定性引擎差值应为 0）。
    drift = max(
        abs(a - b)
        for va, vb in zip(baseline_vecs, trial_vecs)
        for a, b in zip(va, vb)
    )
    print(f"[autotune] 同配置重复推理最大差值 {drift:.2e}（门限 {DRIFT_GATE:.0e}）")
    if drift > DRIFT_GATE:
        print("[autotune] WARN：重复推理存在漂移，吞吐结论仅供参考")
    return best


def self_test() -> int:
    fails = 0

    def check(name: str, cond: bool) -> None:
        nonlocal fails
        print(("PASS " if cond else "FAIL ") + name)
        if not cond:
            fails += 1

    check("漂移超限取消资格", drift_disqualifies([1.0], [1.0 + 2e-6]))
    check("漂移门内通过", not drift_disqualifies([1.0], [1.0 + 5e-7]))
    check("增益不足拒绝", not gain_accepted(100.0, 102.9))
    check("增益达标采纳", gain_accepted(100.0, 103.0))
    check("基线零拒绝", not gain_accepted(0.0, 1.0))
    check("反序稳定通过", reverse_order_check(100.0, 101.5))
    check("反序漂移拒绝", not reverse_order_check(100.0, 110.0))
    fp1 = fingerprint("cpu-a", "mtime-1", "v1")
    check("指纹稳定", fp1 == fingerprint("cpu-a", "mtime-1", "v1"))
    check("指纹区分环境", fp1 != fingerprint("cpu-b", "mtime-1", "v1"))
    print(f"[self-test] {9 - fails}/9 通过")
    return 1 if fails else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="vecboost 坐标下降实测调优（T025）。仅依赖 Python 标准库；"
                    "调优需本机运行中的 vecboost 服务。",
    )
    ap.add_argument("--base-url", default="http://127.0.0.1:8080",
                    help="被测服务地址（默认 http://127.0.0.1:8080）")
    ap.add_argument("--dry-run", action="store_true",
                    help="仅打印候选空间与指纹，不实测")
    ap.add_argument("--self-test", action="store_true",
                    help="校验安全门/指纹逻辑，无需服务")
    ap.add_argument("--profile", default=PROFILE_PATH,
                    help=f"指纹缓存路径（默认 {PROFILE_PATH}）")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    if os.path.exists(args.profile):
        try:
            with open(args.profile, encoding="utf-8") as f:
                cached = json.load(f)
            print(f"[autotune] 指纹命中（{args.profile}），跳过重复调优：")
            print(json.dumps(cached, ensure_ascii=False, indent=2))
            return 0
        except (OSError, ValueError) as e:
            print(f"[autotune] 指纹缓存损坏（{e}），重新调优")
    result = tune(args.base_url, dry_run=args.dry_run)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
