#!/usr/bin/env python3
# Copyright (c) 2025-2026 Kirky.X🌠
# SPDX-License-Identifier: Apache-2.0

"""warmup_corpus.py — 暖语料回放（仅 Python 标准库）。

读取语料文件（每行一条文本），去重保序后分批调 `/embed/batch` 预热
服务端结果缓存——`.coli_usage` 暖语料思想的 embedding 映射：高频语料
提前打热缓存，真实流量首请求即命中。

用法：
    python3 scripts/warmup_corpus.py corpus.txt --base-url http://127.0.0.1:8080 --batch 16

退出码：0 = 全部批次成功；1 = 任一批次失败（打印失败批次号与原因）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="暖语料回放：预热 vecboost embedding 缓存")
    ap.add_argument("corpus", help="语料文件路径（UTF-8，每行一条文本）")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080",
                    help="服务地址（默认 http://127.0.0.1:8080）")
    ap.add_argument("--batch", type=int, default=16, help="每批文本数（默认 16）")
    args = ap.parse_args()

    corpus = Path(args.corpus)
    if not corpus.is_file():
        print(f"FAIL 语料文件不存在: {corpus}", file=sys.stderr)
        return 1

    lines = [ln.strip() for ln in corpus.read_text(encoding="utf-8").splitlines() if ln.strip()]
    # 去重保序：重复文本只推理一次（服务端缓存同样按内容键）
    unique = list(dict.fromkeys(lines))
    if not unique:
        print("FAIL 语料为空", file=sys.stderr)
        return 1

    batch_size = max(1, args.batch)
    url = args.base_url.rstrip("/") + "/embed/batch"
    start = time.perf_counter()
    failed_batches: list[str] = []
    total = 0
    for i in range(0, len(unique), batch_size):
        chunk = unique[i:i + batch_size]
        payload = json.dumps({"texts": chunk}).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                resp.read()
            total += len(chunk)
            print(f"  batch {i // batch_size + 1}: {len(chunk)} texts OK")
        except Exception as e:  # noqa: BLE001 — 脚本要打印任何失败原因
            failed_batches.append(f"batch {i // batch_size + 1}: {e}")

    elapsed = time.perf_counter() - start
    print(
        f"warmup done: {len(lines)} 行 → 去重 {len(unique)} 条"
        f"（省 {len(lines) - len(unique)} 次），成功 {total} 条，"
        f"{elapsed:.2f}s（{total / elapsed:.1f} texts/s）"
    )
    if failed_batches:
        for f in failed_batches:
            print(f"FAIL {f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
