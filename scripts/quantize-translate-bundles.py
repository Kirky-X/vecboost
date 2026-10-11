#!/usr/bin/env python3
"""quantize-translate-bundles.py — 翻译 bundle int8 动态量化（translation-int8-optin D1）。

对 bundle 的 onnx/encoder_model.onnx 与 onnx/decoder_model_merged.onnx 逐图
quantize_dynamic(weight_type=QInt8)，产出同名 *_quantized.onnx（与
ENCODER_CANDIDATES/DECODER_CANDIDATES 的 _quantized 次选位对齐）。

图契约不漂移：生成即断言输入/输出的名与形状和源完全一致（喂值契约 +
past KV 名表锚点），onnx.checker 校验通过；不一致时非零退出且不落半成品。

用法：
    python3.12 scripts/quantize-translate-bundles.py [--bundle models/m2m100_418M]...
    # 缺省处理 4 个配置 pair 的 3 个 bundle；--no-skip-existing 强制重产出
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys

import onnx
from onnxruntime.quantization import quantize_dynamic

DEFAULT_BUNDLES = [
    "models/m2m100_418M",
    "models/opus-mt-en-fr",
    "models/nllb-200-distilled-600M",
]

def io_signature(model_path: str) -> list[tuple[str, str, str]]:
    """图 I/O 名+形状签名（顺序保持）：(kind, name, shape) 三元组列表。"""
    model = onnx.load(model_path, load_external_data=False)

    def value_info(entries):
        out = []
        for v in entries:
            dims = []
            for d in v.type.tensor_type.shape.dim:
                dims.append(str(d.dim_value) if d.HasField("dim_value") else d.dim_param)
            out.append((v.name, str(v.type.tensor_type.elem_type), str(dims)))
        return out

    return [("in", *vi) for vi in value_info(model.graph.input)] + [
        ("out", *vi) for vi in value_info(model.graph.output)
    ]


def quantize_one(src: str, dst: str) -> tuple[int, int, str]:
    """量化单图并校验契约；返回 (src_bytes, dst_bytes, report)。

    契约不一致或校验失败抛异常（调用方统一转非零退出；半成品由 check
    失败前的 os.remove 清理——quantize_dynamic 先写临时再替换，此处防
    部分写留档）。
    """
    before = io_signature(src)
    quantize_dynamic(
        model_input=src,
        model_output=dst,
        weight_type=onnx.TensorProto.INT8,  # QInt8
    )
    after = io_signature(dst)
    if before != after:
        if os.path.exists(dst):
            os.remove(dst)
        raise RuntimeError(f"{dst}: I/O 契约漂移\n  before={before}\n  after={after}")
    onnx.checker.check_model(dst, full_check=False)
    return os.path.getsize(src), os.path.getsize(dst), f"  ok {dst}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", action="append", default=None,
                        help="bundle 目录（可多选；缺省 3 个配置 bundle）")
    parser.add_argument("--no-skip-existing", action="store_true",
                        help="已存在 *_quantized.onnx 也重新产出（缺省跳过）")
    args = parser.parse_args()
    bundles = args.bundle if args.bundle else DEFAULT_BUNDLES

    report_lines: list[str] = []
    total_src = total_dst = 0
    failures: list[str] = []
    for bundle in bundles:
        for graph in ("encoder_model.onnx", "decoder_model_merged.onnx"):
            src = os.path.join(bundle, "onnx", graph)
            dst = os.path.join(bundle, "onnx", graph.replace(".onnx", "_quantized.onnx"))
            if not os.path.isfile(src):
                report_lines.append(f"  skip {src}（源缺失）")
                continue
            if os.path.isfile(dst) and not args.no_skip_existing:
                report_lines.append(f"  skip {dst}（已存在）")
                continue
            try:
                s, d, ok = quantize_one(src, dst)
            except Exception as exc:  # 契约漂移/校验失败：显性失败
                failures.append(f"{dst}: {exc}")
                report_lines.append(f"  FAIL {dst}（详见上方；半成品已清理）")
                continue
            total_src += s
            total_dst += d
            report_lines.append(f"{ok}  {s / 1e6:.0f}MB -> {d / 1e6:.0f}MB "
                               f"({100 * d / s:.1f}%)")
            sys.stderr.write(report_lines[-1] + "\n")

    ratio = f"{100 * total_dst / total_src:.1f}%" if total_src else "n/a"
    summary = (f"total {total_src / 1e6:.0f}MB -> {total_dst / 1e6:.0f}MB ({ratio}); "
               f"failures={len(failures)}")
    os.makedirs("logs", exist_ok=True)
    log_path = f"logs/int8-quantize-{datetime.date.today().isoformat()}.log"
    with open(log_path, "a") as fh:
        fh.write(summary + "\n" + "\n".join(report_lines) + "\n\n")
    print(summary)
    print(f"report: {log_path}")
    if failures:
        for f in failures:
            print(f"ERROR {f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
