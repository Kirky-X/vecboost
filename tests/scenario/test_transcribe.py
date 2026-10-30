# Copyright (c) 2025-2026 Kirky.X🌠
# SPDX-License-Identifier: Apache-2.0

"""转录服务（task=transcribe / `POST /api/v1/transcribe`）黑盒场景测试。

覆盖（R-transcribe 域）：
- TR-01 线上契约与真机 ASR：16kHz 单声道 f32 样本数组 → 200 +
  text/segments 形状、text == 各段拼接、段起止有序；
- TR-02 空样本 400（域层 validate 显性拒绝）；
- TR-03 非有限值样本 400（NaN 直发非法 JSON 字面，serde 拒绝）。

运行条件（缺任一按设计 SKIP，不误报红）：
- 二进制含 `transcribe` feature：无 feature 构建下服务端点为 unconfigured
  （main.rs `#[cfg(not(feature = "transcribe"))]` 分支），TR-01 以 400
  transcribe-not-configured 响应显性 SKIP；
- 转录资产 assets/models（qwen3 三图 + silero_vad + tokenizer/）与
  assets/smoke10s.wav 就位（大体积资产不入库，缺失时整模块 SKIP）。

构建/运行：
    cargo build --features "http,grpc,cli,auth,db,sqlite,mcp,transcribe"
    python3 -m pytest test_transcribe.py -q
"""

from __future__ import annotations

import json
import pathlib
import sys
import wave

import pytest

from conftest import (
    PROJECT_ROOT,
    http_post,
    spawn_server,
    stop_server,
)

TRANSCRIBE_PORT = 9101
TRANSCRIBE_MODELS = PROJECT_ROOT / "assets" / "models"
# 真实中文播客语音 48s（reference.txt 为同源全文，仅作人工核对参考；
# smoke10s 在 VAD 阈值 0.5 下未检出语音段，不作为正例断言样例）
SMOKE_WAV = PROJECT_ROOT / "assets" / "audio48s.wav"

# 转录推理为 CPU 密集（VAD + 逐段 decode），客户端等待放宽；
# server 侧 request_timeout_seconds 同步放宽到 300
TRANSCRIBE_HTTP_TIMEOUT = 300.0


def _assets_ready() -> bool:
    return (
        (TRANSCRIBE_MODELS / "qwen3-conv_frontend.onnx").exists()
        and (TRANSCRIBE_MODELS / "silero_vad.onnx").exists()
        and (TRANSCRIBE_MODELS / "tokenizer").is_dir()
        and SMOKE_WAV.exists()
    )


def _config() -> str:
    lines = [
        "[server]",
        'host = "127.0.0.1"',
        f"port = {TRANSCRIBE_PORT}",
        "request_timeout_seconds = 300",
        # 48s 音频 = 768k f32 样本 ≈ 7-8MB JSON，超过缺省 5MiB body 上限
        "body_limit_mb = 32",
        "",
        "[model]",
        f'model_path = "{PROJECT_ROOT / "models" / "BAAI-bge-small-en-v1.5"}"',
        # onnx-primary：主模型资产为 safetensors，显式钉 candle（配套 build 需 --features candle）
        'engine_type = "candle"',
        "expected_dimension = 384",
        "",
        "[embedding]",
        "cache_enabled = true",
        "",
        "[rate_limit]",
        "enabled = false",
        "",
        "[auth]",
        "enabled = false",
        "",
        "[database]",
        'url = "sqlite::memory:"',
        "",
        "[transcribe]",
        "enabled = true",
        f'models_dir = "{TRANSCRIBE_MODELS}"',
        "vad_threshold = 0.5",
        "",
    ]
    return "\n".join(lines)


def _wav_samples(path: pathlib.Path) -> list[float]:
    """16-bit PCM mono WAV → f32 归一化样本（wave 标准库，零额外依赖）。"""
    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 1, "转录输入必须为单声道"
        assert w.getframerate() == 16000, "转录输入必须为 16kHz"
        assert w.getsampwidth() == 2, "转录输入必须为 16-bit PCM"
        raw = w.readframes(w.getnframes())
    import struct

    count = len(raw) // 2
    ints = struct.unpack(f"<{count}h", raw)
    return [s / 32768.0 for s in ints]


@pytest.fixture(scope="module")
def transcribe_server():
    if not _assets_ready():
        pytest.skip("转录资产缺失（assets/models 或 assets/smoke10s.wav），SKIP")
    # ORT load-dynamic 定位：服务器进程显式指向仓内 3rdparty 动态库，
    # 库名按目标平台选择（与 src/engine/session.rs ort_dylib_default 同口径）
    ort_lib = "onnxruntime.dll" if sys.platform == "win32" else (
        "libonnxruntime.dylib" if sys.platform == "darwin" else "libonnxruntime.so"
    )
    s = spawn_server(
        "transcribe",
        TRANSCRIBE_PORT,
        _config(),
        env_extra={
            "ORT_DYLIB_PATH": str(PROJECT_ROOT / "3rdparty" / "onnxruntime" / ort_lib)
        },
    )
    yield s
    stop_server(s)


def _skip_if_unconfigured(status: int, body: dict) -> None:
    """无 transcribe feature 构建下服务返回 400 transcribe-not-configured——
    按设计 SKIP（feature 矩阵中该组合不适用本套件）。"""
    if status == 400 and "not configured" in json.dumps(body).lower():
        pytest.skip("二进制未启用 transcribe feature（端点 unconfigured），SKIP")


def test_tr01_wire_contract_and_real_asr(transcribe_server):
    """TR-01：真机 ASR 线上契约——text 拼接不变式、段起止有序、内容非空。"""
    # 取前 10s（1.5MB body）做真机断言：避开大 body 一锤子测试（48s 全长
    # ≈ 7-8MB 超缺省 5MiB 上限，属独立背压维度，另立用例若需要）
    samples = _wav_samples(SMOKE_WAV)[:160000]
    assert samples, "样例音频不得为空"
    status, body = http_post(
        transcribe_server["port"],
        "/api/v1/transcribe",
        {"samples": samples},
        timeout=TRANSCRIBE_HTTP_TIMEOUT,
    )
    _skip_if_unconfigured(status, body)
    assert status == 200, f"转录应 200，实际 {status}: {str(body)[:300]}"
    segments = body["segments"]
    assert isinstance(segments, list) and segments, (
        f"48s 真实语音样例必须产出至少一段，got: {str(body)[:500]}"
    )
    for seg in segments:
        assert seg["start_sec"] < seg["end_sec"], f"段起止须有序: {seg}"
    assert body["text"] == "".join(s["text"] for s in segments), "text 必须等于各段按序拼接"
    assert body["text"].strip(), "样例语音的转录文本不得为空"
    # 真实中文语音：转录文本应含 CJK 字符（防 tokenizer 退化/乱码防线）
    assert any("\u4e00" <= ch <= "\u9fff" for ch in body["text"]), (
        f"中文语音转录应含 CJK 字符，got: {body['text'][:200]}"
    )


def test_tr02_empty_samples_rejected(transcribe_server):
    """TR-02：空样本数组 → 400（域层显性拒绝，不落引擎）。"""
    status, body = http_post(
        transcribe_server["port"],
        "/api/v1/transcribe",
        {"samples": []},
        timeout=60,
    )
    _skip_if_unconfigured(status, body)
    assert status == 400, f"空样本应 400，实际 {status}: {str(body)[:200]}"


def test_tr03_non_finite_samples_rejected(transcribe_server):
    """TR-03：非有限值样本 → 400（NaN 字面被 serde 拒绝或域校验拒绝）。"""
    status, body = http_post(
        transcribe_server["port"],
        "/api/v1/transcribe",
        raw_body='{"samples": [NaN]}',
        timeout=60,
    )
    _skip_if_unconfigured(status, body)
    assert status == 400, f"非有限值应 400，实际 {status}: {str(body)[:200]}"


def test_tr04_full_length_body_within_configured_limit(transcribe_server):
    """TR-04：48s 全量样本（JSON ≈14MB，超缺省 5MiB 但小于配置 32MiB）→ 200。

    体量维度回归钉：axum 提取器层 DefaultBodyLimit 2MB 隐藏硬顶修复后，
    `body_limit_mb` 配置值真实生效（修复前该请求 413）且大音频真机转录成功。
    """
    samples = _wav_samples(SMOKE_WAV)
    assert len(samples) == 16000 * 48
    status, body = http_post(
        transcribe_server["port"],
        "/api/v1/transcribe",
        {"samples": samples},
        timeout=TRANSCRIBE_HTTP_TIMEOUT,
    )
    _skip_if_unconfigured(status, body)
    assert status == 200, f"48s 全量转录应 200，实际 {status}: {str(body)[:300]}"
    assert body["segments"], "48s 真实语音必须产出至少一段"
    assert body["text"].strip(), "48s 转录文本不得为空"
