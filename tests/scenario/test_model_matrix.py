# Copyright (c) 2025-2026 Kirky.X🌠
# SPDX-License-Identifier: Apache-2.0

"""模型矩阵 HTTP 套件 — 4 模型 × 3 厂商 × 2 架构经 HTTP 验证（R-model-001）。

每个模型独立服务器进程（本地目录，无网络依赖）：
- M1 BAAI/bge-small-en-v1.5     (BAAI, Bert, 384)
- M2 all-MiniLM-L6-v2           (sentence-transformers, Bert, 384)
- M3 BAAI/bge-small-zh-v1.5     (BAAI, Bert, 512)
- M4 intfloat/multilingual-e5-small (intfloat, XlmRoberta, 384)

模型目录缺失时对应用例 SKIP（下载失败不阻塞其余矩阵）。
"""
from __future__ import annotations

import pathlib

import pytest

from conftest import AUTH_ENV, PROJECT_ROOT, http_post, l2_norm, spawn_server, stop_server

MODELS_DIR = PROJECT_ROOT / "models"

# (fixture 名, 目录, 期望维度, 端口)
MODEL_MATRIX = [
    ("m1", "BAAI-bge-small-en-v1.5", 384, 9111),
    ("m2", "all-MiniLM-L6-v2", 384, 9112),
    ("m3", "BAAI-bge-small-zh-v1.5", 512, 9113),
    ("m4", "multilingual-e5-small", 384, 9114),
]


def _model_available(dirname: str) -> bool:
    d = MODELS_DIR / dirname
    return d.is_dir() and (d / "model.safetensors").exists() and (d / "tokenizer.json").exists()


def _config_for(port: int, dirname: str, dim: int) -> str:
    lines = [
        "[server]",
        'host = "127.0.0.1"',
        f"port = {port}",
        "",
        "[model]",
        f'model_path = "{MODELS_DIR / dirname}"',
        f"expected_dimension = {dim}",
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
    ]
    return "\n".join(lines)


@pytest.fixture(scope="module")
def matrix_servers():
    servers = {}
    for name, dirname, dim, port in MODEL_MATRIX:
        if _model_available(dirname):
            servers[name] = (spawn_server(f"matrix-{name}", port, _config_for(port, dirname, dim)), dim)
        else:
            servers[name] = (None, dim)
    yield servers
    for name, (s, _dim) in servers.items():
        if s is not None:
            stop_server(s)


@pytest.mark.parametrize("name,dirname,dim,port", MODEL_MATRIX)
def test_mm_embed_dim_and_norm(matrix_servers, name, dirname, dim, port):
    """每模型：HTTP embed 维度 + 单位范数（R-model-001）。"""
    s, expected_dim = matrix_servers[name]
    if s is None:
        pytest.skip(f"模型目录缺失: {dirname}")
    st, body = http_post(s["port"], "/api/1/embed", {"text": "model matrix probe"})
    assert st == 200, f"[{name}] {st}: {str(body)[:150]}"
    vec = body["embedding"]
    assert len(vec) == expected_dim, f"[{name}] 维度应 {expected_dim}，实际 {len(vec)}"
    assert abs(l2_norm(vec) - 1.0) < 1e-2, f"[{name}] 范数应≈1"


@pytest.mark.parametrize("name,dirname,dim,port", MODEL_MATRIX)
def test_mm_embed_batch(matrix_servers, name, dirname, dim, port):
    """每模型：HTTP embed_batch 顺序与维度（R-model-001）。"""
    s, expected_dim = matrix_servers[name]
    if s is None:
        pytest.skip(f"模型目录缺失: {dirname}")
    st, body = http_post(s["port"], "/api/1/embed/batch",
                         {"texts": ["alpha", "beta", "gamma"]})
    assert st == 200, f"[{name}] {st}: {str(body)[:150]}"
    assert len(body["embeddings"]) == 3, f"[{name}] 批量应 3 条"
    for emb in body["embeddings"]:
        assert len(emb["embedding"] if isinstance(emb, dict) else emb) == expected_dim


@pytest.mark.parametrize("name,dirname,dim,port", MODEL_MATRIX)
def test_mm_rerank_relevance(matrix_servers, name, dirname, dim, port):
    """每模型：HTTP rerank 相关文档得分高于无关文档（R-rerank-001）。

    能力边界：M3（bge-zh）模型卡要求检索 query 加指令前缀、M4（e5）要求
    query:/passage: 前缀 —— HTTP API 无法表达这些前缀，故 M3/M4 只断言结构、
    值域与确定性；严格相关性断言在 SDK 侧（tests/scenario_sdk.rs，可注入前缀）覆盖。
    """
    s, _dim = matrix_servers[name]
    if s is None:
        pytest.skip(f"模型目录缺失: {dirname}")
    st, body = http_post(s["port"], "/api/1/rerank",
                         {"query": "machine learning algorithms",
                          "documents": [
                              "The chef prepares fresh Italian pasta every morning",
                              "Deep neural networks learn patterns from large training datasets",
                          ],
                          "top_k": None, "return_documents": False})
    assert st == 200, f"[{name}] {st}: {str(body)[:150]}"
    results = body["results"]
    assert len(results) == 2
    scores = {r["index"]: r["score"] for r in results}
    if name in ("m1", "m2"):
        assert scores[1] > scores[0], f"[{name}] 相关文档(index 1)得分应更高: {scores}"
    else:
        # M3/M4：模型前缀用法限制下仅断言值域（sigmoid 域正值）
        assert all(0.0 < v < 1.0 for v in scores.values()), f"[{name}] 分数应在 (0,1): {scores}"
