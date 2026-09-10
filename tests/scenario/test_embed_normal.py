"""嵌入服务正常场景（R-embed-001 ~ R-embed-008）。M1 = BAAI/bge-small-en-v1.5，384 维。"""
from __future__ import annotations

import concurrent.futures

import pytest

from conftest import (
    http_get, http_post, find_list_of_vectors, find_vector, find_scalar_score, l2_norm,
)

DIM = 384
TEXTS = {
    "en": "The quick brown fox jumps over the lazy dog.",
    "zh": "今天天气非常好，适合出去爬山。",
    "mixed": "使用 Rust 编写的高性能 embedding 服务 vecboost",
    "emoji": "这个产品真不错 👍🔥 下次还来！",
}


def _embed(port, text):
    return http_post(port, "/api/1/embed", {"text": text})


def test_r001_single_embed_en_zh_mixed_emoji(base_server):
    """R-embed-001: 中/英/混合/emoji 单句嵌入，维度 384、非全零、确定性、L2≈1。"""
    port = base_server["port"]
    vecs = {}
    for name, text in TEXTS.items():
        st, body = _embed(port, text)
        assert st == 200, f"{name}: HTTP {st}: {str(body)[:200]}"
        vec = find_vector(body)
        assert vec, f"{name}: 响应中未找到向量: {str(body)[:200]}"
        assert len(vec) == DIM, f"{name}: 维度 {len(vec)} != {DIM}"
        assert any(abs(x) > 1e-6 for x in vec), f"{name}: 向量全零"
        vecs[name] = vec
    st, body = _embed(port, TEXTS["en"])
    assert find_vector(body) == vecs["en"], "相同文本两次嵌入结果不一致（确定性失败）"
    n = l2_norm(vecs["zh"])
    assert abs(n - 1.0) < 0.05, f"L2 范数 {n} 偏离 1（归一化预期）"


def test_r002_batch_32_order_and_consistency(base_server):
    """R-embed-002: 批量 32 条——数量、维度、顺序保持、与逐条一致。"""
    port = base_server["port"]
    batch = [f"批量测试文本 {i}：vecboost scenario {i}" for i in range(32)]
    st, body = http_post(port, "/api/1/embed/batch", {"texts": batch})
    assert st == 200, f"HTTP {st}: {str(body)[:200]}"
    # 实际结构：{"embeddings": [{"text_preview":..., "embedding":[...]}, ...]}
    items = body.get("embeddings") if isinstance(body, dict) else None
    assert isinstance(items, list) and len(items) == 32, f"数量异常: {len(items) if items else 0}"
    embs = [it["embedding"] for it in items]
    assert all(len(v) == DIM for v in embs), "批量中存在维度不一致"
    st1, b1 = _embed(port, batch[0])
    assert find_vector(b1) == embs[0], "批量首条与单句结果不一致"


def test_r003_similarity_same_pair_high_score(base_server):
    """R-embed-003: 相似度端点——相同文本≈1.0、不同文本有区分。

    能力边界（发现）：HTTP 未暴露 metric 选择字段（SimilarityRequest 仅 source/target），
    服务层 4 种度量无法经 API 触达，记入报告。
    注：M1 为英文模型，对中文文本区分能力有限（CJK token 在嵌入层几乎无差异），
    故无关文本对使用英文验证。
    """
    port = base_server["port"]
    st, body = http_post(port, "/api/1/similarity",
                         {"source": "I love Beijing", "target": "I love Beijing"})
    assert st == 200, f"HTTP {st}: {str(body)[:200]}"
    same = find_scalar_score(body)
    assert same is not None, f"未找到分数: {str(body)[:200]}"
    assert same > 0.99, f"相同文本相似度 {same} 应≈1.0"
    st2, body2 = http_post(port, "/api/1/similarity",
                           {"source": "machine learning model training", "target": "today lunch is noodles"})
    assert st2 == 200, f"HTTP {st2}"
    diff = find_scalar_score(body2)
    assert diff is not None and diff < same, f"无关文本相似度 {diff} 不应高于相同文本 {same}"


def test_r004_search_not_exposed(base_server):
    """R-embed-004: 搜索能力 HTTP 暴露性验证。

    发现：服务层有 process_search 能力，但未注册 HTTP 路由（仅 /embed /embed/batch
    /similarity /embed/file /v1/embeddings）——POST /embed/search 返回 404。
    """
    port = base_server["port"]
    st, _ = http_post(port, "/api/1/embed/search",
                      {"query": "可爱的猫咪", "texts": [f"文档{i}" for i in range(10)], "top_k": 3})
    if st == 404:
        pytest.skip("能力记录：embed/search 未暴露 HTTP 路由（服务层能力存在，记入报告）")
    assert st == 200, f"HTTP {st}"


def test_r005_file_embed_txt(base_server):
    """R-embed-005: 文件嵌入——合法 txt 分块向量化。"""
    port = base_server["port"]
    from conftest import write_file
    content = "\n\n".join(
        f"第 {i} 段：vecboost 支持文件级嵌入，自动分块并保持重叠语义。section {i}。" for i in range(6)
    )
    path = write_file("base", "files/sample.txt", content)
    st, body = http_post(port, "/api/1/embed/file", {"path": path})
    assert st == 200, f"HTTP {st}: {str(body)[:300]}"
    # 实际结构：{"mode":"document","stats":{"total_chunks":N,...},"embedding":[...]}（聚合向量）
    stats = body.get("stats", {}) if isinstance(body, dict) else {}
    chunks = stats.get("total_chunks") or 0
    vec = find_vector(body)
    assert chunks >= 2, f"分块数异常: {str(body)[:200]}"
    assert vec and len(vec) == DIM, f"聚合向量维度异常: {len(vec) if vec else None}"


def test_r006_matryoshka_via_openai_dimensions(base_server):
    """R-embed-006: Matryoshka 降维（OpenAI 端点 dimensions=128/256）+ 非法 dimensions 拒绝。"""
    port = base_server["port"]
    for d in (128, 256):
        st, body = http_post(port, "/v1/embeddings",
                             {"input": "matryoshka 降维测试", "model": "bge-small-en-v1.5",
                              "dimensions": d, "encoding_format": "float"})
        assert st == 200, f"dimensions={d}: HTTP {st}: {str(body)[:200]}"
        vec = find_vector(body)
        assert vec and len(vec) == d, f"dimensions={d} 返回 {len(vec) if vec else None} 维"
        assert abs(l2_norm(vec) - 1.0) < 0.05, f"dimensions={d} 未重新归一化: L2={l2_norm(vec)}"
    for bad in (0, 4096):
        st, body = http_post(port, "/v1/embeddings",
                             {"input": "x", "model": "bge-small-en-v1.5", "dimensions": bad})
        assert st in (400, 422), f"非法 dimensions={bad} 返回 {st}（预期 400/422）"


def test_r007_openai_compat_endpoint(base_server):
    """R-embed-007: OpenAI 兼容 /v1/embeddings——结构与 /embed 一致性。"""
    port = base_server["port"]
    st, body = http_post(port, "/v1/embeddings",
                         {"input": ["hello world", "你好世界"], "model": "bge-small-en-v1.5",
                          "encoding_format": "float"})
    assert st == 200, f"HTTP {st}: {str(body)[:200]}"
    data = body.get("data") if isinstance(body, dict) else None
    assert isinstance(data, list) and len(data) == 2, f"OpenAI 批量结构异常: {str(body)[:200]}"
    embs = [item.get("embedding") for item in data]
    assert all(isinstance(e, list) and len(e) == DIM for e in embs), "embedding 结构异常"
    st1, b1 = _embed(port, "hello world")
    # 批量与单条推理存在浮点 ULP 级差异（batched matmul 求和顺序不同），
    # 语义契约用余弦相似度≈1 断言，不做逐位相等比较
    v_native = find_vector(b1)
    dot = sum(a * b for a, b in zip(v_native, embs[0]))
    na = sum(a * a for a in v_native) ** 0.5
    nb = sum(b * b for b in embs[0]) ** 0.5
    cos = dot / (na * nb)
    assert cos > 0.999999, f"OpenAI 端点与原生端点向量语义不一致: cos={cos}"


def test_r008_repeat_request_cache_effect(base_server):
    """R-embed-008: 重复请求——向量一致；/metrics 可用性检查（缓存指标存在性记录）。"""
    port = base_server["port"]
    st1, b1 = _embed(port, "缓存命中测试文本 cache probe")
    st2, b2 = _embed(port, "缓存命中测试文本 cache probe")
    assert st1 == st2 == 200
    assert find_vector(b1) == find_vector(b2), "缓存路径下向量漂移"
    mst, metrics = http_get(port, "/metrics")
    assert mst == 200, f"/metrics HTTP {mst}"
    has_cache = "cache" in str(metrics).lower()
    if not has_cache:
        pytest.skip("能力记录：/metrics 中未发现 cache 相关指标（记入报告）")
