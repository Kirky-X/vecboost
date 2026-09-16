"""容错与 API 逻辑契约套件（fault-tolerance-contract，FT-*）。

容错语境：pipeline（队列/拼批/OOM 回退）默认开启；本套件在"容错开启"的默认形态下
验证响应内容与请求参数的一致性——维度、排序、数量、错误码族与限流响应头。

- FT-01/02: embed/batch 上界 64（恰好在界）与批内去重 scatter 的顺序/一致性
- FT-03: 并发同文请求经流水线拼批后逐位一致
- FT-04: rerank 严格契约——降序、index 指回原文档、top_k 截断保持最优前缀、
  return_documents 回显正确
- FT-05/06: rerank/batch 容错语义——单 query 失败静默跳过（响应数≤请求数），全失败
  不产生 5xx
- FT-07: search 契约——top_k>n 收敛到 n、降序、index 越界防护
- FT-08: /metrics Prometheus 文本格式（# HELP/# TYPE/vecboost_ 前缀样本）
- FT-09: embed 单条与批量响应的 dimension 字段与实际向量长度自洽
- FT-10: OpenAI dimensions 参数与响应维度一致 + Matryoshka 前缀一致性
- FT-11/12: [rate_limit] headers_enabled=true 时 200 携带 IETF RateLimit-* 头、
  超限 429 携带 Retry-After 且响应体为合法 JSON 错误
"""
from __future__ import annotations

import concurrent.futures
import http.client
import json
import pathlib

import pytest

from conftest import (
    PROJECT_ROOT, RUN_DIR, M1_REPO, spawn_server, stop_server,
    http_get, http_post, find_vector, l2_norm,
)

DIM = 384
M1_PATH = str(PROJECT_ROOT / "models" / "BAAI-bge-small-en-v1.5")
RL_HEADERS_PORT = 9165

RL_HEADERS_CONFIG = """[server]
host = "127.0.0.1"
port = 9165

[model]
model_path = "{m1}"
expected_dimension = 384

[rate_limit]
enabled = true
global_requests_per_minute = 6
ip_requests_per_minute = 6
window_secs = 60
headers_enabled = true
ip_whitelist = []

[auth]
enabled = false

[database]
url = "sqlite::memory:"
""".format(m1=M1_PATH)


@pytest.fixture(scope="module")
def rl_headers_server():
    """限流响应头探针：6/min、白名单为空、headers_enabled=true（端口 9165）。"""
    s = spawn_server("ft_rl_headers", RL_HEADERS_PORT, RL_HEADERS_CONFIG)
    yield s
    stop_server(s)


def _post_raw(port: int, path: str, js=None):
    """POST 并保留原始响应头，供限流头断言使用。"""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    body = json.dumps(js).encode() if js is not None else None
    conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    raw = resp.read().decode("utf-8", errors="replace")
    headers = {k.lower(): v for k, v in resp.getheaders()}
    conn.close()
    try:
        return resp.status, headers, json.loads(raw)
    except json.JSONDecodeError:
        return resp.status, headers, raw


# ------------------------------------------------------------------ embed/batch

def test_ft01_batch_boundary_64_exact(base_server):
    """FT-01: 批量恰好 64 条（=max_batch_size 上界）应成功，且每条维度与
    响应 dimension 字段一致。"""
    port = base_server["port"]
    batch = [f"边界测试 {i}: fault tolerance batch boundary probe" for i in range(64)]
    st, body = http_post(port, "/api/1/embed/batch", {"texts": batch})
    assert st == 200, f"恰在界的 64 条批量应 200: HTTP {st}: {str(body)[:200]}"
    items = body["embeddings"]
    assert len(items) == 64, f"应返回 64 条，实际 {len(items)}"
    assert body["dimension"] == DIM, f"dimension 字段 {body['dimension']} != {DIM}"
    for i, it in enumerate(items):
        assert len(it["embedding"]) == DIM, f"第 {i} 条维度 {len(it['embedding'])} != {DIM}"


def test_ft02_batch_duplicates_order_and_scatter(base_server):
    """FT-02: 批内重复文本经去重 scatter 后——顺序保持、重复项逐位一致、
    且与单条推理语义一致（cosine≈1）。"""
    port = base_server["port"]
    a = "duplicate scatter probe alpha sentence for vecboost"
    b = "duplicate scatter probe beta sentence with different words"
    batch = [a, b, a, a, b]
    st, body = http_post(port, "/api/1/embed/batch", {"texts": batch})
    assert st == 200, f"HTTP {st}: {str(body)[:200]}"
    embs = [it["embedding"] for it in body["embeddings"]]
    assert len(embs) == 5, "去重 scatter 不得丢失条目"
    previews = [it.get("text_preview", "") for it in body["embeddings"]]
    assert previews[0] == previews[2] == previews[3], "重复项顺序应与请求一致"
    assert embs[0] == embs[2] == embs[3], "同文本批内向量应逐位一致（去重回填）"
    assert embs[1] == embs[4], "同文本批内向量应逐位一致（去重回填）"
    assert embs[0] != embs[1], "不同文本向量不应相同"
    st2, single = http_post(port, "/api/1/embed", {"text": a})
    assert st2 == 200
    v_single = find_vector(single)
    dot = sum(x * y for x, y in zip(v_single, embs[0]))
    cos = dot / (l2_norm(v_single) * l2_norm(embs[0]))
    assert cos > 0.999999, f"批内向量与单条语义不一致: cos={cos}"


def test_ft03_concurrent_identical_embeds_consistent(base_server):
    """FT-03: 8 并发同文请求经流水线（拼批/去重/缓存）后全部 200 且逐位一致。"""
    port = base_server["port"]
    text = "concurrent consistency probe: pipeline drain window and dedup scatter"

    def hit(_):
        return http_post(port, "/api/1/embed", {"text": text})

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(hit, range(8)))
    vectors = []
    for i, (st, body) in enumerate(results):
        assert st == 200, f"并发请求 {i}: HTTP {st}: {str(body)[:150]}"
        vectors.append(find_vector(body))
    assert all(v == vectors[0] for v in vectors), "并发同文向量逐位不一致"


# ------------------------------------------------------------------ rerank

EN_QUERY = "What is machine learning?"
EN_DOCS = [
    "Machine learning is a subfield of artificial intelligence that learns from data.",
    "I had a sandwich for lunch at the cafeteria downstairs.",
    "Deep learning models are trained with gradient descent on large datasets.",
    "The weather in Paris is lovely during spring.",
    "Supervised learning uses labeled examples to train predictive models.",
    "My cat sleeps on the keyboard every afternoon.",
]


def test_ft04_rerank_strict_contract(base_server):
    """FT-04: rerank 严格契约——降序、index 指回原文档位置、return_documents
    回显正确、top_k 截断等于全量结果最优前缀。"""
    port = base_server["port"]
    st, full = http_post(port, "/api/1/rerank",
                         {"query": EN_QUERY, "documents": EN_DOCS, "return_documents": True})
    assert st == 200, f"HTTP {st}: {str(full)[:200]}"
    results = full["results"]
    assert len(results) == len(EN_DOCS), "无 top_k 时应返回全部文档"
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True), f"未降序: {scores}"
    for r in results:
        assert 0 <= r["index"] < len(EN_DOCS), f"index 越界: {r['index']}"
        assert r["document"] == EN_DOCS[r["index"]], "return_documents 未回显原文档"
    st2, top3 = http_post(port, "/api/1/rerank",
                          {"query": EN_QUERY, "documents": EN_DOCS, "top_k": 3})
    assert st2 == 200
    r3 = top3["results"]
    assert len(r3) == 3, "top_k=3 应截断为 3 条"
    assert [(r["index"], r["score"]) for r in r3] == \
           [(r["index"], r["score"]) for r in results[:3]], "top_k 截断应保持全量最优前缀"


def test_ft05_rerank_batch_partial_failure_tolerance(base_server):
    """FT-05: rerank/batch 容错语义——单个 query 失败（空文本）不产生响应，
    但在 statuses 状态位中可见（index 对位 + ok=false + error 原因）。"""
    port = base_server["port"]
    st, body = http_post(port, "/api/1/rerank/batch", {"queries": [
        {"query": EN_QUERY, "documents": EN_DOCS[:4], "top_k": 4},
        {"query": "", "documents": EN_DOCS[:2]},
        {"query": "database systems and query processing",
         "documents": EN_DOCS[2:], "top_k": 2},
    ]})
    assert st == 200, f"部分失败应整体 200: HTTP {st}: {str(body)[:200]}"
    responses = body["responses"]
    assert len(responses) == 2, f"无效 query 不产生响应: {len(responses)} 条"
    statuses = body.get("statuses")
    assert isinstance(statuses, list) and len(statuses) == 3, \
        f"statuses 应与请求一一对位: {str(statuses)[:200]}"
    assert statuses[0]["ok"] and statuses[2]["ok"], f"有效 query 应 ok: {statuses}"
    assert not statuses[1]["ok"] and statuses[1]["index"] == 1 \
        and statuses[1].get("error"), f"失败 query 状态位应可见: {statuses[1]}"
    for resp in responses:
        scores = [r["score"] for r in resp["results"]]
        assert scores == sorted(scores, reverse=True), f"批内结果未降序: {scores}"


def test_ft06_rerank_batch_all_invalid_no_5xx(base_server):
    """FT-06: rerank/batch 全部 query 无效——仍 200 + 空响应 + 全失败状态位
    （不得 5xx/崩溃）。"""
    port = base_server["port"]
    st, body = http_post(port, "/api/1/rerank/batch", {"queries": [
        {"query": "", "documents": EN_DOCS[:2]},
        {"query": "   ", "documents": EN_DOCS[:2]},
    ]})
    assert st == 200, f"全失败不应 5xx: HTTP {st}: {str(body)[:200]}"
    assert body["responses"] == [], "全失败应返回空响应列表"
    statuses = body.get("statuses")
    assert isinstance(statuses, list) and len(statuses) == 2 \
        and all(not s["ok"] for s in statuses), f"全失败状态位应可见: {str(statuses)[:200]}"


# ------------------------------------------------------------------ search

def test_ft07_search_contract_topk_and_order(base_server):
    """FT-07: search 契约——top_k>n 收敛到 n、按分数降序、index 在候选范围内、
    分数在余弦值域内。"""
    port = base_server["port"]
    st, body = http_post(port, "/api/1/search", {
        "query": "machine learning",
        "texts": [
            "machine learning models learn from data",
            "the cat sat on the mat",
            "deep learning uses neural networks",
            "grocery shopping list for the weekend",
        ],
        "top_k": 10,
    })
    assert st == 200, f"HTTP {st}: {str(body)[:200]}"
    results = body["results"]
    assert len(results) == 4, f"top_k>n 应收敛到候选数 4: {len(results)}"
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True), f"未降序: {scores}"
    for r in results:
        assert 0 <= r["index"] < 4, f"index 越界: {r['index']}"
        assert -1.001 <= r["score"] <= 1.001, f"分数越界: {r['score']}"
    assert results[0]["index"] in (0, 2), "相关候选应排前"


# ------------------------------------------------------------------ metrics / 自洽

def test_ft08_metrics_prometheus_format(base_server):
    """FT-08: /metrics 是合法 Prometheus 文本格式——# HELP/# TYPE 注释、
    vecboost_ 前缀样本行。"""
    port = base_server["port"]
    st, metrics = http_get(port, "/metrics")
    assert st == 200, f"/metrics HTTP {st}"
    text = metrics if isinstance(metrics, str) else str(metrics)
    assert "# HELP" in text and "# TYPE" in text, "缺 Prometheus HELP/TYPE 注释"
    assert "vecboost_" in text, "缺 vecboost_ 前缀指标"
    sample_lines = [ln for ln in text.splitlines()
                    if ln and not ln.startswith("#")]
    assert sample_lines, "无指标样本行"
    assert all(len(ln.split()) >= 2 for ln in sample_lines), "样本行应为 name value 形式"


def test_ft09_embed_response_dimension_self_consistent(base_server):
    """FT-09: 响应自洽——单条与批量的 dimension 字段必须等于实际向量长度。"""
    port = base_server["port"]
    st, single = http_post(port, "/api/1/embed", {"text": "dimension self consistency"})
    assert st == 200
    assert single["dimension"] == len(single["embedding"]) == DIM, \
        f"单条 dimension 字段与向量长度不自洽: {single['dimension']}"
    st2, batch = http_post(port, "/api/1/embed/batch",
                           {"texts": ["alpha consistency", "beta consistency"]})
    assert st2 == 200
    assert batch["dimension"] == DIM
    for it in batch["embeddings"]:
        assert len(it["embedding"]) == batch["dimension"], "批量维度与字段不自洽"


def test_ft10_openai_dimensions_consistency_and_prefix(base_server):
    """FT-10: OpenAI dimensions 参数一致性——响应维度=请求维度、重归一化、
    截断向量与全维向量前缀 cos≈1、usage 为正整数、model 字段回显。"""
    port = base_server["port"]
    model = M1_REPO
    text = "matryoshka prefix consistency probe for dimension contract"
    st, full = http_post(port, "/v1/embeddings",
                         {"input": text, "model": model, "encoding_format": "float"})
    assert st == 200
    v_full = full["data"][0]["embedding"]
    assert full["model"] == model, f"model 字段未回显: {full.get('model')}"
    assert isinstance(full["usage"]["prompt_tokens"], int) and full["usage"]["prompt_tokens"] > 0
    st2, cut = http_post(port, "/v1/embeddings",
                         {"input": text, "model": model, "dimensions": 128,
                          "encoding_format": "float"})
    assert st2 == 200
    v_cut = cut["data"][0]["embedding"]
    assert len(v_cut) == 128, f"dimensions=128 返回 {len(v_cut)} 维"
    assert abs(l2_norm(v_cut) - 1.0) < 0.05, "截断后未重归一化"
    prefix = v_full[:128]
    dot = sum(x * y for x, y in zip(prefix, v_cut))
    cos = dot / (l2_norm(prefix) * l2_norm(v_cut))
    assert cos > 0.999999, f"截断向量与全维前缀不一致: cos={cos}"


# ------------------------------------------------------------------ rate limit headers

def test_ft11_rate_limit_headers_on_success_and_429(rl_headers_server):
    """FT-11: headers_enabled=true——200 携带 ratelimit-limit/remaining/reset/policy
    且 remaining 单调递减；超限后 429 携带 retry-after。"""
    port = rl_headers_server["port"]
    statuses = []
    remainings = []
    retry_after = None
    for _ in range(10):
        st, headers, _body = _post_raw(port, "/api/1/embed",
                                       {"text": "rate limit header contract probe"})
        statuses.append(st)
        if st == 200:
            assert "ratelimit-limit" in headers, f"200 缺 ratelimit-limit: {headers}"
            assert "ratelimit-remaining" in headers, f"200 缺 ratelimit-remaining: {headers}"
            assert "ratelimit-reset" in headers, f"200 缺 ratelimit-reset: {headers}"
            assert "ratelimit-policy" in headers, f"200 缺 ratelimit-policy: {headers}"
            remainings.append(int(headers["ratelimit-remaining"]))
        elif st == 429:
            retry_after = headers.get("retry-after")
    assert 429 in statuses, f"6/min 限制下 10 连发应触发 429: {statuses}"
    assert remainings == sorted(remainings, reverse=True), \
        f"remaining 应单调递减: {remainings}"
    assert retry_after is not None, "429 响应缺 retry-after 头"
    assert int(retry_after) > 0, f"retry-after 应为正: {retry_after}"


def test_ft12_rate_limit_429_body_json_and_server_alive(rl_headers_server):
    """FT-12: 429 响应体为合法 JSON 错误结构（非空信息）；服务仍然存活。"""
    port = rl_headers_server["port"]
    st, _headers, body = _post_raw(port, "/api/1/embed",
                                   {"text": "429 body shape probe"})
    if st != 429:
        pytest.skip("窗口已滚动未触发 429（前一用例已覆盖触发路径）")
    text = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)
    assert len(text) > 0, "429 响应体不应为空"
    if not isinstance(body, str):
        joined = json.dumps(body).lower()
        assert any(k in joined for k in ("error", "message", "rate")), \
            f"429 响应体应含错误信息: {text[:200]}"
    hst, _hbody = http_get(port, "/health")
    assert hst == 200, f"限流后 /health 应存活: HTTP {hst}"
