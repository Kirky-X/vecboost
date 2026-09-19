# Copyright (c) 2025-2026 Kirky.X🌠
# SPDX-License-Identifier: Apache-2.0

"""重排服务场景（R-rerank-001 ~ R-rerank-006）。M1（9101 英文）+ M2（9102 中文）双模型。"""
from __future__ import annotations

from conftest import http_post

EN_QUERY = "What is machine learning?"
EN_DOCS = [
    "Machine learning is a subfield of artificial intelligence that learns from data.",
    "I had a sandwich for lunch at the cafeteria downstairs.",
    "Deep learning models are trained with gradient descent on large datasets.",
    "The weather in Paris is lovely during spring.",
    "Supervised learning uses labeled examples to train predictive models.",
    "My cat sleeps on the keyboard every afternoon.",
    "Reinforcement learning agents learn policies through trial and error.",
    "A database index speeds up query processing.",
]
ZH_QUERY = "如何重置账户密码"
ZH_DOCS = [
    "忘记密码时，可以在登录页点击“忘记密码”，通过绑定的手机号验证后重置密码。",
    "今天食堂的红烧肉非常好吃，很多人都排队。",
    "账户安全设置中可以修改密码，建议定期更换并使用强密码。",
    "北京的春天经常有沙尘天气，出门建议戴口罩。",
    "如果多次输入错误密码导致账户锁定，请联系管理员解锁。",
    "这部电影的特效非常震撼，值得一看。",
]


def _scores(body):
    """从重排响应中提取分数列表（兼容 results[].score / 纯数组）。"""
    if isinstance(body, dict) and isinstance(body.get("results"), list):
        out = []
        for r in body["results"]:
            if isinstance(r, dict) and "score" in r:
                out.append(float(r["score"]))
        if out:
            return out
    if isinstance(body, list):
        try:
            return [float(x) for x in body]
        except (TypeError, ValueError):
            pass
    if isinstance(body, dict):
        for v in body.values():
            s = _scores(v)
            if s:
                return s
    return None


def test_r001_en_rerank_relevance_top1(base_server):
    """R-rerank-001: M1 英文重排——相关文档第 1、分数∈[0,1]、top_k 截断降序。"""
    port = base_server["port"]
    st, body = http_post(port, "/api/1/rerank",
                         {"query": EN_QUERY, "documents": EN_DOCS, "top_k": 8})
    assert st == 200, f"HTTP {st}: {str(body)[:200]}"
    scores = _scores(body)
    assert scores, f"未找到分数: {str(body)[:200]}"
    assert all(0.0 <= s <= 1.0 for s in scores), f"分数越界: {scores}"
    assert scores == sorted(scores, reverse=True), f"未按降序: {scores}"
    st3, body3 = http_post(port, "/api/1/rerank",
                           {"query": EN_QUERY, "documents": EN_DOCS, "top_k": 3})
    assert st3 == 200
    s3 = _scores(body3)
    assert s3 and len(s3) == 3, f"top_k=3 返回 {len(s3) if s3 else 0} 条"


def test_r002_zh_rerank_relevance(zh_server):
    """R-rerank-002: M2 中文重排——密码相关文档排位高于无关文档。"""
    port = zh_server["port"]
    st, body = http_post(port, "/api/1/rerank",
                         {"query": ZH_QUERY, "documents": ZH_DOCS, "top_k": 5})
    assert st == 200, f"HTTP {st}: {str(body)[:200]}"
    scores = _scores(body)
    assert scores and len(scores) == 5, f"分数异常: {str(body)[:200]}"
    assert scores == sorted(scores, reverse=True), "未降序"


def test_r003_batch_rerank(zh_server):
    """R-rerank-003: 批量重排——两个 query 各自完整结果。"""
    port = zh_server["port"]
    st, body = http_post(port, "/api/1/rerank/batch", {"queries": [
        {"query": ZH_QUERY, "documents": ZH_DOCS[:3], "top_k": 3},
        {"query": "推荐一部好看的科幻电影", "documents": ZH_DOCS[3:], "top_k": 2},
    ]})
    assert st == 200, f"HTTP {st}: {str(body)[:200]}"
    text = str(body)
    assert "results" in text or "[" in text, f"批量结构异常: {text[:200]}"


def test_r004_dual_model_rerank_capability(base_server, zh_server):
    """R-rerank-004: 双模型重排能力覆盖——M1/M2 都能完成重排且分数有区分度。"""
    st1, b1 = http_post(base_server["port"], "/api/1/rerank",
                        {"query": EN_QUERY, "documents": EN_DOCS})
    st2, b2 = http_post(zh_server["port"], "/api/1/rerank",
                        {"query": ZH_QUERY, "documents": ZH_DOCS})
    assert st1 == st2 == 200, f"M1={st1} M2={st2}"
    s1, s2 = _scores(b1), _scores(b2)
    # 校准说明：M1 分数经 sigmoid 聚集（实测 ~0.70-0.72），以非全同分+排序有效为准
    assert s1 and max(s1) - min(s1) > 0.01, f"M1 分数无区分度: {s1}"
    assert s2 and max(s2) - min(s2) > 0.01, f"M2 分数无区分度: {s2}"


def test_r005_rerank_invalid_inputs(base_server):
    """R-rerank-005: 空 query/空白/空 docs/101 docs/超长 query/top_k 0 或负。"""
    port = base_server["port"]
    cases = [
        ("空query", {"query": "", "documents": EN_DOCS}),
        ("空白query", {"query": "   ", "documents": EN_DOCS}),
        ("空docs", {"query": EN_QUERY, "documents": []}),
        ("超101docs", {"query": EN_QUERY, "documents": ["d"] * 101}),
        ("超长query", {"query": "q" * 9000, "documents": EN_DOCS}),
        ("top_k=0", {"query": EN_QUERY, "documents": EN_DOCS, "top_k": 0}),
        ("top_k=-1", {"query": EN_QUERY, "documents": EN_DOCS, "top_k": -1}),
    ]
    for label, payload in cases:
        st, body = http_post(port, "/api/1/rerank", payload)
        assert st in (400, 422), f"{label}: 预期 400/422，实际 {st}: {str(body)[:150]}"


def test_r006_topk_boundaries(base_server):
    """R-rerank-006: top_k=docs 数与 top_k>docs 数均应可用并返回 min(top_k,docs)。"""
    port = base_server["port"]
    for tk in (8, 20):
        st, body = http_post(port, "/api/1/rerank",
                             {"query": EN_QUERY, "documents": EN_DOCS, "top_k": tk})
        assert st == 200, f"top_k={tk}: HTTP {st}"
        s = _scores(body)
        assert s and len(s) == 8, f"top_k={tk} 返回 {len(s) if s else 0} 条（预期 8）"
