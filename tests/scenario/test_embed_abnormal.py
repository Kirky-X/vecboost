# Copyright (c) 2025-2026 Kirky.X🌠
# SPDX-License-Identifier: Apache-2.0

"""嵌入服务异常场景（R-embed-009 ~ R-embed-010）。"""
from __future__ import annotations

from conftest import http_get, http_post

LONG_TEXT = "超" * 9000  # > max_text_length 8192


def test_r009_empty_and_blank_text(base_server):
    """R-embed-009a: 空文本/纯空白 → 400 族，不得 5xx。"""
    port = base_server["port"]
    for label, payload in [("empty", {"text": ""}), ("blank", {"text": "   \n\t "})]:
        st, body = http_post(port, "/api/1/embed", payload)
        assert st in (400, 422), f"{label}: 预期 400/422，实际 {st}: {str(body)[:150]}"
        assert st < 500


def test_r009_overlong_text(base_server):
    """R-embed-009b: 超长文本（9000 > 8192）→ 400。"""
    st, body = http_post(base_server["port"], "/api/1/embed", {"text": LONG_TEXT})
    assert st in (400, 422), f"超长文本返回 {st}（预期 400/422）: {str(body)[:150]}"


def test_r009_batch_oversize_and_empty_item(base_server):
    """R-embed-009c: 批量 65 条（>64）与含空项批量 → 400。"""
    port = base_server["port"]
    st, _ = http_post(port, "/api/1/embed/batch", {"texts": [f"t{i}" for i in range(65)]})
    assert st in (400, 422), f"批量 65 条返回 {st}"
    st2, _ = http_post(port, "/api/1/embed/batch", {"texts": ["正常", ""]})
    assert st2 in (400, 422), f"含空项批量返回 {st2}"


def test_r009_missing_field_and_bad_types(base_server):
    """R-embed-009d: 缺字段/类型错误/非法 JSON → 400 族。"""
    port = base_server["port"]
    st, _ = http_post(port, "/api/1/embed", {})
    assert st in (400, 422), f"缺 text 字段返回 {st}"
    st2, _ = http_post(port, "/api/1/embed", {"text": 12345})
    assert st2 in (400, 422), f"text 数字类型返回 {st2}"
    st3, _ = http_post(port, "/api/1/embed", raw_body="{not json")
    assert st3 in (400, 422), f"非法 JSON 返回 {st3}"


def test_r010_unknown_route_and_wrong_method(base_server):
    """R-embed-010: 未知路由 404；方法错误 405。"""
    port = base_server["port"]
    st, _ = http_get(port, "/definitely-not-a-route-xyz")
    assert st == 404, f"未知路由返回 {st}"
    st2, _ = http_get(port, "/api/1/embed")
    assert st2 == 405, f"GET /api/1/embed 返回 {st2}（预期 405）"
    st3, _ = http_post(port, "/health", {"x": 1})
    assert st3 == 405, f"POST /health 返回 {st3}（预期 405）"
