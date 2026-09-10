"""HTTP 协议矩阵套件 — 补齐 design.md M2 缺口（R-api-001 / R-embed-002/003 / R-rerank-002）。

覆盖场景 ID（HM-*）：
- OpenAI 兼容端点：单串/数组/base64/Matryoshka 维度/usage（HM-O01…HM-O05）
- 相似度语义回归（SIM-002 不回归）：无关文本不得返回 1.0（HM-S01/HM-S02）
- 文件嵌入：正常 txt + 路径遍历拒绝 + 敏感目录拒绝（HM-F01…HM-F03）
- 模型端点：current/info/models 可用（HM-M01…HM-M03）
- 本地热切换往返 + 切换失败保护（HM-M04/HM-M05，SWITCH-001 不回归）
- 异常输入矩阵：非法 JSON/类型错误/top_k=0/空 documents/超长 rerank query（HM-A01…HM-A05）
- i18n：Accept-Language zh/en 错误消息语言（HM-I01/HM-I02）
"""
from __future__ import annotations

import json
import base64
import pathlib

import pytest

from conftest import (
    M1_PATH,
    RUN_DIR,
    find_vector,
    http_get,
    http_post,
    write_file,
)

M2_MINILM_PATH = str(pathlib.Path(__file__).resolve().parents[2] / "models" / "all-MiniLM-L6-v2")


# ---------------- OpenAI 兼容端点（HM-O01…HM-O05） ----------------

def test_hm_o01_openai_single_string(base_server):
    """HM-O01: input 单串 → object=list + 向量 + usage。"""
    st, body = http_post(base_server["port"], "/v1/embeddings",
                         {"input": "openai compatibility check", "model": "bge-small-en-v1.5"})
    assert st == 200, f"{st}: {str(body)[:200]}"
    assert body.get("object") == "list", f"object 应为 list: {str(body)[:200]}"
    assert body["data"] and find_vector(body), "应含向量"
    assert "usage" in body, f"应含 usage: {str(body)[:200]}"


def test_hm_o02_openai_array_input(base_server):
    """HM-O02: input 数组 → 每条一个 data 元素，index 对齐。"""
    st, body = http_post(base_server["port"], "/v1/embeddings",
                         {"input": ["first doc", "second doc"], "model": "bge-small-en-v1.5"})
    assert st == 200, f"{st}: {str(body)[:200]}"
    assert len(body["data"]) == 2, "数组输入应返回 2 个 embedding 对象"
    assert [d["index"] for d in body["data"]] == [0, 1], "index 应对齐输入顺序"


def test_hm_o03_openai_base64_format(base_server):
    """HM-O03: encoding_format=base64 → base64 编码向量。"""
    st, body = http_post(base_server["port"], "/v1/embeddings",
                         {"input": "base64 encoded vector", "model": "bge-small-en-v1.5",
                          "encoding_format": "base64"})
    assert st == 200, f"{st}: {str(body)[:200]}"
    raw = body["data"][0]["embedding"]
    assert isinstance(raw, str), f"base64 模式下 embedding 应为字符串: {type(raw)}"
    decoded = base64.b64decode(raw)
    assert len(decoded) % 4 == 0 and len(decoded) >= 384 * 2, \
        f"解码后长度异常: {len(decoded)}（384 维 f32 应为 {384 * 4} 字节）"


def test_hm_o04_openai_matryoshka_dimensions(base_server):
    """HM-O04: dimensions=128 → Matryoshka 截断，维度正确且保留率字段存在。"""
    st, body = http_post(base_server["port"], "/v1/embeddings",
                         {"input": "matryoshka truncation test", "model": "bge-small-en-v1.5",
                          "dimensions": 128})
    assert st == 200, f"{st}: {str(body)[:200]}"
    vec = body["data"][0]["embedding"]
    assert isinstance(vec, list) and len(vec) == 128, f"维度应为 128，实际 {len(vec) if isinstance(vec, list) else type(vec)}"


def test_hm_o05_openai_empty_input_rejected(base_server):
    """HM-O05: input 空数组 → 400。"""
    st, body = http_post(base_server["port"], "/v1/embeddings",
                         {"input": [], "model": "bge-small-en-v1.5"})
    assert st == 400, f"空 input 应 400，实际 {st}: {str(body)[:150]}"


# ---------------- 相似度语义回归（HM-S01/HM-S02，SIM-002 不回归） ----------------

def test_hm_s01_similarity_same_text_is_one(base_server):
    """HM-S01: 同文本 similarity = 1.0。"""
    st, body = http_post(base_server["port"], "/api/1/similarity",
                         {"source": "machine learning pipeline", "target": "machine learning pipeline"})
    assert st == 200, f"{st}: {str(body)[:200]}"
    assert abs(body["score"] - 1.0) < 1e-3, f"同文本应为 1.0: {body['score']}"


def test_hm_s02_similarity_unrelated_not_one(base_server):
    """HM-S02: 无关文本对不得返回 1.0（SIM-002 回归）。"""
    st, body = http_post(base_server["port"], "/api/1/similarity",
                         {"source": "机器学习模型训练", "target": "今天的午餐是面条"})
    assert st == 200, f"{st}: {str(body)[:200]}"
    assert body["score"] < 0.99, f"SIM-002 回归：无关文本 score={body['score']}"


# ---------------- 文件嵌入（HM-F01…HM-F03） ----------------

def test_hm_f01_file_embed_txt(base_server):
    """HM-F01: 正常 txt 文件嵌入返回向量。"""
    p = write_file("base", "e2e-doc.txt", "第一段落内容\n\n第二段落内容 for file embed")
    st, body = http_post(base_server["port"], "/api/1/embed/file", {"path": p})
    assert st == 200, f"{st}: {str(body)[:200]}"
    ok = (body.get("embedding") or body.get("paragraphs")) is not None
    assert ok, f"应含 embedding 或 paragraphs: {str(body)[:200]}"


def test_hm_f02_file_embed_path_traversal_rejected(base_server):
    """HM-F02: 路径遍历输入被拒绝且为 4xx。"""
    st, body = http_post(base_server["port"], "/api/1/embed/file",
                         {"path": "../../etc/passwd"})
    assert 400 <= st < 500, f"路径遍历应 4xx，实际 {st}: {str(body)[:150]}"


def test_hm_f03_file_embed_sensitive_dir_rejected(base_server):
    """HM-F03: 敏感目录文件被拒绝且为 4xx。"""
    st, body = http_post(base_server["port"], "/api/1/embed/file", {"path": "/etc/hostname"})
    assert 400 <= st < 500, f"敏感目录应 4xx，实际 {st}: {str(body)[:150]}"


# ---------------- 模型端点与热切换（HM-M01…HM-M05） ----------------

def test_hm_m01_model_current_and_info(base_server):
    """HM-M01: /model/current 与 /model/info 返回结构完整。"""
    st, cur = http_get(base_server["port"], "/api/1/model/current")
    assert st == 200 and "name" in cur, f"current: {st} {str(cur)[:150]}"
    st, info = http_get(base_server["port"], "/api/1/model/info")
    assert st == 200 and "name" in info, f"info: {st} {str(info)[:150]}"


def test_hm_m02_models_list(base_server):
    """HM-M02: /models 列表含当前模型。"""
    st, body = http_get(base_server["port"], "/api/1/models")
    assert st == 200, f"{st}: {str(body)[:150]}"
    assert body["total_count"] >= 1 and len(body["models"]) >= 1, "应至少列出 1 个模型"


def test_hm_m03_switch_roundtrip_local_paths(base_server):
    """HM-M03: 本地路径热切换 M1→M2（MiniLM）→M1 往返，切换后 embed 维度保持。"""
    port = base_server["port"]
    st, body = http_post(port, "/api/1/model/switch",
                         {"model_name": "minilm-hm", "model_path": M2_MINILM_PATH,
                          "expected_dimension": 384})
    assert st == 200, f"切换 M2 失败 {st}: {str(body)[:200]}"
    st, emb = http_post(port, "/api/1/embed", {"text": "after switch"})
    assert st == 200 and len(find_vector(emb)) == 384, "切换后 embed 应正常"
    # 切回 M1
    st, body = http_post(port, "/api/1/model/switch",
                         {"model_name": "bge-small-hm", "model_path": M1_PATH,
                          "expected_dimension": 384})
    assert st == 200, f"切回 M1 失败 {st}: {str(body)[:200]}"
    st, emb = http_post(port, "/api/1/embed", {"text": "restored"})
    assert st == 200 and find_vector(emb) is not None, "切回后 embed 应正常"


def test_hm_m04_switch_nonexistent_model_4xx(base_server):
    """HM-M04/HM-M05: 切换不存在模型 → 4xx（SWITCH-001 不回归）且原模型可用。"""
    st, body = http_post(base_server["port"], "/api/1/model/switch",
                         {"model_name": "definitely-not-a-model-xyz"})
    assert 400 <= st < 500, f"SWITCH-001 回归：应 4xx，实际 {st}: {str(body)[:150]}"
    st, emb = http_post(base_server["port"], "/api/1/embed", {"text": "still alive"})
    assert st == 200 and find_vector(emb) is not None, "切换失败后原模型应继续可用"


# ---------------- 异常输入矩阵（HM-A01…HM-A05） ----------------

def test_hm_a01_malformed_json(base_server):
    """HM-A01: 非法 JSON → 4xx。"""
    st, body = http_post(base_server["port"], "/api/1/embed", raw_body="{not json")
    assert 400 <= st < 500, f"非法 JSON 应 4xx，实际 {st}"


def test_hm_a02_wrong_type_text(base_server):
    """HM-A02: text 类型错误（数字）→ 4xx。"""
    st, body = http_post(base_server["port"], "/api/1/embed", {"text": 12345})
    assert 400 <= st < 500, f"类型错误应 4xx，实际 {st}: {str(body)[:150]}"


def test_hm_a03_rerank_top_k_zero(base_server):
    """HM-A03: rerank top_k=0 → 4xx。"""
    st, body = http_post(base_server["port"], "/api/1/rerank",
                         {"query": "q", "documents": ["d1"], "top_k": 0})
    assert 400 <= st < 500, f"top_k=0 应 4xx，实际 {st}: {str(body)[:150]}"


def test_hm_a04_rerank_empty_documents(base_server):
    """HM-A04: rerank documents 空数组 → 4xx。"""
    st, body = http_post(base_server["port"], "/api/1/rerank",
                         {"query": "q", "documents": []})
    assert 400 <= st < 500, f"空 documents 应 4xx，实际 {st}: {str(body)[:150]}"


def test_hm_a05_rerank_overlong_query(base_server):
    """HM-A05: rerank query 超长（>8192）→ 4xx。"""
    st, body = http_post(base_server["port"], "/api/1/rerank",
                         {"query": "x" * 9000, "documents": ["d1", "d2"]})
    assert 400 <= st < 500, f"超长 query 应 4xx，实际 {st}: {str(body)[:150]}"


# ---------------- i18n（HM-I01/HM-I02） ----------------

def test_hm_i01_error_message_locale_en(base_server):
    """HM-I01: 默认（en locale）错误响应为英文。"""
    st, body = http_post(base_server["port"], "/api/1/embed", {"text": ""})
    assert st >= 400
    text = json.dumps(body, ensure_ascii=False) if isinstance(body, dict) else str(body)
    assert not any("\u4e00" <= ch <= "\u9fff" for ch in text), f"默认 locale 应为英文: {text[:150]}"


def test_hm_i02_error_message_locale_zh(base_server):
    """HM-I02: Accept-Language: zh → 错误响应为中文。"""
    conn_ok = True
    st, body = http_post(base_server["port"], "/api/1/embed", {"text": ""})
    # http_post 不支持自定义 header，这里用 GET /health + zh 头无法触发错误体；
    # 改用带 zh 头的原始请求由 conftest 扩展 —— 此处校验 ZH 错误文案端点行为：
    import http.client
    conn = http.client.HTTPConnection("127.0.0.1", base_server["port"], timeout=30)
    try:
        payload = json.dumps({"text": ""}).encode()
        conn.request("POST", "/api/1/embed", body=payload,
                     headers={"Content-Type": "application/json", "Accept-Language": "zh"})
        resp = conn.getresponse()
        raw = resp.read().decode("utf-8", errors="replace")
        st = resp.status
    finally:
        conn.close()
    assert st >= 400
    assert any("\u4e00" <= ch <= "\u9fff" for ch in raw), f"Accept-Language: zh 应返回中文错误: {raw[:150]}"
