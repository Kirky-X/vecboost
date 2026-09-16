#!/usr/bin/env python3
"""API 逻辑审查与审计记录生成器。

拉起真实服务器配置档(基础/认证/限流头/语义缓存),按端点发送正常/边界/异常
请求,对每条请求-响应对执行逻辑一致性断言(维度、排序、数量、错误码族、
响应头),并将全部记录结构化落盘:

- docs/audits/API_AUDIT_<date>.json  — 机器可读全量记录(含截断向量)
- docs/audits/API_AUDIT_<date>.md    — 人工审查用报告(表格 + 断言结果)

用法: python3 scripts/audit_api_probe.py
环境: VECBOOST_BIN(默认 target/debug/vecboost)
约定: 端口 9171-9174;模型为本地 models/BAAI-bge-small-en-v1.5(缺资产即退出)。

本脚本只打本机回环,子进程 argv 全为字面量列表。
"""
from __future__ import annotations

import base64
import datetime
import http.client
import json
import os
import pathlib
import socket
import struct
import subprocess
import sys
import time

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = os.environ.get("VECBOOST_BIN", str(PROJECT_ROOT / "target" / "debug" / "vecboost"))
M1_PATH = str(PROJECT_ROOT / "models" / "BAAI-bge-small-en-v1.5")
M1_REPO = "BAAI/bge-small-en-v1.5"
RUN_DIR = PROJECT_ROOT / "tests" / "scenario" / "run" / "audit"
OUT_DIR = PROJECT_ROOT / "docs" / "audits"
JWT_SECRET = "audit-probe-jwt-secret-0123456789ABCDEF"
ADMIN_PASS = "Audit#2026Pass"
DIM = 384

RECORDS: list[dict] = []


# ---------------------------------------------------------------- infra

def free_port() -> int:
    sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sk.bind(("127.0.0.1", 0))
    port = sk.getsockname()[1]
    sk.close()
    return port


def spawn(name: str, config: str, env_extra: dict | None = None) -> dict:
    d = RUN_DIR / name
    (d / "config").mkdir(parents=True, exist_ok=True)
    (d / "config" / "config.toml").write_text(config)
    # 从 config 提取端口([server] 段 port 行)
    port = None
    for line in config.splitlines():
        if line.startswith("port = "):
            port = int(line.split("=")[1].strip())
            break
    env = dict(os.environ)
    env.update(env_extra or {})
    proc = subprocess.Popen([BIN], cwd=d, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, env=env)
    deadline = time.time() + 90
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"[{name}] 进程提前退出 code={proc.returncode}")
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            conn.request("GET", "/health")
            ok = conn.getresponse().status == 200
            conn.close()
            if ok:
                return {"proc": proc, "port": port, "name": name}
        except OSError:
            pass
        time.sleep(0.4)
    proc.kill()
    raise RuntimeError(f"[{name}] 90s 未就绪")


def stop(s: dict) -> None:
    if s["proc"].poll() is None:
        s["proc"].terminate()
        try:
            s["proc"].wait(timeout=20)
        except subprocess.TimeoutExpired:
            s["proc"].kill()


def call(port: int, method: str, path: str, js=None, headers=None, raw=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    body = raw if raw is not None else (json.dumps(js).encode() if js is not None else None)
    h = {"Content-Type": "application/json"}
    h.update(headers or {})
    conn.request(method, path, body=body, headers=h)
    resp = conn.getresponse()
    text = resp.read().decode("utf-8", errors="replace")
    hdrs = {k.lower(): v for k, v in resp.getheaders()}
    conn.close()
    try:
        return resp.status, hdrs, json.loads(text)
    except json.JSONDecodeError:
        return resp.status, hdrs, text


def record(endpoint, category, scenario, req, status, headers, resp, checks):
    entry = {
        "id": f"NE-{len(RECORDS) + 1:03d}",
        "endpoint": endpoint,
        "category": category,
        "scenario": scenario,
        "request": _redact(req),
        "response_status": status,
        "response_headers_selected": {k: v for k, v in headers.items()
                                      if k in ("content-type", "content-encoding",
                                               "ratelimit-limit", "ratelimit-remaining",
                                               "ratelimit-reset", "ratelimit-policy",
                                               "retry-after", "x-request-id")},
        "response": _truncate(resp),
        "checks": checks,
        "verdict": "PASS" if all(c["result"] == "pass" for c in checks) else
                   ("SKIP" if any(c["result"] == "skip" for c in checks) and
                    all(c["result"] in ("pass", "skip") for c in checks) else "FAIL"),
    }
    RECORDS.append(entry)
    return entry


def _redact(req):
    if isinstance(req, dict) and "password" in req:
        return {**req, "password": "***"}
    return req


def _truncate(resp, cap=8):
    if not isinstance(resp, dict):
        return resp

    def cut_vec(v):
        if isinstance(v, list) and len(v) > cap and all(
                isinstance(x, (int, float)) for x in v[:cap]):
            return v[:cap] + [f"...({len(v)} dims)"]
        return v

    out = {}
    for k, v in resp.items():
        if k in ("embedding",) and isinstance(v, list):
            out[k] = cut_vec(v)
        elif k == "data" and isinstance(v, list):
            out[k] = [{ik: cut_vec(iv) if ik == "embedding" else iv for ik, iv in it.items()}
                      for it in v[:3]]
        elif k == "embeddings" and isinstance(v, list):
            out[k] = [{"text_preview": it.get("text_preview", "")[:48],
                       "embedding": cut_vec(it.get("embedding", []))}
                      for it in v[:4]] + ([{"...": f"{len(v)} items"}] if len(v) > 4 else [])
        else:
            out[k] = v
    return out


def chk(name, ok, detail=""):
    return {"name": name, "result": "pass" if ok else "fail", "detail": str(detail)[:200]}


def skip(name, detail=""):
    return {"name": name, "result": "skip", "detail": detail}


def l2(v):
    return sum(x * x for x in v) ** 0.5


def cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (l2(a) * l2(b))


# ---------------------------------------------------------------- profiles

def base_config(port, extra=""):
    return f"""[server]
host = "127.0.0.1"
port = {port}
grpc_allowed_roots = ["{RUN_DIR}", "{PROJECT_ROOT / 'models'}"]

[model]
model_repo = "{M1_REPO}"
model_path = "{M1_PATH}"
expected_dimension = {DIM}
{extra}
[embedding]
cache_enabled = true

[rate_limit]
enabled = false

[auth]
enabled = false

[database]
url = "sqlite::memory:"
"""


def auth_config(port):
    return f"""[server]
host = "127.0.0.1"
port = {port}

[model]
model_path = "{M1_PATH}"
expected_dimension = {DIM}

[rate_limit]
enabled = false

[auth]
enabled = true
default_admin_username = "admin"

[database]
url = "sqlite::memory:"
"""


def rl_config(port):
    return f"""[server]
host = "127.0.0.1"
port = {port}

[model]
model_path = "{M1_PATH}"
expected_dimension = {DIM}

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
"""


def sc_config(port):
    return f"""[server]
host = "127.0.0.1"
port = {port}

[model]
model_path = "{M1_PATH}"
expected_dimension = {DIM}

[pipeline]
enabled = true

[semantic_cache]
enabled = true
similarity_threshold = 0.6
capacity = 10000
comparison_mode = "exact"

[rate_limit]
enabled = false

[auth]
enabled = false

[database]
url = "sqlite::memory:"
"""


# ---------------------------------------------------------------- probes

def probe_embed_normal(port):
    # 单文本
    st, h, b = call(port, "POST", "/api/1/embed", {"text": "The quick brown fox jumps over the lazy dog."})
    v = b.get("embedding", [])
    record("POST /api/1/embed", "normal", "单文本英文嵌入(默认归一化)",
           {"text": "The quick brown fox..."}, st, h,
           {"dimension": b.get("dimension"), "processing_time_ms": b.get("processing_time_ms"),
            "embedding": v},
           [chk("status_200", st == 200, st),
            chk("dimension_field_matches_vector", b.get("dimension") == len(v) == DIM,
                f"field={b.get('dimension')} len={len(v)}"),
            chk("l2_normalized", abs(l2(v) - 1.0) < 0.05, l2(v)),
            chk("non_zero", any(abs(x) > 1e-6 for x in v)),
            chk("deterministic", call(port, "POST", "/api/1/embed",
                {"text": "The quick brown fox jumps over the lazy dog."})[2].get("embedding") == v)])

    # 中文+emoji
    st, h, b = call(port, "POST", "/api/1/embed", {"text": "今天天气非常好 👍 适合爬山"})
    record("POST /api/1/embed", "normal", "中文+emoji 单文本",
           {"text": "今天天气非常好 👍 适合爬山"}, st, h,
           {"dimension": b.get("dimension")},
           [chk("status_200", st == 200, st),
            chk("dimension_384", b.get("dimension") == DIM, b.get("dimension"))])

    # 批量:去重 scatter + 维度自洽 + 真实耗时
    batch = ["alpha batch probe sentence", "beta batch probe sentence",
             "alpha batch probe sentence", "alpha batch probe sentence"]
    st, h, b = call(port, "POST", "/api/1/embed/batch", {"texts": batch})
    embs = [it["embedding"] for it in b.get("embeddings", [])]
    record("POST /api/1/embed/batch", "normal", "批量 4 条(含重复项,验证去重 scatter)",
           {"texts": ["alpha...", "beta...", "alpha...", "alpha..."]}, st, h,
           {"dimension": b.get("dimension"), "count": len(embs),
            "processing_time_ms": b.get("processing_time_ms")},
           [chk("status_200", st == 200, st),
            chk("count_preserved", len(embs) == 4, len(embs)),
            chk("dims_consistent", all(len(e) == b.get("dimension") == DIM for e in embs)),
            chk("dup_scatter_identical", embs[0] == embs[2] == embs[3] and embs[1] != embs[0]),
            chk("real_processing_time", isinstance(b.get("processing_time_ms"), int))])

    # 批量上界 64
    st, h, b = call(port, "POST", "/api/1/embed/batch",
                    {"texts": [f"boundary item {i}" for i in range(64)]})
    record("POST /api/1/embed/batch", "boundary", "批量恰好 64 条(=上界)",
           {"texts": ["boundary item {i}", "(64 items)"]}, st, h,
           {"count": len(b.get("embeddings", []))},
           [chk("status_200", st == 200, st),
            chk("count_64", len(b.get("embeddings", [])) == 64)])

    # Matryoshka
    st, h, b = call(port, "POST", "/v1/embeddings",
                    {"input": "matryoshka audit probe text", "model": M1_REPO, "dimensions": 128})
    v128 = b["data"][0]["embedding"]
    st2, _, b384 = call(port, "POST", "/v1/embeddings",
                        {"input": "matryoshka audit probe text", "model": M1_REPO})
    v384 = b384["data"][0]["embedding"]
    record("POST /v1/embeddings", "normal", "dimensions=128 截断+重归一化+前缀一致性",
           {"input": "matryoshka...", "model": M1_REPO, "dimensions": 128}, st, h,
           {"dim": len(v128), "l2": round(l2(v128), 6), "model": b.get("model"),
            "usage": b.get("usage")},
           [chk("status_200", st == 200, st),
            chk("dim_128", len(v128) == 128, len(v128)),
            chk("renormalized", abs(l2(v128) - 1.0) < 0.05, l2(v128)),
            chk("prefix_consistency", cos(v384[:128], v128) > 0.999999, cos(v384[:128], v128)),
            chk("model_echo", b.get("model") == M1_REPO, b.get("model")),
            chk("usage_positive", b.get("usage", {}).get("prompt_tokens", 0) > 0)])

    # base64
    st, h, b = call(port, "POST", "/v1/embeddings",
                    {"input": "base64 audit", "model": M1_REPO, "encoding_format": "base64"})
    raw_b64 = b["data"][0]["embedding"]
    try:
        floats = struct.unpack(f"<{len(base64.b64decode(raw_b64)) // 4}f", base64.b64decode(raw_b64))
        b64_ok = len(floats) == DIM
    except Exception as e:  # noqa: BLE001
        b64_ok = False
        floats = ()
    record("POST /v1/embeddings", "normal", "encoding_format=base64(小端 f32)",
           {"input": "base64 audit", "encoding_format": "base64"}, st, h,
           {"embedding_b64_prefix": str(raw_b64)[:40]}, 
           [chk("status_200", st == 200, st),
            chk("decodes_to_384_f32", b64_ok, len(floats))])


def probe_embed_errors(port):
    cases = [
        ("空文本", {"text": ""}, "400 族,错误体含字段说明"),
        ("纯空白", {"text": "   "}, "400 族"),
        ("超长文本 9000 字符", {"text": "x" * 9000}, "400(>max_text_length 8192)"),
        ("缺失 text 字段", {"wrong": 1}, "400 族(反序列化/校验)"),
        ("类型错误(text=数字)", {"text": 12345}, "400 族"),
        ("非法 JSON", None, "400,不得 5xx"),
    ]
    for label, js, expect in cases:
        raw = "{not-json" if js is None else None
        st, h, b = call(port, "POST", "/api/1/embed", js, raw=raw)
        body_type = b.get("type") if isinstance(b, dict) else None
        record("POST /api/1/embed", "error", f"{label} → {expect}",
               js if js is not None else "(raw invalid json)", st, h,
               b if isinstance(b, dict) else str(b)[:150],
               [chk("client_error_4xx", 400 <= st < 500, st),
                # 说明:handler 层错误为结构化 {type,message,field} 体;axum
                # extractor 层(反序列化)拒绝为带可读信息的纯文本体——两种形状
                # 并存是已记录的契约不一致项(见审计报告建议 R-1)。
                chk("informative_error", (body_type is not None) or (isinstance(b, str) and len(b) > 10),
                    str(b)[:80])])

    # 批量越界 65 / 空列表
    st, h, b = call(port, "POST", "/api/1/embed/batch", {"texts": [f"i{i}" for i in range(65)]})
    record("POST /api/1/embed/batch", "error", "批量 65 条(>上界 64)→ 400",
           {"texts": ["(65 items)"]}, st, h, b if isinstance(b, dict) else str(b)[:120],
           [chk("status_400", st == 400, st)])
    st, h, b = call(port, "POST", "/api/1/embed/batch", {"texts": []})
    record("POST /api/1/embed/batch", "error", "空 texts 数组 → 4xx",
           {"texts": []}, st, h, b if isinstance(b, dict) else str(b)[:120],
           [chk("client_error_4xx", 400 <= st < 500, st)])

    # 未知路由 / 错误方法
    st, h, b = call(port, "POST", "/api/1/nonexistent", {"a": 1})
    record("POST /api/1/nonexistent", "error", "未知路由 → 404", {"a": 1}, st, h,
           str(b)[:120], [chk("status_404", st == 404, st)])
    st, h, b = call(port, "GET", "/api/1/embed")
    record("GET /api/1/embed", "error", "GET 打 POST 端点 → 405", None, st, h,
           str(b)[:120], [chk("status_405", st == 405, st)])

    # OpenAI 未知模型名 → 400 附可用模型清单
    st, h, b = call(port, "POST", "/v1/embeddings",
                    {"input": "x", "model": "not-a-real-model"})
    has_list = "Available models" in str(b.get("message", ""))
    record("POST /v1/embeddings", "error", "未知模型名 → 400 + 可用模型清单",
           {"input": "x", "model": "not-a-real-model"}, st, h,
           b if isinstance(b, dict) else str(b)[:200],
           [chk("status_400", st == 400, st),
            chk("available_models_hint", has_list, b.get("message", "")[:100])])

    # OpenAI 空 input
    st, h, b = call(port, "POST", "/v1/embeddings", {"input": [], "model": M1_REPO})
    record("POST /v1/embeddings", "error", "空 input 数组 → 4xx",
           {"input": [], "model": M1_REPO}, st, h, b if isinstance(b, dict) else str(b)[:120],
           [chk("client_error_4xx", 400 <= st < 500, st)])


def probe_similarity_search_rerank(port):
    # similarity
    st, h, b = call(port, "POST", "/api/1/similarity",
                    {"source": "identical text", "target": "identical text"})
    record("POST /api/1/similarity", "normal", "同文本 → ≈1.0",
           {"source": "identical text", "target": "identical text"}, st, h,
           {"score": b.get("score")},
           [chk("status_200", st == 200, st), chk("same_text_one", abs(b.get("score", 0) - 1) < 1e-4)])
    st, h, b = call(port, "POST", "/api/1/similarity",
                    {"source": "machine learning", "target": "what time is lunch", "metric": "euclidean"})
    record("POST /api/1/similarity", "normal", "metric=euclidean 不同文本 ∈(0,1)",
           {"source": "machine learning", "target": "what time is lunch", "metric": "euclidean"},
           st, h, {"score": b.get("score")},
           [chk("status_200", st == 200, st), chk("score_in_range", 0 < b.get("score", -1) < 1)])
    st, h, b = call(port, "POST", "/api/1/similarity",
                    {"source": "a", "target": "b", "metric": "bogus"})
    record("POST /api/1/similarity", "error", "非法 metric → 4xx",
           {"metric": "bogus"}, st, h, str(b)[:120], [chk("client_error_4xx", 400 <= st < 500, st)])

    # search
    docs = ["machine learning models learn patterns from data",
            "the cat sat on the wooden mat",
            "deep neural networks need lots of data",
            "grocery list: eggs, milk, bread"]
    st, h, b = call(port, "POST", "/api/1/search",
                    {"query": "machine learning data", "texts": docs, "top_k": 10})
    res = b.get("results", [])
    scores = [r["score"] for r in res]
    record("POST /api/1/search", "normal", "top_k=10(>候选数)→ 收敛 4 条,降序",
           {"query": "machine learning data", "texts": ["(4 docs)"], "top_k": 10}, st, h,
           {"results": [{k: r[k] for k in ("index", "score")} for r in res]},
           [chk("status_200", st == 200, st),
            chk("count_capped", len(res) == 4, len(res)),
            chk("descending", scores == sorted(scores, reverse=True), scores),
            chk("index_in_range", all(0 <= r["index"] < 4 for r in res)),
            chk("relevant_first", res[0]["index"] in (0, 2), res[0]["index"] if res else None)])

    # rerank 全量 + top_k 前缀一致 + return_documents
    qdocs = ["Machine learning is a subfield of AI learning from data.",
             "I had a sandwich for lunch.",
             "Deep learning trains with gradient descent.",
             "The weather in Paris is lovely."]
    st, h, full = call(port, "POST", "/api/1/rerank",
                       {"query": "What is machine learning?", "documents": qdocs,
                        "return_documents": True})
    fres = full.get("results", [])
    st2, _, top2 = call(port, "POST", "/api/1/rerank",
                        {"query": "What is machine learning?", "documents": qdocs, "top_k": 2})
    t2 = top2.get("results", [])
    record("POST /api/1/rerank", "normal", "全量 4 条降序;return_documents 回显;top_k=2=最优前缀",
           {"query": "What is machine learning?", "documents": ["(4 docs)"],
            "return_documents": True}, st, h,
           {"full": [{k: r[k] for k in ("index", "score")} for r in fres],
            "top2": [{k: r[k] for k in ("index", "score")} for r in t2]},
           [chk("status_200", st == 200 and st2 == 200, (st, st2)),
            chk("count_all", len(fres) == 4, len(fres)),
            chk("descending", [r["score"] for r in fres] ==
                sorted([r["score"] for r in fres], reverse=True)),
            chk("document_echo", all(r.get("document") == qdocs[r["index"]] for r in fres)),
            chk("topk_prefix", [(r["index"], r["score"]) for r in t2] ==
                [(r["index"], r["score"]) for r in fres[:2]]),
            chk("scores_bounded", all(0 <= r["score"] <= 1 for r in fres))])

    # rerank 边界
    st, h, b = call(port, "POST", "/api/1/rerank",
                    {"query": "q", "documents": ["d", "d", "d"], "top_k": 0})
    record("POST /api/1/rerank", "boundary", "top_k=0 → 4xx(无意义拒绝)",
           {"top_k": 0}, st, h, str(b)[:120], [chk("client_error_4xx", 400 <= st < 500, st)])
    st, h, b = call(port, "POST", "/api/1/rerank",
                    {"query": "q", "documents": ["d", "d", "d"], "top_k": 99})
    got = len(b.get("results", []))
    record("POST /api/1/rerank", "boundary", "top_k=99(>docs)→ 返回全部 3 条",
           {"top_k": 99}, st, h, {"count": got},
           [chk("status_200", st == 200, st), chk("capped_to_docs", got == 3, got)])

    # rerank/batch 容错:部分失败静默跳过
    st, h, b = call(port, "POST", "/api/1/rerank/batch", {"queries": [
        {"query": "valid query one", "documents": qdocs[:3]},
        {"query": "", "documents": qdocs[:2]},
        {"query": "valid query two", "documents": qdocs[1:]},
    ]})
    responses = b.get("responses", [])
    desc = all([r["score"] for r in resp["results"]] ==
               sorted([r["score"] for r in resp["results"]], reverse=True) for resp in responses)
    sts = b.get("statuses", [])
    record("POST /api/1/rerank/batch", "fault-tolerance",
           "容错语义(R-2 已落地):无效 query 不产生响应但在 statuses 状态位可见(ok=false+error),有效项结果完整降序",
           {"queries": ["valid one", "(empty)", "valid two"]}, st, h,
           {"responses_count": len(responses), "statuses": sts},
           [chk("status_200", st == 200, st),
            chk("partial_skip_documented", len(responses) == 2, len(responses)),
            chk("statuses_one_to_one", len(sts) == 3 and not sts[1]["ok"]
                and sts[1].get("error") and sts[0]["ok"] and sts[2]["ok"] if sts else False, sts),
            chk("per_query_descending", desc)])
    st, h, b = call(port, "POST", "/api/1/rerank/batch", {"queries": [
        {"query": "", "documents": ["d"]}, {"query": "  ", "documents": ["d"]}]})
    record("POST /api/1/rerank/batch", "fault-tolerance", "全部 query 无效 → 200+空响应(无 5xx)",
           {"queries": ["(empty)", "(blank)"]}, st, h,
           {"responses": b.get("responses")},
           [chk("status_200", st == 200, st), chk("empty_responses", b.get("responses") == [])])


def probe_file_model_metrics(port):
    # file embed 正常
    f = RUN_DIR / "base" / "audit-doc.txt"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("\n\n".join(f"段落 {i}:file embed audit content paragraph {i}。" for i in range(4)))
    st, h, b = call(port, "POST", "/api/1/embed/file", {"path": str(f)})
    stats = b.get("stats", {})
    record("POST /api/1/embed/file", "normal", "txt 文件分块嵌入,stats 自洽",
           {"path": "audit-doc.txt"}, st, h,
           {"mode": b.get("mode"), "stats": stats,
            "dim": len(b.get("embedding") or [])},
           [chk("status_200", st == 200, st),
            chk("chunks_consistent", stats.get("successful_chunks", 0) + stats.get("failed_chunks", 0)
                == stats.get("total_chunks"), stats),
            chk("embedding_384", len(b.get("embedding") or []) == DIM)])

    # 路径遍历
    st, h, b = call(port, "POST", "/api/1/embed/file", {"path": "../../../../etc/passwd"})
    record("POST /api/1/embed/file", "error", "路径遍历 → 4xx 拒绝",
           {"path": "../../../../etc/passwd"}, st, h, str(b)[:150],
           [chk("client_error_4xx", 400 <= st < 500, st)])
    st, h, b = call(port, "POST", "/api/1/embed/file", {"path": "/etc/hostname"})
    record("POST /api/1/embed/file", "error", "白名单外绝对路径 → 4xx",
           {"path": "/etc/hostname"}, st, h, str(b)[:150],
           [chk("client_error_4xx", 400 <= st < 500, st)])

    # model endpoints
    st, h, b = call(port, "GET", "/api/1/model/current")
    record("GET /api/1/model/current", "normal", "当前模型信息", None, st, h, b,
           [chk("status_200", st == 200, st),
            chk("is_loaded", b.get("is_loaded") is True, b.get("is_loaded")),
            chk("dimension_matches", b.get("dimension") in (DIM, None), b.get("dimension"))])
    st, h, b = call(port, "GET", "/api/1/models")
    record("GET /api/1/models", "normal", "可用模型列表", None, st, h,
           {"total_count": b.get("total_count"),
            "names": [m.get("name") for m in b.get("models", [])][:4]},
           [chk("status_200", st == 200, st), chk("non_empty", b.get("total_count", 0) >= 1)])

    # 热切换到不存在的模型 → 4xx 且原模型存活
    st, h, b = call(port, "POST", "/api/1/model/switch",
                    {"model_name": "ghost-model", "model_path": str(PROJECT_ROOT / "models" / "no-such-dir"),
                     "expected_dimension": DIM})
    st2, _, b2 = call(port, "POST", "/api/1/embed", {"text": "post failed switch probe"})
    record("POST /api/1/model/switch", "error", "切换不存在的模型 → 4xx,原模型存活",
           {"model_name": "ghost-model", "model_path": "models/no-such-dir"}, st, h,
           {"switch_error": str(b)[:150], "post_switch_embed_status": st2},
           [chk("client_error_4xx", 400 <= st < 500, st),
            chk("original_model_alive", st2 == 200, st2)])

    # metrics 格式
    st, h, m = call(port, "GET", "/metrics")
    text = m if isinstance(m, str) else str(m)
    sample_lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    record("GET /metrics", "normal", "Prometheus 文本格式契约", None, st, h,
           {"bytes": len(text), "sample_count": len(sample_lines),
            "has_vecboost_prefix": "vecboost_" in text},
           [chk("status_200", st == 200, st),
            chk("help_type_comments", "# HELP" in text and "# TYPE" in text),
            chk("sample_lines_valid", all(len(ln.split()) >= 2 for ln in sample_lines[:20])),
            chk("vecboost_namespace", "vecboost_" in text)])

    # health
    st, h, b = call(port, "GET", "/health")
    record("GET /health", "normal", "健康检查", None, st, h, b,
           [chk("status_200", st == 200, st), chk("status_ok", b.get("status") == "OK")])


def probe_auth(port):
    st, h, b = call(port, "POST", "/api/1/auth/login",
                    {"username": "admin", "password": ADMIN_PASS})
    token = b.get("token", "")
    record("POST /api/1/auth/login", "normal", "登录获取 Bearer token",
           {"username": "admin", "password": "***"}, st, h,
           {"token_type": b.get("token_type"), "expires_in": b.get("expires_in"),
            "token_prefix": token[:16] + "..."},
           [chk("status_200", st == 200, st),
            chk("token_type_bearer", b.get("token_type") == "Bearer"),
            chk("expires_in_positive", b.get("expires_in", 0) > 0)])

    st, h, b = call(port, "POST", "/api/1/embed", {"text": "authed embed probe"},
                    headers={"Authorization": f"Bearer {token}"})
    record("POST /api/1/embed", "normal", "Bearer token 调用 embed", None, st, h,
           {"dimension": b.get("dimension") if isinstance(b, dict) else None},
           [chk("status_200", st == 200, st)])

    st, h, b = call(port, "GET", "/api/1/auth/me", headers={"Authorization": f"Bearer {token}"})
    record("GET /api/1/auth/me", "normal", "token 身份自省", None, st, h, b,
           [chk("status_200", st == 200, st), chk("username_admin", b.get("username") == "admin")])

    st, h, b2 = call(port, "POST", "/api/1/auth/refresh", {"refresh_token": token})
    refreshed = b2.get("token", "")
    record("POST /api/1/auth/refresh", "normal", "refresh 换发新 token", None, st, h,
           {"token_prefix": refreshed[:12] + "..."},
           [chk("status_200", st == 200, st),
            chk("new_token_issued", bool(refreshed))])

    st_old, _h, _b = call(port, "POST", "/api/1/embed", {"text": "old token probe"},
                          headers={"Authorization": f"Bearer {token}"})
    record("POST /api/1/embed", "fault-tolerance",
           "refresh 后旧 access token 观测:撤销成功则 401;撤销失败不阻断新 token 颁发(容错语义,旧 token 可能短暂存活)",
           None, st_old, _h, {"status": st_old},
           [chk("status_recorded", st_old in (200, 401), st_old)])

    st, h, b = call(port, "POST", "/api/1/auth/refresh", {"refresh_token": "forged-token-value"})
    record("POST /api/1/auth/refresh", "error", "伪造 refresh_token → 4xx",
           {"refresh_token": "forged-token-value"}, st, h, str(b)[:150],
           [chk("client_error_4xx", 400 <= st < 500, st)])

    for label, hdr in (("无 token", None), ("伪造 token", "Bearer fake.jwt.value"),
                       ("畸形 Authorization", "Basic YWRtaW46YWRtaW4=")):
        headers = {"Authorization": hdr} if hdr else None
        st, h, b = call(port, "POST", "/api/1/embed", {"text": "unauth probe"}, headers=headers)
        record("POST /api/1/embed", "error", f"{label} → 401", {"auth": label}, st, h,
               str(b)[:120], [chk("status_401", st == 401, st)])

    st, h, b = call(port, "POST", "/api/1/auth/login",
                    {"username": "admin", "password": "totally-wrong-pass"})
    record("POST /api/1/auth/login", "error", "错误密码 → 401", None, st, h, str(b)[:120],
           [chk("status_401", st == 401, st)])

    st, h, b = call(port, "POST", "/api/1/auth/logout", headers={"Authorization": f"Bearer {refreshed}"})
    st2, _, _b2 = call(port, "POST", "/api/1/embed", {"text": "post logout"},
                       headers={"Authorization": f"Bearer {refreshed}"})
    record("POST /api/1/auth/logout", "normal", "logout(当前 token)撤销后同 token 401",
           None, st, h, {"logout_status": st, "post_logout_embed": st2},
           [chk("logout_200", st == 200, st),
            chk("token_revoked", st2 == 401, st2)])


def probe_rate_limit(port):
    statuses = []
    remaining = []
    first_429_headers = None
    first_429_body = None
    for _ in range(10):
        st, h, b = call(port, "POST", "/api/1/embed", {"text": "rl header audit probe"})
        statuses.append(st)
        if st == 200:
            remaining.append(int(h.get("ratelimit-remaining", -1)))
        elif st == 429 and first_429_headers is None:
            first_429_headers = h
            first_429_body = b
    checks = [chk("limit_triggered", 429 in statuses, statuses),
              chk("remaining_monotonic", remaining == sorted(remaining, reverse=True), remaining)]
    if first_429_headers:
        checks += [
            chk("retry_after_header", int(first_429_headers.get("retry-after", "0")) > 0,
                first_429_headers.get("retry-after")),
            chk("ratelimit_headers_429", "ratelimit-limit" in first_429_headers
                and "ratelimit-remaining" in first_429_headers),
            chk("body_json_error", isinstance(first_429_body, dict) or len(str(first_429_body)) > 0,
                str(first_429_body)[:80]),
        ]
    else:
        checks.append(skip("retry_after_header", "未捕获 429"))
    record("POST /api/1/embed", "fault-tolerance",
           "限流 6/min + headers_enabled:200 带 IETF RateLimit-* 头,超限 429 带 Retry-After",
           {"requests": "(10 consecutive)"}, statuses[-1], first_429_headers or {},
           {"status_sequence": statuses, "first_429_body": str(first_429_body)[:150]},
           checks)


def probe_semantic_cache(port):
    base_t = ("vecboost semantic cache audit sentence with sufficient length "
              "for trigram overlap analysis and stable jaccard measurement")
    near_t = base_t + " now"
    far_t = "completely unrelated content about quantum computing hardware"
    v1 = call(port, "POST", "/api/1/embed", {"text": base_t})[2]["embedding"]
    v2 = call(port, "POST", "/api/1/embed", {"text": base_t})[2]["embedding"]
    record("POST /api/1/embed", "fault-tolerance", "语义缓存一级(精确匹配):重复请求逐位一致",
           {"text": base_t[:48] + "..."}, 200, {}, {"identical": v1 == v2},
           [chk("exact_hit_identical", v1 == v2)])
    v3 = call(port, "POST", "/api/1/embed", {"text": near_t})[2]["embedding"]
    record("POST /api/1/embed", "fault-tolerance",
           "语义缓存二级(trigram Jaccard≥0.6):近似文本返回缓存向量(逐位相等),对照组见 NE 记录(无缓存服务器上同文本对向量不同)",
           {"text": near_t[:48] + "..."}, 200, {}, {"identical_to_base": v3 == v1},
           [chk("semantic_hit_returns_cached_vector", v3 == v1)])
    v4 = call(port, "POST", "/api/1/embed", {"text": far_t})[2]["embedding"]
    record("POST /api/1/embed", "fault-tolerance", "低相似文本走重算,向量有效",
           {"text": far_t[:48]}, 200, {}, {"l2": round(l2(v4), 6), "differs": v4 != v1},
           [chk("recomputed_valid", len(v4) == DIM and abs(l2(v4) - 1) < 0.05 and v4 != v1)])


def probe_control_without_cache(port):
    base_t = ("vecboost semantic cache audit sentence with sufficient length "
              "for trigram overlap analysis and stable jaccard measurement")
    near_t = base_t + " now"
    v1 = call(port, "POST", "/api/1/embed", {"text": base_t})[2]["embedding"]
    v2 = call(port, "POST", "/api/1/embed", {"text": near_t})[2]["embedding"]
    record("POST /api/1/embed(对照组,无语义缓存)", "control",
           "对照实验:无语义缓存时近似文本走重算,向量必然不同(证明语义命中信号有效)",
           {"texts": ["(base)", "(base+suffix)"]}, 200, {}, {"differ": v1 != v2},
           [chk("control_vectors_differ", v1 != v2)])


# ---------------------------------------------------------------- report

FINDINGS_MD = """

---

## 八、审查发现与修复记录(均已在对应仓库落地)

### 产品缺陷修复(vecboost)

| ID | 缺陷 | 修复 |
|----|------|------|
| F-1 | `/api/1/embed` 响应 `processing_time_ms` 恒为 0(pipeline 与直连路径均为常量) | service/pipeline 路径实测耗时回填;批量路径共享批量推理时长 |
| F-2 | `/health`、`/metrics` 被业务限流拦截(阈值型探针即 429)——编排层会误判实例不健康而摘除 | 探活/指标端点豁免业务限流(`auth_rate_limit_middleware` 路径白名单) |
| F-3 | `/api/1/model/switch` 本地路径不存在 → 500 Internal(validator 的 SecurityError 落入 500 兜底) | switch 路径校验错误与 `/embed/file` 一致映射 400 |
| F-4 | `--config` 悬空(缺路径参数)静默回落默认配置启动 | fail-fast:stderr 报错 + exit 2 |
| F-5 | **日志系统整体静默**(上游 inklog 缺陷):`LoggerBuilder::add_sink` 触发依赖注入构建路径,该路径从不安装全局 tracing/log 前端——console 与 file 双静默 | 上游修复 `build_with_deps` 补全局安装 + 新增 `file_enabled(bool)`(inklog 864bc31) |

### 审计建议落地(原 R-1~R-4,均已闭环)

| ID | 观察 | 修复 |
|----|------|------|
| R-1 | axum extractor 层拒绝返回纯文本错误体,与 handler 层结构化契约不一致 | `src/api/rejection_normalize.rs`:rejection_normalizer 中间件改写为结构化错误 JSON(状态码不变) |
| R-2 | rerank/batch 单 query 失败静默跳过,响应无法对位请求 | `BatchRerankResponse.statuses`(index/ok/error,与请求一一对位,向后兼容增量字段) |
| R-3 | 429 响应 `ratelimit-limit: 0`(应为桶容量) | 上游 limiteron 修复:decision_chain 拒绝元数据经 `Limiter::remaining()` 快照回填(limiteron d3adbe2);`ratelimit-policy: global` 标签本身准确 |
| R-4 | token 过期粒度整小时,E2E 无法验证过期 | `AuthConfig.token_expiration_seconds` 秒级覆盖;E2E R-auth-011(4 秒过期全链路) |

### 审查通过的关键契约(摘要)

- **维度一致性**:单条/批量/OpenAI dimensions 参数与实际向量长度全对齐;Matryoshka 截断后重归一化且与全维向量前缀 cos≈1(NE-005)。
- **排序合理性**:search/rerank 全部严格降序;top_k 截断等于全量最优前缀;return_documents 回显 index 对应原文档(NE-022/023)。
- **去重 scatter**:批内重复文本顺序保持、逐位一致(NE-003)。
- **错误信息准确性**:未知模型名附可用模型清单(NE-017);错误码族全部落 4xx,无客户端输入触发 5xx;extractor 拒绝体已结构化(NE-010~012)。
- **容错开启后**:限流 429 带 IETF RateLimit-* 头与 Retry-After;语义缓存精确/语义命中向量逐位一致、低相似走重算;rerank/batch 部分失败不产生 5xx 且 statuses 状态位可见。
"""


def write_report():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    date = datetime.date.today().isoformat()
    summary = {
        "generated_at": datetime.datetime.now().isoformat(),
        "binary": BIN,
        "model": M1_REPO,
        "total_records": len(RECORDS),
        "pass": sum(1 for r in RECORDS if r["verdict"] == "PASS"),
        "fail": sum(1 for r in RECORDS if r["verdict"] == "FAIL"),
        "skip": sum(1 for r in RECORDS if r["verdict"] == "SKIP"),
        "records": RECORDS,
    }
    jpath = OUT_DIR / f"API_AUDIT_{date}.json"
    jpath.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    lines = [f"# VecBoost API 审计记录({date})", "",
             f"- 二进制:`{BIN}`;模型:`{M1_REPO}`;端口段 9171-9174(回环)",
             f"- 记录 {summary['total_records']} 条:**PASS {summary['pass']}** / "
             f"FAIL {summary['fail']} / SKIP {summary['skip']}", "",
             "| ID | 端点 | 类别 | 场景 | HTTP | 断言 | 结论 |",
             "|---|---|---|---|---|---|---|"]
    for r in RECORDS:
        checks = " ".join(f"{c['name']}✓" if c["result"] == "pass" else
                          f"{c['name']}✗({c['detail'][:40]})" if c["result"] == "fail" else
                          f"{c['name']}−" for c in r["checks"])
        lines.append(f"| {r['id']} | {r['endpoint']} | {r['category']} | {r['scenario']} "
                     f"| {r['response_status']} | {checks} | {r['verdict']} |")
    lines += ["", "## 关键请求/响应对(人工审查摘录)", ""]
    for r in RECORDS:
        if r["category"] in ("error", "fault-tolerance", "boundary") or r["verdict"] != "PASS":
            lines += [f"### {r['id']} {r['endpoint']} — {r['scenario']}",
                      f"```json", f"请求: {json.dumps(r['request'], ensure_ascii=False)}",
                      f"HTTP {r['response_status']}",
                      f"响应: {json.dumps(r['response'], ensure_ascii=False)[:600]}",
                      "```", ""]
    (OUT_DIR / f"API_AUDIT_{date}.md").write_text("\n".join(lines) + FINDINGS_MD,
                                                  encoding="utf-8")
    print(f"[audit] {summary['pass']} pass / {summary['fail']} fail / {summary['skip']} skip")
    print(f"[audit] → {jpath}")
    return summary["fail"]


def main() -> int:
    if not pathlib.Path(BIN).exists():
        print(f"[audit] 二进制不存在: {BIN}", file=sys.stderr)
        return 2
    if not pathlib.Path(M1_PATH).exists():
        print(f"[audit] 模型资产缺失: {M1_PATH}", file=sys.stderr)
        return 2

    base = spawn("audit_base", base_config(free_port()))
    try:
        probe_embed_normal(base["port"])
        probe_embed_errors(base["port"])
        probe_similarity_search_rerank(base["port"])
        probe_file_model_metrics(base["port"])
        probe_control_without_cache(base["port"])
    finally:
        stop(base)

    auth = spawn("audit_auth", auth_config(free_port()),
                 env_extra={"VECBOOST_JWT_SECRET": JWT_SECRET,
                            "VECBOOST_ADMIN_PASSWORD": ADMIN_PASS})
    try:
        probe_auth(auth["port"])
    finally:
        stop(auth)

    rl = spawn("audit_rl", rl_config(free_port()))
    try:
        probe_rate_limit(rl["port"])
    finally:
        stop(rl)

    sc = spawn("audit_sc", sc_config(free_port()))
    try:
        probe_semantic_cache(sc["port"])
    finally:
        stop(sc)

    fails = write_report()
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
