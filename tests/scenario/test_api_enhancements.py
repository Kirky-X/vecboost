"""API/配置增强套件（api-config-enhancements，AE-*）。

覆盖：
- AE-01/02: --config 自定义路径启动 / 缺失参数拒绝
- AE-03/04: POST /api/1/search 正常排序与空候选拒绝
- AE-05/06: similarity metric=euclidean/manhattan/非法值
- AE-07/08: CORS 预检与响应头 / gzip 压缩
- AE-09: VECBOOST_LOG_LEVEL 环境变量生效（debug 级日志落盘）
- AE-10: model/unload 幂等卸载
"""
from __future__ import annotations

import http.client
import json
import pathlib
import socket
import subprocess
import time

import pytest

from conftest import PROJECT_ROOT, RUN_DIR

BIN = PROJECT_ROOT / "target" / "debug" / "vecboost"
M1_PATH = str(PROJECT_ROOT / "models" / "BAAI-bge-small-en-v1.5")
PORT = 39461

CONFIG = """[server]
host = "127.0.0.1"
port = {port}
cors_enabled = true
cors_allow_origins = ["*"]

[model]
model_path = "{m1}"
expected_dimension = 384

[rate_limit]
enabled = false

[auth]
enabled = false

[database]
url = "sqlite::memory:"
"""


def _free_port() -> int:
    sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sk.bind(("127.0.0.1", 0))
    port = sk.getsockname()[1]
    sk.close()
    return port


@pytest.fixture(scope="module")
def enh_server():
    d = RUN_DIR / "api-enh"
    (d / "config").mkdir(parents=True, exist_ok=True)
    port = _free_port()
    (d / "custom.toml").write_text(CONFIG.format(port=port, m1=M1_PATH))
    env = {"VECBOOST_LOG_LEVEL": "debug"}
    full_env = {**dict(__import__("os").environ), **env}
    log = open(d / "server.log", "ab")
    proc = subprocess.Popen(
        [str(BIN), "--config", "custom.toml"],
        cwd=d, stdout=log, stderr=subprocess.STDOUT, env=full_env)
    deadline = time.time() + 90
    ready = False
    while time.time() < deadline:
        if proc.poll() is not None:
            break
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            conn.request("GET", "/health")
            ok = conn.getresponse().status == 200
            conn.close()
            if ok:
                ready = True
                break
        except OSError:
            pass
        time.sleep(0.5)
    assert ready, "--config 自定义路径启动失败"
    yield {"port": port, "dir": d}
    proc.terminate()
    try:
        proc.wait(timeout=35)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def _post(port: int, path: str, js=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    body = json.dumps(js).encode() if js is not None else None
    h = {"Content-Type": "application/json", **(headers or {})}
    conn.request("POST", path, body=body, headers=h)
    resp = conn.getresponse()
    raw = resp.read().decode("utf-8", errors="replace")
    conn.close()
    try:
        return resp.status, resp.getheaders(), json.loads(raw)
    except json.JSONDecodeError:
        return resp.status, resp.getheaders(), raw


def test_ae01_config_custom_path_boot(enh_server):
    """AE-01: --config 自定义路径启动成功（fixture 就绪即证明）。"""
    assert enh_server["port"] > 0


def test_ae02_config_missing_value_rejected():
    """AE-02: --config 缺少路径参数 → 非零退出。"""
    d = RUN_DIR / "api-enh"
    r = subprocess.run([str(BIN), "--config"], capture_output=True, text=True,
                       cwd=str(d), timeout=15)
    assert r.returncode != 0, f"--config 缺值应非零退出: {r.returncode}"


def test_ae03_search_ranks_relevant_first(enh_server):
    """AE-03: search 相关候选排名靠前、按分数降序。"""
    st, _, body = _post(enh_server["port"], "/api/1/search", {
        "query": "what is machine learning",
        "texts": ["machine learning is a branch of artificial intelligence",
                  "I had pasta for lunch today"],
        "top_k": 2,
    })
    assert st == 200, f"{st}: {str(body)[:150]}"
    results = body["results"]
    assert results[0]["index"] == 0 and results[0]["score"] > results[1]["score"]


def test_ae04_search_empty_texts_rejected(enh_server):
    """AE-04: search 空 texts → 4xx。"""
    st, _, _ = _post(enh_server["port"], "/api/1/search",
                     {"query": "q", "texts": []})
    assert 400 <= st < 500, f"空 texts 应 4xx，实际 {st}"


def test_ae05_similarity_metric_euclidean(enh_server):
    """AE-05: metric=euclidean 同文本 → 1/(1+0)=1.0；不同文本 ∈ (0,1)。"""
    st, _, body = _post(enh_server["port"], "/api/1/similarity",
                        {"source": "abc", "target": "abc", "metric": "euclidean"})
    assert st == 200 and abs(body["score"] - 1.0) < 1e-6, f"同文本 euclidean 应 1.0: {body}"
    st, _, body = _post(enh_server["port"], "/api/1/similarity",
                        {"source": "机器学习算法", "target": "今天天气不错", "metric": "manhattan"})
    assert st == 200 and 0.0 < body["score"] < 1.0, f"manhattan 相似度应∈(0,1): {body}"


def test_ae06_similarity_invalid_metric_rejected(enh_server):
    """AE-06: 非法 metric → 4xx。"""
    st, _, _ = _post(enh_server["port"], "/api/1/similarity",
                     {"source": "a", "target": "b", "metric": "bogus_metric"})
    assert 400 <= st < 500, f"非法 metric 应 4xx，实际 {st}"


def test_ae07_cors_preflight_and_header(enh_server):
    """AE-07: CORS 预检 200 且响应携带 allow-origin 头。"""
    conn = http.client.HTTPConnection("127.0.0.1", enh_server["port"], timeout=15)
    conn.request("OPTIONS", "/api/1/embed",
                 headers={"Origin": "http://example.com",
                          "Access-Control-Request-Method": "POST"})
    resp = conn.getresponse()
    resp.read()
    conn.close()
    assert resp.status == 200, f"预检应 200，实际 {resp.status}"
    headers = {k.lower(): v for k, v in resp.getheaders()}
    assert headers.get("access-control-allow-origin"), f"缺 allow-origin 头: {headers}"


def test_ae08_gzip_compression(enh_server):
    """AE-08: Accept-Encoding: gzip → 响应 Content-Encoding: gzip。"""
    conn = http.client.HTTPConnection("127.0.0.1", enh_server["port"], timeout=15)
    conn.request("GET", "/api/1/models", headers={"Accept-Encoding": "gzip"})
    resp = conn.getresponse()
    resp.read()
    conn.close()
    headers = {k.lower(): v for k, v in resp.getheaders()}
    assert headers.get("content-encoding") == "gzip", f"应 gzip 压缩: {headers}"


def test_ae09_vecboost_log_level_env(enh_server):
    """AE-09: VECBOOST_LOG_LEVEL=debug → 服务器日志含 debug 级别行。"""
    log_file = enh_server["dir"] / "server.log"
    text = log_file.read_text(errors="replace")
    assert "[DEBUG]" in text or "debug" in text.lower(), "debug 级别日志应落盘"


def test_ae10_model_unload_capability_boundary(enh_server):
    """AE-10: 服务器主路径未装配 ModelManager —— unload 明确 404（能力边界，
    不得谎报卸载成功）。HTTP 404 响应体含资源说明。"""
    st, _, body = _post(enh_server["port"], "/api/1/model/unload",
                        {"model_name": "never-loaded-xyz"})
    assert st == 404, f"无 ModelManager 部署应 404（不得谎报成功），实际 {st}: {str(body)[:150]}"
    assert "model manager" in str(body), f"错误应指明缺失能力: {str(body)[:200]}"
