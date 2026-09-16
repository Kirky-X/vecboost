"""服务器模式场景（R-server-001 ~ R-server-008）。

gRPC：sdforge 免 proto，统一服务 sdforge.v1.SdForgeService/Call，
CallRequest{method:string=1, parameters:map<string,string>=2, data:string=3}，
CallResponse{success:bool=1, data:string=2, error:string=3, status_code:int32=4}。
本文件手写最小 protobuf 编解码（无需 protoc）。

CLI/MCP/嵌入式 library 的进程探针由 scripts/run-scenario-tests.sh 编排执行，
产物写入 tests/scenario/run/modes/，本套件校验产物（独立运行时 skip 并提示）。
"""
from __future__ import annotations

import concurrent.futures
import json
import pathlib
import socket
import struct
import time

import grpc
import pytest

from conftest import RUN_DIR, M1_PATH, http_get, http_post, find_vector, make_config, probe_lifecycle

MODES = RUN_DIR / "modes"


# ---------------- protobuf 最小编解码 ----------------

def _varint(n: int) -> bytes:
    out = b""
    while True:
        b = n & 0x7F
        n >>= 7
        out += bytes((b | (0x80 if n else 0),))
        if not n:
            return out


def _string_field(tag: int, s: str) -> bytes:
    key = (tag << 3) | 2
    data = s.encode("utf-8")
    return _varint(key) + _varint(len(data)) + data


def _map_field(tag: int, d: dict[str, str]) -> bytes:
    out = b""
    for k, v in d.items():
        entry = _string_field(1, k) + _string_field(2, v)
        out += _varint((tag << 3) | 2) + _varint(len(entry)) + entry
    return out


def encode_call_request(method: str, parameters: dict[str, str], data: str) -> bytes:
    return _string_field(1, method) + _map_field(2, parameters) + _string_field(3, data)


def decode_call_response(payload: bytes) -> dict:
    out = {"success": False, "data": "", "error": "", "status_code": 0}
    i = 0
    while i < len(payload):
        key = 0
        shift = 0
        while True:
            b = payload[i]
            i += 1
            key |= (b & 0x7F) << shift
            shift += 7
            if not (b & 0x80):
                break
        tag, wt = key >> 3, key & 7
        if wt == 0:
            val, shift = 0, 0
            while True:
                b = payload[i]
                i += 1
                val |= (b & 0x7F) << shift
                shift += 7
                if not (b & 0x80):
                    break
            out[{1: "success", 4: "status_code"}.get(tag, f"f{tag}")] = bool(val) if tag == 1 else val
        elif wt == 2:
            ln, shift = 0, 0
            while True:
                b = payload[i]
                i += 1
                ln |= (b & 0x7F) << shift
                shift += 7
                if not (b & 0x80):
                    break
            raw = payload[i:i + ln]
            i += ln
            if tag in (2, 3):
                out["data" if tag == 2 else "error"] = raw.decode("utf-8", errors="replace")
        else:
            raise ValueError(f"unsupported wire type {wt}")
    return out


def grpc_call(port: int, method: str, data: str, token: str | None = None, timeout: float = 30):
    """一次 sdforge Call 调用。返回 (grpc_status_code_or_0, CallResponse_dict_or_None, error_msg)。"""
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    try:
        # 关键：grpcio 对 bytes 消息自动添加 5 字节 gRPC 帧前缀（压缩标志+长度）。
        # 旧实现手动再包一层帧导致服务端解出 "invalid tag value: 0"（双重封装缺陷）。
        unary = channel.unary_unary(
            "/sdforge.v1.SdForgeService/Call",
            request_serializer=lambda x: x,
            response_deserializer=lambda x: x,
        )
        payload = encode_call_request(method, {}, data)
        md = [("authorization", f"Bearer {token}")] if token else None
        try:
            # response_deserializer=identity：grpcio 已剥离帧前缀，raw 即 protobuf 消息
            raw = unary(payload, timeout=timeout, metadata=md)
            if isinstance(raw, tuple):
                raw = raw[0]
            return 0, decode_call_response(raw), ""
        except grpc.RpcError as e:
            return e.code().value[0] if hasattr(e.code(), "value") else -1, None, e.details() or str(e)
    finally:
        channel.close()


# ---------------- 场景 ----------------

def _free_port() -> int:
    import socket
    sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sk.bind(("127.0.0.1", 0))
    port = sk.getsockname()[1]
    sk.close()
    return port


def test_r003_grpc_unauthenticated_rejected():
    """R-server-003a: gRPC 无 token 调用被拒绝。"""
    from conftest import AUTH_ENV, M1_PATH, spawn_server, stop_server
    gport = _free_port()
    hport = _free_port()
    s = spawn_server("grpc", hport,
                     make_config(hport, model_path=M1_PATH, auth=True, grpc=True, grpc_port=gport),
                     env_extra=AUTH_ENV, timeout=60)
    s["grpc_port"] = gport
    last = None
    for method in ("vecboost.embed",):
        # gRPC 监听器可能晚于 HTTP health 就绪，重试穿透启动竞态
        for _ in range(20):
            code, resp, err = grpc_call(gport, method, json.dumps({"text": "grpc unauth"}))
            last = (method, code, resp, err)
            if code != 14:  # 14=UNAVAILABLE(连接拒绝)，非连接类结果即可判定
                break
            time.sleep(1)
        if code == 0 and resp is not None:
            assert not resp["success"], f"无 token 竟然调用成功: {resp}"
            stop_server(s)
            return
    stop_server(s)
    assert last and last[1] in (7, 16, 14, -1), f"gRPC 无 token 行为异常: {last}"


def test_r003_grpc_authenticated_embed():
    """R-server-003b: 带 JWT 的 gRPC embed 成功返回向量。"""
    from conftest import AUTH_ENV, ADMIN_PASS, M1_PATH, spawn_server, stop_server
    gport = _free_port()
    hport = _free_port()
    s = spawn_server("grpc", hport,
                     make_config(hport, model_path=M1_PATH, auth=True, grpc=True, grpc_port=gport),
                     env_extra=AUTH_ENV, timeout=60)
    s["grpc_port"] = gport
    st, body = http_post(s["port"], "/api/1/auth/login",
                         {"username": "admin", "password": ADMIN_PASS})
    if st != 200:
        stop_server(s)
        assert False, f"登录不可达（{st}），无法取 token"
    jwt = body["token"]
    successes = []
    for method in ("vecboost.embed",):
        # gRPC 监听器就绪重试（同 r003a）
        for _ in range(20):
            code, resp, err = grpc_call(gport, method, json.dumps({"text": "grpc embed test"}), token=jwt)
            if code != 14:
                break
            time.sleep(1)
        if code == 0 and resp and resp["success"]:
            vec = find_vector(json.loads(resp["data"])) if resp["data"].lstrip().startswith(("{", "[")) else None
            assert vec, f"成功但无向量: {str(resp)[:200]}"
            successes.append(method)
            break
    stop_server(s)
    assert successes, (
        f"所有候选 method 均失败: last_code={code} resp={str(resp)[:200]} err={str(err)[:200]}"
    )


def test_r002_concurrent_embeds(base_server):
    """R-server-002: 16 线程并发 embed 全部成功。"""
    port = base_server["port"]

    def one(i):
        st, body = http_post(port, "/api/1/embed", {"text": f"并发 {i} concurrency {i}"})
        return st, find_vector(body) is not None

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        results = list(ex.map(one, range(32)))
    assert all(st == 200 and ok for st, ok in results), \
        f"并发失败样例: {[r for r in results if r[0] != 200][:3]}"


def test_r007_metrics_prometheus_format(base_server):
    """R-server-007: /metrics 输出 Prometheus 文本格式（实测返回空 body → 缺陷）。"""
    st, body = http_get(base_server["port"], "/metrics")
    assert st == 200, f"/metrics HTTP {st}"
    text = body if isinstance(body, str) else json.dumps(body)
    assert text.strip(), "/metrics 返回空 body（Prometheus 导出未生效）"


def test_r001_graceful_shutdown_sigterm():
    """R-server-001: 启动→健康→SIGTERM → ≤35s 退出码 0。"""
    result = probe_lifecycle("lifecycle", 9131, make_config(9131, model_path=M1_PATH))
    assert result["health_ok"], "服务未就绪"
    assert result["exit_code"] == 0, f"退出码 {result['exit_code']}（预期 0），耗时 {result['seconds']}s"
    assert (result["seconds"] or 99) <= 35, "优雅关闭超时"


def test_r008_port_conflict_startup_failure():
    """R-server-008: 端口被占用 → 启动失败非 0 退出。"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 9132))
    s.listen(1)
    try:
        from conftest import probe_expect_fail
        failed = probe_expect_fail("conflict", 9132, make_config(9132, model_path=M1_PATH), timeout=45)
        assert failed, "端口被占用时服务仍启动成功"
    finally:
        s.close()


def test_r004_cli_mode_artifacts():
    """R-server-004: CLI 模式产物校验（由编排脚本生成）。"""
    f = MODES / "cli_embed.json"
    if not f.exists():
        pytest.skip("编排脚本未运行（缺 run/modes/cli_embed.json）——由 run-scenario-tests.sh 生成")
    data = json.loads(f.read_text())
    from conftest import find_vector as fv
    assert fv(data), f"CLI embed 输出无向量: {str(data)[:200]}"
    meta = MODES / "cli_rerank.meta"
    if meta.exists():
        assert meta.read_text().strip() == "0", "CLI rerank 退出码非 0"


def test_r005_mcp_mode_artifacts():
    """R-server-005: MCP stdio 产物校验。"""
    f = MODES / "mcp_tools_list.json"
    if not f.exists():
        pytest.skip("编排脚本未运行（缺 run/modes/mcp_tools_list.json）")
    data = json.loads(f.read_text())
    text = str(data)
    assert "embed_text" in text, f"tools/list 无 embed_text: {text[:200]}"


def test_r006_library_mode_artifacts():
    """R-server-006: 嵌入式 library 示例产物校验。"""
    f = MODES / "library.log"
    meta = MODES / "library.meta"
    if not f.exists():
        pytest.skip("编排脚本未运行（缺 run/modes/library.log）")
    text = f.read_text()
    if meta.exists() and meta.read_text().strip() != "0":
        pytest.fail(
            f"library_usage 示例崩溃（退出码 {meta.read_text().strip()}）——"
            f"示例在 async runtime 内再次 block_on。输出尾部: {text[-200:]}"
        )
    assert "SDK 示例完成" in text or "embedding" in text.lower(), f"library 示例输出异常: {text[:200]}"
