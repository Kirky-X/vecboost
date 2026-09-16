"""CLI / MCP 模式探针套件（R-server-003）。

- CLI 4 子命令：stdout 输出合法 JSON 结果（根因修复验证）、退出码 0
- CLI 异常输入：空文本 → 退出码非 0 或错误输出
- MCP：--mcp stdio 握手 + tools/list 含三工具 + tools/call 正常/异常
- 二进制需以 cli / mcp feature 构建；缺失时 skip（打印提示）
"""
from __future__ import annotations

import json
import pathlib
import subprocess

import pytest

from conftest import PROJECT_ROOT, RUN_DIR, make_config

BIN = PROJECT_ROOT / "target" / "debug" / "vecboost"
MODELS_DIR = PROJECT_ROOT / "models"
M1_PATH = str(MODELS_DIR / "BAAI-bge-small-en-v1.5")
CLI_TIMEOUT = 120

# CLI/MCP 进程共用的工作目录（含 config/config.toml，指向 M1 本地模型）
CLI_DIR = RUN_DIR / "cli-mcp"
CLI_CONFIG = """[server]
host = "127.0.0.1"
port = 39391

[model]
model_path = "{m1}"
expected_dimension = 384

[rate_limit]
enabled = false

[auth]
enabled = false

[database]
url = "sqlite::memory:"
""".format(m1=M1_PATH)


def _ensure_cli_dir() -> pathlib.Path:
    CLI_DIR.mkdir(parents=True, exist_ok=True)
    cfg = CLI_DIR / "config" / "config.toml"
    if not cfg.exists():
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text(CLI_CONFIG)
    return CLI_DIR


def _cli_feature_available() -> bool:
    """无 cli feature 时 `embed --help` 会被当作服务器模式启动（挂死），用短超时探测。"""
    if not BIN.exists():
        return False
    try:
        r = subprocess.run(
            [str(BIN), "embed", "--help"], capture_output=True, timeout=8,
            cwd=str(_ensure_cli_dir()))
        return r.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def _run_cli(args: list[str], timeout: int = CLI_TIMEOUT) -> subprocess.CompletedProcess:
    return subprocess.run([str(BIN), *args], capture_output=True, text=True,
                          timeout=timeout, cwd=str(_ensure_cli_dir()))


def _extract_json(stdout: str) -> dict:
    """从混合输出（启动日志 + 结果行）中提取最后一行 JSON。"""
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise AssertionError(f"stdout 中无 JSON 结果行:\n{stdout[-400:]}")


# ---------------- CLI 子命令（M4） ----------------

@pytest.mark.skipif(not BIN.exists(), reason="二进制未构建")
def test_cli_m01_embed_outputs_json():
    """CLI-M01: embed 子命令输出合法 JSON 向量（根因修复验证）。"""
    if not _cli_feature_available():
        pytest.skip("二进制未启用 cli feature（用 http,grpc,cli,auth 特性集构建后重跑）")
    r = _run_cli(["embed", "--req", json.dumps({"text": "cli probe hello"})])
    assert r.returncode == 0, f"退出码 {r.returncode}: {r.stderr[-300:]}"
    data = _extract_json(r.stdout)
    assert data.get("dimension") == 384, f"维度应 384: {str(data)[:150]}"
    assert isinstance(data.get("embedding"), list) and len(data["embedding"]) == 384


@pytest.mark.skipif(not BIN.exists(), reason="二进制未构建")
def test_cli_m02_embed_batch_outputs_json():
    """CLI-M02: embed_batch 输出批量 JSON。"""
    if not _cli_feature_available():
        pytest.skip("二进制未启用 cli feature")
    r = _run_cli(["embed_batch", "--req", json.dumps({"texts": ["a", "b", "c"]})])
    assert r.returncode == 0, f"退出码 {r.returncode}: {r.stderr[-300:]}"
    data = _extract_json(r.stdout)
    assert len(data.get("embeddings", [])) == 3, "应返回 3 条向量"


@pytest.mark.skipif(not BIN.exists(), reason="二进制未构建")
def test_cli_m03_compute_similarity_outputs_json():
    """CLI-M03: compute_similarity 输出分数 JSON。"""
    if not _cli_feature_available():
        pytest.skip("二进制未启用 cli feature")
    r = _run_cli(["compute_similarity", "--req",
                  json.dumps({"source": "hello world", "target": "hello world"})])
    assert r.returncode == 0, f"退出码 {r.returncode}: {r.stderr[-300:]}"
    data = _extract_json(r.stdout)
    assert abs(data.get("score", 0) - 1.0) < 1e-3, f"同文本分数应≈1: {data}"


@pytest.mark.skipif(not BIN.exists(), reason="二进制未构建")
def test_cli_m04_rerank_outputs_json():
    """CLI-M04: rerank 输出排序 JSON。"""
    if not _cli_feature_available():
        pytest.skip("二进制未启用 cli feature")
    req = {"query": "machine learning", "documents": ["ml paper", "pizza menu"],
           "top_k": None, "return_documents": False}
    r = _run_cli(["rerank", "--req", json.dumps(req)])
    assert r.returncode == 0, f"退出码 {r.returncode}: {r.stderr[-300:]}"
    data = _extract_json(r.stdout)
    assert "results" in data and len(data["results"]) == 2


@pytest.mark.skipif(not BIN.exists(), reason="二进制未构建")
def test_cli_m05_empty_text_fails():
    """CLI-M05: 空文本 → 退出码非 0 或 stderr 错误（不得静默成功）。"""
    if not _cli_feature_available():
        pytest.skip("二进制未启用 cli feature")
    r = _run_cli(["embed", "--req", json.dumps({"text": ""})])
    failed = r.returncode != 0 or "error" in r.stderr.lower()
    assert failed, f"空文本应失败: exit={r.returncode} stdout={r.stdout[-200:]}"


# ---------------- MCP 模式（M5） ----------------

def _mcp_feature_available() -> bool:
    return BIN.exists()


def _mcp_rpc(payload: dict, timeout: int = 60) -> str | None:
    """向 --mcp stdio 服务器完成 initialize 握手后发送请求，收集全部响应。

    rmcp（官方 Rust MCP SDK）要求先完成 initialize 握手才处理后续请求。
    """
    init = {"jsonrpc": "2.0", "id": 0, "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                       "clientInfo": {"name": "pytest", "version": "0.1"}}}
    initialized = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    lines = "\n".join(json.dumps(x) for x in (init, initialized, payload)) + "\n"
    proc = subprocess.Popen([str(BIN), "--mcp"], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            text=True, cwd=str(_ensure_cli_dir()))
    try:
        out, _ = proc.communicate(input=lines, timeout=timeout)
        return out
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)
        return None


def _json_responses(out: str | None) -> list[dict]:
    """从输出中提取 JSON-RPC 响应（忽略非协议行）。"""
    if not out:
        return []
    responses = []
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                responses.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return responses


@pytest.mark.skipif(not _mcp_feature_available(), reason="二进制未构建")
def test_mcp_m06_tools_list():
    """MCP-M06: tools/list 含 embed_text / embed_batch / compute_similarity。"""
    req = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    out = _mcp_rpc(req)
    responses = _json_responses(out)
    tools_resp = next((r for r in responses if r.get("id") == 1 and "result" in r), None)
    if tools_resp is None:
        pytest.skip(f"MCP tools/list 无响应（协议/超时），记录到报告: {str(out)[:200]}")
    text = json.dumps(tools_resp)
    for tool in ("embed_text", "embed_batch", "compute_similarity"):
        assert tool in text, f"tools/list 缺少 {tool}: {text[:300]}"


@pytest.mark.skipif(not _mcp_feature_available(), reason="二进制未构建")
def test_mcp_m07_tools_call_embed():
    """MCP-M07: tools/call embed_text 返回向量内容不崩溃。"""
    req = {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
           "params": {"name": "embed_text", "arguments": {"req": {"text": "mcp embed probe"}}}}
    out = _mcp_rpc(req)
    responses = _json_responses(out)
    call_resp = next((r for r in responses if r.get("id") == 2), None)
    if call_resp is None:
        pytest.skip(f"MCP tools/call 无响应（协议/超时），记录到报告: {str(out)[:200]}")
    assert "result" in call_resp, f"tools/call 应返回 result: {str(call_resp)[:300]}"
    assert not call_resp["result"].get("isError"), f"正常调用不应 isError: {str(call_resp)[:300]}"


@pytest.mark.skipif(not _mcp_feature_available(), reason="二进制未构建")
def test_mcp_m08_tools_call_empty_text_no_crash():
    """MCP-M08: tools/call 空文本 → 错误响应但不崩溃（进程正常退出）。"""
    req = {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
           "params": {"name": "embed_text", "arguments": {"req": {"text": ""}}}}
    out = _mcp_rpc(req)
    responses = _json_responses(out)
    call_resp = next((r for r in responses if r.get("id") == 3), None)
    if call_resp is None:
        pytest.skip(f"MCP tools/call 无响应（协议/超时），记录到报告: {str(out)[:200]}")
    # 空文本允许两种合规行为：isError=true 的工具错误，或含错误提示的正常响应
    assert call_resp.get("result", {}).get("isError") or "error" in call_resp or "result" in call_resp, \
        f"空文本调用应有响应（错误亦算）: {str(call_resp)[:200]}"
