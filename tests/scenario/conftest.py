"""full-scenario-testing — 黑盒场景测试骨架。

约定：
- 场景 ID 见 specmark/changes/full-scenario-testing/specs/*/spec.md，用例名以 R-<域>-NNN 开头对账。
- 每个"配置档 profile"是一个独立服务器进程：配置写入 tests/scenario/run/<name>/config/config.toml，
  以该目录为 CWD 启动编译产物二进制（应用从 CWD 读 config/config.toml）。
- 全部断言走 HTTP/进程行为的黑盒观测，不依赖内部状态。

安全说明（本文件为测试基础设施，目标全部为本机回环）：
- HTTP 客户端基于 http.client，主机为常量 ALLOWED_HOST("127.0.0.1")，端口为编译期常量并经
  check_port() 区间校验（1024-65535），路径为调用点字面量；不存在由外部输入构造的 URL。
- 子进程调用为调用点内联 argv 字面量列表 + shell=False，无任何 shell 字符串拼接。
"""
from __future__ import annotations

import http.client
import json
import os
import pathlib
import socket
import subprocess
import time

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
RUN_DIR = pathlib.Path(__file__).resolve().parent / "run"
M1_PATH = str(PROJECT_ROOT / "models" / "BAAI-bge-small-en-v1.5")
M1_REPO = "BAAI/bge-small-en-v1.5"
M2_REPO = "BAAI/bge-small-zh-v1.5"
JWT_SECRET = "scenario-test-jwt-secret-0123456789ABCDEF"
ADMIN_USER = "admin"
ADMIN_PASS = "Scenario#2026Pass"
HTTP_TIMEOUT = 30
ALLOWED_HOST = "127.0.0.1"


def check_port(port: int) -> int:
    p = int(port)
    if not (1024 <= p <= 65535):
        raise ValueError(f"port out of range: {port}")
    return p


def _request(port: int, method: str, path: str, body: bytes | None = None,
             headers: dict | None = None, timeout: float = HTTP_TIMEOUT):
    """向本机回环固定端口发送一次 HTTP 请求，返回 (status, parsed_json_or_text)。"""
    conn = http.client.HTTPConnection(ALLOWED_HOST, check_port(port), timeout=timeout)
    try:
        conn.request(method, path, body=body, headers=headers or {})
        resp = conn.getresponse()
        raw = resp.read().decode("utf-8", errors="replace")
        status = resp.status
    finally:
        conn.close()
    try:
        return status, json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return status, raw


def http_get(port: int, path: str, token: str | None = None, timeout: float = HTTP_TIMEOUT):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return _request(port, "GET", path, headers=headers, timeout=timeout)


def http_post(port: int, path: str, js=None, token: str | None = None,
              raw_body: str | None = None, timeout: float = HTTP_TIMEOUT):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = raw_body.encode("utf-8") if raw_body is not None else json.dumps(js).encode("utf-8")
    return _request(port, "POST", path, body=body, headers=headers, timeout=timeout)


# ---------------------------------------------------------------- config TOML

def make_config(
    port: int,
    *,
    model_repo: str | None = M1_REPO,
    model_path: str | None = None,
    dim: int = 384,
    use_gpu: bool = False,
    auth: bool = False,
    admin_user: str = ADMIN_USER,
    grpc: bool = False,
    grpc_port: int = 9151,
    rl_enabled: bool = True,
    rl_ip_rpm: int = 1000,
    rl_global_rpm: int = 1000,
    rl_whitelist: list[str] | None = None,
) -> str:
    lines = [
        "[server]",
        'host = "127.0.0.1"',
        f"port = {check_port(port)}",
        # /embed/file 与 /model/switch 的路径白名单（安全契约：显式配置整体替换默认根），
        # 覆盖场景产物目录与本地模型库
        f"grpc_allowed_roots = {json.dumps([str(RUN_DIR), str(PROJECT_ROOT / 'models')])}",
    ]
    if grpc:
        lines += [
            "grpc_enabled = true",
            f"grpc_port = {check_port(grpc_port)}",
            "grpc_require_auth = true",
        ]
    lines += ["", "[model]"]
    if model_repo:
        lines.append(f'model_repo = "{model_repo}"')
    if model_path:
        lines.append(f'model_path = "{model_path}"')
    lines += [
        f"use_gpu = {'true' if use_gpu else 'false'}",
        f"expected_dimension = {dim}",
        "",
        "[embedding]",
        "cache_enabled = true",
        "",
        "[rate_limit]",
        f"enabled = {'true' if rl_enabled else 'false'}",
        f"global_requests_per_minute = {rl_global_rpm}",
        f"ip_requests_per_minute = {rl_ip_rpm}",
        "window_secs = 60",
        f"ip_whitelist = {json.dumps(rl_whitelist if rl_whitelist is not None else ['127.0.0.1', '::1'])}",
        "",
        "[auth]",
        f"enabled = {'true' if auth else 'false'}",
    ]
    if auth:
        lines.append(f'default_admin_username = "{admin_user}"')
    lines += ["", "[database]", 'url = "sqlite::memory:"', ""]
    return "\n".join(lines)


AUTH_ENV = {
    "VECBOOST_JWT_SECRET": JWT_SECRET,
    "VECBOOST_ADMIN_PASSWORD": ADMIN_PASS,
}

# ---------------------------------------------------------------- process mgmt

def wait_health(port: int, timeout: float, proc: subprocess.Popen | None = None) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            status, _ = http_get(port, "/health", timeout=2)
            if status == 200:
                return True
        except OSError:
            pass
        time.sleep(0.5)
    return False


def spawn_server(name: str, port: int, config_text: str, env_extra: dict | None = None,
                 timeout: float = 90) -> dict:
    d = RUN_DIR / name
    (d / "config").mkdir(parents=True, exist_ok=True)
    (d / "config" / "config.toml").write_text(config_text)
    # 预检：端口必须空闲，避免打到陈旧实例造成假阳性
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("127.0.0.1", check_port(port)))
    except OSError:
        raise RuntimeError(f"port {port} 已被占用（疑似残留 vecboost 进程），请先清理") from None
    finally:
        # 占用与空闲两条路径都必须关闭 fd，否则 ResourceWarning 在零告警门禁下炸测试
        probe.close()
    env = os.environ.copy()
    env.update(env_extra or {})
    log = open(d / "server.log", "ab")
    try:
        proc = subprocess.Popen(
            ["/home/kirky/projects/vecboost/target/debug/vecboost"],
            cwd=d, stdout=log, stderr=subprocess.STDOUT, env=env, shell=False)
    finally:
        # 子进程已 dup 该 fd,关闭父副本避免 ResourceWarning(零告警门禁)
        log.close()
    if not wait_health(port, timeout, proc):
        tail = tail_log(name)
        stop_server({"proc": proc, "port": port})
        raise RuntimeError(f"server [{name}] 未在 {timeout}s 内就绪（port {port}）。\n日志尾部:\n{tail}")
    return {"proc": proc, "port": check_port(port), "name": name, "dir": d}


def stop_server(s: dict) -> None:
    proc = s.get("proc")
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=35)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def tail_log(name: str, n: int = 25) -> str:
    f = RUN_DIR / name / "server.log"
    if not f.exists():
        return "<no log>"
    return "\n".join(f.read_text(errors="replace").splitlines()[-n:])


def probe_expect_fail(name: str, port: int, config_text: str, env_extra: dict | None = None,
                      timeout: float = 45) -> bool:
    """启动一个预期失败的配置档，返回 True=确实未就绪（启动被拒绝）。进程保证被清理。"""
    try:
        spawn_server(name, port, config_text, env_extra=env_extra, timeout=timeout)
    except RuntimeError:
        return True
    except OSError:
        return True
    return False


def probe_lifecycle(name: str, port: int, config_text: str, env_extra: dict | None = None) -> dict:
    """启动→健康检查→SIGTERM，观测优雅关闭行为。返回 {health_ok, exit_code, seconds}。"""
    d = RUN_DIR / name
    (d / "config").mkdir(parents=True, exist_ok=True)
    (d / "config" / "config.toml").write_text(config_text)
    env = os.environ.copy()
    env.update(env_extra or {})
    log = open(d / "server.log", "ab")
    try:
        proc = subprocess.Popen(
            ["/home/kirky/projects/vecboost/target/debug/vecboost"],
            cwd=d, stdout=log, stderr=subprocess.STDOUT, env=env, shell=False)
    finally:
        log.close()
    health_ok = wait_health(port, 90, proc)
    result = {"health_ok": health_ok, "exit_code": None, "seconds": None}
    if not health_ok:
        proc.kill()
        proc.wait(timeout=10)
        return result
    t0 = time.time()
    proc.terminate()
    try:
        proc.wait(timeout=35)
        result["exit_code"] = proc.returncode
        result["seconds"] = round(time.time() - t0, 1)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)
        result["exit_code"] = "SIGKILL"
    return result


# ---------------------------------------------------------------- fixtures

@pytest.fixture(scope="session")
def base_server():
    """auth off / M1 本地目录 / CPU / 端口 9101。"""
    s = spawn_server("base", 9101, make_config(9101, model_path=M1_PATH))
    yield s
    stop_server(s)


@pytest.fixture(scope="session")
def zh_server():
    """M2 服务端 HF 在线下载 / 512 维 / 端口 9102。

    DEFECT-HUB-001：hf-hub 客户端对 hf-mirror.com 报 "missing ETag header"，
    镜像下载不可用 → 自动回退直连 huggingface.co；实际来源记录于 run/zh/endpoint.txt。
    """
    s = None
    for label, env in (("hf-mirror", {"HF_ENDPOINT": "https://hf-mirror.com"}),
                       ("direct", {})):
        try:
            s = spawn_server("zh", 9102,
                             make_config(9102, model_repo=M2_REPO, dim=512),
                             env_extra=env, timeout=600)
            (RUN_DIR / "zh" / "endpoint.txt").write_text(label)
            break
        except RuntimeError:
            continue
    if s is None:
        raise RuntimeError("zh server 启动失败（镜像与直连均失败）")
    yield s
    stop_server(s)


@pytest.fixture(scope="session")
def auth_server():
    """auth on + JWT env / 端口 9103。"""
    s = spawn_server(
        "auth", 9103,
        make_config(9103, model_path=M1_PATH, auth=True),
        env_extra=AUTH_ENV,
    )
    yield s
    stop_server(s)


@pytest.fixture(scope="session")
def rl_strict_server():
    """限流阈值 6/min 且白名单为空 → 本机也受限。端口 9104。"""
    s = spawn_server(
        "rl_strict", 9104,
        make_config(9104, model_path=M1_PATH, rl_ip_rpm=6, rl_global_rpm=6, rl_whitelist=[]),
    )
    yield s
    stop_server(s)


@pytest.fixture(scope="session")
def rl_pass_server():
    """限流阈值 6/min 但白名单含 127.0.0.1 → 本机直通。端口 9105。"""
    s = spawn_server(
        "rl_pass", 9105,
        make_config(9105, model_path=M1_PATH, rl_ip_rpm=6, rl_global_rpm=6),
    )
    yield s
    stop_server(s)


@pytest.fixture(scope="session")
def grpc_server():
    """HTTP 9106 + gRPC 9151，require_auth=true，auth on。"""
    s = spawn_server(
        "grpc", 9106,
        make_config(9106, model_path=M1_PATH, auth=True, grpc=True),
        env_extra=AUTH_ENV,
    )
    s["grpc_port"] = 9151
    yield s
    stop_server(s)


@pytest.fixture(scope="session")
def nogpu_server():
    """use_gpu=true + CUDA_VISIBLE_DEVICES="" → 模拟无 GPU 回退。端口 9107。"""
    s = spawn_server(
        "nogpu", 9107,
        make_config(9107, model_path=M1_PATH, use_gpu=True),
        env_extra={"CUDA_VISIBLE_DEVICES": ""},
    )
    yield s
    stop_server(s)


# ---------------------------------------------------------------- helpers

def find_list_of_vectors(obj):
    """在任意响应结构中递归定位 [[float,...], ...]。"""
    if isinstance(obj, list) and obj and isinstance(obj[0], list):
        return obj
    if isinstance(obj, list):
        for v in obj:
            r = find_list_of_vectors(v)
            if r:
                return r
    if isinstance(obj, dict):
        for v in obj.values():
            r = find_list_of_vectors(v)
            if r:
                return r
    return None


def find_scalar_score(obj):
    """递归定位第一个标量分数（float/int，非向量元素）。"""
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        return float(obj)
    if isinstance(obj, dict):
        for key in ("score", "similarity", "value"):
            if isinstance(obj.get(key), (int, float)):
                return float(obj[key])
        for v in obj.values():
            r = find_scalar_score(v)
            if r is not None:
                return r
    if isinstance(obj, list):
        for v in obj:
            r = find_scalar_score(v)
            if r is not None:
                return r
    return None


def find_vector(obj):
    """递归定位第一个 [float,...]（长度≥8 视为向量）。"""
    if isinstance(obj, list) and len(obj) >= 8 and all(isinstance(x, (int, float)) for x in obj[:8]):
        return obj
    if isinstance(obj, list):
        for v in obj:
            r = find_vector(v)
            if r:
                return r
    if isinstance(obj, dict):
        for v in obj.values():
            r = find_vector(v)
            if r:
                return r
    return None


def l2_norm(vec) -> float:
    return sum(x * x for x in vec) ** 0.5


def login_token(port: int, username: str = ADMIN_USER, password: str = ADMIN_PASS) -> str:
    st, body = http_post(port, "/api/1/auth/login", {"username": username, "password": password})
    assert st == 200, f"login 失败 {st}: {str(body)[:200]}"
    return body["token"]


def write_file(name: str, rel: str, content: str) -> str:
    d = RUN_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    p = d / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return str(p)
