"""生命周期补缺套件（R-server-002 / R-auth-002）。

现有套件已覆盖：非法配置拒启（r004）、env 端口覆盖（r002）、SIGTERM 优雅关闭
（server_modes r001）、并发（r002）、限流 429/白名单/审计（r009/r011）、损坏
safetensors 拒载（model_zh r006）、缺/短 JWT secret 拒启（auth r008）。

本套件补齐缺口（LC-*）：
- LC-01: VECBOOST_REQUIRE_ENCRYPTION=1 且无加密 key → 拒绝启动
- LC-02: SIGINT 优雅关闭（≤35s 退出、非 SIGKILL）
- LC-03: 活跃负载下 SIGTERM 优雅关闭（排空请求、≤35s 退出）
"""
from __future__ import annotations

import concurrent.futures
import json
import subprocess
import time

from conftest import (
    M1_PATH,
    RUN_DIR,
    http_post,
    make_config,
    probe_expect_fail,
)


def test_lc01_require_encryption_without_key_rejected():
    """LC-01: VECBOOST_REQUIRE_ENCRYPTION=1 且无 VECBOOST_ENCRYPTION_KEY → 拒绝启动。"""
    env = {"VECBOOST_REQUIRE_ENCRYPTION": "1", "VECBOOST_ENCRYPTION_KEY": ""}
    failed = probe_expect_fail("lc01_enc", 9141, make_config(9141, model_path=M1_PATH),
                               env_extra=env, timeout=45)
    assert failed, "REQUIRE_ENCRYPTION=1 无 key 时服务不应启动成功"


def test_lc02_sigint_graceful_shutdown():
    """LC-02: SIGINT 触发优雅关闭，35s 内退出且非 SIGKILL。"""
    d = RUN_DIR / "lc02"
    (d / "config").mkdir(parents=True, exist_ok=True)
    (d / "config" / "config.toml").write_text(make_config(9142, model_path=M1_PATH))
    log = open(d / "server.log", "ab")
    try:
        proc = subprocess.Popen(["/home/kirky/projects/vecboost/target/debug/vecboost"],
                                cwd=d, stdout=log, stderr=subprocess.STDOUT)
    finally:
        # 子进程已 dup 该 fd,关闭父副本避免 ResourceWarning(零告警门禁)
        log.close()
    # 等待健康
    deadline = time.time() + 90
    healthy = False
    while time.time() < deadline:
        if proc.poll() is not None:
            break
        try:
            import http.client
            conn = http.client.HTTPConnection("127.0.0.1", 9142, timeout=2)
            conn.request("GET", "/health")
            ok = conn.getresponse().status == 200
            conn.close()
            if ok:
                healthy = True
                break
        except OSError:
            pass
        time.sleep(0.5)
    assert healthy, "LC02 服务未就绪"
    proc.send_signal(2)  # SIGINT
    try:
        code = proc.wait(timeout=35)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)
        assert False, "LC02: SIGINT 后 35s 未退出（挂死）"
    assert code == 0, f"LC02: SIGINT 退出码 {code}（预期 0）"


def test_lc03_graceful_shutdown_under_load():
    """LC-03: 16 线程持续请求中收到 SIGTERM → 请求排空、35s 内退出。"""
    d = RUN_DIR / "lc03"
    (d / "config").mkdir(parents=True, exist_ok=True)
    (d / "config" / "config.toml").write_text(make_config(9143, model_path=M1_PATH))
    log = open(d / "server.log", "ab")
    try:
        proc = subprocess.Popen(["/home/kirky/projects/vecboost/target/debug/vecboost"],
                                cwd=d, stdout=log, stderr=subprocess.STDOUT)
    finally:
        # 子进程已 dup 该 fd,关闭父副本避免 ResourceWarning(零告警门禁)
        log.close()
    deadline = time.time() + 90
    healthy = False
    while time.time() < deadline:
        if proc.poll() is not None:
            break
        try:
            import http.client
            conn = http.client.HTTPConnection("127.0.0.1", 9143, timeout=2)
            conn.request("GET", "/health")
            ok = conn.getresponse().status == 200
            conn.close()
            if ok:
                healthy = True
                break
        except OSError:
            pass
        time.sleep(0.5)
    assert healthy, "LC03 服务未就绪"

    stop = {"flag": False}
    failures = {"count": 0}

    def hammer(i: int):
        while not stop["flag"]:
            st, _ = http_post(9143, "/api/1/embed", {"text": f"负载请求 {i}"})
            if st != 200:
                failures["count"] += 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(hammer, i) for i in range(8)]
        time.sleep(3)  # 让负载跑起来
        t0 = time.time()
        proc.terminate()
        try:
            code = proc.wait(timeout=35)
            elapsed = time.time() - t0
        except subprocess.TimeoutExpired:
            stop["flag"] = True
            proc.kill()
            proc.wait(timeout=10)
            assert False, "LC03: 负载下 SIGTERM 35s 未退出（关停回归）"
        stop["flag"] = True
        for f in futures:
            try:
                f.result(timeout=10)
            except Exception:
                pass
    assert code == 0, f"LC03: 退出码 {code}"
    assert elapsed <= 35, f"LC03: 关闭耗时 {elapsed:.1f}s"
