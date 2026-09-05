"""配置与设备场景（R-config-001 ~ R-config-006）。"""
from __future__ import annotations

import os
import time

import pytest

from conftest import (
    M1_PATH, M1_REPO, RUN_DIR, http_get, http_post, find_vector,
    make_config, probe_expect_fail, spawn_server, stop_server,
)


def test_r001_config_file_port(base_server):
    """R-config-001: base 配置档指定端口 9101 生效（非默认 3000）。"""
    st, _ = http_get(base_server["port"], "/health")
    assert st == 200 and base_server["port"] == 9101, "配置档端口未生效"


def test_r002_env_overrides_port():
    """R-config-002: VECBOOST_SERVER_PORT=9156 覆盖配置文件端口 9155。"""
    try:
        s = spawn_server("cfg_env", 9156,
                         make_config(9155, model_path=M1_PATH),
                         env_extra={"VECBOOST_SERVER_PORT": "9156"}, timeout=60)
    except RuntimeError as e:
        # 环境变量名可能不同：探测 9155 是否被按配置启动（=覆盖不生效，发现）
        from conftest import http_get as g
        try:
            st, _ = g(9155, "/health", timeout=2)
            if st == 200:
                pytest.fail("环境变量 VECBOOST_SERVER_PORT 未生效：服务按配置文件 9155 启动（发现 R-config-002）")
        except OSError:
            pass
        pytest.fail(f"env 覆盖后服务未启动: {e}")
    stop_server(s)


def test_r003_no_config_file_boot():
    """R-config-003: 无配置文件目录启动（Default 链路）。

    完整 Default 会下载 bge-m3（2.3GB），代价高；默认以 env 指定 M1 验证"无配置文件+
    环境变量"链路。重负载全量验证由 RUN_HEAVY=1 控制另行执行。
    """
    candidates = [
        {"VECBOOST_MODEL_MODEL_REPO": M1_REPO},
        {"VECBOOST_MODEL__MODEL_REPO": M1_REPO},
        {"VECBOOST_MODEL_REPO": M1_REPO},
    ]
    for i, env in enumerate(candidates):
        try:
            s = spawn_server(f"cfg_nofile_{i}", 9157 + i, "#[placeholder]\n", env_extra=env, timeout=60)
        except RuntimeError:
            continue
        st, body = http_get(s["port"], "/api/1/model/current")
        stop_server(s)
        assert st == 200, f"无配置文件启动后 current 不可用: {st}"
        print(f"[info] 无配置文件启动成功，生效 env: {list(env)[0]}")
        return
    pytest.skip("发现记录：三个候选 env 变量名均未生效，无配置文件启动需下载 bge-m3（默认跳过，记入报告 R-config-003）")


def test_r004_invalid_configs_rejected():
    """R-config-004: 非法端口 99999 / batch_size=0 → 拒绝启动。"""
    bad_port_cfg = "[server]\nhost = \"127.0.0.1\"\nport = 99999\n"
    ok1 = probe_expect_fail("cfg_badport", 9160, bad_port_cfg, timeout=30)
    assert ok1, "非法端口 99999 仍启动成功"
    bad_batch_cfg = (
        "[server]\nhost = \"127.0.0.1\"\nport = 9161\n\n"
        f"[model]\nmodel_path = \"{M1_PATH}\"\nbatch_size = 0\nuse_gpu = false\n"
    )
    ok2 = probe_expect_fail("cfg_badbatch", 9161, bad_batch_cfg, timeout=30)
    assert ok2, "batch_size=0 仍启动成功"


def test_r005_no_gpu_fallback(nogpu_server):
    """R-config-005: 无 GPU 环境（CUDA_VISIBLE_DEVICES=""）全功能可用。"""
    port = nogpu_server["port"]
    st, body = http_post(port, "/api/1/embed", {"text": "无 GPU 回退验证"})
    assert st == 200, f"HTTP {st}: {str(body)[:200]}"
    vec = find_vector(body)
    assert vec and len(vec) == 384, "回退后嵌入不可用"
    log = ""
    f = RUN_DIR / "nogpu" / "server.log"
    if f.exists():
        log = f.read_text(errors="replace")
    print("[info] GPU 回退日志关键词:",
          [k for k in ("cpu", "fallback", "回退", "CUDA") if k in log] or "无")


def test_r006_device_field_reported(base_server):
    """R-config-006: 设备信息可观测性——/model/info 无 device 字段（能力缺口记录），
    但启动日志明确报告设备（Using CPU / GPU），以日志为准。"""
    st, body = http_get(base_server["port"], "/api/1/model/info")
    assert st == 200, f"info HTTP {st}: {str(body)[:150]}"
    if "device" not in str(body).lower():
        # 日志证据：启动日志含 "Using CPU"/"Using FP32"
        from conftest import tail_log
        log = tail_log("base", 100)
        assert "Using CPU" in log or "Using GPU" in log or "cuda" in log.lower(), \
            "模型信息端点与日志均未报告设备（R-config-006 失败）"
        print("[info] 能力记录：/model/info 无 device 字段（发现），设备经启动日志报告")
