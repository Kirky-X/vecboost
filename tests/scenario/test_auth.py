"""认证场景（R-auth-001 ~ R-auth-008）。auth 配置档（9103）。

凭据来源：测试专用凭证统一由 conftest 常量提供（VECBOOST_ADMIN_PASSWORD 环境变量注入服务器），
本文件不出现明文密码字面量；错误密码由正确常量派生。
"""
from __future__ import annotations

import pytest

from conftest import (
    ADMIN_PASS, ADMIN_USER, AUTH_ENV, M1_PATH,
    http_get, http_post, find_vector, make_config, probe_expect_fail,
    spawn_server, stop_server,
)
import time

WRONG_PASSWORD = ADMIN_PASS + "-wrong-suffix"


def _token_or_skip(port) -> str:
    """登录拿 token。"""
    st, body = http_post(port, "/api/1/auth/login",
                         {"username": ADMIN_USER, "password": ADMIN_PASS})
    assert st == 200, f"login 返回 {st}: {str(body)[:200]}"
    return body["token"]


def test_r001_login_success_tokens(auth_server):
    """R-auth-001: 正确凭证登录 → token + token_type + expires_in。"""
    port = auth_server["port"]
    st, body = http_post(port, "/api/1/auth/login",
                         {"username": ADMIN_USER, "password": ADMIN_PASS})
    assert st == 200, f"login 返回 {st}: {str(body)[:200]}"
    assert body.get("token"), f"缺 token: {str(body)[:150]}"
    assert body.get("token_type") == "Bearer", f"token_type 非 Bearer: {str(body)[:150]}"
    assert body.get("expires_in"), f"缺 expires_in: {str(body)[:150]}"


def test_r002_bearer_embed(auth_server):
    """R-auth-002: Bearer token 调 /embed → 200。"""
    port = auth_server["port"]
    token = _token_or_skip(port)
    st, body = http_post(port, "/api/1/embed", {"text": "认证后嵌入"}, token=token)
    assert st == 200, f"带 token embed 返回 {st}: {str(body)[:150]}"
    assert find_vector(body), "未返回向量"


def test_r003_refresh_and_logout_revocation(auth_server):
    """R-auth-003: refresh 换新 token 可用；logout 后旧 token 401。"""
    port = auth_server["port"]
    st, body = http_post(port, "/api/1/auth/login",
                         {"username": ADMIN_USER, "password": ADMIN_PASS})
    assert st == 200
    access = body["token"]
    # 先验证新 token 可用：refresh 换新 token
    st2, r2 = http_post(port, "/api/1/auth/refresh", {"refresh_token": access})
    assert st2 == 200, f"refresh 返回 {st2}: {str(r2)[:150]}"
    new_access = r2.get("token")
    assert new_access, "refresh 未返回新 token"
    st3, _ = http_post(port, "/api/1/embed", {"text": "新token"}, token=new_access)
    assert st3 == 200, "refresh 后的新 token 不可用"
    # logout 新 token，验证撤销后不可用
    st4, _ = http_post(port, "/api/1/auth/logout", {}, token=new_access)
    assert st4 in (200, 204), f"logout 返回 {st4}"
    st5, _ = http_post(port, "/api/1/embed", {"text": "登出后复用"}, token=new_access)
    assert st5 == 401, f"logout 后 token 仍可用（返回 {st5}）"


def test_r004_auth_me(auth_server):
    """R-auth-004: /auth/me 返回登录用户信息。"""
    port = auth_server["port"]
    token = _token_or_skip(port)
    st, body = http_get(port, "/api/1/auth/me", token=token)
    assert st == 200, f"me 返回 {st}: {str(body)[:150]}"
    assert ADMIN_USER in str(body), f"me 未返回用户名: {str(body)[:150]}"


def test_r005_public_paths_no_token(auth_server):
    """R-auth-005: /health 免 token；login/refresh 端点可达（非 404/405）。"""
    port = auth_server["port"]
    st, _ = http_get(port, "/health")
    assert st == 200, f"/health 被拦截: {st}"
    st2, _ = http_post(port, "/api/1/auth/login",
                       {"username": ADMIN_USER, "password": ADMIN_PASS})
    assert st2 not in (404, 405), f"login 端点不可达: {st2}"


def test_r006_unauthenticated_rejected(auth_server):
    """R-auth-006: 无 token/伪造 token → 401。"""
    port = auth_server["port"]
    st, body = http_post(port, "/api/1/embed", {"text": "x"})
    assert st == 401, f"无 token 返回 {st}（预期 401）: {str(body)[:150]}"
    st2, _ = http_post(port, "/api/1/embed", {"text": "x"}, token="forged.token.value")
    assert st2 == 401, f"伪造 token 返回 {st2}"


def test_r007_wrong_password(auth_server):
    """R-auth-007: 错误密码 → 401，服务不崩溃。"""
    port = auth_server["port"]
    st, body = http_post(port, "/api/1/auth/login",
                         {"username": ADMIN_USER, "password": WRONG_PASSWORD})
    assert st == 401, f"错误密码返回 {st}（预期 401）"
    st2, _ = http_get(port, "/health")
    assert st2 == 200, "错误密码后服务不健康"


def test_r008_startup_requires_jwt_secret():
    """R-auth-008: auth 开启但缺/短 VECBOOST_JWT_SECRET → 拒绝启动。"""
    cfg = make_config(9130, model_path="/nonexistent-unused", auth=True)
    ok1 = probe_expect_fail("auth_nosecret", 9130, cfg, env_extra={"VECBOOST_JWT_SECRET": ""}, timeout=45)
    assert ok1, "缺 JWT secret 仍启动成功（安全缺陷）"
    ok2 = probe_expect_fail("auth_shortsecret", 9130,
                            make_config(9130, model_path="/nonexistent-unused", auth=True),
                            env_extra={"VECBOOST_JWT_SECRET": "short"}, timeout=45)
    assert ok2, "短 JWT secret 仍启动成功（安全缺陷）"


def test_r009_invalid_refresh_token_rejected(auth_server):
    """R-auth-009: 无效/伪造 refresh_token → 4xx（不得 5xx），服务存活。"""
    port = auth_server["port"]
    for label, tok in (("随机串", "not-a-real-refresh-token-0123456789"),
                       ("空串", "")):
        st, body = http_post(port, "/api/1/auth/refresh", {"refresh_token": tok})
        assert 400 <= st < 500, f"{label} refresh_token 返回 {st}（预期 4xx）: {str(body)[:150]}"
    st2, _ = http_get(port, "/health")
    assert st2 == 200, "无效 refresh 后服务不健康"


def test_r010_malformed_authorization_header_rejected(auth_server):
    """R-auth-010: 畸形 Authorization 头——缺失 scheme/未知 scheme/空 Bearer
    → 401 且不崩溃。"""
    port = auth_server["port"]
    import http.client
    import json as _json
    from conftest import ALLOWED_HOST
    for label, header in (("无scheme", "justarawtoken"),
                          ("未知scheme", "Basic YWRtaW46YWRtaW4="),
                          ("空Bearer", "Bearer "),
                          ("空Bearer无空格", "Bearer")):
        conn = http.client.HTTPConnection(ALLOWED_HOST, port, timeout=15)
        conn.request("POST", "/api/1/embed",
                     body=_json.dumps({"text": "malformed auth header probe"}).encode(),
                     headers={"Content-Type": "application/json", "Authorization": header})
        resp = conn.getresponse()
        resp.read()
        conn.close()
        assert resp.status == 401, f"{label}: 返回 {resp.status}（预期 401）"
    st2, _ = http_get(port, "/health")
    assert st2 == 200, "畸形 Authorization 后服务不健康"


def test_r011_token_expiration_seconds_graceful():
    """R-auth-011: token_expiration_seconds 秒级过期——登录可用、
    过期后 401、服务存活。刷新 token 与 access token 同生命周期。"""
    port = 9133
    # token_expiration_seconds 属 [auth] 段:插在 [database] 段之前
    cfg = make_config(port, model_path=M1_PATH, auth=True).replace(
        "[database]", "token_expiration_seconds = 4\n\n[database]")
    s = spawn_server("auth_expiry", port, cfg, env_extra=AUTH_ENV, timeout=60)
    try:
        st, body = http_post(port, "/api/1/auth/login",
                             {"username": ADMIN_USER, "password": ADMIN_PASS})
        assert st == 200, f"login 应 200: {st}: {str(body)[:150]}"
        token = body["token"]
        expires_in = body.get("expires_in")
        assert isinstance(expires_in, int) and 0 < expires_in <= 10, \
            f"expires_in 应为秒级(≤10): {expires_in}"
        st2, _ = http_post(port, "/api/1/embed", {"text": "fresh token"}, token=token)
        assert st2 == 200, f"未过期 token 应可用: {st2}"
        time.sleep(6)
        st3, body3 = http_post(port, "/api/1/embed", {"text": "expired token"}, token=token)
        assert st3 == 401, f"过期 token 应 401: {st3}: {str(body3)[:150]}"
        st4, _ = http_get(port, "/health")
        assert st4 == 200, "过期后服务应存活"
    finally:
        stop_server(s)
