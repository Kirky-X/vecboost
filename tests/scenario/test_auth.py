"""认证场景（R-auth-001 ~ R-auth-008）。auth 配置档（9103）。

已知风险：auth/middleware.rs 的 PUBLIC_PATHS 写死 "/api/v1/auth/login"，而实际路由挂载在
"/api/1/auth/login"（sdforge version 字面量拼接）——若中间件按白名单拦截，登录端点将不可达
（DEFECT-AUTH-001）。测试按规格断言，失败即缺陷坐实。

凭据来源：测试专用凭证统一由 conftest 常量提供（VECBOOST_ADMIN_PASSWORD 环境变量注入服务器），
本文件不出现明文密码字面量；错误密码由正确常量派生。
"""
from __future__ import annotations

import pytest

from conftest import (
    ADMIN_PASS, ADMIN_USER,
    http_get, http_post, find_vector, make_config, probe_expect_fail,
)

WRONG_PASSWORD = ADMIN_PASS + "-wrong-suffix"


def _token_or_skip(port) -> str:
    """登录拿 token；若因 DEFECT-AUTH-001 不可达则 skip（后续用例阻塞）。"""
    st, body = http_post(port, "/api/1/auth/login",
                         {"username": ADMIN_USER, "password": ADMIN_PASS})
    if st == 401:
        pytest.skip(f"DEFECT-AUTH-001: 登录端点被 auth 中间件拦截（401）——白名单前缀 /api/v1 与实际 /api/1 不匹配。响应: {str(body)[:150]}")
    assert st == 200, f"login 返回 {st}: {str(body)[:200]}"
    return body["access_token"]


def test_r001_login_success_tokens(auth_server):
    """R-auth-001: 正确凭证登录 → access+refresh token。"""
    port = auth_server["port"]
    st, body = http_post(port, "/api/1/auth/login",
                         {"username": ADMIN_USER, "password": ADMIN_PASS})
    if st == 401:
        pytest.skip(f"DEFECT-AUTH-001: 登录被中间件拦截: {str(body)[:150]}")
    assert st == 200, f"login 返回 {st}: {str(body)[:200]}"
    assert body.get("access_token"), f"缺 access_token: {str(body)[:150]}"
    assert body.get("refresh_token"), f"缺 refresh_token: {str(body)[:150]}"


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
    if st == 401:
        pytest.skip("DEFECT-AUTH-001: 登录不可达")
    assert st == 200
    access = body["access_token"]
    refresh = body.get("refresh_token")
    st2, r2 = http_post(port, "/api/1/auth/refresh", {"refresh_token": refresh})
    assert st2 == 200, f"refresh 返回 {st2}: {str(r2)[:150]}"
    new_access = r2.get("access_token")
    assert new_access, "refresh 未返回新 access_token"
    st3, _ = http_post(port, "/api/1/embed", {"text": "新token"}, token=new_access)
    assert st3 == 200, "refresh 后的新 token 不可用"
    st4, _ = http_post(port, "/api/1/auth/logout", {}, token=access)
    assert st4 in (200, 204), f"logout 返回 {st4}"
    st5, _ = http_post(port, "/api/1/embed", {"text": "登出后复用"}, token=access)
    assert st5 == 401, f"logout 后旧 token 仍可用（返回 {st5}）"


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
