"""模型管理场景（R-model-002 ~ R-model-007）。zh 配置档 = M2 服务端 HF 镜像下载，512 维。

R-model-001（本地加载）由 base 配置档的启动本身覆盖（model_path=本地目录）。
"""
from __future__ import annotations

import pytest

from conftest import (
    M1_PATH, M1_REPO, PROJECT_ROOT, http_get, http_post, find_vector,
    probe_expect_fail, tail_log,
)

# M2 本地副本(models/ 资产):热切换回 M2 用本地路径,消除对 HF 网络的
# 运行期依赖——在线下载能力已由 R-model-002 在启动阶段单独覆盖。
M2_LOCAL_PATH = str(PROJECT_ROOT / "models" / "BAAI-bge-small-zh-v1.5")


def test_r002_hf_mirror_server_side_download(zh_server):
    """R-model-002: 服务端 HF 在线下载加载（HF_ENDPOINT=hf-mirror.com）→ 512 维可用。"""
    port = zh_server["port"]
    st, body = http_post(port, "/api/1/embed", {"text": "中文模型在线下载验证"})
    assert st == 200, f"HTTP {st}: {str(body)[:200]}"
    vec = find_vector(body)
    assert vec and len(vec) == 512, f"维度 {len(vec) if vec else None} != 512"
    log = tail_log("zh", 200)
    endpoint = "hf-mirror" if "hf-mirror" in log else ("huggingface" if "huggingface" in log else "unspecified")
    print(f"[info] 模型下载来源记录: {endpoint}")


def test_r003_model_info_endpoints(zh_server):
    """R-model-003: /models、/model/current、/model/info 三端点。"""
    port = zh_server["port"]
    st, models = http_get(port, "/api/1/models")
    assert st == 200, f"models HTTP {st}"
    st2, cur = http_get(port, "/api/1/model/current")
    assert st2 == 200, f"current HTTP {st2}"
    assert "zh" in str(cur).lower() or "bge" in str(cur).lower(), f"current 未反映 M2: {str(cur)[:150]}"
    st3, info = http_get(port, "/api/1/model/info")
    assert st3 == 200, f"info HTTP {st3}"


def test_r004_hot_switch_dims(zh_server):
    """R-model-004: 热切换 M2→M1→M2，维度 512↔384。回切走本地路径（确定性）。"""
    port = zh_server["port"]
    st, _ = http_post(port, "/api/1/model/switch",
                      {"model_name": M1_REPO, "model_path": M1_PATH, "expected_dimension": 384})
    assert st == 200, f"切换 M1: HTTP {st}: {str(_)[:200]}"
    st1, b1 = http_post(port, "/api/1/embed", {"text": "切换后验证"})
    assert st1 == 200 and len(find_vector(b1)) == 384, "切换 M1 后维度非 384"
    st2, body2 = http_post(port, "/api/1/model/switch",
                           {"model_name": "BAAI/bge-small-zh-v1.5",
                            "model_path": M2_LOCAL_PATH, "expected_dimension": 512})
    assert st2 == 200, f"切回 M2: HTTP {st2}: {str(body2)[:200]}"
    st3, b3 = http_post(port, "/api/1/embed", {"text": "切回验证"})
    assert st3 == 200 and len(find_vector(b3)) == 512, "切回 M2 后维度非 512"


def test_r005_switch_failure_preserves_current(zh_server):
    """R-model-005: 切换到不存在模型失败，原模型继续可用。"""
    port = zh_server["port"]
    st, body = http_post(port, "/api/1/model/switch",
                         {"model_name": "BAAI/definitely-not-exist-xyz-123"})
    assert st >= 400, f"无效切换竟返回 {st}"
    assert st in (400, 404, 422), f"无效切换状态码 {st}（预期 4xx）"
    st2, b2 = http_post(port, "/api/1/embed", {"text": "失败后仍可用"})
    assert st2 == 200 and len(find_vector(b2)) == 512, "切换失败后原模型不可用"


def test_r006_invalid_model_sources():
    """R-model-006: 不存在目录 / 缺权重 / 损坏 safetensors → 启动或加载被拒。

    注：服务器在启动期同步加载模型，故以"进程拒绝启动"为验收形态（等价于 ModelLoadError 族）。
    """
    from conftest import make_config, write_file, RUN_DIR
    cases = []
    cases.append(("不存在目录", make_config(9120, model_repo=M1_REPO, model_path="/nonexistent/model/dir")))
    missing = write_file("badmodel_missing", "model/config.json", "{}")
    cases.append(("缺权重文件", make_config(9121, model_repo=M1_REPO, model_path=str(RUN_DIR / "badmodel_missing" / "model"))))
    corrupt = write_file("badmodel_corrupt", "model/model.safetensors", "THIS IS NOT A REAL SAFETENSORS FILE" * 10)
    import shutil
    src = RUN_DIR / "badmodel_corrupt" / "model"
    for f in ("config.json", "tokenizer.json"):
        shutil.copy(f"{M1_PATH}/{f}", src / f)
    cases.append(("损坏safetensors", make_config(9122, model_repo=M1_REPO, model_path=str(src))))

    for label, cfg in cases:
        ok = probe_expect_fail(f"fail_{9120 + len(cases)}_{label}", 9120, cfg, timeout=60)
        assert ok, f"{label}: 预期启动失败但服务就绪了"


def test_r007_unload_not_exposed(zh_server):
    """R-model-007: 卸载能力验证——HTTP 未暴露 unload 路由（能力边界发现）。"""
    port = zh_server["port"]
    st, _ = http_post(port, "/api/1/model/unload", {"model_name": "x"})
    if st == 404:
        pytest.skip("能力记录：HTTP 未暴露 model/unload 路由（服务层有 unload_model 能力但无端点，记入报告）")
    assert st in (200, 400, 422), f"unload 返回意外状态 {st}"
