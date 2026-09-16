"""语义缓存端到端套件（semantic-cache-e2e，SC-*）。

黑盒可观测性设计：语义缓存命中时返回的是**已缓存文本**的原始向量（未经当前文本
重算），因此"近似文本返回与原文本逐位相等的向量"是命中信号；对照组（无语义缓存的
base_server）同一文本对向量必然不同（引擎只对同文本确定）。

- SC-01: 精确重复请求逐位一致（第一级精确匹配）
- SC-02: trigram 高重叠近似文本返回缓存向量（第二级语义命中）
- SC-03: 对照组——无语义缓存服务器上同文本对向量不同（证明 SC-02 是缓存命中）
- SC-04: 低相似文本走重算路径，向量有效（384 维、非零、L2≈1）
- SC-05: comparison_mode 非法值拒绝启动（fail-fast，不静默回退）
"""
from __future__ import annotations

import pytest

from conftest import (
    PROJECT_ROOT, probe_expect_fail, http_post, find_vector, l2_norm,
)

DIM = 384
M1_PATH = str(PROJECT_ROOT / "models" / "BAAI-bge-small-en-v1.5")

BASE_TEXT = ("vecboost semantic cache probe sentence with sufficient length "
             "for trigram overlap analysis and stable jaccard measurement")
# 追加短后缀：trigram 集合高度重叠（Jaccard≈0.94），必然越过 0.6 阈值
NEAR_TEXT = BASE_TEXT + " now"
FAR_TEXT = "completely unrelated content about quantum computing hardware"

SC_CONFIG = """[server]
host = "127.0.0.1"
port = 9166

[model]
model_path = "{m1}"
expected_dimension = 384

[embedding]
cache_enabled = true

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
""".format(m1=M1_PATH)

SC_BAD_MODE_CONFIG = SC_CONFIG.replace('comparison_mode = "exact"',
                                       'comparison_mode = "bogus_mode"')


@pytest.fixture(scope="module")
def sc_server():
    """语义缓存服务器：threshold=0.6、pipeline 显式开启（端口 9166）。"""
    from conftest import spawn_server, stop_server
    s = spawn_server("semantic_cache", 9166, SC_CONFIG)
    yield s
    stop_server(s)


def _embed(port, text):
    st, body = http_post(port, "/api/1/embed", {"text": text})
    assert st == 200, f"HTTP {st}: {str(body)[:200]}"
    return find_vector(body)


def test_sc01_exact_repeat_bitwise_identical(sc_server):
    """SC-01: 精确匹配级——同文本重复请求逐位一致。"""
    port = sc_server["port"]
    v1 = _embed(port, BASE_TEXT)
    v2 = _embed(port, BASE_TEXT)
    assert v1 == v2, "精确重复请求向量逐位不一致"
    assert len(v1) == DIM and any(abs(x) > 1e-6 for x in v1)


def test_sc02_near_duplicate_returns_cached_vector(sc_server):
    """SC-02: 语义命中——trigram 高重叠的近似文本返回与原文本逐位相等的缓存向量。"""
    port = sc_server["port"]
    base_vec = _embed(port, BASE_TEXT)
    near_vec = _embed(port, NEAR_TEXT)
    assert near_vec == base_vec, (
        "近似文本应命中语义缓存并返回缓存向量（逐位相等），"
        "若逐位不同则说明走了重算路径（缓存未命中）"
    )


def test_sc03_control_without_semantic_cache_vectors_differ(base_server):
    """SC-03: 对照组——无语义缓存的 base_server 上，同文本对走重算，向量必然不同。"""
    port = base_server["port"]
    base_vec = _embed(port, BASE_TEXT)
    near_vec = _embed(port, NEAR_TEXT)
    assert base_vec != near_vec, (
        "无语义缓存时不同文本不应逐位相等（若相等说明引擎对异文同果，SC-02 信号失效）"
    )


def test_sc04_dissimilar_text_recomputed_valid(sc_server):
    """SC-04: 低相似文本走重算路径——向量有效（384 维、非零、L2≈1）且与缓存项不同。"""
    port = sc_server["port"]
    base_vec = _embed(port, BASE_TEXT)
    far_vec = _embed(port, FAR_TEXT)
    assert len(far_vec) == DIM
    assert any(abs(x) > 1e-6 for x in far_vec), "向量全零"
    assert abs(l2_norm(far_vec) - 1.0) < 0.05, f"L2={l2_norm(far_vec)} 偏离 1"
    assert far_vec != base_vec, "不相关文本不应返回缓存向量"


def test_sc05_invalid_comparison_mode_rejected_at_boot():
    """SC-05: comparison_mode 非法值——启动即拒绝（fail-fast）。"""
    assert probe_expect_fail("semantic_cache_badmode", 9167, SC_BAD_MODE_CONFIG), \
        "非法 comparison_mode 应拒绝启动"
