"""
VecBoost API 接口测试

使用 Mock 服务测试 API 接口的各个功能：
- /health - 健康检查接口
- /api/v1/embed - 文本向量化接口
- /api/v1/similarity - 相似度计算接口
"""

import pytest
from typing import Any
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.services import MockEmbeddingService
from tests.api_simulator import BaseAPISimulator, APIClient, create_api_client
from tests.conftest import model_dimension


class MockAPISimulator(BaseAPISimulator):
    """Mock API 模拟器"""

    pass


@pytest.fixture
def api_simulator() -> MockAPISimulator:
    """提供 API 模拟器实例"""
    return MockAPISimulator(MockEmbeddingService)


@pytest.fixture
def api_client(api_simulator: MockAPISimulator) -> APIClient:
    """提供 API 客户端，用于发送测试请求"""
    return create_api_client(api_simulator)


class TestHealthEndpoint:
    """健康检查接口测试"""

    def test_health_check_normal(self, api_client: Any):
        """TC-HEALTH-001: 验证健康检查接口正常工作"""
        status_code, response = api_client.get("/health")

        assert status_code == 200
        assert response["status"] == "healthy"
        assert response["service"] == "vecboost"
        assert response["version"] == "0.1.0"
        assert "timestamp" in response


class TestEmbedEndpoint:
    """文本向量化接口测试"""

    def test_embed_normal_request(self, api_client: Any, model_dimension: int):
        """TC-EMBED-001: 验证正常文本可以成功向量化"""
        status_code, response = api_client.post(
            "/api/v1/embed", {"text": "Hello, world! This is a test."}
        )

        assert status_code == 200
        assert "embedding" in response
        assert "dimension" in response
        assert response["dimension"] == model_dimension
        assert len(response["embedding"]) == model_dimension

    def test_embed_short_text(self, api_client: Any, short_text: str, model_dimension: int):
        """TC-EMBED-002: 验证短文本可以成功向量化"""
        status_code, response = api_client.post("/api/v1/embed", {"text": short_text})

        assert status_code == 200
        assert "embedding" in response
        assert "dimension" in response
        assert response["dimension"] == model_dimension
        assert len(response["embedding"]) == model_dimension

    def test_embed_empty_text(self, api_client: Any):
        """TC-EMBED-003: 验证空文本返回错误"""
        status_code, response = api_client.post("/api/v1/embed", {"text": ""})

        assert status_code == 400
        assert "error" in response
        assert "message" in response

    def test_embed_missing_text_field(self, api_client: Any):
        """TC-EMBED-004: 验证缺失text字段返回错误"""
        status_code, response = api_client.post("/api/v1/embed", {})

        assert status_code == 400
        assert "error" in response
        assert "message" in response

    def test_embed_special_characters(
        self, api_client: Any, special_char_text: str, model_dimension: int
    ):
        """TC-EMBED-005: 验证包含特殊字符的文本向量化"""
        status_code, response = api_client.post(
            "/api/v1/embed", {"text": special_char_text}
        )

        assert status_code == 200
        assert "embedding" in response
        assert "dimension" in response
        assert response["dimension"] == model_dimension
        assert len(response["embedding"]) == model_dimension

    def test_embed_chinese_text(
        self, api_client: Any, chinese_text: str, model_dimension: int
    ):
        """TC-EMBED-006: 验证中文文本的向量化"""
        status_code, response = api_client.post("/api/v1/embed", {"text": chinese_text})

        assert status_code == 200
        assert "embedding" in response
        assert "dimension" in response
        assert response["dimension"] == model_dimension
        assert len(response["embedding"]) == model_dimension

    def test_embed_long_text(self, api_client: Any, long_text: str, model_dimension: int):
        """TC-EMBED-007: 验证长文本的向量化"""
        status_code, response = api_client.post("/api/v1/embed", {"text": long_text})

        assert status_code == 200
        assert "embedding" in response
        assert "dimension" in response
        assert response["dimension"] == model_dimension
        assert len(response["embedding"]) == model_dimension

    def test_embed_dimension_consistency(
        self, api_client: Any, model_dimension: int
    ):
        """TC-EMBED-008: 验证不同文本返回的向量维度一致"""
        texts = ["Short text", "A slightly longer text for testing", "中文测试文本"]

        for text in texts:
            status_code, response = api_client.post("/api/v1/embed", {"text": text})

            assert status_code == 200
            assert response["dimension"] == model_dimension
            assert len(response["embedding"]) == model_dimension

    def test_embed_max_length_text(self, api_client: Any):
        """TC-EMBED-009: 验证最大长度文本可以处理"""
        max_length_text = "a" * 10000

        status_code, response = api_client.post(
            "/api/v1/embed", {"text": max_length_text}
        )

        assert status_code == 200
        assert "embedding" in response
        assert "dimension" in response

    def test_embed_exceeds_max_length(self, api_client: Any):
        """TC-EMBED-010: 验证超过最大长度的文本返回错误"""
        exceeded_text = "a" * 10001

        status_code, response = api_client.post(
            "/api/v1/embed", {"text": exceeded_text}
        )

        assert status_code == 400
        assert "error" in response
        assert "exceeds" in response.get("message", "").lower()


class TestSimilarityEndpoint:
    """相似度计算接口测试"""

    def test_similarity_identical_texts(self, api_client: Any):
        """TC-SIM-001: 验证相同文本的相似度为1.0"""
        status_code, response = api_client.post(
            "/api/v1/similarity", {"source": "Hello world", "target": "Hello world"}
        )

        assert status_code == 200
        assert "score" in response
        assert "metric" in response
        assert response["metric"] == "cosine"
        assert abs(response["score"] - 1.0) < 0.001

    def test_similarity_different_texts(self, api_client: Any):
        """TC-SIM-002: 验证不同文本的相似度小于1.0"""
        status_code, response = api_client.post(
            "/api/v1/similarity", {"source": "Hello world", "target": "Goodbye world"}
        )

        assert status_code == 200
        assert "score" in response
        assert 0.0 <= response["score"] < 1.0

    def test_similarity_missing_source(self, api_client: Any):
        """TC-SIM-003: 验证缺失源文本返回错误"""
        status_code, response = api_client.post(
            "/api/v1/similarity", {"source": "", "target": "Some text"}
        )

        assert status_code == 400
        assert "error" in response
        assert "message" in response

    def test_similarity_missing_target(self, api_client: Any):
        """TC-SIM-004: 验证缺失目标文本返回错误"""
        status_code, response = api_client.post(
            "/api/v1/similarity", {"source": "Some text", "target": ""}
        )

        assert status_code == 400
        assert "error" in response
        assert "message" in response

    def test_similarity_missing_both_fields(self, api_client: Any):
        """TC-SIM-005: 验证两个字段都缺失时返回错误"""
        status_code, response = api_client.post(
            "/api/v1/similarity", {"source": "", "target": ""}
        )

        assert status_code == 400
        assert "error" in response
        assert "message" in response

    def test_similarity_empty_request_body(self, api_client: Any):
        """TC-SIM-006: 验证空请求体返回错误"""
        status_code, response = api_client.post("/api/v1/similarity", {})

        assert status_code == 400
        assert "error" in response
        assert "message" in response

    def test_similarity_chinese_texts(self, api_client: Any):
        """TC-SIM-007: 验证中文文本相似度计算"""
        status_code, response = api_client.post(
            "/api/v1/similarity", {"source": "你好世界", "target": "你好中国"}
        )

        assert status_code == 200
        assert "score" in response
        assert "metric" in response
        assert -1.0 <= response["score"] <= 1.0

    def test_similarity_similar_chinese_phrases(self, api_client: Any):
        """TC-SIM-008: 验证相似中文短语的相似度计算"""
        status_code, response = api_client.post(
            "/api/v1/similarity",
            {
                "source": "机器学习是人工智能的子领域",
                "target": "机器学习属于人工智能范畴",
            },
        )

        assert status_code == 200
        assert "score" in response
        assert "metric" in response
        assert -1.0 <= response["score"] <= 1.0

    def test_similarity_score_range(self, api_client: Any):
        """TC-SIM-009: 验证相似度分数在有效范围内"""
        test_cases = [
            {"source": "Hello", "target": "World"},
            {"source": "Same text", "target": "Same text"},
            {"source": "Apple fruit", "target": "Banana fruit"},
        ]

        for case in test_cases:
            status_code, response = api_client.post("/api/v1/similarity", case)

            assert status_code == 200
            assert -1.0 <= response["score"] <= 1.0


class TestErrorHandling:
    """错误处理测试"""

    def test_404_not_found(self, api_client: Any):
        """TC-BOUNDARY-001: 验证访问不存在的端点返回404"""
        status_code, response = api_client.post("/api/v1/nonexistent", {})

        assert status_code == 404
        assert "error" in response
        assert "message" in response

    def test_405_method_not_allowed(self, api_client: Any):
        """TC-BOUNDARY-002: 验证使用错误的HTTP方法返回405"""
        status_code, response = api_client.get("/api/v1/embed")

        assert status_code == 404
        assert "error" in response

    def test_400_bad_request_empty_body(self, api_client: Any):
        """TC-BOUNDARY-003: 验证空请求体触发正确的错误"""
        status_code, response = api_client.post("/api/v1/embed", None)

        assert status_code == 400
        assert "error" in response
