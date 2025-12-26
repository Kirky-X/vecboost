"""
真实模型集成测试

使用 sentence-transformers 库进行真实的文本向量化运算测试。
需要安装 sentence-transformers: pip install sentence-transformers

运行方式:
    python -m tests.test_api_real
    或
    pytest tests/test_api_real.py -v
"""

import os
import sys
from typing import Any, Dict, List
from dataclasses import dataclass

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.config import get_model_name, get_model_dimension
from tests.api_simulator import BaseAPISimulator, APIClient, create_api_client

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


@dataclass
class CaseData:
    """测试用例数据类"""
    name: str
    description: str
    endpoint: str
    method: str
    request: Dict[str, Any]
    expected_status_code: int
    expected_response_keys: List[str]
    validate_response: Any = None


class RealEmbeddingService:
    """真实向量化服务"""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or get_model_name()
        self.model = None
        self.dimension = get_model_dimension()
        self._initialize_model()

    def _initialize_model(self):
        """初始化模型"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("警告: sentence-transformers 未安装")
            self.model = None
            return

        try:
            print(f"正在加载模型: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"模型加载成功，输出维度: {self.dimension}")
        except Exception as e:
            print(f"警告: 模型加载失败: {e}")
            self.model = None

    def embed(self, text: str) -> Dict[str, Any]:
        """生成文本向量"""
        if self.model is None:
            from tests.services import TestEmbeddingService
            return TestEmbeddingService.embed(text)

        embeddings = self.model.encode(text, normalize_embeddings=True)
        embedding = embeddings.tolist()

        return {
            "embedding": embedding,
            "dimension": len(embedding),
        }

    def embed_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """批量生成文本向量"""
        if self.model is None:
            from tests.services import TestEmbeddingService
            return [TestEmbeddingService.embed(text) for text in texts]

        embeddings = self.model.encode(texts, normalize_embeddings=True)

        return [
            {
                "embedding": emb.tolist(),
                "dimension": len(emb),
            }
            for emb in embeddings
        ]

    def similarity(self, text1: str, text2: str) -> Dict[str, Any]:
        """计算两个文本的相似度"""
        emb1 = self.embed(text1)["embedding"]
        emb2 = self.embed(text2)["embedding"]

        dot_product = sum(a * b for a, b in zip(emb1, emb2))
        score = round(dot_product, 6)

        return {
            "score": score,
            "metric": "cosine",
        }

    @property
    def is_available(self) -> bool:
        """检查模型是否可用"""
        return self.model is not None


@pytest.fixture(scope="module")
def embedding_service() -> RealEmbeddingService:
    """提供真实向量化服务实例"""
    return RealEmbeddingService()


class RealModelAPISimulator(BaseAPISimulator):
    """使用真实模型的 API 模拟器"""

    def __init__(self, embedding_service: RealEmbeddingService):
        self.service = embedding_service

    def embed(self, request: Dict[str, Any]):
        text = request.get("text", "")

        if not text:
            return 400, {
                "error": "Invalid input",
                "message": "text field is required and cannot be empty",
            }

        if len(text) > 10000:
            return 400, {
                "error": "Invalid input",
                "message": "text length exceeds maximum allowed length (10000 characters)",
            }

        result = self.service.embed(text)
        return 200, result

    def similarity(self, request: Dict[str, Any]):
        source = request.get("source", "")
        target = request.get("target", "")

        if not source:
            return 400, {
                "error": "Invalid input",
                "message": "source field is required and cannot be empty",
            }

        if not target:
            return 400, {
                "error": "Invalid input",
                "message": "target field is required and cannot be empty",
            }

        if len(source) > 10000 or len(target) > 10000:
            return 400, {
                "error": "Invalid input",
                "message": "text length exceeds maximum allowed length (10000 characters)",
            }

        result = self.service.similarity(source, target)
        return 200, result


@pytest.fixture
def api_simulator(embedding_service: RealEmbeddingService) -> RealModelAPISimulator:
    """提供使用真实模型的 API 模拟器"""
    return RealModelAPISimulator(embedding_service)


@pytest.fixture
def api_client(api_simulator: RealModelAPISimulator) -> APIClient:
    """提供 API 客户端"""
    return create_api_client(api_simulator)


@pytest.mark.skipif(
    not SENTENCE_TRANSFORMERS_AVAILABLE,
    reason="sentence-transformers not installed"
)
class TestRealModelEmbed:
    """真实模型文本向量化测试"""

    def test_embed_dimension(
        self, api_client: Any, embedding_service: RealEmbeddingService
    ):
        """验证向量维度与模型配置一致"""
        if not embedding_service.is_available:
            pytest.skip("模型未加载")

        status_code, response = api_client.post(
            "/api/v1/embed", {"text": "Test text for dimension validation"}
        )

        assert status_code == 200
        assert response["dimension"] == embedding_service.dimension
        assert len(response["embedding"]) == embedding_service.dimension

    def test_embed_consistency(
        self, api_client: Any, embedding_service: RealEmbeddingService
    ):
        """验证相同文本生成相同的向量"""
        if not embedding_service.is_available:
            pytest.skip("模型未加载")

        text = "Consistency test text"
        status_code1, response1 = api_client.post("/api/v1/embed", {"text": text})
        status_code2, response2 = api_client.post("/api/v1/embed", {"text": text})

        assert status_code1 == 200
        assert status_code2 == 200
        assert response1["embedding"] == response2["embedding"]

    def test_embed_normalization(
        self, api_client: Any, embedding_service: RealEmbeddingService
    ):
        """验证向量已归一化"""
        if not embedding_service.is_available:
            pytest.skip("模型未加载")

        import math

        status_code, response = api_client.post(
            "/api/v1/embed", {"text": "Normalization test"}
        )

        assert status_code == 200
        embedding = response["embedding"]
        magnitude = sum(x**2 for x in embedding)
        assert abs(magnitude - 1.0) < 0.001


@pytest.mark.skipif(
    not SENTENCE_TRANSFORMERS_AVAILABLE,
    reason="sentence-transformers not installed"
)
class TestRealModelSimilarity:
    """真实模型相似度计算测试"""

    def test_similarity_identical(
        self, api_client: Any, embedding_service: RealEmbeddingService
    ):
        """验证相同文本的相似度为1.0"""
        if not embedding_service.is_available:
            pytest.skip("模型未加载")

        status_code, response = api_client.post(
            "/api/v1/similarity", {"source": "Hello world", "target": "Hello world"}
        )

        assert status_code == 200
        assert abs(response["score"] - 1.0) < 0.001

    def test_similarity_different(
        self, api_client: Any, embedding_service: RealEmbeddingService
    ):
        """验证不同文本的相似度小于1.0"""
        if not embedding_service.is_available:
            pytest.skip("模型未加载")

        status_code, response = api_client.post(
            "/api/v1/similarity",
            {"source": "Hello world", "target": "Goodbye world"},
        )

        assert status_code == 200
        assert response["score"] < 1.0

    def test_similarity_range(
        self, api_client: Any, embedding_service: RealEmbeddingService
    ):
        """验证相似度在有效范围内"""
        if not embedding_service.is_available:
            pytest.skip("模型未加载")

        test_cases = [
            {"source": "Apple", "target": "Apple"},
            {"source": "Apple", "target": "Fruit"},
            {"source": "Apple", "target": "Computer"},
        ]

        for case in test_cases:
            status_code, response = api_client.post("/api/v1/similarity", case)
            assert status_code == 200
            assert -1.0 <= response["score"] <= 1.0


class TestTestFallback:
    """测试服务回退测试"""

    def test_embed_with_test(
        self, api_client: Any, embedding_service: RealEmbeddingService
    ):
        """当模型不可用时使用测试服务"""
        from tests.services import TestEmbeddingService

        if embedding_service.is_available:
            # 如果模型可用，验证 API 仍然正常工作
            status_code, response = api_client.post(
                "/api/v1/embed", {"text": "Test text"}
            )
            assert status_code == 200
            assert "embedding" in response
            assert "dimension" in response
        else:
            # 模型不可用时，验证测试服务正常工作
            status_code, response = api_client.post(
                "/api/v1/embed", {"text": "Test text"}
            )
            assert status_code == 200
            assert "embedding" in response
            assert response["dimension"] == TestEmbeddingService.get_dimension()


def run_real_model_tests():
    """运行真实模型测试的主函数"""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("错误: sentence-transformers 未安装")
        print("请安装: pip install sentence-transformers")
        return

    model_name = get_model_name()
    print(f"使用模型: {model_name}")
    print("=" * 60)

    service = RealEmbeddingService(model_name)

    if not service.is_available:
        print("错误: 模型加载失败")
        return

    print(f"模型维度: {service.dimension}")
    print()

    test_cases = create_test_cases(service.dimension)

    print("运行测试用例...")
    print("=" * 60)

    passed = 0
    failed = 0

    simulator = RealModelAPISimulator(service)

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n测试 {i}/{len(test_cases)}: {test_case.name}")

        status_code, response = simulator.handle_request(
            test_case.endpoint, test_case.method, test_case.request
        )

        if status_code == test_case.expected_status_code:
            if test_case.validate_response:
                if callable(test_case.validate_response):
                    result = test_case.validate_response(response)
                else:
                    result = True
            else:
                result = True

            if result:
                print(f"  ✅ 通过")
                passed += 1
            else:
                print(f"  ❌ 失败: 验证未通过")
                failed += 1
        else:
            print(f"  ❌ 失败: 期望状态码 {test_case.expected_status_code}, 实际 {status_code}")
            failed += 1

    print()
    print("=" * 60)
    print(f"测试完成: {passed} 通过, {failed} 失败")
    print("=" * 60)


def create_test_cases(model_dimension: int = 384) -> List[CaseData]:
    """创建测试用例集"""
    test_cases: List[CaseData] = []

    test_cases.append(CaseData(
        name="TC-HEALTH-001: 健康检查",
        description="验证健康检查接口返回正常状态",
        endpoint="/health",
        method="GET",
        request={},
        expected_status_code=200,
        expected_response_keys=["status", "service", "version"],
    ))

    test_cases.append(CaseData(
        name="TC-EMBED-001: 文本向量化",
        description="验证文本向量化返回正确的维度",
        endpoint="/api/v1/embed",
        method="POST",
        request={"text": "Test text"},
        expected_status_code=200,
        expected_response_keys=["embedding", "dimension"],
        validate_response=lambda r: r.get("dimension", 0) == model_dimension
    ))

    test_cases.append(CaseData(
        name="TC-EMBED-002: 短文本向量化",
        description="验证短文本向量化",
        endpoint="/api/v1/embed",
        method="POST",
        request={"text": "AI"},
        expected_status_code=200,
        expected_response_keys=["embedding", "dimension"],
    ))

    test_cases.append(CaseData(
        name="TC-EMBED-003: 空文本",
        description="验证空文本返回错误",
        endpoint="/api/v1/embed",
        method="POST",
        request={"text": ""},
        expected_status_code=400,
        expected_response_keys=["error", "message"],
    ))

    test_cases.append(CaseData(
        name="TC-EMBED-004: 缺失字段",
        description="验证缺失text字段返回错误",
        endpoint="/api/v1/embed",
        method="POST",
        request={},
        expected_status_code=400,
        expected_response_keys=["error", "message"],
    ))

    test_cases.append(CaseData(
        name="TC-EMBED-005: 中文文本",
        description="验证中文文本向量化",
        endpoint="/api/v1/embed",
        method="POST",
        request={"text": "你好世界"},
        expected_status_code=200,
        expected_response_keys=["embedding", "dimension"],
    ))

    test_cases.append(CaseData(
        name="TC-SIM-001: 相同文本相似度",
        description="验证相同文本相似度为1.0",
        endpoint="/api/v1/similarity",
        method="POST",
        request={"source": "Hello world", "target": "Hello world"},
        expected_status_code=200,
        expected_response_keys=["score", "metric"],
        validate_response=lambda r: abs(r.get("score", 0) - 1.0) < 0.001
    ))

    test_cases.append(CaseData(
        name="TC-SIM-002: 不同文本相似度",
        description="验证不同文本相似度小于1.0",
        endpoint="/api/v1/similarity",
        method="POST",
        request={"source": "Hello world", "target": "Goodbye world"},
        expected_status_code=200,
        expected_response_keys=["score", "metric"],
        validate_response=lambda r: 0.0 <= r.get("score", -1) <= 1.0 and r.get("score", 1) < 1.0
    ))

    return test_cases


if __name__ == "__main__":
    run_real_model_tests()
