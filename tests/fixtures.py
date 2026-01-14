"""测试夹具

提供 pytest fixture 支持真实模型推理和 Mock 回退。
"""

import os
import sys
from typing import Any, Dict, List, Optional, Type

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.config import (
    get_api_base_url,
    get_model_dimension,
    get_model_name,
    get_test_mode,
    is_mock_mode,
    is_real_mode,
    TestMode,
)


class EmbeddingServiceInterface:
    """嵌入服务接口（Mock 和 Real 都实现此接口）"""

    @classmethod
    def get_dimension(cls) -> int:
        """获取模型输出维度"""
        raise NotImplementedError

    @classmethod
    def embed(cls, text: str) -> Dict[str, Any]:
        """生成文本向量"""
        raise NotImplementedError

    @classmethod
    def embed_batch(cls, texts: List[str]) -> List[Dict[str, Any]]:
        """批量生成文本向量"""
        raise NotImplementedError

    @classmethod
    def similarity(cls, text1: str, text2: str) -> Dict[str, Any]:
        """计算两个文本的相似度"""
        raise NotImplementedError


def get_embedding_service_class() -> Type[EmbeddingServiceInterface]:
    """根据测试模式返回对应的服务类"""
    mode = get_test_mode()

    if is_mock_mode():
        from tests.services import TestEmbeddingService

        return TestEmbeddingService
    else:
        # 使用真实服务
        try:
            from tests.real_service import RealEmbeddingService

            return RealEmbeddingService
        except ImportError:
            print("Warning: RealEmbeddingService not available, falling back to mock")
            from tests.services import TestEmbeddingService

            return TestEmbeddingService


class MockEmbeddingService:
    """Mock 嵌入服务（用于 fixture）"""

    def __init__(self):
        from tests.services import TestEmbeddingService

        self._service = TestEmbeddingService

    def get_dimension(self) -> int:
        return self._service.get_dimension()

    def embed(self, text: str) -> Dict[str, Any]:
        return self._service.embed(text)

    def embed_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        return self._service.embed_batch(texts)

    def similarity(self, text1: str, text2: str) -> Dict[str, Any]:
        return self._service.similarity(text1, text2)


class RealEmbeddingService:
    """真实嵌入服务（用于 fixture）"""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or get_api_base_url()
        self._session = None

    def _get_session(self):
        """获取 HTTP 会话（懒加载）"""
        if self._session is None:
            import requests

            self._session = requests.Session()
        return self._session

    def get_dimension(self) -> int:
        return get_model_dimension()

    def embed(self, text: str) -> Dict[str, Any]:
        import requests

        session = self._get_session()
        response = session.post(f"{self.base_url}/api/v1/embed", json={"text": text})
        response.raise_for_status()
        return response.json()

    def embed_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        import requests

        session = self._get_session()
        response = session.post(
            f"{self.base_url}/api/v1/embed/batch", json={"texts": texts}
        )
        response.raise_for_status()
        result = response.json()
        return result.get("embeddings", [])

    def similarity(self, text1: str, text2: str) -> Dict[str, Any]:
        import requests

        session = self._get_session()
        response = session.post(
            f"{self.base_url}/api/v1/similarity",
            json={"source": text1, "target": text2},
        )
        response.raise_for_status()
        return response.json()


def create_embedding_service(mode: Optional[str] = None) -> Any:
    """工厂函数：创建嵌入服务实例

    Args:
        mode: 可选，强制使用特定模式 ('mock' 或 'real')

    Returns:
        嵌入服务实例
    """
    if mode == "mock":
        return MockEmbeddingService()
    elif mode == "real":
        return RealEmbeddingService()
    elif mode is not None:
        raise ValueError(f"Unknown service mode: {mode}")

    # 根据配置选择
    if is_mock_mode():
        return MockEmbeddingService()
    else:
        try:
            return RealEmbeddingService()
        except Exception as e:
            print(f"Warning: Failed to create real service: {e}")
            print("Falling back to mock service")
            return MockEmbeddingService()


# Pytest Fixtures


def pytest_configure(config):
    """Pytest 配置钩子：初始化测试配置"""
    mode = get_test_mode()
    print(f"\n{'=' * 60}")
    print(f"Test Configuration:")
    print(f"  Mode: {mode.value}")
    print(f"  Model: {get_model_name()}")
    print(f"  Dimension: {get_model_dimension()}")
    print(f"  API URL: {get_api_base_url()}")
    print(f"{'=' * 60}\n")


@pytest.fixture(scope="session")
def test_mode() -> str:
    """会话级 fixture：返回测试模式"""
    return get_test_mode().value


@pytest.fixture(scope="session")
def model_name() -> str:
    """会话级 fixture：返回模型名称"""
    return get_model_name()


@pytest.fixture(scope="session")
def model_dimension() -> int:
    """会话级 fixture：返回模型维度"""
    return get_model_dimension()


@pytest.fixture
def embedding_service() -> Any:
    """返回配置的嵌入服务实例"""
    return create_embedding_service()


@pytest.fixture
def mock_embedding_service() -> MockEmbeddingService:
    """返回 Mock 嵌入服务实例"""
    return MockEmbeddingService()


@pytest.fixture
def real_embedding_service() -> RealEmbeddingService:
    """返回真实嵌入服务实例

    注意：需要运行 VecBoost 服务才能使用此 fixture
    """
    return RealEmbeddingService()


@pytest.fixture
def api_base_url() -> str:
    """返回 API 基础 URL"""
    return get_api_base_url()


@pytest.fixture
def sample_text() -> str:
    """提供测试用的示例文本"""
    return "Hello, world! This is a test."


@pytest.fixture
def sample_texts() -> list:
    """提供测试用的文本列表"""
    return [
        "Hello world",
        "Machine learning is great",
        "Artificial intelligence is the future",
    ]


@pytest.fixture
def short_text() -> str:
    """提供短文本用于测试"""
    return "AI"


@pytest.fixture
def chinese_text() -> str:
    """提供中文文本用于测试"""
    return "你好世界"


@pytest.fixture
def long_text() -> str:
    """提供长文本用于边界测试"""
    return (
        "Machine learning is a subset of artificial intelligence that enables "
        "systems to learn and improve from experience without being explicitly "
        "programmed. It focuses on developing computer programs that can access "
        "data and use it to learn for themselves."
    )


@pytest.fixture
def special_char_text() -> str:
    """提供包含特殊字符的文本用于测试"""
    return "Hello! @#$% &*() World"
