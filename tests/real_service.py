"""
真实嵌入服务

提供通过 HTTP API 调用真实推理引擎的能力。
用于测试模式设置为 light 或 full 时。
"""

import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.config import get_api_base_url, get_model_dimension, get_model_name


class RealEmbeddingService:
    """真实嵌入服务

    通过 HTTP API 调用 VecBoost 服务进行真实推理。

    Attributes:
        base_url: API 服务基础 URL
        model_name: 使用的模型名称
        dimension: 模型输出维度
    """

    def __init__(self, base_url: Optional[str] = None):
        """初始化真实嵌入服务

        Args:
            base_url: API 服务基础 URL，如果为 None 则从环境变量读取
        """
        self.base_url = base_url or get_api_base_url()
        self.model_name = get_model_name()
        self.dimension = get_model_dimension()
        self._session = None

    def _get_session(self):
        """获取 HTTP 会话（懒加载）"""
        if self._session is None:
            import requests

            self._session = requests.Session()
        return self._session

    def get_dimension(self) -> int:
        """获取模型输出维度"""
        return self.dimension

    def embed(self, text: str) -> Dict[str, Any]:
        """生成文本向量

        Args:
            text: 输入文本

        Returns:
            包含 embedding 和 dimension 的字典

        Raises:
            requests.RequestException: API 请求失败时
        """
        import requests

        session = self._get_session()
        response = session.post(
            f"{self.base_url}/api/v1/embed", json={"text": text}, timeout=30.0
        )
        response.raise_for_status()
        return response.json()

    def embed_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """批量生成文本向量

        Args:
            texts: 输入文本列表

        Returns:
            包含 embeddings 的字典
        """
        import requests

        session = self._get_session()
        response = session.post(
            f"{self.base_url}/api/v1/embed/batch", json={"texts": texts}, timeout=60.0
        )
        response.raise_for_status()
        result = response.json()
        return result.get("embeddings", [])

    def similarity(self, text1: str, text2: str) -> Dict[str, Any]:
        """计算两个文本的相似度

        Args:
            text1: 源文本
            text2: 目标文本

        Returns:
            包含 score 和 metric 的字典
        """
        import requests

        session = self._get_session()
        response = session.post(
            f"{self.base_url}/api/v1/similarity",
            json={"source": text1, "target": text2},
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算两个向量的余弦相似度

        Args:
            vec1: 第一个向量
            vec2: 第二个向量

        Returns:
            余弦相似度分数
        """
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same length")

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)


class RealEmbeddingServiceWithFallback:
    """带回退的真实嵌入服务

    在真实服务不可用时自动回退到 Mock 实现。
    """

    def __init__(self, base_url: Optional[str] = None):
        """初始化带回退的服务"""
        self.base_url = base_url or get_api_base_url()
        self._real_service = None
        self._mock_service = None
        self._use_fallback = False
        self._fallback_reason = None

    def _get_real_service(self) -> Optional[RealEmbeddingService]:
        """获取真实服务实例"""
        if self._real_service is None:
            try:
                self._real_service = RealEmbeddingService(self.base_url)
                # 验证服务可用性
                self._real_service.get_dimension()
                self._use_fallback = False
                self._fallback_reason = None
            except Exception as e:
                self._use_fallback = True
                self._fallback_reason = str(e)
                print(f"Warning: Real service unavailable: {e}")
                print("Falling back to mock service")
                return None
        return self._real_service

    def _get_mock_service(self):
        """获取 Mock 服务实例"""
        if self._mock_service is None:
            from tests.services import TestEmbeddingService

            self._mock_service = TestEmbeddingService
        return self._mock_service

    def get_dimension(self) -> int:
        """获取模型输出维度"""
        real = self._get_real_service()
        if real is not None and not self._use_fallback:
            return real.get_dimension()
        return self._get_mock_service().get_dimension()

    def embed(self, text: str) -> Dict[str, Any]:
        """生成文本向量"""
        real = self._get_real_service()
        if real is not None and not self._use_fallback:
            return real.embed(text)
        return self._get_mock_service().embed(text)

    def embed_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """批量生成文本向量"""
        real = self._get_real_service()
        if real is not None and not self._use_fallback:
            return real.embed_batch(texts)
        return self._get_mock_service().embed_batch(texts)

    def similarity(self, text1: str, text2: str) -> Dict[str, Any]:
        """计算两个文本的相似度"""
        real = self._get_real_service()
        if real is not None and not self._use_fallback:
            return real.similarity(text1, text2)
        return self._get_mock_service().similarity(text1, text2)

    def is_using_fallback(self) -> bool:
        """检查是否使用回退"""
        return self._use_fallback

    def get_fallback_reason(self) -> Optional[str]:
        """获取回退原因"""
        return self._fallback_reason


# 便捷函数
def create_real_service(base_url: Optional[str] = None) -> RealEmbeddingService:
    """创建真实嵌入服务实例

    Args:
        base_url: 可选的基础 URL

    Returns:
        RealEmbeddingService 实例
    """
    return RealEmbeddingService(base_url)


def create_service_with_fallback(
    base_url: Optional[str] = None,
) -> RealEmbeddingServiceWithFallback:
    """创建带回退的嵌入服务实例

    Args:
        base_url: 可选的基础 URL

    Returns:
        RealEmbeddingServiceWithFallback 实例
    """
    return RealEmbeddingServiceWithFallback(base_url)


# 测试代码
if __name__ == "__main__":
    import sys

    print("=== RealEmbeddingService Test ===")
    print()

    base_url = sys.argv[1] if len(sys.argv) > 1 else None

    try:
        service = RealEmbeddingService(base_url)

        print(f"Service URL: {service.base_url}")
        print(f"Model: {service.model_name}")
        print(f"Dimension: {service.dimension}")
        print()

        # 测试 embed
        print("Testing embed...")
        result = service.embed("Hello, world!")
        print(f"  Embedding dimension: {result.get('dimension')}")
        print(f"  Vector sample: {result.get('embedding', [])[:5]}...")
        print()

        # 测试 batch
        print("Testing embed_batch...")
        texts = ["Text 1", "Text 2", "Text 3"]
        results = service.embed_batch(texts)
        print(f"  Batch size: {len(results)}")
        for i, r in enumerate(results):
            print(f"  Text {i + 1} dimension: {r.get('dimension')}")
        print()

        # 测试 similarity
        print("Testing similarity...")
        result = service.similarity("Hello world", "Hello there")
        print(f"  Similarity score: {result.get('score')}")
        print(f"  Metric: {result.get('metric')}")

        print()
        print("All tests passed!")

    except Exception as e:
        print(f"Error: {e}")
        print()
        print("Make sure VecBoost service is running at the specified URL")
        sys.exit(1)
