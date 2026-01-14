"""
测试向量化服务

模拟 BGE-M3 模型的输出：
- 维度: 由配置决定（默认 1024）
- 向量值: 归一化的随机值，范围 [-1, 1]

用于测试环境，使用确定性哈希算法生成向量。

支持通过 TEST_MODE 环境变量配置：
- mock (默认): 使用确定性哈希算法
- light: 尝试使用真实推理，失败时回退
- full: 强制使用真实推理
"""

import hashlib
import random
from typing import Any, Dict, List, Optional
from tests.config import get_model_dimension, get_test_mode, is_mock_mode


class TestEmbeddingService:
    """测试向量化服务（Mock 实现）

    使用确定性哈希算法生成向量，确保测试的确定性和可重复性。
    """

    @classmethod
    def get_dimension(cls) -> int:
        return get_model_dimension()

    @classmethod
    def embed(cls, text: str) -> Dict[str, Any]:
        """生成模拟的文本向量"""
        dimension = cls.get_dimension()
        seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16) % (2**31)
        random.seed(seed)

        vector = [random.uniform(-1, 1) for _ in range(dimension)]
        magnitude = sum(v**2 for v in vector) ** 0.5
        if magnitude > 0:
            vector = [v / magnitude for v in vector]

        return {
            "embedding": vector,
            "dimension": len(vector),
        }

    @classmethod
    def embed_batch(cls, texts: List[str]) -> List[Dict[str, Any]]:
        """批量生成模拟的文本向量"""
        return [cls.embed(text) for text in texts]

    @classmethod
    def similarity(cls, text1: str, text2: str) -> Dict[str, Any]:
        """计算模拟的相似度"""
        emb1 = cls.embed(text1)
        emb2 = cls.embed(text2)

        dot_product = sum(a * b for a, b in zip(emb1["embedding"], emb2["embedding"]))

        return {
            "score": round(dot_product, 6),
            "metric": "cosine",
        }

    @classmethod
    def cosine_similarity(cls, vec1: List[float], vec2: List[float]) -> float:
        """计算两个向量的余弦相似度"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        return round(dot_product, 6)


class AdaptiveEmbeddingService:
    """自适应嵌入服务

    根据测试模式自动选择使用真实推理或 Mock。

    - TEST_MODE=mock: 始终使用 Mock
    - TEST_MODE=light: 尝试真实推理，失败时回退 Mock
    - TEST_MODE=full: 强制使用真实推理
    """

    def __init__(self):
        self.test_mode = get_test_mode()
        self._mock_service = TestEmbeddingService
        self._real_service: Optional[Any] = None

    def _get_real_service(self) -> Optional[Any]:
        """获取真实服务实例（懒加载）"""
        if self._real_service is None:
            try:
                from tests.real_service import RealEmbeddingServiceWithFallback

                self._real_service = RealEmbeddingServiceWithFallback()
            except ImportError:
                self._real_service = None
        return self._real_service

    def get_dimension(self) -> int:
        """获取模型输出维度"""
        if is_mock_mode():
            return self._mock_service.get_dimension()

        real = self._get_real_service()
        if real is not None and not real.is_using_fallback():
            return real.get_dimension()

        return self._mock_service.get_dimension()

    def embed(self, text: str) -> Dict[str, Any]:
        """生成文本向量"""
        if is_mock_mode():
            return self._mock_service.embed(text)

        real = self._get_real_service()
        if real is not None and not real.is_using_fallback():
            return real.embed(text)

        return self._mock_service.embed(text)

    def embed_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """批量生成文本向量"""
        if is_mock_mode():
            return self._mock_service.embed_batch(texts)

        real = self._get_real_service()
        if real is not None and not real.is_using_fallback():
            return real.embed_batch(texts)

        return self._mock_service.embed_batch(texts)

    def similarity(self, text1: str, text2: str) -> Dict[str, Any]:
        """计算两个文本的相似度"""
        if is_mock_mode():
            return self._mock_service.similarity(text1, text2)

        real = self._get_real_service()
        if real is not None and not real.is_using_fallback():
            return real.similarity(text1, text2)

        return self._mock_service.similarity(text1, text2)

    def is_using_real_engine(self) -> bool:
        """检查是否使用真实推理"""
        if is_mock_mode():
            return False

        real = self._get_real_service()
        if real is not None:
            return not real.is_using_fallback()

        return False

    def is_using_fallback(self) -> bool:
        """检查是否使用回退"""
        if is_mock_mode():
            return True

        real = self._get_real_service()
        if real is not None:
            return real.is_using_fallback()

        return True


# 便捷函数
def get_embedding_service() -> AdaptiveEmbeddingService:
    """获取自适应嵌入服务实例

    Returns:
        AdaptiveEmbeddingService 实例
    """
    return AdaptiveEmbeddingService()


def is_real_mode() -> bool:
    """检查是否使用真实推理模式

    Returns:
        True 如果 TEST_MODE 是 light 或 full
    """
    return not is_mock_mode()
