"""
Mock 向量化服务

模拟 BGE-M3 模型的输出：
- 维度: 1024 (BGE-M3 默认维度)
- 向量值: 归一化的随机值，范围 [-1, 1]

用于测试环境，无需加载真实模型。
"""

import hashlib
import random
from typing import Any, Dict, List
from tests.config import get_model_dimension


class MockEmbeddingService:
    """Mock 向量化服务"""

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

        dot_product = sum(
            a * b for a, b in zip(emb1["embedding"], emb2["embedding"])
        )

        return {
            "score": round(dot_product, 6),
            "metric": "cosine",
        }

    @classmethod
    def cosine_similarity(cls, vec1: List[float], vec2: List[float]) -> float:
        """计算两个向量的余弦相似度"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        return round(dot_product, 6)
