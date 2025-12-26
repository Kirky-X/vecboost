"""API 客户端和模拟器

提供统一的 API 客户端和模拟器基类，用于 API 接口测试。
"""

from datetime import datetime
from typing import Any, Dict, List, Tuple, Type, TypeVar, Callable

from tests.config import MAX_TEXT_LENGTH

T = TypeVar("T")


class APIClient:
    """统一的 API 客户端"""

    def __init__(self, simulator: Any):
        self.simulator = simulator

    def get(self, endpoint: str) -> Tuple[int, Dict[str, Any]]:
        return self.simulator.handle_request(endpoint, "GET", {})

    def post(self, endpoint: str, data: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        return self.simulator.handle_request(endpoint, "POST", data)


class BaseAPISimulator:
    """API 模拟器基类"""

    def __init__(self, embedding_service: Type[T]):
        self.service = embedding_service

    def health_check(self) -> Tuple[int, Dict[str, Any]]:
        return 200, {
            "status": "healthy",
            "service": "vecboost",
            "version": "0.1.0",
            "timestamp": datetime.now().isoformat(),
        }

    def embed(self, request: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        if request is None:
            request = {}
        text = request.get("text", "")
        if not text:
            return 400, {
                "error": "Invalid input",
                "message": "text field is required and cannot be empty",
            }
        if len(text) > MAX_TEXT_LENGTH:
            return 400, {
                "error": "Invalid input",
                "message": f"text length exceeds maximum allowed length ({MAX_TEXT_LENGTH} characters)",
            }
        result = self.service.embed(text)
        return 200, result

    def similarity(self, request: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
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
        if len(source) > MAX_TEXT_LENGTH or len(target) > MAX_TEXT_LENGTH:
            return 400, {
                "error": "Invalid input",
                "message": f"text length exceeds maximum allowed length ({MAX_TEXT_LENGTH} characters)",
            }
        result = self.service.similarity(source, target)
        return 200, result

    def embed_batch(self, request: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        texts = request.get("texts", [])
        if not texts:
            return 400, {
                "error": "Invalid input",
                "message": "texts field is required and cannot be empty",
            }
        if len(texts) > 100:
            return 400, {
                "error": "Invalid input",
                "message": "batch size exceeds maximum allowed (100 texts)",
            }
        if any(len(text) > MAX_TEXT_LENGTH for text in texts):
            return 400, {
                "error": "Invalid input",
                "message": f"text length exceeds maximum allowed length ({MAX_TEXT_LENGTH} characters)",
            }
        results = self.service.embed_batch(texts)
        return 200, {"embeddings": results}

    def handle_request(
        self, endpoint: str, method: str, request: Dict[str, Any]
    ) -> Tuple[int, Dict[str, Any]]:
        handler_map: Dict[Tuple[str, str], Callable[[], Tuple[int, Dict[str, Any]]]] = {
            ("/health", "GET"): self.health_check,
            ("/api/v1/embed", "POST"): lambda: self.embed(request),
            ("/api/v1/similarity", "POST"): lambda: self.similarity(request),
            ("/api/v1/embed/batch", "POST"): lambda: self.embed_batch(request),
        }
        key = (endpoint, method)
        if key in handler_map:
            return handler_map[key]()
        return 404, {"error": "Not Found", "message": f"Endpoint {endpoint} not found"}


class TestAPISimulator(BaseAPISimulator):
    """测试 API 模拟器"""

    pass


def create_api_client(simulator: BaseAPISimulator) -> APIClient:
    """工厂函数：创建 API 客户端"""
    return APIClient(simulator)
