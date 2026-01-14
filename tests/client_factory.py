"""
客户端工厂

提供创建不同类型 API 客户端的工厂函数。

支持:
- Mock 客户端：使用模拟服务
- Real 客户端：使用真实 API
- Adaptive 客户端：根据配置自动选择
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Type, Union

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.config import get_api_base_url, is_mock_mode


class APIClient:
    """统一的 API 客户端接口"""

    def __init__(self, service: Any):
        self.service = service

    def get(self, endpoint: str) -> Tuple[int, Dict[str, Any]]:
        raise NotImplementedError

    def post(self, endpoint: str, data: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        raise NotImplementedError


class MockAPIClient(APIClient):
    """Mock API 客户端

    使用 TestEmbeddingService 进行本地处理。
    """

    def get(self, endpoint: str) -> Tuple[int, Dict[str, Any]]:
        if endpoint == "/health":
            from datetime import datetime

            return 200, {
                "status": "healthy",
                "service": "vecboost",
                "version": "0.1.0",
                "timestamp": datetime.now().isoformat(),
            }
        return 404, {"error": "Not Found", "message": f"Endpoint {endpoint} not found"}

    def post(self, endpoint: str, data: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        from tests.services import AdaptiveEmbeddingService
        from tests.config import MAX_TEXT_LENGTH

        service = AdaptiveEmbeddingService()

        if endpoint == "/api/v1/embed":
            text = data.get("text", "")
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
            result = service.embed(text)
            return 200, result

        elif endpoint == "/api/v1/similarity":
            source = data.get("source", "")
            target = data.get("target", "")
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
            result = service.similarity(source, target)
            return 200, result

        elif endpoint == "/api/v1/embed/batch":
            texts = data.get("texts", [])
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
            results = service.embed_batch(texts)
            return 200, {"embeddings": results}

        return 404, {"error": "Not Found", "message": f"Endpoint {endpoint} not found"}


class RealAPIClient(APIClient):
    """Real API 客户端

    通过 HTTP 请求调用真实的 VecBoost 服务。
    """

    def __init__(self, base_url: Optional[str] = None):
        import requests

        self.base_url = base_url or get_api_base_url()
        self._session = requests.Session()

    def _request(
        self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None
    ) -> Tuple[int, Dict[str, Any]]:
        import requests

        url = f"{self.base_url}{endpoint}"
        try:
            if method == "GET":
                response = self._session.get(url, timeout=30.0)
            elif method == "POST":
                response = self._session.post(url, json=data, timeout=60.0)
            else:
                return 405, {
                    "error": "Method Not Allowed",
                    "message": f"Unsupported method: {method}",
                }

            return response.status_code, response.json()
        except requests.exceptions.ConnectionError as e:
            return 503, {
                "error": "Service Unavailable",
                "message": f"Could not connect to {self.base_url}: {e}",
            }
        except requests.exceptions.Timeout as e:
            return 504, {
                "error": "Gateway Timeout",
                "message": f"Request timed out: {e}",
            }
        except requests.exceptions.RequestException as e:
            return 500, {"error": "Internal Server Error", "message": str(e)}

    def get(self, endpoint: str) -> Tuple[int, Dict[str, Any]]:
        return self._request("GET", endpoint)

    def post(self, endpoint: str, data: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        return self._request("POST", endpoint, data)


class AdaptiveAPIClient(APIClient):
    """自适应 API 客户端

    根据测试模式自动选择使用 Mock 或 Real 客户端。
    """

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or get_api_base_url()
        self._mock_client = MockAPIClient(None)
        self._real_client: Optional[RealAPIClient] = None
        self._use_real = False

    def _ensure_real_client(self):
        """确保 RealAPIClient 已初始化"""
        if self._real_client is None:
            try:
                self._real_client = RealAPIClient(self.base_url)
                # 验证服务可用性
                status, _ = self._real_client.get("/health")
                if status == 200:
                    self._use_real = True
                else:
                    self._use_real = False
            except Exception:
                self._use_real = False

    def get(self, endpoint: str) -> Tuple[int, Dict[str, Any]]:
        if is_mock_mode():
            return self._mock_client.get(endpoint)

        self._ensure_real_client()
        if self._use_real and self._real_client is not None:
            return self._real_client.get(endpoint)

        return self._mock_client.get(endpoint)

    def post(self, endpoint: str, data: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        if is_mock_mode():
            return self._mock_client.post(endpoint, data)

        self._ensure_real_client()
        if self._use_real and self._real_client is not None:
            return self._real_client.post(endpoint, data)

        return self._mock_client.post(endpoint, data)

    def is_using_real(self) -> bool:
        """检查是否使用真实服务"""
        if is_mock_mode():
            return False

        self._ensure_real_client()
        return self._use_real


# 工厂函数


def create_client(
    mode: Optional[str] = None, base_url: Optional[str] = None
) -> APIClient:
    """创建指定类型的 API 客户端

    Args:
        mode: 客户端模式 ("mock", "real", "adaptive" 或 None)
        base_url: Real 客户端的基础 URL

    Returns:
        API 客户端实例

    Examples:
        # 使用 Mock 模式
        client = create_client("mock")

        # 使用 Real 模式
        client = create_client("real", "http://localhost:9002")

        # 根据环境变量自动选择
        client = create_client()
    """
    if mode == "mock":
        return MockAPIClient(None)
    elif mode == "real":
        return RealAPIClient(base_url)
    elif mode is None or mode == "adaptive":
        return AdaptiveAPIClient(base_url)
    else:
        raise ValueError(
            f"Unknown client mode: {mode}. Valid options: mock, real, adaptive"
        )


def create_mock_client() -> MockAPIClient:
    """创建 Mock API 客户端

    Returns:
        MockAPIClient 实例
    """
    return MockAPIClient(None)


def create_real_client(base_url: Optional[str] = None) -> RealAPIClient:
    """创建 Real API 客户端

    Args:
        base_url: 可选的基础 URL

    Returns:
        RealAPIClient 实例
    """
    return RealAPIClient(base_url)


def create_adaptive_client(base_url: Optional[str] = None) -> AdaptiveAPIClient:
    """创建自适应 API 客户端

    Args:
        base_url: 可选的基础 URL

    Returns:
        AdaptiveAPIClient 实例
    """
    return AdaptiveAPIClient(base_url)


# 测试代码
if __name__ == "__main__":
    import sys

    print("=== Client Factory Test ===")
    print()

    # 测试 Mock 客户端
    print("1. Testing MockAPIClient...")
    mock_client = create_client("mock")
    status, response = mock_client.get("/health")
    print(f"   GET /health: {status}")
    assert status == 200
    print("   Mock client works!")
    print()

    # 测试 Mock embed
    print("2. Testing Mock embed...")
    status, response = mock_client.post("/api/v1/embed", {"text": "Hello world"})
    print(f"   POST /api/v1/embed: {status}")
    assert status == 200
    assert "embedding" in response
    print(f"   Dimension: {response.get('dimension')}")
    print("   Mock embed works!")
    print()

    # 测试 Real 客户端（如果服务可用）
    print("3. Testing RealAPIClient...")
    try:
        real_client = create_real_client()
        status, response = real_client.get("/health")
        if status == 200:
            print(f"   GET /health: {status}")
            print("   Real client works!")
        else:
            print(f"   Service returned status: {status}")
            print("   (This is OK if service is not running)")
    except Exception as e:
        print(f"   Could not connect: {e}")
        print("   (This is OK if service is not running)")
    print()

    # 测试 Adaptive 客户端
    print("4. Testing AdaptiveAPIClient...")
    adaptive_client = create_adaptive_client()
    status, response = adaptive_client.get("/health")
    print(f"   GET /health: {status}")
    print(f"   Using real: {adaptive_client.is_using_real()}")
    print("   Adaptive client works!")
    print()

    print("All tests passed!")
