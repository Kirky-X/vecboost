"""测试配置"""

import os
from enum import Enum
from typing import Optional


# 支持的测试模式
class TestMode(Enum):
    MOCK = "mock"
    LIGHT = "light"
    FULL = "full"


# 模型配置映射：模型名称 -> 输出维度
MODEL_CONFIGS = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-m3": 1024,
    "BAAI/bge-small-zh-v1.5": 512,
    "BAAI/bge-base-zh-v1.5": 768,
    "BAAI/bge-large-zh-v1.5": 1024,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
}

# 默认配置
DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_MODEL_DIMENSION = 384
DEFAULT_TEST_MODE = TestMode.MOCK
MAX_TEXT_LENGTH = 10000


def get_test_mode() -> TestMode:
    """获取测试模式配置"""
    mode_str = os.environ.get("TEST_MODE", DEFAULT_TEST_MODE.value)
    try:
        return TestMode(mode_str)
    except ValueError:
        print(
            f"Warning: Invalid TEST_MODE '{mode_str}', using default '{DEFAULT_TEST_MODE.value}'"
        )
        return DEFAULT_TEST_MODE


def is_mock_mode() -> bool:
    """检查是否使用 Mock 模式"""
    return get_test_mode() == TestMode.MOCK


def is_real_mode() -> bool:
    """检查是否使用真实推理模式"""
    return get_test_mode() in (TestMode.LIGHT, TestMode.FULL)


def get_model_name() -> str:
    """获取测试使用的模型名称"""
    return os.environ.get("TEST_MODEL_NAME", DEFAULT_MODEL_NAME)


def get_model_path() -> Optional[str]:
    """获取本地模型路径（如果设置）"""
    return os.environ.get("TEST_MODEL_PATH")


def get_model_dimension() -> int:
    """获取测试模型的输出维度"""
    # 优先使用环境变量覆盖
    dim_override = os.environ.get("TEST_MODEL_DIMENSION")
    if dim_override:
        try:
            return int(dim_override)
        except ValueError:
            print(f"Warning: Invalid TEST_MODEL_DIMENSION '{dim_override}'")

    # 从模型配置获取
    model_name = get_model_name()
    return MODEL_CONFIGS.get(model_name, DEFAULT_MODEL_DIMENSION)


def get_device_type() -> str:
    """获取测试使用的设备类型"""
    return os.environ.get("TEST_DEVICE", "cpu")


def get_api_base_url() -> str:
    """获取 API 服务基础 URL"""
    return os.environ.get("TEST_API_BASE_URL", "http://localhost:9002")
