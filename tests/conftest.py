"""
Pytest 配置文件和共享 fixture

提供测试所需的公共 fixture 和配置。
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.config import get_model_name, get_model_dimension


@pytest.fixture(scope="session")
def model_name() -> str:
    """获取测试使用的模型名称"""
    return get_model_name()


@pytest.fixture(scope="session")
def model_dimension() -> int:
    """获取测试模型的输出维度"""
    return get_model_dimension()


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
