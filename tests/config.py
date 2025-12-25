"""测试配置"""

import os

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

DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_MODEL_DIMENSION = 384
MAX_TEXT_LENGTH = 10000


def get_model_name() -> str:
    """获取测试使用的模型名称"""
    return os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME)


def get_model_dimension() -> int:
    """获取测试模型的输出维度"""
    model_name = get_model_name()
    return MODEL_CONFIGS.get(model_name, DEFAULT_MODEL_DIMENSION)
