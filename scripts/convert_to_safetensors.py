#!/usr/bin/env python3
"""
将 PyTorch 模型转换为 safetensors 格式
用于解决大型 PyTorch 模型在 Rust 中的加载问题
"""

import os
import sys

def convert_model_to_safetensors(model_path: str, output_path: str):
    """
    将模型转换为 safetensors 格式

    Args:
        model_path: 原始模型路径 (HuggingFace repo ID 或本地路径)
        output_path: 输出目录路径
    """
    try:
        from transformers import AutoModel
        import safetensors.torch as st

        print(f"正在加载模型: {model_path}")

        # 加载模型 (这会下载模型如果需要)
        model = AutoModel.from_pretrained(model_path)
        print("模型加载成功!")

        # 获取模型权重
        state_dict = model.state_dict()

        print(f"正在保存权重到 safetensors 格式: {output_path}")
        st.save_file(state_dict, os.path.join(output_path, "model.safetensors"))
        print("权重保存成功!")

        # 复制配置文件
        import shutil

        print("正在复制配置文件...")
        files_to_copy = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "sentencepiece.bpe.model",
            "config_sentence_transformers.json"
        ]

        # 如果 model_path 是本地路径
        if os.path.exists(model_path):
            source_dir = model_path
        else:
            # 如果是 HuggingFace repo ID，从缓存获取
            from huggingface_hub import hf_hub_download
            source_dir = hf_hub_download(repo_id=model_path, filename="config.json", repo_type="model")
            source_dir = os.path.dirname(source_dir)

        for file in files_to_copy:
            src = os.path.join(source_dir, file)
            if os.path.exists(src):
                dst = os.path.join(output_path, file)
                shutil.copy2(src, dst)
                print(f"  复制: {file}")

        print(f"\n转换完成! 输出目录: {output_path}")
        print("现在可以使用这个路径作为 model_path 配置项的值。")

        return True

    except ImportError as e:
        print(f"缺少必要的库: {e}")
        print("请安装: pip install transformers safetensors")
        return False
    except Exception as e:
        print(f"转换失败: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="将 PyTorch 模型转换为 safetensors 格式")
    parser.add_argument("--model", type=str, default="BAAI/bge-m3",
                        help="模型 ID 或本地路径")
    parser.add_argument("--output", type=str, default="./bge-m3-safetensors",
                        help="输出目录路径")

    args = parser.parse_args()

    success = convert_model_to_safetensors(args.model, args.output)
    sys.exit(0 if success else 1)
