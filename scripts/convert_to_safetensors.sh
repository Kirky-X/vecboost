#!/usr/bin/env bash
# convert_to_safetensors.sh — 将 HuggingFace 模型转换为 safetensors 格式
#
# 背景:
#   BAAI/bge-m3 等模型仅提供 pytorch_model.bin（pickle/zip 格式），vecboost 依赖的
#   candle 0.11 通过 zip crate 加载大 pytorch 文件时会报 "header too large"。
#   本脚本用 transformers 把权重转换为 safetensors，配合 config.toml 的
#   model_path 本地加载即可绕过该限制。
#
# 依赖:
#   pip install transformers torch
#
# 用法:
#   ./scripts/convert_to_safetensors.sh [MODEL_REPO] [OUTPUT_DIR] [--offline]
#
# 参数:
#   MODEL_REPO   HuggingFace 仓库 ID（默认 BAAI/bge-m3）
#   OUTPUT_DIR   输出目录（默认 models/<repo-with-dashes>-st）
#   --offline    仅使用本地缓存，不联网（需模型已下载过）
#
# 示例:
#   ./scripts/convert_to_safetensors.sh
#   ./scripts/convert_to_safetensors.sh BAAI/bge-m3 models/bge-m3-st
#   ./scripts/convert_to_safetensors.sh BAAI/bge-m3 models/bge-m3-st --offline

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_REPO=""
OUTPUT_DIR=""
OFFLINE=0

for arg in "$@"; do
    case "$arg" in
        --offline) OFFLINE=1 ;;
        -h|--help) sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)
            if [[ -z "$MODEL_REPO" ]]; then
                MODEL_REPO="$arg"
            elif [[ -z "$OUTPUT_DIR" ]]; then
                OUTPUT_DIR="$arg"
            fi
            ;;
    esac
done

MODEL_REPO="${MODEL_REPO:-BAAI/bge-m3}"
OUTPUT_DIR="${OUTPUT_DIR:-models/$(echo "$MODEL_REPO" | sed 's|/|-|g')-st}"

echo "==> Model repo : $MODEL_REPO"
echo "==> Output dir : $OUTPUT_DIR"
echo "==> Offline    : $([ "$OFFLINE" = 1 ] && echo yes || echo no)"
echo

if ! python3 -c "import transformers, torch" 2>/dev/null; then
    echo "ERROR: transformers/torch 未安装。请运行: pip install transformers torch" >&2
    exit 1
fi

if [ "$OFFLINE" = 1 ]; then
    export HF_HUB_OFFLINE=1
fi

mkdir -p "$OUTPUT_DIR"

export CONVERT_MODEL_REPO="$MODEL_REPO"
export CONVERT_OUTPUT_DIR="$OUTPUT_DIR"

python3 -c "
import os
from transformers import AutoModel, AutoTokenizer

repo = os.environ['CONVERT_MODEL_REPO']
out = os.environ['CONVERT_OUTPUT_DIR']

print(f'loading model {repo} ...')
model = AutoModel.from_pretrained(repo)
print('loading tokenizer ...')
tokenizer = AutoTokenizer.from_pretrained(repo)
print(f'saving safetensors to {out} ...')
model.save_pretrained(out, safe_serialization=True)
tokenizer.save_pretrained(out)
print('done')
"

echo
echo "==> Converted files:"
ls -lh "$OUTPUT_DIR" | awk 'NR>1 {print "    " $5 "\t" $9}'
echo
echo "==> 下一步: 在 config.toml 设置"
echo "    model_path = \"$OUTPUT_DIR\""
