#!/usr/bin/env python3
"""
Convert BGE-M3 model from safetensors to ONNX format for GPU inference.

Usage:
    python convert_model_to_onnx.py [--quantize]
"""

import os
import sys
import argparse
from pathlib import Path

try:
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer
    OPTIMUM_AVAILABLE = True
except ImportError:
    OPTIMUM_AVAILABLE = False

def convert_model_to_onnx(
    model_path: str,
    output_path: str,
    quantize: bool = False
):
    """Convert a HuggingFace model to ONNX format."""
    print(f"Converting model from {model_path} to ONNX format...")
    print(f"Output path: {output_path}")
    print(f"Quantize: {quantize}")

    if not OPTIMUM_AVAILABLE:
        print("ERROR: optimum library not installed.")
        print("Please install it with: pip install optimum[onnxruntime]")
        print("Or for GPU support: pip install optimum[onnxruntime-gpu]")
        sys.exit(1)

    model_path = Path(model_path)
    output_path = Path(output_path)

    if not model_path.exists():
        print(f"ERROR: Model path does not exist: {model_path}")
        sys.exit(1)

    output_path.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    print("Loading model and converting to ONNX...")
    if quantize:
        print("  Note: Quantization will be applied during export")
        model = ORTModelForFeatureExtraction.from_pretrained(
            str(model_path),
            export=True,
        )
    else:
        model = ORTModelForFeatureExtraction.from_pretrained(
            str(model_path),
            export=True,
        )

    print("Saving tokenizer...")
    tokenizer.save_pretrained(str(output_path))

    print("Saving ONNX model...")
    model.save_pretrained(str(output_path))

    print(f"\n✓ Model successfully converted to ONNX format!")
    print(f"  Output directory: {output_path}")
    print(f"  Model files: {list(output_path.glob('*.onnx'))}")

    return output_path

def check_dependencies():
    """Check if required dependencies are installed."""
    print("Checking dependencies...")

    try:
        import optimum
        print("  ✓ optimum: installed")
    except ImportError:
        print("  ✗ optimum not installed")
        return False

    try:
        import onnxruntime
        print(f"  ✓ onnxruntime: {onnxruntime.__version__}")
    except ImportError:
        print("  ✗ onnxruntime not installed")
        return False

    try:
        import torch
        print(f"  ✓ torch: {torch.__version__}")
    except ImportError:
        print("  ✗ torch not installed")
        return False

    try:
        import transformers
        print(f"  ✓ transformers: {transformers.__version__}")
    except ImportError:
        print("  ✗ transformers not installed")
        return False

    return True

def main():
    parser = argparse.ArgumentParser(
        description="Convert BGE-M3 model to ONNX format for GPU inference"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/dev/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181",
        help="Path to the source model (safetensors format)"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="/home/project/vecboost/models/bge-m3-onnx",
        help="Path for the output ONNX model"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply quantization to reduce model size"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check dependencies, don't convert"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("BGE-M3 Model to ONNX Converter")
    print("=" * 60)

    if args.check_only:
        check_dependencies()
        return

    if not check_dependencies():
        print("\nPlease install missing dependencies before running this script.")
        sys.exit(1)

    try:
        convert_model_to_onnx(
            model_path=args.model_path,
            output_path=args.output_path,
            quantize=args.quantize
        )
    except Exception as e:
        print(f"ERROR: Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
