#!/usr/bin/env python3
"""
Convert PyTorch model to safetensors format for VecBoost.

Usage:
    python3 convert_pytorch_to_safetensors.py <model_path> [output_path]

Example:
    python3 convert_pytorch_to_safetensors.py "/home/dev/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/xxx"
"""

import os
import sys
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def get_model_weights(pt_path: str) -> dict:
    """Load PyTorch model weights."""
    print(f"Loading weights from: {pt_path}")
    state_dict = torch.load(pt_path, map_location="cpu")
    return state_dict


def convert_model(model_dir: str, output_path: str = None) -> str:
    """Convert PyTorch model to safetensors format."""
    model_path = Path(model_dir)
    
    # Find PyTorch model file
    pt_files = list(model_path.glob("pytorch_model*.bin"))
    if not pt_files:
        raise FileNotFoundError(f"No PyTorch model file found in {model_dir}")
    
    pt_path = str(pt_files[0])
    print(f"Found PyTorch model: {pt_path}")
    
    # Load weights
    state_dict = get_model_weights(pt_path)
    print(f"Loaded {len(state_dict)} state dict entries")
    
    # Determine output path
    if output_path is None:
        output_path = str(model_path / "model.safetensors")
    
    # Save as safetensors
    print(f"Saving to safetensors format: {output_path}")
    save_file(state_dict, output_path)
    print(f"Successfully saved {len(state_dict)} tensors to {output_path}")
    
    # Also save metadata
    metadata = {
        "format": "pt",
        "original_files": [os.path.basename(pt_path)],
        "tensor_count": len(state_dict),
    }
    metadata_path = output_path + ".metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")
    
    return output_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 convert_pytorch_to_safetensors.py <model_path> [output_path]")
        sys.exit(1)
    
    model_dir = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        output_file = convert_model(model_dir, output_path)
        print(f"\nConversion complete! Output: {output_file}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
