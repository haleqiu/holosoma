#!/usr/bin/env python3
"""Script to inspect pickle file and print mean and shape for all elements."""

import pickle
import numpy as np
from pathlib import Path


def print_statistics(data, prefix=""):
    """Recursively print statistics for arrays in a nested structure."""
    if isinstance(data, dict):
        for key, value in data.items():
            new_prefix = f"{prefix}.{key}" if prefix else key
            print_statistics(value, new_prefix)
    elif isinstance(data, (list, tuple)):
        for i, item in enumerate(data):
            new_prefix = f"{prefix}[{i}]" if prefix else f"[{i}]"
            print_statistics(item, new_prefix)
    elif isinstance(data, np.ndarray):
        print(f"{prefix}:")
        print(f"  shape: {data.shape}")
        print(f"  dtype: {data.dtype}")
        if data.size > 0:
            print(f"  mean: {np.mean(data)}")
            print(f"  min: {np.min(data)}")
            print(f"  max: {np.max(data)}")
        else:
            print(f"  mean: N/A (empty array)")
        print()
    elif isinstance(data, (int, float)):
        print(f"{prefix}: {data} (scalar)")
    elif isinstance(data, str):
        print(f"{prefix}: '{data}' (string)")
    else:
        print(f"{prefix}: {type(data).__name__}")


def main():
    import sys
    
    if len(sys.argv) > 1:
        pkl_path = Path(sys.argv[1])
    else:
        # Default path if no argument provided
        pkl_path = Path("/home/yuhengq/workspace/data/locomotion_collection/holosoma/rotatewalking2/rotatewalking2_lowstate.pkl")
    
    # If just filename provided, try to find it in common locations
    if not pkl_path.is_absolute() and not pkl_path.exists():
        # Try current directory
        if (Path.cwd() / pkl_path).exists():
            pkl_path = Path.cwd() / pkl_path
        # Try workspace root
        elif (Path("/home/yuhengq/workspace/holosoma") / pkl_path).exists():
            pkl_path = Path("/home/yuhengq/workspace/holosoma") / pkl_path
    
    if not pkl_path.exists():
        print(f"Error: File not found: {pkl_path}")
        print(f"Usage: {sys.argv[0]} <path_to_pkl_file>")
        return
    
    print(f"Reading pickle file: {pkl_path}")
    print("=" * 80)
    print()
    
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    
    print("Statistics for all elements:")
    print("=" * 80)
    print()
    
    print_statistics(data)


if __name__ == "__main__":
    main()

