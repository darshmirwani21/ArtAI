"""
Extract ViT features for all images in a dataset folder organized by class.
Saves: features.npy, labels.npy, style_names.joblib

Usage:
    python scripts/extract_features.py --data_dir "C:\path\to\Paintings" --out_dir .
"""
import os
import argparse
import numpy as np
import joblib
try:
    from tqdm import tqdm
except Exception:
    # Fallback if tqdm is not installed — simple passthrough iterator
    def tqdm(iterable, **kwargs):
        return iterable

# Ensure the repository root is on sys.path so imports like `from model import ...` work
import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from model import ArtStyleAnalyzer


def main(data_dir: str, out_dir: str, device: str = None):
    if device is None:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    analyzer = ArtStyleAnalyzer(device=device)

    style_names = []
    # Discover styles as subdirectories (sorted for consistency)
    for name in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, name)
        if os.path.isdir(path):
            style_names.append(name)

    X = []
    y = []

    for idx, style in enumerate(style_names):
        style_dir = os.path.join(data_dir, style)
        files = [f for f in os.listdir(style_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'))]
        print(f"Processing style: {style} ({len(files)} images)")
        for fn in tqdm(files, desc=f"{style}"):
            fp = os.path.join(style_dir, fn)
            try:
                feat = analyzer.extract_features(fp)  # torch tensor on CPU
                feat_np = feat.squeeze(0).numpy()
                X.append(feat_np)
                y.append(idx)
            except Exception as e:
                print(f"Skipping {fp}: {e}")

    if len(X) == 0:
        print("No features extracted. Exiting.")
        return

    X = np.vstack(X)
    y = np.array(y)

    os.makedirs(out_dir, exist_ok=True)
    feat_path = os.path.join(out_dir, 'features.npy')
    labels_path = os.path.join(out_dir, 'labels.npy')
    styles_path = os.path.join(out_dir, 'style_names.joblib')

    np.save(feat_path, X)
    np.save(labels_path, y)
    joblib.dump(style_names, styles_path)

    print(f"Saved features ({X.shape}) to: {feat_path}")
    print(f"Saved labels ({y.shape}) to: {labels_path}")
    print(f"Saved style names to: {styles_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset root (folders per style)')
    parser.add_argument('--out_dir', type=str, default='.', help='Output directory to save features and labels')
    parser.add_argument('--device', type=str, default=None, help='Device to run feature extraction on (cuda or cpu)')
    args = parser.parse_args()
    main(args.data_dir, args.out_dir, args.device)
