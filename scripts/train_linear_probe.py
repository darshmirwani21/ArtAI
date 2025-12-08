"""
Train a linear probe (logistic regression) on pre-extracted features.
Saves: linear_probe.joblib

Usage:
    python scripts/train_linear_probe.py --features features.npy --labels labels.npy
"""
import argparse
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def main(features_path: str, labels_path: str, style_names_path: str = None, out_path: str = 'linear_probe.joblib'):
    X = np.load(features_path)
    y = np.load(labels_path)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Train samples: {X_train.shape[0]}  Val samples: {X_val.shape[0]}")

    clf = LogisticRegression(max_iter=2000, solver='saga', multi_class='multinomial', n_jobs=-1)
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    val_acc = clf.score(X_val, y_val)
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")

    joblib.dump(clf, out_path)
    print(f"Saved classifier to: {out_path}")

    if style_names_path is not None:
        style_names = joblib.load(style_names_path)
        joblib.dump(style_names, 'style_names.joblib')
        print(f"Saved style names copy to: style_names.joblib")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=str, default='features.npy')
    parser.add_argument('--labels', type=str, default='labels.npy')
    parser.add_argument('--style_names', type=str, default='style_names.joblib')
    parser.add_argument('--out', type=str, default='linear_probe.joblib')
    args = parser.parse_args()
    main(args.features, args.labels, args.style_names, args.out)
