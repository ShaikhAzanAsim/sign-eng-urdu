"""Train a RandomForest static-letter classifier from data/landmarks/<lang>/*.csv."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src.console import configure_utf8_stdout
from src.dataset_build import matrices
from src.normalization import augment_coords, joint_angles
from src.paths import ENGLISH_MODELS_DIR, PROJECT_ROOT, URDU_MODELS_DIR


def _model_dir(language: str) -> Path:
    return ENGLISH_MODELS_DIR if language == "english" else URDU_MODELS_DIR


def _augment_split(
    x: np.ndarray,
    y: np.ndarray,
    copies: int,
    include_angles: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if copies <= 0 or len(x) == 0:
        return x, y
    rng = np.random.default_rng(seed)
    extra_x: list[np.ndarray] = []
    extra_y: list[int] = []
    for row, label in zip(x, y):
        coords = row[:63].reshape(21, 3)
        for _ in range(copies):
            aug = augment_coords(coords, rng)
            vec = aug.reshape(-1)
            if include_angles:
                vec = np.concatenate([vec, joint_angles(aug)])
            extra_x.append(vec.astype(np.float32))
            extra_y.append(int(label))
    x_out = np.concatenate([x, np.stack(extra_x)], axis=0)
    y_out = np.concatenate([y, np.asarray(extra_y, dtype=np.int32)], axis=0)
    return x_out, y_out


def train(
    language: str,
    n_estimators: int = 250,
    augment_copies: int = 2,
    seed: int = 42,
) -> Path:
    include_angles = language == "urdu"
    x, y, labels = matrices(language, include_angles=include_angles)
    if len(x) == 0:
        raise SystemExit(
            f"No static landmark CSVs for {language}. "
            f"Run: python -m src.data_collection --lang {language}"
        )
    counts = {labels[i]: int(np.sum(y == i)) for i in range(len(labels))}
    print("samples per class:")
    for name, n in counts.items():
        print(f"  {name}: {n}")

    x, y = _augment_split(x, y, augment_copies, include_angles, seed)
    stratify = y if len(set(y.tolist())) > 1 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=seed, stratify=stratify
    )
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    print(classification_report(y_test, pred, target_names=labels, zero_division=0))
    matrix = confusion_matrix(y_test, pred, labels=list(range(len(labels))))
    print("confusion matrix (rows=true, cols=pred):")
    print(matrix)

    out_dir = _model_dir(language)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "static_classifier.pkl"
    labels_path = out_dir / "static_labels.json"
    joblib.dump(clf, model_path)
    payload = {
        "language": language,
        "labels": labels,
        "include_angles": include_angles,
        "n_features": int(x.shape[1]),
        "confusion_matrix": matrix.tolist(),
        "class_counts": counts,
    }
    labels_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {model_path}")
    print(f"saved {labels_path}")
    return model_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train static RandomForest per language.")
    parser.add_argument("--lang", choices=("english", "urdu"), default="english")
    parser.add_argument("--trees", type=int, default=250)
    parser.add_argument("--augment", type=int, default=2, help="Augmented copies per sample.")
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_utf8_stdout()
    args = build_parser().parse_args(argv)
    print(f"Project root: {PROJECT_ROOT}")
    train(args.lang, n_estimators=args.trees, augment_copies=args.augment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
