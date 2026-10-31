import argparse
from pathlib import Path
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from .config import CFG, DATA_DIR, MODEL_PATH, CLASSES
from .features import extract_features
from .triangulation import extract_fourier_triangulation_vector

def evaluate_saved_model(model_file: Path, target_dir: Path = DATA_DIR, holdout_only: bool = False):
    saved = joblib.load(model_file)
    model = saved["model"]
    triangulator = saved.get("triangulator", None)
    w = saved.get("fourier_weight", 0.50)
    feature_set = saved.get("feature_set", "enhanced")
    classes = saved.get("classes", CLASSES)

    X_ml, X_fourier, y = [], [], []
    for idx, label in enumerate(classes):
        folder = target_dir / label
        if not folder.exists():
            continue
        for file in sorted(folder.glob("*.npy")):
            waveform = np.load(file)
            X_ml.append(extract_features(waveform, feature_set=feature_set))
            X_fourier.append(extract_fourier_triangulation_vector(waveform))
            y.append(idx)

    X_ml, X_fourier, y = np.array(X_ml), np.array(X_fourier), np.array(y)
    if len(y) == 0:
        raise ValueError(f"No samples found in {target_dir}")

    eval_label = "FULL DATASET"
    if holdout_only:
        tr_cfg = CFG.get("training", {})
        seed = int(tr_cfg.get("random_seed", 42))
        test_size = float(tr_cfg.get("test_size", 0.20))
        indices = np.arange(len(y))
        _, test_idx = train_test_split(indices, test_size=test_size, stratify=y, random_state=seed)
        X_ml, X_fourier, y = X_ml[test_idx], X_fourier[test_idx], y[test_idx]
        eval_label = f"UNSEEN HOLDOUT SET ({len(test_idx)} samples)"

    p_ml = model.predict_proba(X_ml)
    if triangulator is not None:
        p_fourier = triangulator.predict_proba(X_fourier)
        probs = (1.0 - w) * p_ml + w * p_fourier
    else:
        probs = p_ml

    preds = np.argmax(probs, axis=1)
    print(f"\n[EVALUATE - {eval_label}] Model: {model_file.name} | Samples: {len(y)}")
    print(f"Accuracy: {accuracy_score(y, preds)*100:.2f}% | Macro F1: {f1_score(y, preds, average='macro'):.4f}\n")
    print(classification_report(y, preds, target_names=classes))
    print("Confusion Matrix:")
    print(confusion_matrix(y, preds))

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate any saved .joblib model on a dataset")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Path to .joblib model file")
    parser.add_argument("--data-dir", "--dataset", dest="data_dir", type=Path, default=DATA_DIR, help="Path to evaluation dataset directory")
    parser.add_argument("--holdout-only", action="store_true", help="Evaluate strictly on the isolated test partition")
    return parser

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    evaluate_saved_model(args.model, args.data_dir, args.holdout_only)
