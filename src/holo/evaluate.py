import argparse
from pathlib import Path
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from .config import DATA_DIR, MODEL_PATH, CLASSES
from .features import extract_features
from .triangulation import extract_fourier_triangulation_vector

def evaluate_saved_model(model_file: Path):
    saved = joblib.load(model_file)
    model = saved["model"]
    triangulator = saved.get("triangulator", None)
    w = saved.get("fourier_weight", 0.50)
    feature_set = saved.get("feature_set", "enhanced")
    classes = saved.get("classes", CLASSES)

    X_ml, X_fourier, y = [], [], []
    for idx, label in enumerate(classes):
        folder = DATA_DIR / label
        if not folder.exists():
            continue
        for file in sorted(folder.glob("*.npy")):
            waveform = np.load(file)
            X_ml.append(extract_features(waveform, feature_set=feature_set))
            X_fourier.append(extract_fourier_triangulation_vector(waveform))
            y.append(idx)

    X_ml, X_fourier, y = np.array(X_ml), np.array(X_fourier), np.array(y)
    p_ml = model.predict_proba(X_ml)
    if triangulator is not None:
        p_fourier = triangulator.predict_proba(X_fourier)
        probs = (1.0 - w) * p_ml + w * p_fourier
    else:
        probs = p_ml

    preds = np.argmax(probs, axis=1)
    print(f"\n[EVALUATE] Model: {model_file.name} | Total Samples: {len(y)}")
    print(f"Accuracy: {accuracy_score(y, preds)*100:.2f}% | Macro F1: {f1_score(y, preds, average='macro'):.4f}\n")
    print(classification_report(y, preds, target_names=classes))
    print("Confusion Matrix:")
    print(confusion_matrix(y, preds))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate any saved .joblib model on ./dataset")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Path to .joblib model file")
    args = parser.parse_args()
    evaluate_saved_model(args.model)
