import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from .config import MODEL_PATH, SAMPLE_RATE
from .features import extract_features
from .train import load_dataset
from .ultrasonic import UltrasonicSonarGate

def main():
    if not MODEL_PATH.exists():
        from .train import train_and_benchmark
        train_and_benchmark()
    bundle = joblib.load(MODEL_PATH)
    model, tri, w_four, classes = bundle["model"], bundle["triangulator"], float(bundle.get("fourier_weight", 0.50)), list(bundle["classes"])
    waveforms, y_true = load_dataset(sr=SAMPLE_RATE)
    X_feat = np.vstack([extract_features(w, sr=SAMPLE_RATE) for w in waveforms])
    from sklearn.model_selection import train_test_split
    y_true = np.asarray(y_true)
    _, te_idx = train_test_split(np.arange(len(y_true)), test_size=0.20, stratify=y_true, random_state=42)
    X_feat, y_true = X_feat[te_idx], y_true[te_idx]
    waveforms = [waveforms[i] for i in te_idx]
    p_hyb = (1.0 - w_four) * model.predict_proba(X_feat) + w_four * tri.predict_proba(waveforms)
    sonar = UltrasonicSonarGate(sr=SAMPLE_RATE)
    y_pred = []
    for i, w in enumerate(waveforms):
        probs = {cls: float(p_hyb[i, j]) for j, cls in enumerate(classes)}
        us = sonar.evaluate_window(w)
        if us["active"]:
            if "left_tap" in probs: probs["left_tap"] *= us["gate_left"]
            if "right_tap" in probs: probs["right_tap"] *= us["gate_right"]
        y_pred.append(max(probs, key=probs.get))
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f} | F1: {f1_score(y_true, y_pred, average="macro"):.4f}")

if __name__ == "__main__":
    main()
