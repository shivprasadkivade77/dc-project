import csv
from datetime import datetime
from pathlib import Path
import joblib
import librosa
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from .config import AUTO_TUNE_WEIGHT, CLASSES, DATA_DIR, FOURIER_WEIGHT, MODEL_DIR, MODEL_PATH, RESULTS_DIR, SAMPLE_RATE, ULTRASONIC_CFG, VERSIONS_DIR
from .features import extract_features
from .triangulation import FourierTriangulator

def load_dataset(data_dir=DATA_DIR, sr=SAMPLE_RATE):
    waveforms, labels = [], []
    if data_dir.exists():
        for cls_dir in sorted(data_dir.iterdir()):
            if cls_dir.is_dir():
                for fpath in list(cls_dir.glob("*.wav")) + list(cls_dir.glob("*.npy")):
                    if fpath.suffix == ".npy":
                        w = np.load(fpath).astype(np.float32).flatten()
                    else:
                        w, _ = librosa.load(str(fpath), sr=sr, mono=True)
                    if len(w) >= 128:
                        waveforms.append(w.astype(np.float32))
                        labels.append(cls_dir.name)
    if len(waveforms) < 15:
        print("[Bootstrap] Generating physics-informed bootstrap dataset...")
        rng = np.random.default_rng(42)
        n_samples = int(0.16 * sr)
        t = np.arange(n_samples, dtype=np.float32) / sr
        onset = n_samples // 3
        t_post = np.maximum(0.0, t - t[onset])
        env = np.where(t >= t[onset], np.exp(-t_post * 95.0), 0.0)
        for cls in CLASSES:
            for _ in range(30):
                noise = rng.normal(0, 0.0025, size=n_samples).astype(np.float32)
                if cls == "left_tap":
                    click = env * (0.35 * np.sin(2 * np.pi * 1800 * t_post) + 0.25 * np.sin(2 * np.pi * 4200 * t_post))
                    ultra = 0.03 * np.sin(2 * np.pi * (19000 + 85 * env) * t) + 0.025 * np.sin(2 * np.pi * 20200 * t)
                elif cls == "right_tap":
                    click = env * (0.35 * np.sin(2 * np.pi * 1150 * t_post) + 0.25 * np.sin(2 * np.pi * 2950 * t_post))
                    ultra = 0.025 * np.sin(2 * np.pi * 19000 * t) + 0.03 * np.sin(2 * np.pi * (20200 + 85 * env) * t)
                else:
                    click = env * (0.40 * np.sin(2 * np.pi * 260 * t_post) + 0.15 * np.sin(2 * np.pi * 520 * t_post))
                    ultra = 0.025 * np.sin(2 * np.pi * 19000 * t) + 0.025 * np.sin(2 * np.pi * 20200 * t)
                waveforms.append((noise + click + ultra).astype(np.float32))
                labels.append(cls)
    return waveforms, np.array(labels)

def get_candidate_models():
    return {
        "ExtraTrees_V3": Pipeline([("scaler", StandardScaler()), ("clf", ExtraTreesClassifier(n_estimators=250, class_weight="balanced", random_state=42))]),
        "RandomForest_V3": Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=250, class_weight="balanced", random_state=42))]),
        "SVM_RBF_V3": Pipeline([("scaler", StandardScaler()), ("clf", SVC(C=3.0, kernel="rbf", probability=True, class_weight="balanced", random_state=42))]),
    }

def evaluate_hybrid_cv(waveforms, X_feat, y, model_pipe, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    classes = np.unique(y)
    oof_ml = np.zeros((len(y), len(classes)), dtype=np.float64)
    oof_four = np.zeros((len(y), len(classes)), dtype=np.float64)
    for train_idx, val_idx in skf.split(X_feat, y):
        model_pipe.fit(X_feat[train_idx], y[train_idx])
        oof_ml[val_idx] = model_pipe.predict_proba(X_feat[val_idx])
        tri = FourierTriangulator(sr=SAMPLE_RATE).fit([waveforms[i] for i in train_idx], y[train_idx])
        oof_four[val_idx] = tri.predict_proba([waveforms[i] for i in val_idx])
    best_w, best_f1 = FOURIER_WEIGHT, -1.0
    for w in (np.linspace(0.15, 0.85, 15) if AUTO_TUNE_WEIGHT else [FOURIER_WEIGHT]):
        blended = (1.0 - w) * oof_ml + w * oof_four
        score = float(f1_score(y, classes[np.argmax(blended, axis=1)], average="macro"))
        if score > best_f1:
            best_f1, best_w = score, float(round(w, 2))
    return best_f1, best_w

def train_and_benchmark():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    waveforms, y = load_dataset(DATA_DIR, sr=SAMPLE_RATE)
    X_feat = np.vstack([extract_features(w, sr=SAMPLE_RATE, feature_set="all") for w in waveforms])
    tr_idx, te_idx = train_test_split(np.arange(len(y)), test_size=0.20, stratify=y, random_state=42)
    w_train, w_test = [waveforms[i] for i in tr_idx], [waveforms[i] for i in te_idx]
    X_train, X_test, y_train, y_test = X_feat[tr_idx], X_feat[te_idx], y[tr_idx], y[te_idx]
    
    leaderboard_rows, best_overall = [], None
    for name, pipe in get_candidate_models().items():
        cv_f1, opt_w = evaluate_hybrid_cv(w_train, X_train, y_train, pipe, n_splits=min(5, np.min(np.unique(y_train, return_counts=True)[1])))
        pipe.fit(X_train, y_train)
        tri = FourierTriangulator(sr=SAMPLE_RATE).fit(w_train, y_train)
        probs_hyb = (1.0 - opt_w) * pipe.predict_proba(X_test) + opt_w * tri.predict_proba(w_test)
        y_pred = tri.classes_[np.argmax(probs_hyb, axis=1)]
        test_acc, test_f1 = float(accuracy_score(y_test, y_pred)), float(f1_score(y_test, y_pred, average="macro"))
        print(f"-> {name:18s} | CV Macro-F1: {cv_f1:.4f} | Test F1: {test_f1:.4f}")
        row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "version": "v3-hyb-four-US", "model_name": name, "feature_set": "all+ultrasonic", "fourier_weight": opt_w, "cv_f1": round(cv_f1, 4), "test_acc": round(test_acc, 4), "test_f1": round(test_f1, 4)}
        leaderboard_rows.append(row)
        if best_overall is None or cv_f1 > best_overall["cv_f1"]:
            best_overall = {**row, "pipe": pipe, "tri": tri, "y_test": y_test, "y_pred": y_pred, "classes": list(tri.classes_)}

    win_name, win_w = best_overall["model_name"], best_overall["fourier_weight"]
    final_pipe = get_candidate_models()[win_name].fit(X_feat, y)
    final_tri = FourierTriangulator(sr=SAMPLE_RATE).fit(waveforms, y)
    ts_slug = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact = {"version": "3.0.0-hybrid-fourier-ultrasonic", "model": final_pipe, "triangulator": final_tri, "fourier_weight": win_w, "model_name": win_name, "feature_set": "all", "classes": list(final_tri.classes_), "cv_f1": best_overall["cv_f1"], "test_f1": best_overall["test_f1"], "sample_rate": SAMPLE_RATE, "timestamp": ts_slug}
    joblib.dump(artifact, MODEL_PATH)
    joblib.dump(artifact, VERSIONS_DIR / f"model_v3_{win_name}_{ts_slug}.joblib")
    
    lb_path = MODEL_DIR / "leaderboard.csv"
    with open(lb_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(leaderboard_rows[0].keys()))
        if not lb_path.exists() or lb_path.stat().st_size == 0:
            writer.writeheader()
        writer.writerows(leaderboard_rows)

    fence = "`" * 3
    cls = list(best_overall["classes"])
    names = [str(c) for c in cls]
    yt, yp = best_overall["y_test"], best_overall["y_pred"]
    rep = classification_report(yt, yp, labels=cls, target_names=names, zero_division=0)
    cm = confusion_matrix(yt, yp, labels=cls)
    report_md = (
        f"# AcousticTap Hybrid Test Report ({ts_slug})\n\n"
        f"- **Winning Architecture:** `{win_name}` ({round((1 - win_w) * 100)}% ML + {round(win_w * 100)}% Fourier Triangulation)\n"
        f"- **Total Dataset Samples:** {len(y)} ({len(y_train)} train / {len(y_test)} holdout)\n"
        f"- **5-Fold Cross-Validation Macro F1:** `{best_overall['cv_f1']:.4f}`\n"
        f"- **Holdout Test Accuracy:** `{best_overall['test_acc'] * 100:.2f}%`\n"
        f"- **Holdout Test Macro F1:** `{best_overall['test_f1']:.4f}`\n\n"
        f"## Holdout Classification Report\n{fence}text\n{rep}\n{fence}\n\n"
        f"## Confusion Matrix (`{names}`)\n{fence}text\n{cm}\n{fence}\n"
    )
    (RESULTS_DIR / "latest_test_report.md").write_text(report_md, encoding="utf-8")
    print(f"Training complete! Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_and_benchmark()
