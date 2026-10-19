import csv
import json
import argparse
from datetime import datetime
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from .config import DATA_DIR, MODEL_DIR, VERSIONS_DIR, RESULTS_DIR, MODEL_PATH, CLASSES, FOURIER_WEIGHT
from .features import extract_features
from .triangulation import extract_fourier_triangulation_vector, FourierTriangulator

def evaluate_hybrid_cv(ml_pipeline, X_ml, X_fourier, y, fourier_weight=FOURIER_WEIGHT, n_splits=5):
    """Evaluates ML-only, Fourier-Triangulation-only, and 50/50 Hybrid across 5 stratified folds."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    ml_f1s, fourier_f1s, hybrid_f1s = [], [], []

    for train_idx, val_idx in cv.split(X_ml, y):
        ml_pipeline.fit(X_ml[train_idx], y[train_idx])
        triangulator = FourierTriangulator().fit(X_fourier[train_idx], y[train_idx])

        p_ml = ml_pipeline.predict_proba(X_ml[val_idx])
        p_fourier = triangulator.predict_proba(X_fourier[val_idx])
        p_hybrid = (1.0 - fourier_weight) * p_ml + fourier_weight * p_fourier

        ml_f1s.append(f1_score(y[val_idx], np.argmax(p_ml, axis=1), average="macro"))
        fourier_f1s.append(f1_score(y[val_idx], np.argmax(p_fourier, axis=1), average="macro"))
        hybrid_f1s.append(f1_score(y[val_idx], np.argmax(p_hybrid, axis=1), average="macro"))

    return float(np.mean(ml_f1s)), float(np.mean(fourier_f1s)), float(np.mean(hybrid_f1s)), float(np.std(hybrid_f1s))

def train_and_benchmark(feature_set: str = "enhanced", fourier_weight: float = FOURIER_WEIGHT):
    X_ml, X_fourier, y = [], [], []
    for idx, label in enumerate(CLASSES):
        folder = DATA_DIR / label
        if not folder.exists():
            continue
        for file in sorted(folder.glob("*.npy")):
            waveform = np.load(file)
            X_ml.append(extract_features(waveform, feature_set=feature_set))
            X_fourier.append(extract_fourier_triangulation_vector(waveform))
            y.append(idx)

    X_ml, X_fourier, y = np.array(X_ml), np.array(X_fourier), np.array(y)
    if len(np.unique(y)) < 2:
        raise ValueError("Collect samples for at least 2 classes inside ./dataset before training!")

    print(f"\n[TRAIN HYBRID] Loaded {len(y)} samples across {len(np.unique(y))} classes.")
    print(f"ML Feature Dim: {X_ml.shape[1]} | Fourier Triangulation Dim: {X_fourier.shape[1]} | Fusion Weight: {fourier_weight:.2f}\n")

    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=y, random_state=42)

    candidates = {
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, max_depth=14, random_state=42))
        ]),
        "ExtraTrees": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", ExtraTreesClassifier(n_estimators=200, random_state=42))
        ]),
        "SVM_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10.0, probability=True, random_state=42))
        ]),
    }

    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    leaderboard_path = MODEL_DIR / "leaderboard.csv"
    write_header = not leaderboard_path.exists()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_name, best_hybrid_cv, best_ml_pipe = None, -1.0, None

    with open(leaderboard_path, mode="a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow([
                "timestamp", "model_name", "feature_set", "num_samples",
                "ml_only_cv_f1", "fourier_only_cv_f1", "hybrid_cv_f1", "holdout_hybrid_f1", "saved_path"
            ])

        for name, pipeline in candidates.items():
            ml_cv, fourier_cv, hybrid_cv, hybrid_std = evaluate_hybrid_cv(
                pipeline, X_ml, X_fourier, y, fourier_weight=fourier_weight
            )

            # Holdout evaluation
            pipeline.fit(X_ml[train_idx], y[train_idx])
            tri_holdout = FourierTriangulator().fit(X_fourier[train_idx], y[train_idx])
            p_ml_test = pipeline.predict_proba(X_ml[test_idx])
            p_tri_test = tri_holdout.predict_proba(X_fourier[test_idx])
            p_hyb_test = (1.0 - fourier_weight) * p_ml_test + fourier_weight * p_tri_test
            preds_test = np.argmax(p_hyb_test, axis=1)
            holdout_f1 = float(f1_score(y[test_idx], preds_test, average="macro"))

            # Refit both ML and Fourier Triangulator on 100% of dataset and save version
            pipeline.fit(X_ml, y)
            tri_full = FourierTriangulator().fit(X_fourier, y)
            version_filename = f"Hybrid_{name}_{feature_set}_{timestamp}.joblib"
            version_path = VERSIONS_DIR / version_filename

            artifact = {
                "model": pipeline,
                "triangulator": tri_full,
                "fourier_weight": fourier_weight,
                "model_name": f"Hybrid_{name}",
                "feature_set": feature_set,
                "classes": CLASSES,
                "cv_f1": hybrid_cv,
                "holdout_f1": holdout_f1,
                "timestamp": timestamp
            }
            joblib.dump(artifact, version_path)

            writer.writerow([
                timestamp, f"Hybrid_{name}", feature_set, len(y),
                f"{ml_cv:.4f}", f"{fourier_cv:.4f}", f"{hybrid_cv:.4f}", f"{holdout_f1:.4f}",
                str(version_path.relative_to(MODEL_DIR.parent))
            ])

            print(f"  {name:13s} | ML-Only F1: {ml_cv:.3f} | Fourier-Only F1: {fourier_cv:.3f} | 50/50 Hybrid F1: {hybrid_cv:.3f} (+/- {hybrid_std:.3f}) | Holdout: {holdout_f1:.3f}")

            if hybrid_cv > best_hybrid_cv:
                best_name, best_hybrid_cv, best_ml_pipe = name, hybrid_cv, pipeline

    # Detailed evaluation report for the winning hybrid model on the 20% holdout set
    best_ml_pipe.fit(X_ml[train_idx], y[train_idx])
    best_tri = FourierTriangulator().fit(X_fourier[train_idx], y[train_idx])
    p_hyb = (1.0 - fourier_weight) * best_ml_pipe.predict_proba(X_ml[test_idx]) + fourier_weight * best_tri.predict_proba(X_fourier[test_idx])
    best_preds = np.argmax(p_hyb, axis=1)
    holdout_acc = float(accuracy_score(y[test_idx], best_preds))
    report_str = classification_report(y[test_idx], best_preds, target_names=CLASSES)
    cm = confusion_matrix(y[test_idx], best_preds)

    print(f"\n--> Winner: Hybrid_{best_name} (5-Fold Hybrid Macro F1: {best_hybrid_cv:.4f}, Holdout Acc: {holdout_acc*100:.1f}%)")
    print("\nHoldout Test Set Classification Report:")
    print(report_str)
    print("Confusion Matrix (Rows = Actual, Columns = Predicted):")
    print(cm)

    # Save test result report in models/test_results/ so it can be committed to GitHub
    report_md_path = RESULTS_DIR / f"test_report_{timestamp}.md"
    latest_md_path = RESULTS_DIR / "latest_test_report.md"
    md_content = (
        f"# AcousticTap Hybrid Test Report ({timestamp})\n\n"
        f"- **Winning Architecture:** `Hybrid_{best_name}` ({int((1-fourier_weight)*100)}% ML + {int(fourier_weight*100)}% Fourier Transform Triangulation)\n"
        f"- **Total Dataset Samples:** {len(y)}\n"
        f"- **5-Fold Cross-Validation Macro F1:** `{best_hybrid_cv:.4f}`\n"
        f"- **Holdout Test Accuracy:** `{holdout_acc*100:.2f}%`\n\n"
        f"## Holdout Classification Report\n```text\n{report_str}\n```\n\n"
        f"## Confusion Matrix (`{CLASSES}`)\n```text\n{cm}\n```\n"
    )
    report_md_path.write_text(md_content)
    latest_md_path.write_text(md_content)

    # Refit winner on 100% of data and save as default production model
    best_ml_pipe.fit(X_ml, y)
    best_tri_full = FourierTriangulator().fit(X_fourier, y)
    joblib.dump({
        "model": best_ml_pipe,
        "triangulator": best_tri_full,
        "fourier_weight": fourier_weight,
        "model_name": f"Hybrid_{best_name}",
        "feature_set": feature_set,
        "classes": CLASSES,
        "cv_f1": best_hybrid_cv,
        "timestamp": timestamp
    }, MODEL_PATH)

    print(f"\nSaved default hybrid model -> {MODEL_PATH}")
    print(f"Saved GitHub test report   -> {latest_md_path}")
    print(f"Updated leaderboard        -> {leaderboard_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 50/50 Hybrid ML + Fourier Triangulation models")
    parser.add_argument("--features", choices=["baseline", "enhanced"], default="enhanced")
    parser.add_argument("--fourier-weight", type=float, default=FOURIER_WEIGHT, help="Weight for Fourier Triangulator (0.0 to 1.0)")
    args = parser.parse_args()
    train_and_benchmark(args.features, args.fourier_weight)
