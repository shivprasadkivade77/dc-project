import csv
import argparse
from datetime import datetime
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from .config import DATA_DIR, MODEL_DIR, MODEL_PATH, CLASSES
from .features import extract_features

def train_and_benchmark(feature_set: str = "enhanced"):
    X, y = [], []
    for idx, label in enumerate(CLASSES):
        folder = DATA_DIR / label
        if not folder.exists():
            continue
        for file in folder.glob("*.npy"):
            waveform = np.load(file)
            X.append(extract_features(waveform, feature_set=feature_set))
            y.append(idx)

    X, y = np.array(X), np.array(y)
    if len(np.unique(y)) < 2:
        raise ValueError("Collect samples for at least 2 classes inside ./dataset before training!")

    print(f"\n[TRAIN] Loaded {len(X)} total samples across {len(np.unique(y))} classes.")
    print(f"Feature vector dimension: {X.shape[1]} ({feature_set} mode)\n")

    # Strict isolation: 80% train / 20% holdout test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Dataset split: {len(X_train)} training samples | {len(X_test)} holdout test samples\n")

    candidates = {
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42))
        ]),
        "ExtraTrees": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", ExtraTreesClassifier(n_estimators=150, random_state=42))
        ]),
        "SVM_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10.0, probability=True, random_state=42))
        ]),
    }

    versions_dir = MODEL_DIR / "versions"
    versions_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_path = MODEL_DIR / "leaderboard.csv"
    write_header = not leaderboard_path.exists()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_name, best_cv_score, best_pipeline = None, -1.0, None

    with open(leaderboard_path, mode="a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow([
                "timestamp", "model_name", "feature_set", "num_samples",
                "cv_f1_mean", "cv_f1_std", "holdout_f1", "saved_path"
            ])

        for name, pipeline in candidates.items():
            # Problem 1 Fix: Cross-validation runs strictly on X_train / y_train
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1_macro")
            cv_mean = float(np.mean(cv_scores))
            cv_std = float(np.std(cv_scores))

            # Fit candidate strictly on training split and evaluate unseen holdout
            pipeline.fit(X_train, y_train)
            test_preds = pipeline.predict(X_test)
            holdout_f1 = float(f1_score(y_test, test_preds, average="macro"))

            version_filename = f"{name}_{feature_set}_{timestamp}.joblib"
            version_path = versions_dir / version_filename

            joblib.dump({
                "model": pipeline,
                "model_name": name,
                "feature_set": feature_set,
                "classes": CLASSES,
                "cv_f1": cv_mean,
                "holdout_f1": holdout_f1
            }, version_path)

            writer.writerow([
                timestamp, name, feature_set, len(X_train),
                f"{cv_mean:.4f}", f"{cv_std:.4f}", f"{holdout_f1:.4f}",
                str(version_path.relative_to(MODEL_DIR.parent))
            ])

            print(f"  {name:14s} | Train 5-Fold F1: {cv_mean:.4f} (+/- {cv_std:.4f}) | Holdout F1: {holdout_f1:.4f}")
            print(f"                 -> Saved version: models/versions/{version_filename}")

            if cv_mean > best_cv_score:
                best_name, best_cv_score, best_pipeline = name, cv_mean, pipeline

    # Final holdout evaluation of winning pipeline
    best_preds = best_pipeline.predict(X_test)
    print(f"\n--> Winner: {best_name} (Train 5-Fold Macro F1: {best_cv_score:.4f})")
    print("\nHoldout Test Set Classification Report (Evaluated on untouched test split):")
    print(classification_report(y_test, best_preds, target_names=CLASSES))
    print("Confusion Matrix (Rows = Actual, Columns = Predicted):")
    print(confusion_matrix(y_test, best_preds))

    # Production model refit on full (X, y) for deployment binary
    best_pipeline.fit(X, y)
    joblib.dump({
        "model": best_pipeline,
        "model_name": best_name,
        "feature_set": feature_set,
        "classes": CLASSES,
        "cv_f1": best_cv_score
    }, MODEL_PATH)
    print(f"\nUpdated default production model -> {MODEL_PATH}")
    print(f"Logged all model metrics to        -> {leaderboard_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark, train, and save all model versions")
    parser.add_argument("--features", choices=["baseline", "enhanced"], default="enhanced")
    args = parser.parse_args()
    train_and_benchmark(args.features)
