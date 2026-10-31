import csv
import argparse
from datetime import datetime
import numpy as np
import joblib
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from .config import CFG, DATA_DIR, MODEL_DIR, VERSIONS_DIR, RESULTS_DIR, MODEL_PATH, CLASSES
from .features import extract_features
from .triangulation import extract_fourier_triangulation_vector, FourierTriangulator

def build_candidates_from_config():
    tr_cfg = CFG["training"]
    m_cfg = tr_cfg["models"]
    seed = int(tr_cfg["random_seed"])

    svm_cw = m_cfg["SVM_RBF"].get("class_weight")
    if isinstance(svm_cw, dict):
        svm_cw = {int(k): float(v) for k, v in svm_cw.items()}

    candidates = {
        "SVM_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                C=float(m_cfg["SVM_RBF"]["C"]),
                gamma=m_cfg["SVM_RBF"]["gamma"],
                kernel=m_cfg["SVM_RBF"].get("kernel", "rbf"),
                class_weight=svm_cw,
                probability=True,
                random_state=seed
            ))
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=int(m_cfg["RandomForest"]["n_estimators"]),
                max_depth=m_cfg["RandomForest"]["max_depth"],
                min_samples_split=int(m_cfg["RandomForest"].get("min_samples_split", 2)),
                class_weight=m_cfg["RandomForest"].get("class_weight"),
                random_state=seed
            ))
        ]),
        "ExtraTrees": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", ExtraTreesClassifier(
                n_estimators=int(m_cfg["ExtraTrees"]["n_estimators"]),
                max_depth=m_cfg["ExtraTrees"]["max_depth"],
                min_samples_split=int(m_cfg["ExtraTrees"].get("min_samples_split", 2)),
                class_weight=m_cfg["ExtraTrees"].get("class_weight"),
                random_state=seed
            ))
        ]),
    }
    return candidates

def evaluate_hybrid_cv(ml_pipeline, X_ml, X_fourier, y, weight_grid, n_splits=5, seed=42):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_ml = np.zeros((len(y), len(np.unique(y))))
    oof_fourier = np.zeros((len(y), len(np.unique(y))))

    for train_idx, val_idx in cv.split(X_ml, y):
        pipe_fold = clone(ml_pipeline)
        pipe_fold.fit(X_ml[train_idx], y[train_idx])
        tri_fold = FourierTriangulator().fit(X_fourier[train_idx], y[train_idx])

        oof_ml[val_idx] = pipe_fold.predict_proba(X_ml[val_idx])
        oof_fourier[val_idx] = tri_fold.predict_proba(X_fourier[val_idx])

    ml_f1 = float(f1_score(y, np.argmax(oof_ml, axis=1), average="macro"))
    fourier_f1 = float(f1_score(y, np.argmax(oof_fourier, axis=1), average="macro"))

    best_w, best_hyb_f1 = weight_grid[0], -1.0
    for w in weight_grid:
        p_hyb = (1.0 - w) * oof_ml + w * oof_fourier
        score = float(f1_score(y, np.argmax(p_hyb, axis=1), average="macro"))
        if score > best_hyb_f1:
            best_hyb_f1 = score
            best_w = float(w)

    return ml_f1, fourier_f1, best_hyb_f1, best_w

def train_and_benchmark(feature_set: str | None = None, fourier_weight: float | None = None):
    tr_cfg = CFG["training"]
    hyb_cfg = CFG["hybrid_fusion"]
    mode = feature_set if feature_set is not None else CFG["features"]["mode"]
    cv_folds = int(tr_cfg["cv_folds"])
    seed = int(tr_cfg["random_seed"])
    do_opt = bool(tr_cfg.get("optimize_hyperparameters", True))

    if fourier_weight is not None:
        weight_grid = [float(fourier_weight)]
    elif hyb_cfg.get("auto_tune_weight", True):
        weight_grid = [float(w) for w in hyb_cfg["weight_search_grid"]]
    else:
        weight_grid = [float(hyb_cfg["fourier_weight"])]

    X_ml, X_fourier, y = [], [], []
    for idx, label in enumerate(CLASSES):
        folder = DATA_DIR / label
        if not folder.exists():
            continue
        for file in sorted(folder.glob("*.npy")):
            waveform = np.load(file)
            X_ml.append(extract_features(waveform, feature_set=mode))
            X_fourier.append(extract_fourier_triangulation_vector(waveform))
            y.append(idx)

    X_ml, X_fourier, y = np.array(X_ml), np.array(X_fourier), np.array(y)
    if len(np.unique(y)) < 2:
        raise ValueError("Collect samples for at least 2 classes inside ./dataset before training!")

    # Strict isolation: partition 80% train / 20% holdout test before any optimization
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices, test_size=float(tr_cfg["test_size"]), stratify=y, random_state=seed
    )
    X_ml_tr, y_tr = X_ml[train_idx], y[train_idx]
    X_fourier_tr = X_fourier[train_idx]
    X_ml_te, y_te = X_ml[test_idx], y[test_idx]
    X_fourier_te = X_fourier[test_idx]

    print(f"\n[TRAIN HYBRID] Total Samples: {len(y)} | Train: {len(train_idx)} | Holdout: {len(test_idx)}")
    print(f"ML Feature Dim: {X_ml.shape[1]} | Fourier Dim: {X_fourier.shape[1]}")
    print(f"Hyperparameter Tuning: {'ENABLED' if do_opt else 'DISABLED'} | Weight Grid: {weight_grid}\n")

    candidates = build_candidates_from_config()
    param_grids = tr_cfg.get("param_grids", {})

    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    leaderboard_path = MODEL_DIR / "leaderboard.csv"
    write_header = not leaderboard_path.exists()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_name, best_hybrid_cv, best_ml_pipe, best_opt_w, best_params = None, -1.0, None, 0.5, {}

    with open(leaderboard_path, mode="a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow([
                "timestamp", "model_name", "feature_set", "num_samples",
                "ml_only_cv_f1", "fourier_only_cv_f1", "hybrid_cv_f1", "holdout_hybrid_f1",
                "optimal_fourier_weight", "saved_path"
            ])

        for name, base_pipeline in candidates.items():
            if do_opt and name in param_grids:
                raw_grid = param_grids[name]
                clean_grid = {}
                for k, v_list in raw_grid.items():
                    clean_list = []
                    for item in v_list:
                        if isinstance(item, dict):
                            clean_list.append({int(ck): float(cv) for ck, cv in item.items()})
                        else:
                            clean_list.append(item)
                    clean_grid[k] = clean_list

                # Critical Fix 1: GridSearchCV runs strictly on (X_ml_tr, y_tr)
                cv_obj = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
                gs = GridSearchCV(base_pipeline, clean_grid, cv=cv_obj, scoring="f1_macro", n_jobs=-1)
                gs.fit(X_ml_tr, y_tr)
                tuned_pipeline = gs.best_estimator_
                tuned_params = gs.best_params_
            else:
                tuned_pipeline = base_pipeline
                tuned_params = {}

            # Critical Fix 2: evaluate_hybrid_cv runs strictly on the training split
            ml_cv, fourier_cv, hybrid_cv, opt_w = evaluate_hybrid_cv(
                tuned_pipeline, X_ml_tr, X_fourier_tr, y_tr, weight_grid=weight_grid, n_splits=cv_folds, seed=seed
            )

            # Holdout evaluation on untouched test split
            holdout_pipe = clone(tuned_pipeline).fit(X_ml_tr, y_tr)
            tri_holdout = FourierTriangulator().fit(X_fourier_tr, y_tr)
            p_ml_test = holdout_pipe.predict_proba(X_ml_te)
            p_tri_test = tri_holdout.predict_proba(X_fourier_te)
            p_hyb_test = (1.0 - opt_w) * p_ml_test + opt_w * p_tri_test
            preds_test = np.argmax(p_hyb_test, axis=1)
            holdout_f1 = float(f1_score(y_te, preds_test, average="macro"))

            # Refit on 100% dataset strictly for deployment artifact version
            full_pipe = clone(tuned_pipeline).fit(X_ml, y)
            tri_full = FourierTriangulator().fit(X_fourier, y)
            version_filename = f"Hybrid_{name}_{mode}_{timestamp}.joblib"
            version_path = VERSIONS_DIR / version_filename

            joblib.dump({
                "model": full_pipe,
                "triangulator": tri_full,
                "fourier_weight": opt_w,
                "model_name": f"Hybrid_{name}",
                "best_params": tuned_params,
                "feature_set": mode,
                "classes": CLASSES,
                "cv_f1": hybrid_cv,
                "holdout_f1": holdout_f1,
                "timestamp": timestamp
            }, version_path)

            writer.writerow([
                timestamp, f"Hybrid_{name}", mode, len(train_idx),
                f"{ml_cv:.4f}", f"{fourier_cv:.4f}", f"{hybrid_cv:.4f}", f"{holdout_f1:.4f}",
                f"{opt_w:.2f}", str(version_path.relative_to(MODEL_DIR.parent))
            ])

            print(f"  {name:13s} | ML F1: {ml_cv:.3f} | FFT F1: {fourier_cv:.3f} | Hybrid F1: {hybrid_cv:.4f} (w={opt_w:.2f}) | Holdout: {holdout_f1:.4f}")
            if tuned_params:
                print(f"                  Best Params: {tuned_params}")

            if hybrid_cv > best_hybrid_cv:
                best_name, best_hybrid_cv, best_ml_pipe, best_opt_w, best_params = (
                    name, hybrid_cv, tuned_pipeline, opt_w, tuned_params
                )

    # Holdout report for winning architecture (evaluated strictly on untouched test split)
    win_holdout_pipe = clone(best_ml_pipe).fit(X_ml_tr, y_tr)
    win_tri = FourierTriangulator().fit(X_fourier_tr, y_tr)
    p_hyb = (1.0 - best_opt_w) * win_holdout_pipe.predict_proba(X_ml_te) + best_opt_w * win_tri.predict_proba(X_fourier_te)
    best_preds = np.argmax(p_hyb, axis=1)
    holdout_acc = float(accuracy_score(y_te, best_preds))
    report_str = classification_report(y_te, best_preds, target_names=CLASSES)
    cm = confusion_matrix(y_te, best_preds)

    print(f"\n--> Winner: Hybrid_{best_name} | Train 5-Fold F1: {best_hybrid_cv:.4f} | Holdout Acc: {holdout_acc*100:.2f}% | Optimal FFT Weight: {best_opt_w:.2f}")
    print("\nHoldout Test Set Classification Report (Evaluated on untouched test split):")
    print(report_str)
    print("Confusion Matrix (Rows = Actual, Columns = Predicted):")
    print(cm)

    report_md_path = RESULTS_DIR / f"test_report_{timestamp}.md"
    latest_md_path = RESULTS_DIR / "latest_test_report.md"
    md_content = (
        f"# AcousticTap Hybrid Test Report ({timestamp})\n\n"
        f"- **Winning Architecture:** `Hybrid_{best_name}` ({int((1-best_opt_w)*100)}% ML + {int(best_opt_w*100)}% Fourier Transform Triangulation)\n"
        f"- **Best Tuned Hyperparameters:** `{best_params}`\n"
        f"- **Total Dataset Samples:** {len(y)} ({len(train_idx)} train / {len(test_idx)} holdout)\n"
        f"- **5-Fold Cross-Validation Macro F1:** `{best_hybrid_cv:.4f}`\n"
        f"- **Holdout Test Accuracy:** `{holdout_acc*100:.2f}%`\n\n"
        f"## Holdout Classification Report\n```text\n{report_str}\n```\n\n"
        f"## Confusion Matrix (`{CLASSES}`)\n```text\n{cm}\n```\n"
    )
    report_md_path.write_text(md_content)
    latest_md_path.write_text(md_content)

    final_ml = clone(best_ml_pipe).fit(X_ml, y)
    final_tri = FourierTriangulator().fit(X_fourier, y)
    joblib.dump({
        "model": final_ml,
        "triangulator": final_tri,
        "fourier_weight": best_opt_w,
        "model_name": f"Hybrid_{best_name}",
        "best_params": best_params,
        "feature_set": mode,
        "classes": CLASSES,
        "cv_f1": best_hybrid_cv,
        "timestamp": timestamp
    }, MODEL_PATH)

    print(f"\nSaved production model -> {MODEL_PATH}")
    print(f"Saved test report      -> {latest_md_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and optimize Hybrid AcousticTap models via config/config.yaml")
    parser.add_argument("--features", choices=["baseline", "enhanced"], default=None)
    parser.add_argument("--fourier-weight", type=float, default=None)
    args = parser.parse_args()
    train_and_benchmark(args.features, args.fourier_weight)
