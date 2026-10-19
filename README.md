# AcousticTap (holo) — V2 Hybrid Fourier Model

AcousticTap turns the aluminum palm rest of a MacBook into virtual, programmable macro buttons using structure-borne acoustic sensing. V2 classifies each tap with a **hybrid model**: a classical machine-learning classifier on MFCC/spectral features, fused with a Fourier-transform triangulation classifier that works on the tap's frequency-domain signature.

## How the V2 Hybrid Model Works

1. **Capture.** A pre-trigger ring buffer records a 128 ms acoustic transient around each tap.
2. **Two feature branches** run on the same waveform:
   - **ML branch:** short-window (512-sample) MFCC, Delta-MFCC and spectral features (`baseline` or `enhanced` set) go through a `StandardScaler` and one of three candidate classifiers: RBF-SVM, Random Forest or Extra Trees.
   - **Fourier branch:** a Fourier-transform triangulation vector is extracted from the waveform and classified by `FourierTriangulator`, which outputs its own class probabilities.
3. **Probability fusion.** The two probability vectors are blended with a single weight `w`:

   ```text
   P_hybrid = (1 - w) * P_ml + w * P_fourier
   ```

   The predicted class is the `argmax` of `P_hybrid`. The live listener compares the winning probability against the `--confidence` threshold before triggering a macro.
4. **Weight tuning.** Unless you fix it with `--fourier-weight`, `w` is searched over `weight_search_grid` from the config. Each candidate weight is scored on out-of-fold predictions from 5-fold stratified cross-validation, and the weight with the best macro F1 wins.
5. **Model selection.** Every candidate classifier is optionally tuned with `GridSearchCV` (macro F1), combined with the Fourier branch, and benchmarked. The architecture with the best hybrid cross-validated F1 becomes the production model.
6. **Evaluation and final fit.** Each candidate is also scored on a stratified holdout split. The saved models are then refit on 100% of the dataset.

## Architecture

- `src/holo/config.py`: Central constants and path definitions (also reads `config/config.yaml` for training and fusion settings).
- `src/holo/detector.py`: Pre-trigger ring buffer for capturing 128 ms acoustic transients.
- `src/holo/features.py`: Short-window (512-sample) MFCC, Delta-MFCC and spectral feature extraction for the ML branch.
- `src/holo/triangulation.py`: Fourier-transform triangulation vector extraction and the `FourierTriangulator` classifier for the Fourier branch.
- `src/holo/collect.py`: Interactive CLI tool for recording training taps.
- `src/holo/train.py`: Hybrid training and benchmarking (SVM, Random Forest, Extra Trees, each fused with the Fourier branch), hyperparameter tuning, fusion-weight search, version logging and test-report generation.
- `src/holo/update_readme.py`: Copies the latest test report into this README.
- `src/holo/listen.py`: Low-latency macOS listener daemon with keyboard typing suppression and dry-run testing.

## Usage with uv

1. Collect training samples:

   ```bash
   uv run python -m src.holo.collect --label left_tap --samples 50
   uv run python -m src.holo.collect --label right_tap --samples 50
   uv run python -m src.holo.collect --label noise --samples 50
   ```

2. Train and benchmark the hybrid models (fusion weight is auto-tuned):

   ```bash
   uv run python -m src.holo.train --features enhanced
   ```

   To fix the Fourier fusion weight instead of tuning it (here 50% ML / 50% Fourier):

   ```bash
   uv run python -m src.holo.train --features enhanced --fourier-weight 0.50
   ```

   Training writes:
   - a versioned bundle per candidate, named `Hybrid_<model>_<features>_<timestamp>.joblib`
   - a row per candidate in `leaderboard.csv`
   - `latest_test_report.md` (plus a timestamped copy)
   - the production model used by the listener

3. Update this README with the latest test report:

   ```bash
   uv run python -m src.holo.update_readme
   ```

4. Test live predictions safely (without triggering macOS shortcuts):

   ```bash
   uv run python -m src.holo.listen --dry-run
   ```

5. Run the live macOS macro daemon:

   ```bash
   uv run python -m src.holo.listen --confidence 0.75
   ```

<!-- TEST_REPORT_START -->
## Latest Test Report

## AcousticTap Hybrid Test Report (20261019_165703)

- **Winning Architecture:** `Hybrid_SVM_RBF` (50% ML + 50% Fourier Transform Triangulation)
- **Best Tuned Hyperparameters:** `{'clf__C': 5.0, 'clf__class_weight': {0: 1.3, 1: 1.0, 2: 1.0}, 'clf__gamma': 'scale'}`
- **Total Dataset Samples:** 589
- **5-Fold Cross-Validation Macro F1:** `0.8771`
- **Holdout Test Accuracy:** `88.98%`

### Holdout Classification Report
```text
              precision    recall  f1-score   support

    left_tap       0.92      0.84      0.88        43
   right_tap       0.89      0.97      0.93        40
       noise       0.86      0.86      0.86        35

    accuracy                           0.89       118
   macro avg       0.89      0.89      0.89       118
weighted avg       0.89      0.89      0.89       118

```

### Confusion Matrix (`['left_tap', 'right_tap', 'noise']`)
```text
[[36  2  5]
 [ 1 39  0]
 [ 2  3 30]]
```
<!-- TEST_REPORT_END -->