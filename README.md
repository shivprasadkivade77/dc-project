# AcousticTap (holo) — V3 Hybrid Fourier + Ultrasonic Model

AcousticTap turns the aluminum palm rest of a MacBook into virtual, programmable macro buttons using structure-borne acoustic sensing. V3 classifies each tap with a hybrid model: a classical machine-learning classifier on a combined audio + ultrasonic feature set, fused with a Fourier-transform triangulation classifier, and checked by a soft-floored ultrasonic sonar gate before a tap is accepted.

## What's New in V3

- **Scale-invariant feature set & Mel filterbank fix.** The ML branch (`features.py`) normalizes aligned waveforms by peak amplitude (`norm_aligned = aligned / peak`) and clips SNR to `[0.0, 25.0] dB` so light and hard taps share the same spectral and temporal scale. The Mel filterbank uses `n_mels=40` (at `n_fft=512`, `48 kHz`) alongside temporal, spectral, and ultrasonic features, logged in the leaderboard as `all+ultrasonic`.
- **Higher sensitivity for light taps.** `ImpulseDetector` (`detector.py`, `config.py`) uses a lowered base `RMS_THRESHOLD` of `0.004`, a `DYNAMIC_MULTIPLIER` of `2.0`, and a relaxed crest factor threshold of `1.85` so gentle taps trigger capture reliably.
- **Soft-floored ultrasonic sonar gate.** `UltrasonicSonarGate` evaluates each audio window and modulates fused tap probabilities using `0.5 + 0.5 * gate` (`listen.py`), preventing low-Doppler light taps from being zeroed out while still penalizing non-tap noise.
- **Fixed candidate models.** V3 benchmarks three fixed-hyperparameter candidates (`ExtraTrees_V3`, `RandomForest_V3`, `SVM_RBF_V3`) instead of running a `GridSearchCV` per model.
- **Model selection by cross-validation only.** The production model is the candidate with the best cross-validated macro F1. The holdout set is used only for reporting.
- **Richer test report & live diagnostics.** `latest_test_report.md` includes the classification report and confusion matrix, and `listen.py` logs both `>>> TRIGGER` and `[Rejected]` events with confidence and gate scores.

## How the V3 Hybrid Model Works

1. **Capture.** A pre-trigger ring buffer monitors `512`-sample blocks at `48 kHz` (`lowpass_for_v2` at `16 kHz`) and records a 17-block (~181 ms) acoustic transient around each tap (`6` pre-blocks + `1` trigger block + `10` post-blocks) when `rms > max(0.004, ambient_floor * 2.0)` and `crest > 1.85`.
2. **Two probability branches** run on the same waveform:
   - **ML branch:** `extract_features(..., feature_set="all")` slices and peak-normalizes a `43 ms` (`8 ms` pre-peak, `35 ms` post-peak) window to produce an 89-dimensional feature vector (`6` temporal + `30` spectral with `n_mels=40` + `53` Fourier/ultrasonic features) that goes through a `StandardScaler` and one of three candidate classifiers.
   - **Fourier branch:** `FourierTriangulator` extracts a Fourier-transform triangulation vector (`pooled_mag`, `phase_stats`, and `ultra_vec`) from the waveform and outputs its own class probabilities (`70%` shrinkage LDA + `30%` cosine template similarity).
3. **Probability fusion.** The two probability vectors are blended with a single weight `w`:
   ```text
   P_hybrid = (1 - w) * P_ml + w * P_fourier
   ```
   The initial predicted distribution comes from `P_hybrid`.
4. **Ultrasonic gate.** `UltrasonicSonarGate.evaluate_window(waveform)` is evaluated for each window alongside the fused probabilities. In live listening (`listen.py`), left (`19.0 kHz`) and right (`20.2 kHz`) tap probabilities are scaled by `(0.5 + 0.5 * gate_side)` and accepted if the winning class is a tap with confidence `>= 0.58` (`CONFIDENCE_THRESHOLD`).
5. **Weight tuning.** If `AUTO_TUNE_WEIGHT` is enabled in the config, `w` is searched over 15 values from `0.15` to `0.85`. Each candidate weight is scored on out-of-fold predictions from stratified cross-validation (up to 5 folds, limited by the smallest class size), and the weight with the best macro F1 wins. Otherwise the fixed `FOURIER_WEIGHT` is used.
6. **Model selection.** Each candidate is cross-validated on the training split (80%), then fit on it and scored on the stratified 20% holdout. The candidate with the highest cross-validated macro F1 becomes the production model.
7. **Final fit.** The winning classifier and the Fourier triangulator are refit on 100% of the dataset and saved.

## Candidate Models

| Model | Configuration |
| :--- | :--- |
| `ExtraTrees_V3` | `StandardScaler` + `ExtraTreesClassifier` (250 trees, balanced class weights) |
| `RandomForest_V3` | `StandardScaler` + `RandomForestClassifier` (250 trees, balanced class weights) |
| `SVM_RBF_V3` | `StandardScaler` + `SVC` (RBF kernel, `C=3.0`, probability outputs, balanced class weights) |

All use `random_state=42`, so training is reproducible for a given dataset.

## Benchmark Results

Latest run (stratified 80/20 split on 839 samples, macro F1):

| Model | Optimal Fusion | CV Macro-F1 | Holdout Accuracy | Holdout Macro-F1 |
| :--- | :--- | :--- | :--- | :--- |
| `ExtraTrees_V3` (winner) | 85% ML + 15% Fourier (`w=0.15`) | `0.9017` | `91.67%` | `0.9180` |

The cross-validated score is the more conservative number, because the holdout set is smaller (`168` clips). See [Interpreting Results](#interpreting-results).

## Architecture

- `src/holo/config.py`: Central constants and path definitions (`RMS_THRESHOLD = 0.004`, `DYNAMIC_MULTIPLIER = 2.0`, `CONFIDENCE_THRESHOLD = 0.58`; also reads `config/config.yaml` for training and fusion settings).
- `src/holo/detector.py`: Pre-trigger ring buffer for capturing ~181 ms (`17` blocks) acoustic transients with `crest > 1.85`.
- `src/holo/features.py`: Peak-normalized temporal, 40-bin Mel (`n_mels=40`) MFCC, spectral, and Fourier/ultrasonic feature extraction for the ML branch.
- `src/holo/triangulation.py`: Waveform alignment (`43 ms`), Fourier-transform triangulation vector extraction, and the `FourierTriangulator` classifier for the Fourier branch.
- `src/holo/ultrasonic.py`: `19.0 kHz` (Left) and `20.2 kHz` (Right) continuous ultrasonic emitter, baseline calibrator, and Doppler/reflection sonar gate.
- `src/holo/collect.py`: Interactive CLI tool for recording training taps.
- `src/holo/train.py`: Hybrid training and benchmarking, fusion-weight search, model selection, versioned model saving, leaderboard logging, and test-report generation.
- `src/holo/evaluate.py`: Loads the saved model and runs the hybrid pipeline (ML + Fourier fusion and the ultrasonic gate) over the dataset.
- `src/holo/update_readme.py`: Copies the latest test report into this README.
- `src/holo/listen.py`: Low-latency macOS listener daemon with hands-off ultrasonic calibration, softened 2FA gating (`0.5 + 0.5 * gate`), and live `>>> TRIGGER` / `[Rejected]` output.

## Usage with `uv`

1. **Collect training samples:**
   ```bash
   uv run python -m src.holo.collect
   ```
   Record the samples across several sessions and tap intensities (light to firm) if you can (different times of day, different hand positions). Clips from the same session look very similar, which makes holdout scores look better than real-world performance.

2. **Train and benchmark the hybrid models:**
   ```bash
   uv run python -m src.holo.train
   ```
   Training writes:
   - the production model to `models/acoustictap_model.joblib`
   - a versioned copy named `model_v3_<model>_<timestamp>.joblib` in the `models/versions` directory
   - one row per candidate appended to `models/leaderboard.csv`
   - `latest_test_report.md` in the `models/test_results` directory

3. **Run the evaluation script:**
   ```bash
   uv run python -m src.holo.evaluate
   ```
   This is a check that the saved model loads and runs end to end. It is not an unseen accuracy measurement (see below).

4. **Update this README with the latest test report:**
   ```bash
   uv run python -m src.holo.update_readme
   ```

5. **Run the live macOS listener:**
   ```bash
   uv run python -m src.holo.listen
   ```

## Interpreting Results

- **Holdout score from `train.py`.** The reported holdout metrics come from a model fit only on the 80% training split (`671` samples) and scored on the unseen 20% (`168` samples). This is the number to quote, together with the cross-validated macro F1.
- **`evaluate.py` is not a fair accuracy test.** The saved production model is refit on 100% of the data, so every clip in the dataset, including the holdout clips, was seen during training. Scoring it on dataset clips gives an inflated result (often near `1.0`).
- **Trigger vs. Rejected in `listen.py`.** A captured impulse prints `>>> TRIGGER` when the highest-scoring post-gate class is `left_tap` or `right_tap` and its probability is `>= 0.58`. Otherwise, it prints `[Rejected]` along with the winning class, score, and ultrasonic gate values (`Gates L/R`).

## Known & Resolved Warnings

- **Resolved — `librosa` empty Mel filter warning:** Setting `n_mels=40` in `src/holo/features.py` prevents empty filter channels when computing `n_fft=512` spectrograms at `48 kHz`.
- **Known — `scikit-learn` `SVC(probability=True)` deprecation:** `scikit-learn` prints a `FutureWarning` about `SVC(probability=True)` being deprecated. It does not affect results today. The suggested replacement is `CalibratedClassifierCV(SVC(), ensemble=False)`.

<!-- TEST_REPORT_START -->
# AcousticTap Hybrid Test Report (20261023_004409)

- **Winning Architecture:** `ExtraTrees_V3` (85% ML + 15% Fourier Triangulation)
- **Total Dataset Samples:** 839 (671 train / 168 holdout)
- **5-Fold Cross-Validation Macro F1:** `0.9017`
- **Holdout Test Accuracy:** `91.67%`
- **Holdout Test Macro F1:** `0.9180`

## Holdout Classification Report
```text
              precision    recall  f1-score   support

    left_tap       0.88      0.90      0.89        63
       noise       0.91      0.93      0.92        45
   right_tap       0.96      0.92      0.94        60

    accuracy                           0.92       168
   macro avg       0.92      0.92      0.92       168
weighted avg       0.92      0.92      0.92       168
```

## Confusion Matrix (`['left_tap', 'noise', 'right_tap']`)
```text
[[57  4  2]
 [ 3 42  0]
 [ 5  0 55]]
```
<!-- TEST_REPORT_END -->