# AcousticTap (holo)

AcousticTap turns the aluminum palm rest of a MacBook into virtual, programmable macro buttons using structure-borne acoustic sensing and machine learning.

## Architecture
- `src/holo/config.py`: Central constants and path definitions.
- `src/holo/detector.py`: Pre-trigger ring buffer for capturing 128 ms acoustic transients.
- `src/holo/features.py`: Short-window (512-sample) MFCC, Delta-MFCC, and spectral feature extraction.
- `src/holo/collect.py`: Interactive CLI tool for recording training taps.
- `src/holo/train.py`: Multi-model benchmarking (Random Forest, Extra Trees, RBF-SVM) with version logging.
- `src/holo/listen.py`: Low-latency macOS listener daemon with keyboard typing suppression and dry-run testing.

## Usage with uv
1. Collect training samples:
    uv run python -m src.holo.collect --samples 50

2. Collect single group training samples:
    uv run python -m src.holo.collect --label left_tap --samples 50
    uv run python -m src.holo.collect --label right_tap --samples 50
    uv run python -m src.holo.collect --label noise --samples 50

3. Train and benchmark all models:
    uv run python -m src.holo.train --features enhanced

4. Test live predictions safely (without triggering macOS shortcuts):
    uv run python -m src.holo.listen --dry-run

5. Run the live macOS macro daemon:
    uv run python -m src.holo.listen --confidence 0.75
