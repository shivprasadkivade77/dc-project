from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SAMPLE_RATE = 44100
BLOCK_SIZE = 512          # ~11.6 ms per audio block
PRE_BLOCKS = 3            # ~34.8 ms captured BEFORE the impulse trigger
POST_BLOCKS = 8           # ~92.8 ms captured AFTER the impulse trigger
COOLDOWN_BLOCKS = 25      # ~290 ms cooldown between taps
RMS_THRESHOLD = 0.015     # Base RMS trigger threshold

# Hybrid Fusion Weight: 0.50 means 50% Fourier Transform Triangulation + 50% ML Model
FOURIER_WEIGHT = 0.50

DATA_DIR = PROJECT_ROOT / "dataset"
MODEL_DIR = PROJECT_ROOT / "models"
VERSIONS_DIR = MODEL_DIR / "versions"
RESULTS_DIR = MODEL_DIR / "test_results"
MODEL_PATH = MODEL_DIR / "acoustictap_model.joblib"

CLASSES = ["left_tap", "right_tap", "noise"]
