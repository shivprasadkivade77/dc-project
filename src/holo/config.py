from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SAMPLE_RATE = 44100
BLOCK_SIZE = 512          # ~11.6 ms per audio block
PRE_BLOCKS = 2            # ~23.2 ms captured BEFORE the impulse trigger
POST_BLOCKS = 8           # ~92.8 ms captured AFTER the impulse trigger
COOLDOWN_BLOCKS = 25      # ~290 ms cooldown between taps
RMS_THRESHOLD = 0.015     # Adjust if mic is too sensitive or not sensitive enough

DATA_DIR = PROJECT_ROOT / "dataset"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "acoustictap_model.joblib"

CLASSES = ["left_tap", "right_tap", "noise"]
