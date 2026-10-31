from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SAMPLE_RATE = 44100
BLOCK_SIZE = 512          # ~11.6 ms per audio block
PRE_BLOCKS = 2            # ~23.2 ms captured BEFORE the impulse trigger
POST_BLOCKS = 8           # ~92.8 ms captured AFTER the impulse trigger
COOLDOWN_BLOCKS = 25      # ~290 ms cooldown between taps
RMS_THRESHOLD = 0.015     # Base trigger threshold

# Dynamic noise floor and crest factor settings
PEAK_MULTIPLIER = 1.8     # Peak must exceed rms * peak_multiplier (crest check)
BG_SNR_RATIO = 2.0        # RMS must exceed rolling background floor * bg_snr_ratio
BG_EMA_ALPHA = 0.95       # Rolling background floor exponential smoothing

# Transient alignment window constants
WINDOW_PRE_PEAK = 32      # Samples before peak included in active window
WINDOW_WAKE = 1024        # Active impact transient and resonance window (1024 samples ≈ 23.2 ms)

DATA_DIR = PROJECT_ROOT / "dataset"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "acoustictap_model.joblib"

CLASSES = ["left_tap", "right_tap", "noise"]
