from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}

_cfg = load_config()

SAMPLE_RATE = int(_cfg.get("audio", {}).get("sample_rate", 48000))
BLOCK_SIZE = int(_cfg.get("audio", {}).get("block_size", 512))
PRE_BLOCKS = int(_cfg.get("audio", {}).get("pre_blocks", 6))
POST_BLOCKS = int(_cfg.get("audio", {}).get("post_blocks", 10))
COOLDOWN_BLOCKS = int(_cfg.get("audio", {}).get("cooldown_blocks", 25))
RMS_THRESHOLD = float(_cfg.get("audio", {}).get("rms_threshold", 0.004))
DYNAMIC_MULTIPLIER = float(_cfg.get("audio", {}).get("dynamic_multiplier", 2.0))
LOWPASS_CUTOFF_HZ = float(_cfg.get("audio", {}).get("lowpass_cutoff_hz", 16000.0))

ULTRASONIC_CFG = _cfg.get("ultrasonic", {})
ULTRASONIC_ENABLED = bool(ULTRASONIC_CFG.get("enabled", True))
LEFT_ULTRA_HZ = float(ULTRASONIC_CFG.get("left_freq_hz", 19000.0))
RIGHT_ULTRA_HZ = float(ULTRASONIC_CFG.get("right_freq_hz", 20200.0))
SPEAKER_AMPLITUDE = float(ULTRASONIC_CFG.get("speaker_amplitude", 0.03))
CARRIER_BW_HZ = float(ULTRASONIC_CFG.get("carrier_bw_hz", 25.0))
DOPPLER_MIN_HZ = float(ULTRASONIC_CFG.get("doppler_min_hz", 25.0))
DOPPLER_MAX_HZ = float(ULTRASONIC_CFG.get("doppler_max_hz", 180.0))
ALPHA_DOPPLER = float(ULTRASONIC_CFG.get("alpha_doppler", 0.65))
GATE_THRESHOLD = float(ULTRASONIC_CFG.get("gate_threshold", 2.0))
GATE_STEEPNESS = float(ULTRASONIC_CFG.get("gate_steepness", 2.5))
CALIBRATION_BLOCKS = int(ULTRASONIC_CFG.get("calibration_blocks", 35))

FOURIER_WEIGHT = float(_cfg.get("fusion", {}).get("fourier_weight", 0.50))
AUTO_TUNE_WEIGHT = bool(_cfg.get("fusion", {}).get("auto_tune_weight", True))
ULTRASONIC_2FA_ENABLED = bool(_cfg.get("fusion", {}).get("ultrasonic_2fa_enabled", True))

DATA_DIR = PROJECT_ROOT / "dataset"
MODEL_DIR = PROJECT_ROOT / "models"
VERSIONS_DIR = MODEL_DIR / "versions"
RESULTS_DIR = MODEL_DIR / "test_results"
MODEL_PATH = MODEL_DIR / "acoustictap_model.joblib"

CLASSES = _cfg.get("classes", ["left_tap", "right_tap", "noise"])
CONFIDENCE_THRESHOLD = float(_cfg.get("listener", {}).get("confidence_threshold", 0.58))
ENABLE_SHORTCUTS = bool(_cfg.get("listener", {}).get("enable_shortcuts", False))
