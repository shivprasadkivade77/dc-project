from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

def load_config(path: Path = CONFIG_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

CFG = load_config()

# Directory Paths
DATA_DIR = PROJECT_ROOT / "dataset"
MODEL_DIR = PROJECT_ROOT / "models"
VERSIONS_DIR = MODEL_DIR / "versions"
RESULTS_DIR = MODEL_DIR / "test_results"
MODEL_PATH = MODEL_DIR / "acoustictap_model.joblib"

# Backwards-compatible top-level constants read from config.yaml
SAMPLE_RATE = int(CFG["audio"]["sample_rate"])
BLOCK_SIZE = int(CFG["audio"]["block_size"])
PRE_BLOCKS = int(CFG["audio"]["pre_blocks"])
POST_BLOCKS = int(CFG["audio"]["post_blocks"])
COOLDOWN_BLOCKS = int(CFG["audio"]["cooldown_blocks"])
RMS_THRESHOLD = float(CFG["detector"]["rms_threshold"])
FOURIER_WEIGHT = float(CFG["hybrid_fusion"]["fourier_weight"])
CLASSES = list(CFG["training"]["classes"])
