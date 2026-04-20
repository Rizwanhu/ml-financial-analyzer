from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"
REPORT_DIR = PROJECT_ROOT / "artifacts" / "reports"


def ensure_directories() -> None:
    """Create standard project directories if they do not exist."""
    for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)

