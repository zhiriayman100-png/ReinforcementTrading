from pathlib import Path

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
CKPT_DIR = ROOT / "checkpoints"
TENSORBOARD_DIR = ROOT / "tensorboard_log"

for d in (MODELS_DIR, CKPT_DIR, TENSORBOARD_DIR):
    d.mkdir(parents=True, exist_ok=True)
