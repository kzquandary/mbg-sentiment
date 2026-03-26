import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np


SEED = 42


def set_global_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(data: dict, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def backup_if_exists(path: Path) -> str | None:
    if not path.exists():
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_name(f"{path.stem}_{timestamp}.bak{path.suffix}")
    path.rename(backup_path)
    return str(backup_path)
