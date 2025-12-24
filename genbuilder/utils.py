import hashlib
import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np


@dataclass
class CachePaths:
    base_dir: Path

    def texture_dir(self) -> Path:
        d = self.base_dir / "textures"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def model_dir(self) -> Path:
        d = self.base_dir / "models"
        d.mkdir(parents=True, exist_ok=True)
        return d


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


def deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def sha256_of_dict(data: Dict[str, Any]) -> str:
    payload = json.dumps(data, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))
