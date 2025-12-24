from dataclasses import dataclass, field
from pathlib import Path

from .segmentation import FacadeMaskConfig


@dataclass
class GenParams:
    """Configurable parameters for the building generation pipeline."""

    texel_density: float = 512.0
    model: str = "sdxl"
    seed: int = 0
    batch_size: int = 1
    device: str = "cpu"
    dry_run_geometry: bool = False
    cache_dir: Path | None = None
    facade_config: FacadeMaskConfig = field(default_factory=FacadeMaskConfig)
