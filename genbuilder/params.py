from dataclasses import dataclass, field
from pathlib import Path

from .segmentation import FacadeMaskConfig


@dataclass
class GenParams:
    """Configurable parameters for the building generation pipeline."""

    texel_density: float = 512.0
    seed: int = 0
    batch_size: int = 1
    device: str = "cpu"
    dry_run_geometry: bool = False
    cache_dir: Path | None = None
    facade_config: FacadeMaskConfig = field(default_factory=FacadeMaskConfig)
    # Prompt placeholder hints (optional)
    city_hint: str | None = None
    climate_hint: str | None = None
    era_hint: str | None = None
    style_hint: str | None = None
    material_hint: str | None = None
    seed_hint: str | None = None
    recipe: str | None = None

    def placeholder_metadata(self, floors_count: int, floor_height: float) -> dict[str, str]:
        """Collect runtime prompt placeholders as a metadata mapping."""

        data: dict[str, str] = {
            "floor_count": str(floors_count),
            "floor_height_m": str(floor_height),
        }

        optional_fields = {
            "city_hint": self.city_hint,
            "climate_hint": self.climate_hint,
            "era_hint": self.era_hint,
            "style_hint": self.style_hint,
            "material_hint": self.material_hint,
            "seed_hint": self.seed_hint,
            "recipe": self.recipe,
        }

        for key, value in optional_fields.items():
            if value:
                data[key] = value

        return data
