import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .utils import ensure_dir, clamp

LOGGER = logging.getLogger(__name__)


@dataclass
class FacadeMaskConfig:
    plinth_height: float = 0.6
    door_height: float = 2.2
    window_width: float = 1.2
    window_height: float = 1.4
    horizontal_margin: float = 0.8
    vertical_margin: float = 0.5


@dataclass
class MaskBundle:
    plinth: Path
    floors: Path
    openings: Path


class SegmentationGenerator:
    def __init__(self, texel_density: float, config: FacadeMaskConfig | None = None):
        self.texel_density = texel_density
        self.config = config or FacadeMaskConfig()

    def _blank_mask(self, size: Tuple[int, int]) -> Image.Image:
        return Image.new("L", size, 0)

    def generate(self, wall_size: Tuple[int, int], properties: Dict[str, float], output_dir: Path) -> MaskBundle:
        width, height = wall_size
        ensure_dir(output_dir)

        plinth_mask = self._blank_mask((width, height))
        floor_mask = self._blank_mask((width, height))
        opening_mask = self._blank_mask((width, height))

        plinth_px = int(self.config.plinth_height * self.texel_density)
        ImageDraw.Draw(plinth_mask).rectangle([(0, height - plinth_px), (width, height)], fill=255)

        floor_height_px = int(properties["floor_height"] * self.texel_density)
        floors_count = int(properties["floors_count"])
        draw_floors = ImageDraw.Draw(floor_mask)
        for i in range(floors_count):
            y_top = clamp(height - (i + 1) * floor_height_px, 0, height)
            y_bottom = clamp(height - i * floor_height_px, 0, height)
            draw_floors.rectangle([(0, y_top), (width, y_bottom)], fill=int(255 * (i % 2 == 0)))

        # Openings grid
        draw_openings = ImageDraw.Draw(opening_mask)
        window_w_px = int(self.config.window_width * self.texel_density)
        window_h_px = int(self.config.window_height * self.texel_density)
        margin_x = int(self.config.horizontal_margin * self.texel_density)
        margin_y = int(self.config.vertical_margin * self.texel_density)

        y = height - floor_height_px + margin_y
        for floor in range(floors_count):
            x = margin_x
            while x + window_w_px < width - margin_x:
                draw_openings.rectangle(
                    [(x, y - window_h_px), (x + window_w_px, y)],
                    fill=255,
                )
                x += window_w_px + margin_x
            y -= floor_height_px

        plinth_path = output_dir / "plinth.png"
        floors_path = output_dir / "floors.png"
        openings_path = output_dir / "openings.png"

        plinth_mask.save(plinth_path)
        floor_mask.save(floors_path)
        opening_mask.save(openings_path)

        LOGGER.debug("Facade masks saved to %s", output_dir)
        return MaskBundle(plinth=plinth_path, floors=floors_path, openings=openings_path)
