import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .utils import ensure_dir, clamp

LOGGER = logging.getLogger(__name__)


@dataclass
class FacadeMaskConfig:
    plinth_height: float = 0.6
    door_height: float = 2.2
    door_width: float = 1.1
    window_width: float = 1.2
    window_height: float = 1.4
    horizontal_margin: float = 0.8
    vertical_margin: float = 0.5
    seam_margin: float = 0.5
    min_opening_spacing_horizontal: float = 1.0
    min_opening_spacing_vertical: float = 0.9


@dataclass
class MaskBundle:
    plinth: Path
    floors: Path
    openings: Path


@dataclass
class OpeningPlacement:
    x: int
    y: int
    width: int
    height: int
    kind: str
    variant: int


@dataclass
class FacadeLayout(MaskBundle):
    seam_positions: Tuple[int, ...]
    placements: List[OpeningPlacement]


class SegmentationGenerator:
    def __init__(self, texel_density: float, config: FacadeMaskConfig | None = None):
        self.texel_density = texel_density
        self.config = config or FacadeMaskConfig()

    def _blank_mask(self, size: Tuple[int, int]) -> Image.Image:
        return Image.new("L", size, 0)

    def generate(
        self,
        wall_size: Tuple[int, int],
        properties: Dict[str, float],
        output_dir: Path,
        seams: Tuple[int, ...] | None = None,
    ) -> FacadeLayout:
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
            tone = int(255 * (1 - i / max(floors_count, 1)))
            draw_floors.rectangle([(0, y_top), (width, y_bottom)], fill=tone)

        # Openings grid with seam-aware margins that respect Russian spacing norms
        draw_openings = ImageDraw.Draw(opening_mask)
        window_w_px = int(self.config.window_width * self.texel_density)
        window_h_px = int(self.config.window_height * self.texel_density)
        margin_x = int(
            max(self.config.horizontal_margin, self.config.min_opening_spacing_horizontal) * self.texel_density
        )
        margin_y = int(max(self.config.vertical_margin, self.config.min_opening_spacing_vertical) * self.texel_density)
        door_w_px = int(self.config.door_width * self.texel_density)
        door_h_px = int(self.config.door_height * self.texel_density)
        seam_guard = int(self.config.seam_margin * self.texel_density)

        seam_positions = tuple(sorted(seams or (0, width)))
        placements: List[OpeningPlacement] = []

        def _segment_slots(span_start: int, span_end: int, element_width: int) -> List[int]:
            slots: List[int] = []
            cursor = span_start + margin_x
            while cursor + element_width <= span_end - margin_x:
                if cursor - seam_guard <= span_start or cursor + element_width + seam_guard >= span_end:
                    cursor += element_width + margin_x
                    continue
                slots.append(cursor)
                cursor += element_width + margin_x
            return slots

        segment_spans = list(zip(seam_positions[:-1], seam_positions[1:]))
        y = height - floor_height_px + margin_y
        for floor in range(floors_count):
            for segment_idx, (seg_start, seg_end) in enumerate(segment_spans):
                window_slots = _segment_slots(seg_start + seam_guard, seg_end - seam_guard, window_w_px)
                for slot_idx, x in enumerate(window_slots):
                    draw_openings.rectangle(
                        [(x, y - window_h_px), (x + window_w_px, y)],
                        fill=255,
                    )
                    placements.append(
                        OpeningPlacement(
                            x=x,
                            y=y - window_h_px,
                            width=window_w_px,
                            height=window_h_px,
                            kind="balcony" if slot_idx % 3 == 2 else "window",
                            variant=(slot_idx + floor + segment_idx) % 5,
                        )
                    )
            y -= floor_height_px

        door_y = height - plinth_px - margin_y
        for seg_start, seg_end in segment_spans:
            center = (seg_start + seg_end) // 2
            door_x0 = max(seg_start + seam_guard + margin_x, center - door_w_px // 2)
            door_x1 = min(seg_end - seam_guard - margin_x, door_x0 + door_w_px)
            if door_x1 - door_x0 <= 0:
                continue
            draw_openings.rectangle(
                [(door_x0, door_y - door_h_px), (door_x1, door_y)],
                fill=255,
            )
            placements.append(
                OpeningPlacement(
                    x=door_x0,
                    y=door_y - door_h_px,
                    width=door_x1 - door_x0,
                    height=door_h_px,
                    kind="door",
                    variant=(seg_start // max(door_w_px, 1)) % 5,
                )
            )

        plinth_path = output_dir / "plinth.png"
        floors_path = output_dir / "floors.png"
        openings_path = output_dir / "openings.png"

        plinth_mask.save(plinth_path)
        floor_mask.save(floors_path)
        opening_mask.save(openings_path)

        LOGGER.debug("Facade masks saved to %s", output_dir)
        return FacadeLayout(
            plinth=plinth_path,
            floors=floors_path,
            openings=openings_path,
            seam_positions=seam_positions,
            placements=placements,
        )
