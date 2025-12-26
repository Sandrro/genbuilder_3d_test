import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from shapely.geometry import Polygon

from .geometry import Mesh

LOGGER = logging.getLogger(__name__)


@dataclass
class UVAtlas:
    wall_size: Tuple[int, int]
    roof_size: Tuple[int, int]
    wall_uvs: List[Tuple[float, float]]
    roof_uvs: List[Tuple[float, float]]


@dataclass
class UVMetadata:
    perimeter: float
    height: float
    texel_density: float


class UVGenerator:
    def __init__(self, texel_density: float = 512.0):
        self.texel_density = texel_density

    def wall_strip_dimensions(self, perimeter: float, height: float) -> Tuple[int, int]:
        width_px = max(int(perimeter * self.texel_density), 16)
        height_px = max(int(height * self.texel_density), 16)
        return width_px, height_px

    def map_wall_uvs(self, polygon: Polygon, mesh: Mesh) -> UVAtlas:
        exterior = list(polygon.exterior.coords)
        lengths = [np.linalg.norm(np.array(exterior[i + 1]) - np.array(exterior[i])) for i in range(len(exterior) - 1)]
        perimeter = float(sum(lengths))
        height = float(max(v[2] for v in mesh.vertices))
        width_px, height_px = self.wall_strip_dimensions(perimeter, height)

        wall_uvs: List[Tuple[float, float]] = []
        cumulative = 0.0
        for segment_length in lengths:
            x0 = cumulative / perimeter
            x1 = (cumulative + segment_length) / perimeter
            cumulative += segment_length
            wall_uvs.extend(
                [
                    (x0, 0.0),
                    (x1, 0.0),
                    (x1, 1.0),
                    (x0, 1.0),
                ]
            )
        roof_uvs = [(0.5, 0.5) for _ in range(len(mesh.vertices))]
        return UVAtlas(
            wall_size=(width_px, height_px),
            roof_size=(max(width_px // 2, 16), max(height_px // 2, 16)),
            wall_uvs=wall_uvs,
            roof_uvs=roof_uvs,
        )

    def annotate_mesh_uvs(self, mesh: Mesh, atlas: UVAtlas) -> Mesh:
        mesh.uvs = atlas.wall_uvs + atlas.roof_uvs
        # Rebuild uv_indices to align with uv list
        mesh.uv_indices = []
        wall_face_counts: Dict[int, int] = defaultdict(int)
        roof_offset = len(atlas.wall_uvs)
        roof_cursor = 0

        for label in mesh.face_labels:
            if label.startswith("wall"):
                wall_idx = int(label.split("_")[1])
                face_number = wall_face_counts[wall_idx]
                base = wall_idx * 4
                if face_number == 0:
                    indices = (base, base + 1, base + 2)
                else:
                    indices = (base, base + 2, base + 3)
                wall_face_counts[wall_idx] += 1
                mesh.uv_indices.append(indices)
            else:
                indices = (roof_offset + roof_cursor, roof_offset + roof_cursor + 1, roof_offset + roof_cursor + 2)
                mesh.uv_indices.append(indices)
                roof_cursor += 3
        return mesh


