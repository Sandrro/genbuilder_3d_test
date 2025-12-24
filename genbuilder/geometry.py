import logging
from dataclasses import dataclass
from typing import List, Tuple

from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.ops import transform
from pyproj import CRS, Transformer

LOGGER = logging.getLogger(__name__)


@dataclass
class BuildingProperties:
    floors_count: int
    floor_height: float
    building_height: float
    roof_type: str | None = None
    roof_material: str | None = None


@dataclass
class Mesh:
    vertices: List[Tuple[float, float, float]]
    faces: List[Tuple[int, int, int]]
    face_labels: List[str]
    uvs: List[Tuple[float, float]]
    uv_indices: List[Tuple[int, int, int]]


@dataclass
class PreparedGeometry:
    polygon: Polygon
    crs: CRS



def select_local_crs(polygon: Polygon) -> CRS:
    lon, lat = polygon.centroid.x, polygon.centroid.y
    utm_zone = int((lon + 180) / 6) + 1
    hemisphere = "north" if lat >= 0 else "south"
    return CRS.from_dict({"proj": "utm", "zone": utm_zone, "south": hemisphere == "south"})


def validate_polygon(feature_geometry: dict) -> Polygon:
    geom = shape(feature_geometry)
    if isinstance(geom, MultiPolygon):
        geom = max(geom.geoms, key=lambda g: g.area)
    if not isinstance(geom, Polygon):
        raise ValueError("Geometry must be Polygon or MultiPolygon")
    if not geom.is_valid:
        LOGGER.warning("Invalid geometry, attempting fix via buffer(0)")
        geom = geom.buffer(0)
    if not geom.is_valid:
        raise ValueError("Geometry could not be validated")
    return geom


def reproject_polygon(polygon: Polygon, target_crs: CRS) -> Tuple[Polygon, Transformer]:
    transformer = Transformer.from_crs(CRS.from_epsg(4326), target_crs, always_xy=True)
    projected = transform(transformer.transform, polygon)
    return projected, transformer


def extrude_building(polygon: Polygon, properties: BuildingProperties) -> Mesh:
    height = properties.building_height or properties.floors_count * properties.floor_height
    exterior = list(polygon.exterior.coords)
    vertices: List[Tuple[float, float, float]] = []
    faces: List[Tuple[int, int, int]] = []
    uvs: List[Tuple[float, float]] = []
    uv_indices: List[Tuple[int, int, int]] = []
    face_labels: List[str] = []

    def add_vertex(x: float, y: float, z: float) -> int:
        vertices.append((x, y, z))
        return len(vertices) - 1

    # Walls
    wall_labels = []
    for idx in range(len(exterior) - 1):
        p0 = exterior[idx]
        p1 = exterior[idx + 1]
        bottom0 = add_vertex(p0[0], p0[1], 0.0)
        bottom1 = add_vertex(p1[0], p1[1], 0.0)
        top0 = add_vertex(p0[0], p0[1], height)
        top1 = add_vertex(p1[0], p1[1], height)
        faces.extend(
            [
                (bottom0, bottom1, top1),
                (bottom0, top1, top0),
            ]
        )
        uv_indices.extend([(len(uvs), len(uvs) + 1, len(uvs) + 2), (len(uvs), len(uvs) + 2, len(uvs) + 3)])
        uvs.extend([(0, 0), (1, 0), (1, 1), (0, 1)])
        wall_labels.extend([f"wall_{idx}", f"wall_{idx}"])

    # Roof fan triangulation around first point
    roof_start = add_vertex(exterior[0][0], exterior[0][1], height)
    roof_base_index = len(vertices)
    for p in exterior[1:-1]:
        add_vertex(p[0], p[1], height)
    for i in range(roof_base_index, len(vertices) - 1):
        faces.append((roof_start, i, i + 1))
        uv_indices.append((len(uvs), len(uvs) + 1, len(uvs) + 2))
        uvs.extend([(0.5, 0.5), (0.6, 0.5), (0.5, 0.6)])
        face_labels.append("roof")

    face_labels = wall_labels + face_labels
    return Mesh(vertices=vertices, faces=faces, face_labels=face_labels, uvs=uvs, uv_indices=uv_indices)
