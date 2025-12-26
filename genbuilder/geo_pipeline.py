import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

from .exporter import export_glb, write_index
from .geometry import (
    BuildingProperties,
    PreparedGeometry,
    extrude_building,
    reproject_polygon,
    select_local_crs,
    validate_polygon,
)
from .params import GenParams
from .segmentation import SegmentationGenerator
from .texture import TextureGenerator, TextureResult
from .uv import UVGenerator
from .utils import CachePaths, deterministic_seed, setup_logging

LOGGER = logging.getLogger(__name__)


class BuildingPipeline:
    def __init__(self, params: GenParams | None = None) -> None:
        self.params = params or GenParams()
        self.cache = CachePaths(self.params.cache_dir or Path(".cache"))
        self.uv_generator = UVGenerator(texel_density=self.params.texel_density)
        self.segmentation = SegmentationGenerator(
            texel_density=self.params.texel_density, config=self.params.facade_config
        )
        self.texture_generator = TextureGenerator(
            cache_paths=self.cache, device=self.params.device, seed=self.params.seed
        )
        self.seed = self.params.seed
        self.dry_run_geometry = self.params.dry_run_geometry
        self._shape_textures: Dict[Tuple[float, float, float], TextureResult] = {}
        setup_logging()
        deterministic_seed(self.params.seed)

    def _prepare_geometry(self, feature: Dict) -> PreparedGeometry:
        polygon_wgs84 = validate_polygon(feature["geometry"])
        local_crs = select_local_crs(polygon_wgs84)
        projected, transformer = reproject_polygon(polygon_wgs84, local_crs)
        return PreparedGeometry(polygon=projected, crs=local_crs)

    def _properties_from_feature(self, feature: Dict) -> BuildingProperties:
        props = feature.get("properties", {})
        floors = int(props.get("floors_count", 1))
        floor_height = float(props.get("floor_height", 3.0))
        building_height = float(props.get("building_height", floors * floor_height))
        return BuildingProperties(
            floors_count=floors,
            floor_height=floor_height,
            building_height=building_height,
            roof_type=props.get("roof_type"),
            roof_material=props.get("roof_material"),
        )

    def _shape_signature(self, polygon, properties: BuildingProperties) -> Tuple[float, float, float]:
        minx, miny, maxx, maxy = polygon.bounds
        width, length = sorted((maxx - minx, maxy - miny), reverse=True)
        height = properties.building_height
        return tuple(round(value, 3) for value in (width, length, height))

    def _shape_mask_dir(self, output_dir: Path, shape_key: Tuple[float, float, float]) -> Path:
        key_str = "_".join(f"{dim:.3f}" for dim in shape_key)
        return output_dir / "masks" / key_str

    def process_feature(self, feature: Dict, output_dir: Path) -> Dict[str, object]:
        feature_id = feature.get("id", "building")
        LOGGER.info("Preparing geometry for feature %s", feature_id)
        prepared = self._prepare_geometry(feature)
        properties = self._properties_from_feature(feature)
        LOGGER.info(
            "Extruding building %s: %s floors at %.2f m per floor", feature_id, properties.floors_count, properties.floor_height
        )
        mesh = extrude_building(prepared.polygon, properties)
        LOGGER.info("Mapping UVs for feature %s", feature_id)
        atlas = self.uv_generator.map_wall_uvs(prepared.polygon, mesh)
        mesh = self.uv_generator.annotate_mesh_uvs(mesh, atlas)

        shape_key = self._shape_signature(prepared.polygon, properties)
        cached_textures = self._shape_textures.get(shape_key)
        if cached_textures:
            LOGGER.info("Reusing textures for shape %s", shape_key)
            textures = cached_textures
        else:
            LOGGER.info("Generating segmentation masks for feature %s", feature_id)
            masks = self.segmentation.generate(
                wall_size=atlas.wall_size,
                properties={"floors_count": properties.floors_count, "floor_height": properties.floor_height},
                output_dir=self._shape_mask_dir(output_dir, shape_key),
            )
            metadata = self.params.placeholder_metadata(properties.floors_count, properties.floor_height)
            metadata.update({"roof": properties.roof_type or "flat", "material": properties.roof_material or "default"})
            textures = self.texture_generator.synthesize_facade(
                wall_size=atlas.wall_size,
                masks=masks,
                metadata=metadata,
                dry_run=self.dry_run_geometry,
            )
            self._shape_textures[shape_key] = textures

        glb_output = output_dir / f"{feature_id}.glb"
        LOGGER.info("Exporting GLB for feature %s to %s", feature_id, glb_output)
        export_glb(mesh, {"baseColor": textures.base_color}, glb_output)

        centroid = prepared.polygon.centroid
        return {
            "id": feature.get("id"),
            "glb": str(glb_output),
            "centroid": {
                "x": centroid.x,
                "y": centroid.y,
            },
            "height": properties.building_height,
        }

    def run(self, geojson_path: Path, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        collection = json.loads(Path(geojson_path).read_text())
        if collection.get("type") != "FeatureCollection":
            raise ValueError("Input must be a FeatureCollection")
        features = collection.get("features", [])
        LOGGER.info("Starting pipeline for %s features", len(features))
        results: List[Dict[str, object]] = []
        for feature in tqdm(features, desc="Processing buildings"):
            feature_id = feature.get("id", "unknown")
            LOGGER.info("Processing feature %s", feature_id)
            try:
                record = self.process_feature(feature, output_dir)
                results.append(record)
                LOGGER.info("Finished feature %s", feature_id)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("Failed to process feature %s: %s", feature_id, exc)
        index_path = output_dir / "index.json"
        write_index(results, index_path)
        LOGGER.info("Wrote index for %s features to %s", len(results), index_path)
        return index_path


