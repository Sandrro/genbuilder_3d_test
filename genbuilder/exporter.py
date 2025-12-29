import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from PIL import Image
import numpy as np
import trimesh

from .geometry import Mesh
from .utils import ensure_dir

LOGGER = logging.getLogger(__name__)


@dataclass
class ExportResult:
    glb_path: Path
    textures: Dict[str, Path] = field(default_factory=dict)


def export_glb(
    mesh: Mesh,
    texture_paths: Dict[str, Path | None],
    output_path: Path,
    texture_export_dir: Path | None = None,
    texture_name_prefix: str | None = None,
) -> ExportResult:
    if not texture_paths:
        raise ValueError("At least one texture is required to export a textured GLB")

    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)

    base_color_path = texture_paths.get("baseColor")
    if base_color_path is None:
        raise ValueError("baseColor texture is required for GLB export")

    uv = np.zeros((len(mesh.vertices), 2), dtype=float)
    uv_assigned = np.zeros(len(mesh.vertices), dtype=bool)
    if mesh.uv_indices and len(mesh.uv_indices) == len(mesh.faces):
        for face_idx, (face, uv_idx) in enumerate(zip(mesh.faces, mesh.uv_indices)):
            for vertex_id, uv_id in zip(face, uv_idx):
                uv_coord = mesh.uvs[uv_id]
                if not uv_assigned[vertex_id]:
                    uv[vertex_id] = uv_coord
                    uv_assigned[vertex_id] = True
                elif not np.allclose(uv[vertex_id], uv_coord):
                    LOGGER.warning(
                        "Conflicting UVs for vertex %s on face %s; keeping first assignment",
                        vertex_id,
                        face_idx,
                    )

    with Image.open(base_color_path) as img:
        texture_image = img.convert("RGBA")
    visuals = trimesh.visual.texture.TextureVisuals(uv=uv, image=texture_image)
    tm = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visuals, process=False)

    ensure_dir(output_path.parent)
    tm.export(output_path, file_type="glb")

    exported_textures: Dict[str, Path] = {}
    if texture_export_dir is not None:
        name_prefix = texture_name_prefix or output_path.stem
        exported_textures = export_textures(
            texture_paths=texture_paths,
            output_dir=texture_export_dir,
            name_prefix=name_prefix,
        )

    return ExportResult(glb_path=output_path, textures=exported_textures)


def export_textures(texture_paths: Dict[str, Path | None], output_dir: Path, name_prefix: str) -> Dict[str, Path]:
    ensure_dir(output_dir)
    exported: Dict[str, Path] = {}
    for texture_type, source_path in texture_paths.items():
        if source_path is None:
            continue
        extension = source_path.suffix or ".png"
        destination = output_dir / f"{name_prefix}_{texture_type}{extension}"
        shutil.copy2(source_path, destination)
        exported[texture_type] = destination
    return exported


def write_index(records: List[Dict[str, object]], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(records, indent=2))
    LOGGER.info("Index written to %s", output_path)
