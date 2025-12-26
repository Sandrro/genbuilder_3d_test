import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import trimesh

from .geometry import Mesh
from .utils import ensure_dir

LOGGER = logging.getLogger(__name__)


@dataclass
class ExportResult:
    glb_path: Path


def export_glb(mesh: Mesh, texture_paths: Dict[str, Path], output_path: Path) -> ExportResult:
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

    texture_image = trimesh.visual.texture.load_image(str(base_color_path))
    visuals = trimesh.visual.texture.TextureVisuals(uv=uv, image=texture_image)
    tm = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visuals, process=False)

    ensure_dir(output_path.parent)
    tm.export(output_path, file_type="glb")
    return ExportResult(glb_path=output_path)


def write_index(records: List[Dict[str, object]], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(records, indent=2))
    LOGGER.info("Index written to %s", output_path)
