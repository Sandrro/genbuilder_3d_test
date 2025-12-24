import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .geometry import Mesh
from .utils import ensure_dir

LOGGER = logging.getLogger(__name__)


@dataclass
class ExportResult:
    glb_path: Path


def export_glb(mesh: Mesh, texture_paths: Dict[str, Path], output_path: Path) -> ExportResult:
    try:
        import trimesh
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("trimesh unavailable (%s), writing JSON stub instead", exc)
        data = {
            "vertices": mesh.vertices,
            "faces": mesh.faces,
            "textures": {k: str(v) for k, v in texture_paths.items()},
        }
        output_path.write_text(json.dumps(data))
        return ExportResult(glb_path=output_path)

    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    tm.visual.vertex_colors = [200, 200, 200, 255]

    ensure_dir(output_path.parent)
    tm.export(output_path, file_type="glb")
    return ExportResult(glb_path=output_path)


def write_index(records: List[Dict[str, object]], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(records, indent=2))
    LOGGER.info("Index written to %s", output_path)
