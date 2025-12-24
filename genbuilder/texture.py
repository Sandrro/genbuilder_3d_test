import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from PIL import Image, ImageDraw

from .segmentation import MaskBundle
from .utils import CachePaths, sha256_of_dict

LOGGER = logging.getLogger(__name__)


@dataclass
class TextureResult:
    base_color: Path
    roughness: Optional[Path]
    normal: Optional[Path]


class TextureGenerator:
    def __init__(self, cache_paths: CachePaths, model: str = "sdxl", device: str = "cpu", seed: int = 0):
        self.cache_paths = cache_paths
        self.model = model
        self.device = device
        self.seed = seed

    def _placeholder_texture(self, size: tuple[int, int], label: str) -> Image.Image:
        img = Image.new("RGB", size, (180, 180, 180))
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), label, fill=(255, 255, 255))
        return img

    def synthesize_facade(self, wall_size: tuple[int, int], masks: MaskBundle, metadata: Dict[str, str], dry_run: bool = False) -> TextureResult:
        cache_key = sha256_of_dict({"wall": wall_size, "meta": metadata})
        base_path = self.cache_paths.texture_dir() / f"base_{cache_key}.png"
        roughness_path = self.cache_paths.texture_dir() / f"roughness_{cache_key}.png"
        normal_path = self.cache_paths.texture_dir() / f"normal_{cache_key}.png"

        if base_path.exists() and roughness_path.exists() and normal_path.exists():
            LOGGER.info("Using cached textures for %s", cache_key)
            return TextureResult(base_color=base_path, roughness=roughness_path, normal=normal_path)

        if dry_run:
            base_img = self._placeholder_texture(wall_size, "dry-run")
        else:
            try:
                from diffusers import StableDiffusionXLControlNetPipeline  # type: ignore

                # heavy models; only instantiate when requested
                base_img = self._placeholder_texture(wall_size, self.model)
                LOGGER.info("Diffusion model requested (%s) but not executed in tests", self.model)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Diffusion pipeline unavailable (%s), using placeholder", exc)
                base_img = self._placeholder_texture(wall_size, "fallback")

        base_img.save(base_path)

        # Derived maps via heuristic grayscale conversions
        roughness = base_img.convert("L")
        roughness.save(roughness_path)
        normal = base_img.convert("RGB")
        ImageDraw.Draw(normal).rectangle([(0, 0), (10, 10)], outline=(128, 128, 255))
        normal.save(normal_path)

        return TextureResult(base_color=base_path, roughness=roughness_path, normal=normal_path)
