import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from PIL import Image, ImageDraw

from .model_downloader import ensure_sd15_controlnet
from .prompt_library import PromptLibrary
from .segmentation import MaskBundle
from .utils import CachePaths, sha256_of_dict

LOGGER = logging.getLogger(__name__)


@dataclass
class TextureResult:
    base_color: Path
    roughness: Optional[Path]
    normal: Optional[Path]


TEXTURE_CACHE_VERSION = "no-placeholder"


class TextureGenerator:
    def __init__(
        self,
        cache_paths: CachePaths,
        device: str = "cpu",
        seed: int = 0,
        prompt_library_path: Path | None = None,
    ):
        self.cache_paths = cache_paths
        self.device = device
        self.seed = seed
        default_library = Path(__file__).resolve().parents[1] / "tex_prompts.yaml"
        self.prompt_library_path = prompt_library_path or default_library
        self.prompt_library = self._load_prompt_library()
        self.model_paths = ensure_sd15_controlnet(self.cache_paths.model_dir())

    def _load_prompt_library(self) -> Optional[PromptLibrary]:
        if self.prompt_library_path.exists():
            try:
                library = PromptLibrary.from_file(self.prompt_library_path)
                LOGGER.info(
                    "Loaded prompt library from %s with recipes: %s",
                    self.prompt_library_path,
                    ", ".join(library.recipe_names()) or "<none>",
                )
                return library
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Failed to load prompt library (%s)", exc)
        else:
            LOGGER.info("Prompt library not found at %s", self.prompt_library_path)
        return None

    def _select_recipe(self, metadata: Dict[str, str]) -> str:
        if self.prompt_library is None:
            return metadata.get("recipe", "default")

        requested = metadata.get("recipe")
        if requested and self.prompt_library.has_recipe(requested):
            return requested

        try:
            return self.prompt_library.default_recipe()
        except Exception:  # noqa: BLE001
            return requested or "default"

    def _placeholder_texture(self, size: tuple[int, int], label: str) -> Image.Image:
        img = Image.new("RGB", size, (180, 180, 180))
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), label, fill=(255, 255, 255))
        return img

    def synthesize_facade(
        self, wall_size: tuple[int, int], masks: MaskBundle, metadata: Dict[str, str], dry_run: bool = False
    ) -> TextureResult:
        recipe = self._select_recipe(metadata)
        prompt_context = {"recipe": recipe, **metadata}
        cache_key = sha256_of_dict(
            {
                "wall": wall_size,
                "meta": prompt_context,
                "version": TEXTURE_CACHE_VERSION,
            }
        )
        base_path = self.cache_paths.texture_dir() / f"base_{cache_key}.png"
        roughness_path = self.cache_paths.texture_dir() / f"roughness_{cache_key}.png"
        normal_path = self.cache_paths.texture_dir() / f"normal_{cache_key}.png"

        if base_path.exists() and roughness_path.exists() and normal_path.exists():
            LOGGER.info("Using cached textures for %s", cache_key)
            return TextureResult(base_color=base_path, roughness=roughness_path, normal=normal_path)

        if dry_run:
            raise RuntimeError(
                "Texture synthesis requested in dry-run mode; real model generation is required now that placeholders are removed."
            )

        try:
            # Import solely to ensure the dependency is present. Actual pipeline
            # invocation must be wired by the caller; we deliberately refuse to
            # fabricate placeholder textures.
            from diffusers import StableDiffusionControlNetPipeline  # type: ignore  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Diffusion pipeline unavailable; cannot synthesize facade textures without the model."
            ) from exc

        raise RuntimeError(
            "Texture synthesis is mandatory but no diffusion call was performed; integrate the model execution to proceed."
        )
