import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from PIL import Image, ImageDraw
import numpy as np

import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

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
        self._pipeline: StableDiffusionControlNetPipeline | None = None

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

    def _load_pipeline(self) -> StableDiffusionControlNetPipeline:
        if self._pipeline is not None:
            return self._pipeline

        base_path, controlnet_path = self.model_paths
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=dtype)
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            base_path,
            controlnet=controlnet,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline.to(self.device)
        self._pipeline = pipeline
        return pipeline

    def _build_prompt(self, recipe: str, metadata: Dict[str, str]) -> str:
        if self.prompt_library and self.prompt_library.has_recipe(recipe):
            try:
                recipe_data = self.prompt_library.get_recipe(recipe)
                prompt = recipe_data.get("prompt") or recipe
                try:
                    prompt = prompt.format(**metadata)
                except Exception:  # noqa: BLE001
                    pass
                return str(prompt)
            except Exception:  # noqa: BLE001
                return f"Facade texture, recipe {recipe}"
        return f"Facade texture, recipe {recipe}"

    def _compose_control_image(self, masks: MaskBundle) -> Image.Image:
        plinth = Image.open(masks.plinth).convert("L")
        floors = Image.open(masks.floors).convert("L")
        openings = Image.open(masks.openings).convert("L")

        combined = np.maximum(np.array(plinth), np.array(floors))
        combined = np.maximum(combined, np.array(openings))
        control = Image.fromarray(combined).convert("RGB")
        return control

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

        pipeline = self._load_pipeline()
        prompt = self._build_prompt(recipe, prompt_context)
        control_image = self._compose_control_image(masks)

        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        result = pipeline(
            prompt=prompt,
            image=control_image,
            num_inference_steps=20,
            guidance_scale=5.0,
            generator=generator,
        )

        base_image: Image.Image = result.images[0]
        roughness_image = Image.new("L", base_image.size, 128)
        normal_image = Image.merge(
            "RGB",
            (
                Image.new("L", base_image.size, 128),
                Image.new("L", base_image.size, 128),
                Image.new("L", base_image.size, 255),
            ),
        )

        base_image.save(base_path)
        roughness_image.save(roughness_path)
        normal_image.save(normal_path)

        LOGGER.info("Generated textures using ControlNet pipeline for %s", cache_key)
        return TextureResult(base_color=base_path, roughness=roughness_path, normal=normal_path)
