import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw
import numpy as np

import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

from .model_downloader import ensure_sd15_controlnet
from .prompt_library import PromptLibrary
from .segmentation import FacadeLayout, MaskBundle, OpeningPlacement
from .utils import CachePaths, sha256_of_dict

LOGGER = logging.getLogger(__name__)


@dataclass
class TextureResult:
    base_color: Path
    roughness: Optional[Path]
    normal: Optional[Path]


@dataclass
class FacadeTextures(TextureResult):
    wall_base: Path
    opening_variants: Dict[str, List[Path]]
    windows_applied: Path
    roof: Path


TEXTURE_CACHE_VERSION = "facade-unfold-v1"


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

    def _opening_control_image(self, size: tuple[int, int], kind: str) -> Image.Image:
        base = Image.new("L", size, 0)
        draw = ImageDraw.Draw(base)
        inset = max(2, min(size) // 15)
        shape_bounds = [(inset, inset), (size[0] - inset, size[1] - inset)]
        fill_value = 180 if kind == "window" else 220
        draw.rectangle(shape_bounds, fill=fill_value)
        if kind == "balcony":
            rail_height = max(4, size[1] // 6)
            draw.rectangle([(inset, size[1] - rail_height), (size[0] - inset, size[1] - inset)], fill=255)
        control = Image.merge("RGB", (base, base, base))
        return control

    def _cache_key(self, tag: str, wall_size: tuple[int, int], metadata: Dict[str, str]) -> str:
        prompt_context = {"tag": tag, **metadata}
        return sha256_of_dict(
            {
                "wall": wall_size,
                "meta": prompt_context,
                "version": TEXTURE_CACHE_VERSION,
            }
        )

    def _render_material(
        self,
        masks: MaskBundle,
        wall_size: tuple[int, int],
        metadata: Dict[str, str],
        tag: str,
        dry_run: bool,
    ) -> tuple[Path, Path, Path]:
        cache_key = self._cache_key(tag, wall_size, metadata)
        base_path = self.cache_paths.texture_dir() / f"{tag}_base_{cache_key}.png"
        roughness_path = self.cache_paths.texture_dir() / f"{tag}_roughness_{cache_key}.png"
        normal_path = self.cache_paths.texture_dir() / f"{tag}_normal_{cache_key}.png"

        if base_path.exists() and roughness_path.exists() and normal_path.exists():
            LOGGER.info("Using cached %s textures for %s", tag, cache_key)
            return base_path, roughness_path, normal_path

        if dry_run:
            raise RuntimeError(
                "Texture synthesis requested in dry-run mode; real model generation is required now that placeholders are removed."
            )

        pipeline = self._load_pipeline()
        recipe = self._select_recipe(metadata)
        prompt = self._build_prompt(recipe, {"recipe": recipe, **metadata})
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
        LOGGER.info("Generated %s textures using ControlNet pipeline for %s", tag, cache_key)
        return base_path, roughness_path, normal_path

    def _render_opening_texture(
        self,
        size: Tuple[int, int],
        kind: str,
        metadata: Dict[str, str],
        variant: int,
        dry_run: bool,
    ) -> Path:
        cache_key = sha256_of_dict(
            {
                "kind": kind,
                "size": size,
                "variant": variant,
                "meta": metadata,
                "version": TEXTURE_CACHE_VERSION,
            }
        )
        base_path = self.cache_paths.texture_dir() / f"{kind}_variant_{variant}_{cache_key}.png"
        if base_path.exists():
            return base_path

        if dry_run:
            raise RuntimeError(
                "Texture synthesis requested in dry-run mode; real model generation is required now that placeholders are removed."
            )

        pipeline = self._load_pipeline()
        prompt = f"Realistic {kind} texture with architectural detailing, photorealistic, clean materials"
        control_image = self._opening_control_image(size, kind)
        generator = torch.Generator(device=self.device).manual_seed(self.seed + variant)
        result = pipeline(
            prompt=prompt,
            image=control_image,
            num_inference_steps=15,
            guidance_scale=6.0,
            generator=generator,
        )
        result.images[0].save(base_path)
        return base_path

    def _blank_openings_mask(self, wall_size: tuple[int, int]) -> Path:
        width, height = wall_size
        blank_path = self.cache_paths.texture_dir() / f"blank_{width}x{height}.png"
        if not blank_path.exists():
            Image.new("L", (width, height), 0).save(blank_path)
        return blank_path

    def generate_wall_base(
        self, wall_size: tuple[int, int], layout: FacadeLayout, metadata: Dict[str, str], dry_run: bool = False
    ) -> tuple[Path, Path, Path]:
        empty_openings = self._blank_openings_mask(wall_size)
        neutral_masks = MaskBundle(plinth=layout.plinth, floors=layout.floors, openings=empty_openings)
        return self._render_material(neutral_masks, wall_size, metadata, tag="facade", dry_run=dry_run)

    def _variant_cache_key(self, layout: FacadeLayout, metadata: Dict[str, str]) -> str:
        return sha256_of_dict(
            {
                "seams": layout.seam_positions,
                "placements": [(p.x, p.y, p.width, p.height, p.kind, p.variant) for p in layout.placements],
                "meta": metadata,
                "version": TEXTURE_CACHE_VERSION,
            }
        )

    def generate_opening_variants(
        self, layout: FacadeLayout, metadata: Dict[str, str], dry_run: bool = False
    ) -> Dict[str, List[Path]]:
        cache_key = self._variant_cache_key(layout, metadata)
        variant_paths: Dict[str, List[Path]] = defaultdict(list)

        unique_specs = {(p.kind, p.width, p.height) for p in layout.placements}
        for kind, width, height in unique_specs:
            key = f"{kind}:{width}x{height}"
            for variant in range(3):
                name = f"{key}:v{variant}"
                path = self._render_opening_texture((width, height), kind, metadata, variant, dry_run)
                variant_paths[key].append(path)
                LOGGER.info("Opening texture generated for %s with cache %s", name, cache_key)
        return variant_paths

    def compose_openings(
        self,
        base_texture: Path,
        layout: FacadeLayout,
        variant_paths: Dict[str, List[Path]],
        metadata: Dict[str, str],
    ) -> Path:
        cache_key = self._variant_cache_key(layout, metadata)
        composed_path = self.cache_paths.texture_dir() / f"facade_with_openings_{cache_key}.png"
        if composed_path.exists():
            return composed_path

        facade = Image.open(base_texture).convert("RGB")
        for placement in layout.placements:
            key = f"{placement.kind}:{placement.width}x{placement.height}"
            variant_list = variant_paths.get(key)
            if not variant_list:
                LOGGER.warning("No variant textures found for %s, skipping", key)
                continue
            variant_img = Image.open(variant_list[placement.variant % len(variant_list)]).convert("RGB")
            resized = variant_img.resize((placement.width, placement.height))
            facade.paste(resized, (placement.x, placement.y))

        facade.save(composed_path)
        return composed_path

    def generate_roof_texture(self, roof_size: tuple[int, int], metadata: Dict[str, str], dry_run: bool = False) -> Path:
        cache_key = sha256_of_dict({"roof": roof_size, "meta": metadata, "version": TEXTURE_CACHE_VERSION})
        roof_path = self.cache_paths.texture_dir() / f"roof_{cache_key}.png"
        if roof_path.exists():
            return roof_path

        if dry_run:
            raise RuntimeError(
                "Texture synthesis requested in dry-run mode; real model generation is required now that placeholders are removed."
            )

        base_color = tuple(int(140 + (hash(cache_key) % 40)) for _ in range(3))
        roof_image = Image.new("RGB", roof_size, base_color)
        ImageDraw.Draw(roof_image).line([(0, 0), (roof_size[0], roof_size[1])], fill=(80, 80, 80), width=2)
        roof_image.save(roof_path)
        return roof_path

    def generate_full_facade(
        self,
        wall_size: tuple[int, int],
        layout: FacadeLayout,
        roof_size: tuple[int, int],
        metadata: Dict[str, str],
        dry_run: bool = False,
    ) -> FacadeTextures:
        wall_base, roughness_path, normal_path = self.generate_wall_base(wall_size, layout, metadata, dry_run)
        variant_paths = self.generate_opening_variants(layout, metadata, dry_run)
        windows_applied = self.compose_openings(wall_base, layout, variant_paths, metadata)
        roof_texture = self.generate_roof_texture(roof_size, metadata, dry_run)

        return FacadeTextures(
            base_color=windows_applied,
            roughness=roughness_path,
            normal=normal_path,
            wall_base=wall_base,
            opening_variants=variant_paths,
            windows_applied=windows_applied,
            roof=roof_texture,
        )

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
