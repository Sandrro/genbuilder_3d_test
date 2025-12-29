# Genbuilder 3D Test

Pipeline for generating textured 3D building models from GeoJSON footprints.

## Quick start

```
pip install -r requirements.txt  # if you maintain a requirements file
python -m genbuilder.cli run input.geojson output_dir --texel-density 256 --dry-run-geometry
```

The CLI builds a `GenParams` configuration with the tunable options (texel density, device, seed, batch size, dry-run)
and passes it into the pipeline so all configurable knobs live in one place. The texture generator now targets the
SD 1.5 (pruned EMA-only weights from `sd-legacy/stable-diffusion-v1-5`) + ControlNet pipeline; models are downloaded
automatically to the repository on first use via `genbuilder.model_downloader`.

To run texture synthesis through a remote ComfyUI instance instead of the built-in diffusers pipeline, set
`GenParams(use_comfyui=True, comfyui_base_url="http://127.0.0.1:8188")`. Optional fields let you provide custom
checkpoint and ControlNet names if they differ from the defaults.

## Notes
- The pipeline validates polygons, reprojects to a metric CRS, extrudes meshes, assigns UVs, and synthesizes placeholder textures unless diffusion is available.
- Unit tests cover UV mapping continuity and mask generation.
