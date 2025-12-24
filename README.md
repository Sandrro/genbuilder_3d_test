# Genbuilder 3D Test

Pipeline for generating textured 3D building models from GeoJSON footprints.

## Quick start

```
pip install -r requirements.txt  # if you maintain a requirements file
python -m genbuilder.cli run input.geojson output_dir --texel-density 256 --dry-run-geometry
```

The CLI builds a `GenParams` configuration with the tunable options (texel density, model, device, seed, batch size, dry-run)
and passes it into the pipeline so all configurable knobs live in one place.

## Notes
- The pipeline validates polygons, reprojects to a metric CRS, extrudes meshes, assigns UVs, and synthesizes placeholder textures unless diffusion is available.
- Unit tests cover UV mapping continuity and mask generation.
