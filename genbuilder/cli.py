import json
from pathlib import Path
import typer

from .geo_pipeline import BuildingPipeline
from .params import GenParams
from .utils import setup_logging

app = typer.Typer(help="Convert building GeoJSON into textured 3D Mapbox-ready models.")


@app.command()
def run(
    input_path: Path = typer.Argument(..., exists=True, help="Path to GeoJSON FeatureCollection"),
    output_dir: Path = typer.Argument(..., help="Directory for generated assets"),
    texel_density: float = typer.Option(512.0, help="Texels per meter for UV atlas"),
    model: str = typer.Option("sdxl", help="Texture model choice (sdxl/flux)"),
    seed: int = typer.Option(0, help="Deterministic seed"),
    batch_size: int = typer.Option(1, help="Not used yet; reserved for batching"),
    device: str = typer.Option("cpu", help="Device for diffusion pipeline"),
    dry_run_geometry: bool = typer.Option(False, help="Skip heavy texturing; geometry only"),
):
    setup_logging()
    params = GenParams(
        texel_density=texel_density,
        model=model,
        device=device,
        seed=seed,
        batch_size=batch_size,
        dry_run_geometry=dry_run_geometry,
    )
    pipeline = BuildingPipeline(params=params)
    index_path = pipeline.run(input_path, output_dir)
    typer.echo(json.dumps({"index": str(index_path)}, indent=2))


if __name__ == "__main__":
    app()
