from pathlib import Path

from PIL import Image

from genbuilder.exporter import export_textures


def test_export_textures(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (4, 4), (255, 0, 0)).save(source)

    output_dir = tmp_path / "textures"
    exported = export_textures({"baseColor": source}, output_dir, "feature123")

    expected_path = output_dir / "feature123_baseColor.png"
    assert exported["baseColor"] == expected_path
    assert expected_path.exists()
