from pathlib import Path

from PIL import Image

from genbuilder.segmentation import SegmentationGenerator


def test_segmentation_masks(tmp_path: Path):
    generator = SegmentationGenerator(texel_density=10)
    layout = generator.generate(
        wall_size=(200, 300),
        properties={"floors_count": 3, "floor_height": 3.0},
        output_dir=tmp_path,
    )

    for mask_path in [layout.plinth, layout.floors, layout.openings]:
        assert mask_path.exists()

    plinth_img = Image.open(layout.plinth)
    pixels = plinth_img.getdata()
    assert sum(pixels) > 0

    assert layout.placements, "Should mark window and door placements"
    # ensure openings avoid seams at 0 and width
    width, _ = (200, 300)
    for placement in layout.placements:
        assert placement.x > 0
        assert placement.x + placement.width < width
