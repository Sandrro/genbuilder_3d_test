from pathlib import Path

from PIL import Image

from genbuilder.segmentation import SegmentationGenerator


def test_segmentation_masks(tmp_path: Path):
    generator = SegmentationGenerator(texel_density=10)
    masks = generator.generate(
        wall_size=(200, 300),
        properties={"floors_count": 3, "floor_height": 3.0},
        output_dir=tmp_path,
    )

    for mask_path in [masks.plinth, masks.floors, masks.openings]:
        assert mask_path.exists()

    plinth_img = Image.open(masks.plinth)
    pixels = plinth_img.getdata()
    assert sum(pixels) > 0
