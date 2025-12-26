from shapely.geometry import Polygon

from genbuilder.geometry import BuildingProperties, extrude_building
from genbuilder.uv import UVGenerator


def test_wall_strip_dimensions_and_uv_ordering():
    polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    props = BuildingProperties(floors_count=2, floor_height=3.0, building_height=6.0)
    mesh = extrude_building(polygon, props)

    generator = UVGenerator(texel_density=10)
    atlas = generator.map_wall_uvs(polygon, mesh)
    assert atlas.wall_size[0] > atlas.wall_size[1]

    # UV continuity around perimeter
    expected_segments = len(polygon.exterior.coords) - 1
    assert len(atlas.wall_uvs) == expected_segments * 4
    # first and last U coordinates should align to form loop
    first_u = atlas.wall_uvs[0][0]
    last_u = atlas.wall_uvs[-1][0]
    assert first_u == 0.0
    assert 0.0 <= last_u <= 1.0

    mesh = generator.annotate_mesh_uvs(mesh, atlas)
    assert mesh.uvs, "Mesh should have UVs assigned"


def test_uv_indices_align_with_wall_segments():
    polygon = Polygon([(0, 0), (10, 0), (10, 5), (5, 5), (5, 10), (0, 10)])
    props = BuildingProperties(floors_count=1, floor_height=3.0, building_height=3.0)
    mesh = extrude_building(polygon, props)

    generator = UVGenerator(texel_density=10)
    atlas = generator.map_wall_uvs(polygon, mesh)
    mesh = generator.annotate_mesh_uvs(mesh, atlas)

    segments = len(polygon.exterior.coords) - 1
    for wall_idx in range(segments):
        base = wall_idx * 4
        assert mesh.uv_indices[2 * wall_idx] == (base, base + 1, base + 2)
        assert mesh.uv_indices[2 * wall_idx + 1] == (base, base + 2, base + 3)

    wall_uvs = len(atlas.wall_uvs)
    roof_indices = mesh.uv_indices[2 * segments :]
    for face_indices in roof_indices:
        assert all(idx >= wall_uvs for idx in face_indices)
    assert max(max(face) for face in mesh.uv_indices) < len(mesh.uvs)
