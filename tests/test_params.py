from genbuilder.params import GenParams


def test_placeholder_metadata_includes_optional_hints():
    params = GenParams(
        city_hint="Vienna",
        climate_hint="humid",
        era_hint="1900s",
        style_hint="art nouveau",
        material_hint="limestone",
        seed_hint="custom",
        recipe="brick_classic_city",
    )

    metadata = params.placeholder_metadata(floors_count=5, floor_height=3.2)

    assert metadata["city_hint"] == "Vienna"
    assert metadata["climate_hint"] == "humid"
    assert metadata["era_hint"] == "1900s"
    assert metadata["style_hint"] == "art nouveau"
    assert metadata["material_hint"] == "limestone"
    assert metadata["seed_hint"] == "custom"
    assert metadata["recipe"] == "brick_classic_city"
    assert metadata["floor_count"] == "5"
    assert metadata["floor_height_m"] == "3.2"


def test_placeholder_metadata_skips_empty_values():
    params = GenParams()

    metadata = params.placeholder_metadata(floors_count=2, floor_height=3.0)

    assert metadata == {"floor_count": "2", "floor_height_m": "3.0"}
