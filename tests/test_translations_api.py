from app.main import app


def test_openapi_contains_translation_routes():
    paths = app.openapi()["paths"]

    assert "/translations" in paths
    assert "/translations/{translation_id}" in paths
    assert "/languages" in paths


def test_languages_endpoint_returns_list():
    import asyncio

    from app.api.routes.translations import list_languages

    result = asyncio.run(list_languages())
    assert isinstance(result, list)
    assert len(result) >= 5
    ids = {item["id"] for item in result}
    assert "japanese" in ids
    assert "spanish" in ids
    for item in result:
        assert set(item.keys()) == {"id", "label"}
