from app.main import app
from app.api.routes.health import health
from app.core.config import Settings


async def test_health_endpoint():
    response = await health(Settings())

    assert response["status"] == "ok"


def test_openapi_contains_conversation_routes():
    paths = app.openapi()["paths"]

    assert "/conversations" in paths
    assert "/conversations/{conversation_id}/messages" in paths
