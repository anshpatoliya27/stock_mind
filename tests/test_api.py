"""Unit tests for StockMind AI FastAPI Backend."""

from __future__ import annotations

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_check(client: TestClient):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["app"] == "StockMind AI"


def test_system_status(client: TestClient):
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert "llm" in data
    assert "dataset" in data
    assert "supported_providers" in data["llm"]


def test_conversation_lifecycle(client: TestClient):
    # 1. Create
    res_create = client.post("/api/conversations", data={"title": "Nifty Analysis Test"})
    assert res_create.status_code == 200
    conv = res_create.json()
    conv_id = conv["id"]
    assert conv["title"] == "Nifty Analysis Test"

    # 2. List
    res_list = client.get("/api/conversations")
    assert res_list.status_code == 200
    items = res_list.json()
    assert any(c["id"] == conv_id for c in items)

    # 3. Retrieve
    res_get = client.get(f"/api/conversations/{conv_id}")
    assert res_get.status_code == 200
    assert res_get.json()["id"] == conv_id

    # 4. Delete
    res_del = client.delete(f"/api/conversations/{conv_id}")
    assert res_del.status_code == 200

    # 5. Verify gone
    res_gone = client.get(f"/api/conversations/{conv_id}")
    assert res_gone.status_code == 404


def test_chat_endpoint_missing_credentials(client: TestClient):
    # Without an API key, chat endpoint must return SSE error explaining setup, NOT fake data!
    response = client.post(
        "/api/chat",
        data={"message": "What is the trend for Reliance?"},
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    body = response.text
    # SSE chunk should contain init and error
    assert "data: " in body
    assert "MISSING_CREDENTIALS" in body or "token" in body


def test_serve_index_html(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert "<title>StockMind AI" in response.text


def test_get_market_ticker_tape(client: TestClient):
    response = client.get("/api/market/ticker_tape")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 5
    symbols = [it["symbol"] for it in data]
    assert "NIFTY 50" in symbols
    assert "RELIANCE" in symbols


def test_update_settings(client: TestClient, monkeypatch):
    import os
    from pathlib import Path
    env_file = Path(".env")
    original_env = env_file.read_text(encoding="utf-8") if env_file.exists() else None

    try:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        response = client.post(
            "/api/settings",
            data={"provider": "gemini", "api_key": "test-gemini-key-12345", "model": "gemini-3.5-flash"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["provider"] == "gemini"
        assert data["model"] == "gemini-3.5-flash"
        assert data["is_configured"] is True
    finally:
        if original_env is not None:
            env_file.write_text(original_env, encoding="utf-8")
        os.environ.pop("GEMINI_API_KEY", None)
