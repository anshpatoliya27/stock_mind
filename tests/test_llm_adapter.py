"""Unit tests for Multimodal LLM Adapter & Agent."""

from __future__ import annotations

import pytest

from src.agents.financial_agent import StockMindAgent
from src.agents.provider_adapter import (
    MissingProviderCredentialsError,
    MultimodalLLMAdapter,
)


def test_adapter_initialization_defaults():
    adapter = MultimodalLLMAdapter(provider="gemini", model="gemini-1.5-flash")
    assert adapter.provider == "gemini"
    assert adapter.model == "gemini-1.5-flash"


def test_adapter_grok_defaults():
    adapter = MultimodalLLMAdapter(provider="grok")
    assert adapter.provider == "grok"
    assert adapter.model == "grok-2-vision-1212"


def test_adapter_missing_key_detection():
    # Explicitly empty key
    adapter = MultimodalLLMAdapter(provider="gemini", api_key="")
    assert adapter.is_configured() is False
    status = adapter.get_status()
    assert status["is_configured"] is False
    assert status["setup_hint"] is not None


def test_adapter_grok_missing_key(monkeypatch):
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    adapter = MultimodalLLMAdapter(provider="grok", api_key="")
    assert adapter.is_configured() is False
    status = adapter.get_status()
    assert "GROK_API_KEY" in status["setup_hint"]


def test_adapter_with_key():
    adapter = MultimodalLLMAdapter(provider="gemini", api_key="AIzaSyDummyKeyForTesting12345")
    assert adapter.is_configured() is True
    status = adapter.get_status()
    assert status["is_configured"] is True
    assert status["key_preview"] == "AIza...2345"


def test_adapter_grok_with_key():
    adapter = MultimodalLLMAdapter(provider="grok", api_key="xai-dummy-grok-key-12345678")
    assert adapter.is_configured() is True
    status = adapter.get_status()
    assert status["is_configured"] is True
    assert status["provider"] == "grok"


@pytest.mark.asyncio
async def test_adapter_stream_raises_on_missing_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    adapter = MultimodalLLMAdapter(provider="gemini", api_key="")
    with pytest.raises(MissingProviderCredentialsError) as exc_info:
        async for _ in adapter.generate_stream(prompt="What is Nifty P/E ratio?"):
            pass
    assert "GEMINI_API_KEY" in str(exc_info.value)


@pytest.mark.asyncio
async def test_adapter_grok_stream_raises_on_missing_key(monkeypatch):
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    adapter = MultimodalLLMAdapter(provider="grok", api_key="")
    with pytest.raises(MissingProviderCredentialsError) as exc_info:
        async for _ in adapter.generate_stream(prompt="Analyze this chart"):
            pass
    assert "GROK_API_KEY" in str(exc_info.value)


def test_financial_agent_status():
    adapter = MultimodalLLMAdapter(provider="openai", api_key="")
    agent = StockMindAgent(adapter=adapter)
    status = agent.get_status()
    assert status["provider"] == "openai"
    assert status["is_configured"] is False
