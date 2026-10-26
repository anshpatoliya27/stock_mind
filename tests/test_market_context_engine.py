"""Unit and integration tests for MarketContextEngine and Multimodal Flow."""

from __future__ import annotations

import io
from pathlib import Path
import pytest
from PIL import Image, ImageDraw

from src.agents.financial_agent import StockMindAgent
from src.agents.market_context_engine import MarketContextEngine
from src.agents.provider_adapter import MultimodalLLMAdapter

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def engine():
    return MarketContextEngine()


def test_get_macro_context(engine: MarketContextEngine):
    macro = engine.get_macro_context()
    assert macro["rbi_repo_rate"] == "5.50%"
    assert macro["rbi_sdf_rate"] == "5.25%"
    assert "usd_inr_rate" in macro
    assert float(macro["usd_inr_rate"]) > 80.0


def test_get_latest_ticker_price_real_data(engine: MarketContextEngine):
    res = engine.get_latest_ticker_price("RELIANCE")
    assert res["symbol"] == "RELIANCE"
    assert res["price"] > 0
    assert "date" in res
    assert "source" in res


def test_get_latest_ticker_price_default(engine: MarketContextEngine):
    res = engine.get_latest_ticker_price(None)
    assert res["symbol"] == "NIFTY 50"
    assert round(res["price"], 0) == 24850.0


def test_extract_ticker_from_text():
    assert MarketContextEngine.extract_ticker_from_text("Analyze RELIANCE please") == "RELIANCE"
    assert MarketContextEngine.extract_ticker_from_text("What is outlook for TCS and INFY?") == "TCS"
    assert MarketContextEngine.extract_ticker_from_text("General market forecast") is None


def test_analyze_chart_image_synthetic(engine: MarketContextEngine):
    # Create green image (Bullish)
    img = Image.new("RGB", (100, 100), color=(0, 200, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    res = engine.analyze_chart_image(buf.getvalue())

    assert res["success"] is True
    assert res["market_bias"] == "BULLISH"
    assert "Double Bottom" in res["detected_pattern"]
    assert res["pattern_confidence_pct"] > 70.0


def test_analyze_chart_image_sample_file(engine: MarketContextEngine):
    sample_path = PROJECT_ROOT / "Dataset" / "chart_images" / "Patterns" / "0_00012.jpg"
    if sample_path.exists():
        with open(sample_path, "rb") as f:
            res = engine.analyze_chart_image(f.read())
        assert res["success"] is True
        assert res["detected_pattern"] in ["Double Top (Bearish Reversal)", "Double Bottom (Bullish Reversal)", "Consolidation / Symmetrical Breakout"]
        assert res["pattern_confidence_pct"] >= 70.0


def test_calculate_price_forecast(engine: MarketContextEngine):
    forecast = engine.calculate_price_forecast(
        ticker="RELIANCE",
        pattern_data={"market_bias": "BULLISH"},
        base_price=1000.0,
    )
    assert forecast["base_reference_price"] == 1000.0
    assert forecast["short_term_target"] > 1000.0
    assert forecast["recommended_stop_loss"] < 1000.0
    assert "1 :" in forecast["risk_reward_ratio"]


def test_build_synthesis_report(engine: MarketContextEngine):
    report = engine.build_synthesis_report(
        prompt="What will be future price of Reliance after chart pattern?",
        image_bytes=None,
        ticker="RELIANCE",
    )
    assert "StockMind AI Analysis: RELIANCE" in report
    assert "Technical Chart Pattern & Structure Analysis" in report
    assert "Multi-Horizon Quantitative Price Forecast" in report
    assert "Target 1 (Primary)" in report
    assert "RBI Policy Repo Rate" in report


@pytest.mark.asyncio
async def test_financial_agent_streaming_local_mode(engine: MarketContextEngine):
    agent = StockMindAgent(adapter=MultimodalLLMAdapter(api_key=""), engine=engine)
    tokens = []
    async for chunk in agent.chat_stream("What is the forecast for Tata Motors?"):
        tokens.append(chunk)

    full_output = "".join(tokens)
    assert "StockMind AI Analysis" in full_output
    assert "Multi-Horizon Quantitative Price Forecast" in full_output
    assert len(tokens) > 5  # Confirms real-time token streaming
